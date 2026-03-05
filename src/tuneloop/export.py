"""Conversation reconstruction and dataset export for SFT/DPO training."""

from __future__ import annotations

import json
from pathlib import Path

from sqlmodel import Session, select

from tuneloop.db import get_engine, init_db
from tuneloop.models import Session as ChatSession, Message


def _reconstruct_conversation(db: Session, session_id: str) -> list[dict]:
    """Reconstruct a deduplicated conversation from proxy-logged messages.

    The proxy logs every message from the request body on each API call,
    creating duplicates across turns. Strategy:
    1. Find the last assistant message in the session.
    2. Read raw_request["messages"] from the user message just before it
       — this contains the full deduplicated history up to that point.
    3. Append the final assistant response.
    4. Fallback: deduplicate by (role, content) pairs if raw_request is missing.
    """
    msgs = db.exec(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at)
    ).all()

    if not msgs:
        return []

    # Find the last assistant message
    last_assistant = None
    last_assistant_idx = -1
    for i, m in enumerate(msgs):
        if m.role == "assistant":
            last_assistant = m
            last_assistant_idx = i

    if last_assistant is None:
        return []

    # Skip error responses
    if last_assistant.raw_response and "error" in last_assistant.raw_response:
        return []

    # Find the user message just before the last assistant
    last_user_before = None
    for i in range(last_assistant_idx - 1, -1, -1):
        if msgs[i].role == "user":
            last_user_before = msgs[i]
            break

    # Try to reconstruct from raw_request
    if last_user_before and last_user_before.raw_request:
        raw_msgs = last_user_before.raw_request.get("messages", [])
        if raw_msgs:
            conversation = [
                {"role": m["role"], "content": m.get("content", "")}
                for m in raw_msgs
            ]
            # Append the final assistant response
            assistant_content = last_assistant.content
            if last_assistant.raw_response:
                resp = last_assistant.raw_response
                if resp.get("streamed"):
                    assistant_content = resp.get("content", assistant_content)
                else:
                    choice = resp.get("choices", [{}])[0]
                    assistant_content = choice.get("message", {}).get(
                        "content", assistant_content
                    )
            conversation.append({"role": "assistant", "content": assistant_content})
            return conversation

    # Fallback: deduplicate by (role, content) pairs preserving order
    seen: set[tuple[str, str]] = set()
    conversation = []
    for m in msgs:
        key = (m.role, m.content)
        if key not in seen:
            seen.add(key)
            conversation.append({"role": m.role, "content": m.content})
    return conversation


def get_conversations() -> list[list[dict]]:
    """Return all reconstructed conversations from the database.

    Skips judge_eval sessions so evaluation data doesn't leak into training.
    """
    init_db()
    conversations = []
    with Session(get_engine()) as db:
        sessions = db.exec(
            select(ChatSession).order_by(ChatSession.created_at)
        ).all()
        for session in sessions:
            # Skip judge evaluation sessions
            if session.metadata_ and session.metadata_.get("type") == "judge_eval":
                continue
            conv = _reconstruct_conversation(db, session.id)
            if len(conv) >= 2:  # Need at least one user + one assistant
                # Skip conversations with empty assistant content
                if all(m["content"] for m in conv):
                    conversations.append(conv)
    return conversations


def format_sft(conversations: list[list[dict]]) -> list[dict]:
    """Format conversations for TRL SFTTrainer.

    Returns list of {"messages": [{"role": ..., "content": ...}, ...]}
    """
    return [{"messages": conv} for conv in conversations]


def format_dpo(
    conversations: list[list[dict]], strategy: str = "all"
) -> list[dict]:
    """Format conversations for TRL DPOTrainer.

    Each example has:
      prompt: list of message dicts (context)
      chosen: list of message dicts (better response)
      rejected: list of message dicts (worse response)

    Strategies:
      "first_last" — one pair per session: first draft vs final refined poem
      "consecutive" — pairs for each adjacent refinement step
      "all" — both strategies combined
    """
    examples = []

    for conv in conversations:
        # Extract assistant responses with their positions
        assistant_turns = []
        for i, msg in enumerate(conv):
            if msg["role"] == "assistant":
                assistant_turns.append((i, msg))

        if len(assistant_turns) < 2:
            continue

        if strategy in ("first_last", "all"):
            # First user message as prompt, first response=rejected, last=chosen
            first_user_idx = next(
                (i for i, m in enumerate(conv) if m["role"] == "user"), None
            )
            if first_user_idx is not None:
                prompt_msgs = [conv[first_user_idx]]
                first_resp = assistant_turns[0][1]
                last_resp = assistant_turns[-1][1]
                if first_resp["content"] != last_resp["content"]:
                    examples.append({
                        "prompt": prompt_msgs,
                        "chosen": [last_resp],
                        "rejected": [first_resp],
                    })

        if strategy in ("consecutive", "all"):
            # For each pair of adjacent assistant turns, the later one is preferred
            for j in range(len(assistant_turns) - 1):
                prev_idx, prev_resp = assistant_turns[j]
                next_idx, next_resp = assistant_turns[j + 1]
                if prev_resp["content"] == next_resp["content"]:
                    continue
                # Prompt = everything up to the tweak instruction (the user message
                # that triggered the next assistant response)
                prompt_end = next_idx
                for k in range(next_idx - 1, prev_idx, -1):
                    if conv[k]["role"] == "user":
                        prompt_end = k + 1
                        break
                prompt_msgs = conv[:prompt_end]
                examples.append({
                    "prompt": prompt_msgs,
                    "chosen": [next_resp],
                    "rejected": [prev_resp],
                })

    return examples


def export_jsonl(data: list[dict], output: Path) -> None:
    """Write a list of dicts as JSONL."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
