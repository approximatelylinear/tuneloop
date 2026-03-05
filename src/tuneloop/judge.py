"""LLM-as-judge pipeline: generate responses, compare via blind evaluation, store verdicts."""

from __future__ import annotations

import json
import random

import httpx
from rich.console import Console
from rich.table import Table
from sqlmodel import Session

from tuneloop.config import OLLAMA_BASE_URL
from tuneloop.db import get_engine, init_db
from tuneloop.models import Session as ChatSession, Message, Judgment

console = Console()

JUDGE_SYSTEM_PROMPT = """\
You are an impartial judge comparing two AI assistant responses to the same prompt.
Evaluate both responses for quality, helpfulness, accuracy, and creativity.

You MUST respond with valid JSON only — no other text. Use this exact format:
{"score_a": <1-5>, "score_b": <1-5>, "winner": "a" or "b", "rationale": "<brief explanation>"}
"""


def generate_response(
    model: str,
    prompt: str,
    ollama_base_url: str = OLLAMA_BASE_URL,
) -> str:
    """Generate a single response from an Ollama model."""
    resp = httpx.post(
        f"{ollama_base_url}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def judge_pair(
    prompt: str,
    response_a: str,
    response_b: str,
    judge_model: str,
    ollama_base_url: str = OLLAMA_BASE_URL,
    max_retries: int = 2,
) -> dict:
    """Ask the judge model to compare two responses. Returns parsed verdict dict.

    Retries up to max_retries times if the judge output fails JSON parsing.
    """
    user_content = (
        f"**Prompt:**\n{prompt}\n\n"
        f"**Response A:**\n{response_a}\n\n"
        f"**Response B:**\n{response_b}"
    )

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    for attempt in range(1 + max_retries):
        resp = httpx.post(
            f"{ollama_base_url}/api/chat",
            json={
                "model": judge_model,
                "messages": messages,
                "stream": False,
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        raw = resp.json()["message"]["content"]

        verdict = _parse_verdict(raw)
        if not verdict.get("error"):
            return verdict

        if attempt < max_retries:
            console.print(f"[yellow]  Judge returned invalid JSON, retrying ({attempt + 1}/{max_retries})...[/yellow]")
            # Append the bad response and a correction prompt for the next attempt
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": "That was not valid JSON. Please respond with ONLY a JSON object in this format: {\"score_a\": <1-5>, \"score_b\": <1-5>, \"winner\": \"a\" or \"b\", \"rationale\": \"<brief explanation>\"}"})

    return verdict


def _parse_verdict(raw: str) -> dict:
    """Parse judge output, handling markdown code fences and malformed JSON."""
    text = raw.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        verdict = json.loads(text)
    except json.JSONDecodeError:
        return {
            "score_a": None,
            "score_b": None,
            "winner": None,
            "rationale": f"Failed to parse judge output: {raw[:200]}",
            "error": True,
        }

    # Validate required fields
    for key in ("score_a", "score_b", "winner"):
        if key not in verdict:
            verdict[key] = None

    return verdict


def run_judge(
    base_model: str,
    finetuned_model: str,
    judge_model: str | None = None,
    limit: int = 0,
    ollama_base_url: str = OLLAMA_BASE_URL,
) -> list[dict]:
    """Run the full judge pipeline: generate, compare, store, summarize.

    Returns a list of result dicts for each prompt evaluated.
    """
    from tuneloop.export import get_conversations

    if judge_model is None:
        judge_model = base_model

    init_db()

    # Extract unique first-user prompts from conversations
    conversations = get_conversations()
    prompts = []
    for conv in conversations:
        for msg in conv:
            if msg["role"] == "user":
                prompts.append(msg["content"])
                break

    if not prompts:
        console.print("[yellow]No prompts found in database.[/yellow]")
        return []

    if limit > 0:
        prompts = prompts[:limit]

    console.print(
        f"Judging [bold]{len(prompts)}[/bold] prompts: "
        f"[cyan]{base_model}[/cyan] vs [cyan]{finetuned_model}[/cyan] "
        f"(judge: [cyan]{judge_model}[/cyan])\n"
    )

    rng = random.Random(42)
    results = []

    for i, prompt in enumerate(prompts, 1):
        console.print(f"[dim]── Prompt {i}/{len(prompts)} ──[/dim]")
        console.print(f"[blue]{prompt[:120]}{'...' if len(prompt) > 120 else ''}[/blue]\n")

        # Generate from both models
        try:
            base_resp = generate_response(base_model, prompt, ollama_base_url)
        except Exception as e:
            console.print(f"[red]Error generating from {base_model}: {e}[/red]\n")
            results.append({"prompt": prompt, "error": str(e)})
            continue

        try:
            ft_resp = generate_response(finetuned_model, prompt, ollama_base_url)
        except Exception as e:
            console.print(f"[red]Error generating from {finetuned_model}: {e}[/red]\n")
            results.append({"prompt": prompt, "error": str(e)})
            continue

        # Randomize A/B order to avoid position bias
        swap = rng.random() < 0.5
        if swap:
            resp_a, resp_b = ft_resp, base_resp
        else:
            resp_a, resp_b = base_resp, ft_resp

        # Judge
        try:
            verdict = judge_pair(prompt, resp_a, resp_b, judge_model, ollama_base_url)
        except Exception as e:
            console.print(f"[red]Error from judge: {e}[/red]\n")
            results.append({"prompt": prompt, "error": str(e)})
            continue

        # Map scores back to base/finetuned
        if swap:
            base_score = verdict.get("score_b")
            ft_score = verdict.get("score_a")
            winner_raw = verdict.get("winner")
            if winner_raw == "a":
                winner = "finetuned"
            elif winner_raw == "b":
                winner = "base"
            else:
                winner = None
        else:
            base_score = verdict.get("score_a")
            ft_score = verdict.get("score_b")
            winner_raw = verdict.get("winner")
            if winner_raw == "a":
                winner = "base"
            elif winner_raw == "b":
                winner = "finetuned"
            else:
                winner = None

        result = {
            "prompt": prompt,
            "base_response": base_resp,
            "finetuned_response": ft_resp,
            "base_score": base_score,
            "finetuned_score": ft_score,
            "winner": winner,
            "rationale": verdict.get("rationale", ""),
            "error": verdict.get("error", False),
        }
        results.append(result)

        # Store in DB
        _store_verdict(
            prompt, base_resp, ft_resp, base_model, finetuned_model,
            judge_model, result,
        )

        # Print result
        w_style = {"base": "red", "finetuned": "green"}.get(winner or "", "yellow")
        console.print(f"  Base score: {base_score}  |  Fine-tuned score: {ft_score}  |  Winner: [{w_style}]{winner or 'error'}[/{w_style}]")
        console.print(f"  [dim]{verdict.get('rationale', '')}[/dim]\n")

    return results


def _store_verdict(
    prompt: str,
    base_resp: str,
    ft_resp: str,
    base_model: str,
    finetuned_model: str,
    judge_model: str,
    result: dict,
) -> None:
    """Store judge evaluation in the database."""
    with Session(get_engine()) as db:
        session = ChatSession(metadata_={"type": "judge_eval"})
        db.add(session)
        db.flush()

        # User message (the prompt)
        user_msg = Message(
            session_id=session.id,
            role="user",
            content=prompt,
        )
        db.add(user_msg)

        # Base model response
        base_msg = Message(
            session_id=session.id,
            role="assistant",
            content=base_resp,
            model=base_model,
        )
        db.add(base_msg)

        # Fine-tuned model response
        ft_msg = Message(
            session_id=session.id,
            role="assistant",
            content=ft_resp,
            model=finetuned_model,
        )
        db.add(ft_msg)
        db.flush()

        # Judgment on the fine-tuned message
        judgment = Judgment(
            message_id=ft_msg.id,
            score=result.get("finetuned_score"),
            labels={
                "base_score": result.get("base_score"),
                "winner": result.get("winner"),
                "base_model": base_model,
                "finetuned_model": finetuned_model,
            },
            rationale=result.get("rationale"),
            judge_model=judge_model,
        )
        db.add(judgment)
        db.commit()


def summarize_results(results: list[dict]) -> None:
    """Print a summary table of judge results."""
    if not results:
        console.print("[yellow]No results to summarize.[/yellow]")
        return

    ft_wins = sum(1 for r in results if r.get("winner") == "finetuned")
    base_wins = sum(1 for r in results if r.get("winner") == "base")
    errors = sum(1 for r in results if r.get("error"))
    ties = len(results) - ft_wins - base_wins - errors

    base_scores = [r["base_score"] for r in results if r.get("base_score") is not None]
    ft_scores = [r["finetuned_score"] for r in results if r.get("finetuned_score") is not None]

    avg_base = sum(base_scores) / len(base_scores) if base_scores else 0
    avg_ft = sum(ft_scores) / len(ft_scores) if ft_scores else 0

    table = Table(title="Judge Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    total = len(results)
    table.add_row("Total prompts", str(total))
    table.add_row("Fine-tuned wins", f"[green]{ft_wins}[/green] ({ft_wins/total*100:.0f}%)" if total else "0")
    table.add_row("Base wins", f"[red]{base_wins}[/red] ({base_wins/total*100:.0f}%)" if total else "0")
    table.add_row("Ties", str(ties))
    table.add_row("Errors", str(errors))
    table.add_row("Avg base score", f"{avg_base:.2f}")
    table.add_row("Avg fine-tuned score", f"{avg_ft:.2f}")

    console.print()
    console.print(table)
