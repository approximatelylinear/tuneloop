#!/usr/bin/env python3
"""Generate ~100 multi-turn poem conversations through the tuneloop proxy.

Each conversation:
  1. Ask for a poem about a topic
  2. 1-3 follow-up tweaks (rhyme, shorter, fancier, etc.)

All conversations are logged by the proxy, ready for SFT/DPO export.

Usage:
    uv run python scripts/generate_poems.py
    uv run python scripts/generate_poems.py --count 20 --model qwen2.5:7b
"""

import argparse
import itertools
import json
import random
import sys
import uuid

import httpx

TOPICS = [
    "ships", "clouds", "building a house", "the ocean", "mountains",
    "trains", "bridges", "storms", "gardens", "rivers",
    "lighthouses", "forests", "the moon", "campfires", "harbors",
    "windmills", "cathedrals", "deserts", "snowfall", "sunrise",
]

OPENING_TEMPLATES = [
    "Write a poem about {topic}.",
    "Write me a short poem about {topic}.",
    "Can you write a poem about {topic}?",
    "I'd like a poem about {topic}.",
    "Compose a poem on the subject of {topic}.",
]

TWEAKS = [
    "Make it rhyme.",
    "Make it shorter — just four lines.",
    "Make it fancier, more literary.",
    "Rewrite it as a haiku.",
    "Make the tone darker and more dramatic.",
    "Make it cheerful and lighthearted.",
    "Add more vivid imagery.",
    "Rewrite it as a limerick.",
    "Use simpler language, like for a child.",
    "Make it sound ancient, like an old ballad.",
]


def generate_conversation(
    client: httpx.Client, proxy_url: str, model: str, topic: str
) -> dict:
    """Run one multi-turn poem conversation. Returns summary."""
    session_id = str(uuid.uuid4())
    headers = {"x-session-id": session_id, "Content-Type": "application/json"}

    # Pick opening
    opening = random.choice(OPENING_TEMPLATES).format(topic=topic)

    # Pick 1-3 random tweaks
    num_tweaks = random.randint(1, 3)
    tweaks = random.sample(TWEAKS, num_tweaks)

    messages = []
    turns = [opening] + tweaks

    for i, user_msg in enumerate(turns):
        messages.append({"role": "user", "content": user_msg})

        resp = client.post(
            proxy_url,
            json={"model": model, "messages": messages, "stream": False},
            headers=headers,
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()

        assistant_content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        messages.append({"role": "assistant", "content": assistant_content})

    return {
        "session_id": session_id,
        "topic": topic,
        "turns": len(turns),
        "messages": len(messages),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate poem training conversations")
    parser.add_argument("--count", type=int, default=100, help="Number of conversations")
    parser.add_argument("--model", default="qwen2.5:7b", help="Model name")
    parser.add_argument("--proxy", default="http://localhost:8000", help="Proxy URL")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    proxy_url = f"{args.proxy}/v1/chat/completions"

    # Cycle through topics to get even coverage
    topic_cycle = list(itertools.islice(itertools.cycle(TOPICS), args.count))
    random.shuffle(topic_cycle)

    print(f"Generating {args.count} conversations via {args.proxy} with {args.model}")
    print(f"Topics: {len(TOPICS)} unique, {args.count} conversations\n")

    with httpx.Client() as client:
        for i, topic in enumerate(topic_cycle):
            try:
                result = generate_conversation(client, proxy_url, args.model, topic)
                print(
                    f"[{i+1:3d}/{args.count}] {topic:20s} "
                    f"{result['turns']} turns  session={result['session_id'][:8]}"
                )
            except httpx.ConnectError:
                print("ERROR: Cannot connect to proxy. Is `tuneloop serve` running?")
                sys.exit(1)
            except Exception as e:
                print(f"[{i+1:3d}/{args.count}] {topic:20s} ERROR: {e}")

    print(f"\nDone. Check with: tuneloop sessions | tuneloop stats")


if __name__ == "__main__":
    main()
