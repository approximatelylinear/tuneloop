import json
import logging
import uuid

import httpx
from fastapi import FastAPI, Request, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlmodel import Session

from tuneloop.config import OLLAMA_BASE_URL
from tuneloop.db import init_db, get_session, get_engine
from tuneloop.models import Session as ChatSession, Message

log = logging.getLogger("tuneloop.proxy")

app = FastAPI(title="tuneloop proxy")


@app.on_event("startup")
def on_startup():
    # Configure logging in the server process (reload spawns a child)
    import os
    debug = os.environ.get("TUNELOOP_DEBUG") == "1"
    if not logging.root.handlers:
        logging.basicConfig(
            level=logging.DEBUG if debug else logging.INFO,
            format="%(levelname)s %(name)s: %(message)s",
        )
    init_db()


@app.get("/v1/models")
async def list_models():
    log.debug("GET /v1/models → forwarding to Ollama")
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{OLLAMA_BASE_URL}/v1/models")
        data = resp.json()
    log.debug("GET /v1/models ← %d models", len(data.get("data", [])))
    return data


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    background: BackgroundTasks,
    db: Session = Depends(get_session),
):
    body = await request.json()
    session_id = request.headers.get("x-session-id") or str(uuid.uuid4())

    msg_count = len(body.get("messages", []))
    last_msg = body.get("messages", [{}])[-1].get("content", "")[:80]
    log.debug(
        "POST /v1/chat/completions session=%s model=%s msgs=%d stream=%s last=%r",
        session_id[:8], body.get("model"), msg_count, body.get("stream", False), last_msg,
    )

    # Ensure session exists
    existing = db.get(ChatSession, session_id)
    if not existing:
        db.add(ChatSession(id=session_id))
        db.commit()

    model = body.get("model")
    stream = body.get("stream", False)

    if stream:
        return await _handle_stream(body, session_id, model, background)
    else:
        return await _handle_non_stream(body, session_id, model, db)


async def _handle_non_stream(
    body: dict, session_id: str, model: str | None, db: Session
):
    log.debug("→ Ollama (non-stream) model=%s", model)
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/v1/chat/completions",
            json=body,
            timeout=120.0,
        )
        data = resp.json()

    choice = data.get("choices", [{}])[0]
    assistant_msg = choice.get("message", {})
    usage = data.get("usage", {})
    log.debug(
        "← Ollama (non-stream) tokens=%s/%s content=%r",
        usage.get("prompt_tokens"), usage.get("completion_tokens"),
        assistant_msg.get("content", "")[:80],
    )

    # Log user messages
    for msg in body.get("messages", []):
        db.add(Message(
            session_id=session_id,
            role=msg["role"],
            content=msg.get("content", ""),
            model=model,
            raw_request=body,
        ))

    # Log assistant response
    db.add(Message(
        session_id=session_id,
        role=assistant_msg.get("role", "assistant"),
        content=assistant_msg.get("content", ""),
        model=model,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        raw_response=data,
    ))
    db.commit()

    return data


async def _handle_stream(
    body: dict, session_id: str, model: str | None, background: BackgroundTasks
):
    collected: list[str] = []
    buf = ""

    async def generate():
        nonlocal buf
        log.debug("→ Ollama (stream) model=%s", model)
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_BASE_URL}/v1/chat/completions",
                json=body,
                timeout=120.0,
            ) as resp:
                async for raw in resp.aiter_bytes():
                    yield raw
                    # Parse SSE lines from raw bytes for logging
                    buf += raw.decode("utf-8", errors="replace")
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.strip()
                        if line.startswith("data: ") and line != "data: [DONE]":
                            try:
                                chunk = json.loads(line[6:])
                                delta = (
                                    chunk.get("choices", [{}])[0]
                                    .get("delta", {})
                                    .get("content", "")
                                )
                                if delta:
                                    collected.append(delta)
                            except (json.JSONDecodeError, IndexError):
                                pass

        log.debug(
            "← Ollama (stream) collected %d chunks, %d chars",
            len(collected), sum(len(c) for c in collected),
        )

    # Schedule DB write as a background task so it runs even if client disconnects
    background.add_task(_log_stream_bg, body, session_id, model, collected)

    return StreamingResponse(generate(), media_type="text/event-stream")


def _log_stream_bg(
    body: dict, session_id: str, model: str | None, collected: list[str]
):
    with Session(get_engine()) as db:
        for msg in body.get("messages", []):
            db.add(Message(
                session_id=session_id,
                role=msg["role"],
                content=msg.get("content", ""),
                model=model,
                raw_request=body,
            ))

        full_content = "".join(collected)
        db.add(Message(
            session_id=session_id,
            role="assistant",
            content=full_content,
            model=model,
            raw_response={"streamed": True, "content": full_content},
        ))
        db.commit()
        log.debug(
            "DB logged session=%s: %d input msgs + assistant (%d chars)",
            session_id[:8], len(body.get("messages", [])), len(full_content),
        )
