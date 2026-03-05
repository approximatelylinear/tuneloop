# tuneloop

RL fine-tuning workbench for local LLMs.

Chat with an Ollama model through a logging proxy, collect conversation data in SQLite, then fine-tune with SFT, DPO, or PPO — all from one CLI.

## How it works

1. **Proxy** — an OpenAI-compatible server (`tuneloop serve`) sits in front of Ollama and logs every request/response to `tuneloop.db`
2. **Collect data** — chat through the proxy (interactive CLI or any OpenAI-compatible client) to build up training conversations
3. **Export** — convert conversations to SFT or DPO format (JSONL)
4. **Train** — QLoRA fine-tuning with TRL (SFT, DPO, or PPO with a learned reward model)
5. **Publish** — merge adapter, convert to GGUF, register as an Ollama model
6. **Evaluate** — blind LLM-as-judge comparison between base and fine-tuned models

## Quickstart

```bash
# Install
uv sync                  # proxy, CLI, export
uv sync --extra train    # adds PyTorch, TRL, PEFT, etc.

# Pull a base model
ollama pull qwen2.5:7b

# Start the proxy
tuneloop serve

# Generate training data (in another terminal)
uv run python scripts/generate_poems.py --count 50

# Train
tuneloop train --method sft
```

## CLI reference

| Command | Description |
|---|---|
| `tuneloop serve` | Start the logging proxy (default port 8000) |
| `tuneloop chat` | Interactive chat through the proxy with streaming |
| `tuneloop sessions` | List all chat sessions |
| `tuneloop messages <id>` | Show messages for a session (prefix match) |
| `tuneloop stats` | Show database statistics |
| `tuneloop export` | Export conversations to SFT or DPO JSONL |
| `tuneloop train` | Run QLoRA fine-tuning (SFT, DPO, or PPO) |
| `tuneloop train-reward-model` | Train a scalar reward model from preference pairs |
| `tuneloop runs` | List training runs |
| `tuneloop publish` | Merge adapter → GGUF → Ollama model |
| `tuneloop judge` | Blind A/B evaluation between two models |
| `tuneloop experiment` | Run full PPO vs DPO experiment end-to-end |

Run `tuneloop <command> --help` for detailed options.

## Architecture

The proxy is a FastAPI app that implements the OpenAI `/v1/chat/completions` endpoint (including streaming). It forwards requests to Ollama's local API and logs both sides of every conversation to a SQLite database (`tuneloop.db`) via SQLModel. Session tracking uses a custom `x-session-id` header — any OpenAI-compatible client can generate training data just by pointing at `localhost:8000`.

Training uses 4-bit QLoRA (nf4, double quantization, bfloat16 compute) with LoRA adapters on all attention and MLP projections. Everything fits on a 24GB GPU.

## Detailed experiment guide

See [docs/experiments.md](docs/experiments.md) for the full workflow: data generation, export strategies, training options, PPO setup, publishing, and evaluation — plus notes from debugging PPO's KL divergence under 4-bit quantization.
