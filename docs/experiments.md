# Running Experiments

End-to-end guide: generate data, export it, fine-tune with QLoRA, and inspect results.

## Prerequisites

- [Ollama](https://ollama.com) running locally with a model pulled (e.g. `ollama pull qwen2.5:7b`)
- Python 3.11+
- `uv` package manager

```bash
uv sync                  # base install (proxy, CLI, export)
uv sync --extra train    # adds torch, TRL, PEFT, bitsandbytes, etc.
```

## 1. Generate training data

Start the proxy, then run the data generation script:

```bash
# Terminal 1
tuneloop serve

# Terminal 2
uv run python scripts/generate_poems.py --count 50
```

This creates ~50 multi-turn poem conversations through the proxy, which logs everything to `tuneloop.db`.

Verify the data:

```bash
tuneloop stats       # total sessions & messages
tuneloop sessions    # list all sessions
tuneloop messages <session-id>   # inspect a specific conversation
```

## 2. Export datasets

Export to SFT format (one example per conversation):

```bash
tuneloop export --format sft --output sft.jsonl
```

Export to DPO format (preference pairs from refinement steps):

```bash
tuneloop export --format dpo --output dpo.jsonl
```

DPO supports three strategies via `--strategy`:

| Strategy | Description |
|---|---|
| `first_last` | One pair per session: first draft (rejected) vs final poem (chosen) |
| `consecutive` | One pair per adjacent refinement step |
| `all` (default) | Both strategies combined — more training signal |

```bash
tuneloop export --format dpo --strategy consecutive --output dpo_consecutive.jsonl
```

## 3. Fine-tune

### SFT (Supervised Fine-Tuning)

```bash
# Auto-exports from DB and trains
tuneloop train --method sft

# Or use a pre-exported file
tuneloop train --method sft --data-file sft.jsonl --epochs 3
```

### DPO (Direct Preference Optimization)

```bash
tuneloop train --method dpo --epochs 1
```

### Common options

| Flag | Default | Description |
|---|---|---|
| `--model` | `Qwen/Qwen2.5-7B-Instruct` | HuggingFace model ID |
| `--epochs` | 3 (SFT) / 1 (DPO) | Training epochs |
| `--lr` | 2e-4 (SFT) / 5e-5 (DPO) | Learning rate |
| `--data-file` | *(auto-export from DB)* | Path to JSONL file |
| `--output-dir` | `runs/sft` or `runs/dpo` | Where adapters are saved |

### What happens under the hood

- Model is loaded in **4-bit QLoRA** (nf4, double quantization, bfloat16 compute)
- **LoRA** adapters (r=64, alpha=128) are applied to all attention + MLP projections
- TRL's `SFTTrainer`/`DPOTrainer` handles chat template application automatically
- Training progress is tracked in the database (TrainingRun + Checkpoint tables)

## 4. Inspect runs

```bash
tuneloop runs
```

Shows a table of all training runs with status, method, model, example count, and completion state.

## 5. Load the trained adapter

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model = AutoPeftModelForCausalLM.from_pretrained("runs/sft/final")
tokenizer = AutoTokenizer.from_pretrained("runs/sft/final")
```

## 6. PPO Training

PPO (Proximal Policy Optimization) trains a policy using a learned reward model, rather than static preference pairs like DPO. This requires three steps: train a reward model, then run PPO.

### Step 1: Train a reward model

The reward model is a sequence classifier trained on the same DPO preference pairs:

```bash
tuneloop train-reward-model --epochs 3 --output-dir runs/reward_model
```

This produces a LoRA adapter at `runs/reward_model/final/` that scores responses on a continuous scale.

### Step 2: Run PPO training

```bash
tuneloop train --method ppo \
    --reward-adapter-path runs/reward_model/final \
    --output-dir runs/ppo \
    --kl-coef 0.5 \
    --lr 1e-6 \
    --response-length 128 \
    --lora-r 16 \
    --num-ppo-epochs 1
```

### PPO-specific options

| Flag | Default | Description |
|---|---|---|
| `--reward-adapter-path` | *(required)* | Path to trained reward model adapter |
| `--kl-coef` | 0.05 | KL penalty coefficient (higher = stay closer to base) |
| `--response-length` | 256 | Max tokens generated per prompt during rollouts |
| `--lora-r` | 64 | LoRA rank (16 recommended for PPO to limit drift) |
| `--binary-reward` | False | Clamp reward to ±1 instead of continuous scores |
| `--num-ppo-epochs` | 4 | Optimization passes per rollout batch |
| `--gradient-accumulation-steps` | 4 | Gradient accumulation steps |

### Memory layout (24GB GPU)

PPO loads three models simultaneously:
- **Policy** (QLoRA 4-bit): ~3.7 GB — the model being trained
- **Value** (SeqCls 4-bit): ~3.7 GB — estimates future rewards; shares backbone forward pass with policy
- **Reward** (SeqCls bf16, CPU): ~14 GB system RAM — scores responses, offloaded to CPU to fit in VRAM

Total GPU: ~10 GB + optimizer states and activations. Fits on a 24GB card.

### Key implementation details

**CPU reward model offloading**: The reward model runs on CPU in bf16. A monkey-patched `get_reward` moves query tensors to CPU, scores them, and moves results back to GPU. This avoids loading three 7B models on one GPU.

**Shared policy-value forward pass**: `PolicyAndValueWrapper.forward` is patched to run a single forward pass through the policy with `output_hidden_states=True`, then feeds the last hidden state to the value model's `.score()` head. This halves the activation memory cost.

**Forward-pass KL computation**: TRL's default PPO uses generation `output.scores` (autoregressive, KV-cache path) for policy logprobs, but computes reference logprobs via a single forward pass. With 4-bit quantization, these two computation paths produce slightly different logits (~5 nats KL per token), which accumulates to ~700 KL over 128 response tokens — even when policy and reference are the same model. We patch `ppo_trainer.py` to recompute policy logprobs via forward pass (same path as reference), eliminating this artifact. See [Experiment 2 notes](#experiment-2-ppo-vs-dpo) for the full investigation.

## 7. Publish and compare models

### Publish to Ollama

Merge the adapter, convert to GGUF, and register as an Ollama model:

```bash
tuneloop publish --adapter-path runs/dpo/final --model-name qwen2.5:7b-dpo
tuneloop publish --adapter-path runs/ppo/final --model-name qwen2.5:7b-ppo
```

### LLM-as-judge comparison

Run blind A/B evaluation between base and fine-tuned models:

```bash
tuneloop judge --base-model qwen2.5:7b --finetuned-model qwen2.5:7b-dpo
tuneloop judge --base-model qwen2.5:7b --finetuned-model qwen2.5:7b-ppo
```

### Full automated experiment

Run the entire pipeline (export → reward model → DPO → PPO → publish → judge → compare):

```bash
tuneloop experiment --judge-model qwen2.5:7b
```

---

## Experiment 2: PPO vs DPO

Investigation log from getting PPO working with TRL 0.29.0's `trl.experimental.ppo` on a 24GB GPU.

### Dataset

358 DPO preference pairs from multi-turn poem conversations, exported with the `all` strategy.

### Reward model

| Version | Epochs | Accuracy | Notes |
|---|---|---|---|
| v1 | 1 | 50% (random) | Barely learned; binary clamping destroyed remaining signal |
| v2 | 3 | 67% | 100% train accuracy by epoch 1.3; margins of 15-16 |

Lesson: 1 epoch is insufficient for the reward model. 3 epochs with early stopping at 100% train accuracy works.

### PPO runs

| Run | KL coef | LR | LoRA r | Resp len | PPO epochs | Reward model | Binary | KL @ step 0 | Result |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.05 | 3e-6 | 64 | 256 | 4 | v1 | yes | ~700 | KL exploded (600-1335) |
| 2 | 0.5 | 1e-6 | 16 | 128 | 4 | v1 | yes | ~700 | KL still 300-700; reward signal mostly 0 |
| 3 | 0.5 | 1e-6 | 16 | 128 | 4 | v2 | no | ~700 | Reward signal improved (1-10), KL still broken |
| 4 | 5.0 | 1e-6 | 16 | 128 | 1 | v2 | no | ~700 | Value loss exploded to 100K+ |
| 5 | 0.5 | 1e-6 | 16 | 128 | 1 | v2 | no | ~700 | Attempted temperature fix — didn't help |
| 6 | 0.5 | 1e-6 | 16 | 128 | 1 | v2 | no | **0** | Forward-pass KL fix — **training stable** |

### The KL bug

**Symptom**: KL divergence started at ~700 at step 0 across all runs, even though policy and reference should be identical at initialization.

**Root cause**: TRL computes policy logprobs from `output.scores` (autoregressive generation with KV cache) but reference logprobs from a single forward pass with `disable_adapter()`. With 4-bit quantization (nf4 + double quant), these two computation paths produce numerically different logits:

```
KL(forward_pass, generation_scores) ≈ 5-10 per sequence
× 128 response tokens = 640-700 total KL
```

Even for the exact same model weights, the autoregressive (token-by-token with KV cache) and batched forward (all tokens at once) paths produce slightly different results under 4-bit quantization.

**Fix**: Patch `ppo_trainer.py` to recompute policy logprobs via `forward()` instead of using cached generation scores. Both policy and reference now use the same computation path:

```python
# Before (line 698-699):
logits = logitss[i : i + args.local_rollout_forward_batch_size]
logprob = selective_log_softmax(logits, response)

# After:
policy_output = forward(model.policy, query_response, processing_class.pad_token_id)
policy_logits = policy_output.logits[:, context_length - 1 : -1]
policy_logits /= args.temperature + 1e-7
logprob = selective_log_softmax(policy_logits, response)
```

Cost: one extra forward pass per rollout batch. Worth it for correct KL.

### Working hyperparameters (Run 6)

```bash
tuneloop train --method ppo \
    --reward-adapter-path runs/reward_model_v2/final \
    --kl-coef 0.5 \
    --lr 1e-6 \
    --response-length 128 \
    --lora-r 16 \
    --num-ppo-epochs 1
```

Key observations from the successful run:
- KL stays in [-2.3, +2.0] throughout training
- Reward scores in 2-9 range, averaging ~5
- Policy approxkl: 0.004-0.022 (small, controlled updates)
- Value loss: 7-13 (stable, no explosions)
- Clip fraction: 4-7% (reasonable)

## Typical experiment workflow

```bash
# 1. Generate data
tuneloop serve &
uv run python scripts/generate_poems.py --count 50

# 2. Export + train SFT
tuneloop train --method sft --epochs 3

# 3. Export + train DPO
tuneloop train --method dpo --epochs 1

# 4. Train reward model + PPO
tuneloop train-reward-model --epochs 3
tuneloop train --method ppo --reward-adapter-path runs/reward_model/final

# 5. Publish and compare
tuneloop publish --adapter-path runs/dpo/final --model-name qwen2.5:7b-dpo
tuneloop publish --adapter-path runs/ppo/final --model-name qwen2.5:7b-ppo
tuneloop judge --base-model qwen2.5:7b --finetuned-model qwen2.5:7b-dpo
tuneloop judge --base-model qwen2.5:7b --finetuned-model qwen2.5:7b-ppo

# Or run everything at once
tuneloop experiment --judge-model qwen2.5:7b
```
