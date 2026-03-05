"""QLoRA fine-tuning with TRL, tracking runs via the database."""

from __future__ import annotations

from pathlib import Path

from sqlmodel import Session

from tuneloop.config import DEFAULT_MODEL_ID, RUNS_DIR
from tuneloop.db import get_engine, init_db
from tuneloop.models import TrainingRun, Checkpoint


def _make_bnb_config():
    from transformers import BitsAndBytesConfig
    import torch

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def _make_lora_config(r: int = 64):
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=r,
        lora_alpha=r * 2,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
    )


def _make_db_callback(run_id: str):
    """Create a TrainerCallback that tracks progress in the DB."""
    from transformers import TrainerCallback

    class DBTrackingCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            with Session(get_engine()) as db:
                run = db.get(TrainingRun, run_id)
                if run:
                    run.status = "running"
                    db.add(run)
                    db.commit()

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            with Session(get_engine()) as db:
                run = db.get(TrainingRun, run_id)
                if run:
                    metrics = dict(run.metrics or {})
                    metrics.update({
                        k: v for k, v in logs.items()
                        if isinstance(v, (int, float))
                    })
                    run.metrics = metrics
                    db.add(run)
                    db.commit()

        def on_save(self, args, state, control, **kwargs):
            ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            with Session(get_engine()) as db:
                db.add(Checkpoint(
                    training_run_id=run_id,
                    step=state.global_step,
                    adapter_path=str(ckpt_dir),
                ))
                db.commit()

        def on_train_end(self, args, state, control, **kwargs):
            with Session(get_engine()) as db:
                run = db.get(TrainingRun, run_id)
                if run:
                    run.status = "completed"
                    run.adapter_path = args.output_dir
                    db.add(run)
                    db.commit()

    return DBTrackingCallback()


def run_sft(
    data: list[dict],
    model_id: str = DEFAULT_MODEL_ID,
    epochs: int = 3,
    output_dir: Path | None = None,
    lr: float = 2e-4,
    batch_size: int = 4,
    max_length: int = 2048,
    eval_size: float = 0.2,
    lora_r: int = 64,
) -> str:
    """Run QLoRA SFT training. Returns the training run ID."""
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    init_db()

    output_dir = output_dir or RUNS_DIR / "sft"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create DB record
    with Session(get_engine()) as db:
        run = TrainingRun(
            method="sft",
            base_model=model_id,
            config={
                "epochs": epochs, "lr": lr, "batch_size": batch_size,
                "max_length": max_length, "num_examples": len(data),
                "eval_size": eval_size,
            },
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        run_id = run.id

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=_make_bnb_config(),
        device_map="auto",
    )

    dataset = Dataset.from_list(data)

    # Train/eval split
    eval_dataset = None
    if eval_size > 0 and len(data) >= 5:
        split = dataset.train_test_split(test_size=eval_size, seed=42)
        dataset = split["train"]
        eval_dataset = split["test"]

    eval_kwargs = {}
    if eval_dataset is not None:
        eval_kwargs["eval_strategy"] = "steps"
        eval_kwargs["eval_steps"] = 10
        eval_kwargs["per_device_eval_batch_size"] = 1

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        optim="paged_adamw_8bit",
        logging_dir=str(output_dir / "logs" / run_id[:8]),
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        report_to="tensorboard",
        max_length=max_length,
        **eval_kwargs,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=_make_lora_config(r=lora_r),
        callbacks=[_make_db_callback(run_id)],
    )

    trainer.train()
    trainer.save_model(str(output_dir / "final"))

    return run_id


def _format_for_ppo(data: list[dict], tokenizer) -> list[dict]:
    """Convert DPO-format data to PPO format: tokenized prompts.

    Returns list of {"input_ids": list[int]} ready for the PPO dataloader.
    """
    prompts = []
    seen = set()
    for ex in data:
        prompt_msgs = ex["prompt"]
        rendered = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        if rendered not in seen:
            seen.add(rendered)
            input_ids = tokenizer.encode(rendered, add_special_tokens=False)
            prompts.append({"input_ids": input_ids})
    return prompts


def run_ppo(
    data: list[dict],
    reward_adapter_path: Path,
    model_id: str = DEFAULT_MODEL_ID,
    epochs: int = 1,
    output_dir: Path | None = None,
    lr: float = 3e-6,
    kl_coef: float = 0.05,
    response_length: int = 256,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    lora_r: int = 64,
    binary_reward: bool = False,
    num_ppo_epochs: int = 4,
) -> str:
    """Run PPO training with a reward model. Returns the training run ID."""
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )
    from peft import PeftModel
    from trl.experimental.ppo import PPOConfig, PPOTrainer

    init_db()

    output_dir = output_dir or RUNS_DIR / "ppo"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Format dataset
    ppo_data = _format_for_ppo(data, tokenizer)
    dataset = Dataset.from_list(ppo_data)

    # Create DB record
    with Session(get_engine()) as db:
        run = TrainingRun(
            method="ppo",
            base_model=model_id,
            config={
                "epochs": epochs,
                "lr": lr,
                "kl_coef": kl_coef,
                "response_length": response_length,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "num_examples": len(ppo_data),
                "reward_adapter_path": str(reward_adapter_path),
            },
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        run_id = run.id

    bnb_config = _make_bnb_config()

    # Policy model (QLoRA)
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    torch.cuda.empty_cache()

    # Value model (needs .score() head for TRL's get_reward)
    value_model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=1,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    value_model.config.pad_token_id = tokenizer.pad_token_id
    torch.cuda.empty_cache()

    # Reward model (frozen, bf16 on CPU — GPU can't fit 3 models)
    reward_base = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu"},
    )
    reward_base.config.pad_token_id = tokenizer.pad_token_id
    reward_model = PeftModel.from_pretrained(reward_base, str(reward_adapter_path))
    reward_model.eval()

    # Prevent PPOTrainer.__init__ from moving reward model to GPU
    reward_model.to = lambda *args, **kwargs: reward_model
    reward_model.cuda = lambda *args, **kwargs: reward_model

    # Monkey-patch get_reward in TRL's PPO module so it moves tensors to CPU
    # for the reward model and results back to GPU, with binary ±1 clamping
    import trl.experimental.ppo.ppo_trainer as _ppo_mod
    from trl.experimental.utils import first_true_indices
    _original_get_reward = _ppo_mod.get_reward

    def _patched_get_reward(model, query_responses, pad_token_id, context_length):
        # Check if this is our CPU reward model
        device = next(model.parameters()).device
        if device.type == "cpu":
            gpu_device = query_responses.device
            query_responses_cpu = query_responses.to("cpu")
            attention_mask = query_responses_cpu != pad_token_id
            position_ids = attention_mask.cumsum(1) - attention_mask.long()
            lm_backbone = getattr(model, model.base_model_prefix)
            input_ids = torch.masked_fill(query_responses_cpu, ~attention_mask, 0)
            with torch.no_grad():
                output = lm_backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    return_dict=True,
                    output_hidden_states=True,
                    use_cache=False,
                )
                reward_logits = model.score(output.hidden_states[-1])
                if binary_reward:
                    reward_logits = torch.sign(reward_logits)
            # Move results back to GPU
            reward_logits = reward_logits.to(gpu_device)
            sequence_lengths = (
                first_true_indices(query_responses[:, context_length:] == pad_token_id)
                - 1 + context_length
            )
            return (
                reward_logits,
                reward_logits[
                    torch.arange(reward_logits.size(0), device=gpu_device),
                    sequence_lengths,
                ].squeeze(-1),
                sequence_lengths,
            )
        return _original_get_reward(model, query_responses, pad_token_id, context_length)

    _ppo_mod.get_reward = _patched_get_reward

    # Patch PolicyAndValueWrapper to share policy hidden states with value head,
    # avoiding a second full 7B forward pass (saves ~50% activation memory)
    _OrigWrapper = _ppo_mod.PolicyAndValueWrapper
    _orig_forward = _OrigWrapper.forward

    def _shared_forward(self, **kwargs):
        # Single forward pass through policy with hidden states
        kwargs["output_hidden_states"] = True
        policy_output = self.policy(**kwargs)
        # Reuse policy's hidden states for value estimation
        logits = self.value_model.score(policy_output.hidden_states[-1])
        return policy_output, logits

    _OrigWrapper.forward = _shared_forward

    ppo_config = PPOConfig(
        output_dir=str(output_dir),
        total_episodes=len(ppo_data) * epochs,
        response_length=response_length,
        kl_coef=kl_coef,
        num_ppo_epochs=num_ppo_epochs,
        num_mini_batches=1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        temperature=0.7,
        bf16=True,
        gradient_checkpointing=True,
        logging_dir=str(output_dir / "logs" / run_id[:8]),
        logging_steps=10,
        save_strategy="epoch",
        report_to="tensorboard",
    )

    trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=policy_model,
        ref_model=None,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=dataset,
        peft_config=_make_lora_config(r=lora_r),
        callbacks=[_make_db_callback(run_id)],
    )

    # Cast any float32 params to bfloat16 after PEFT wrapping (fixes lm_head/score dtype mismatch)
    for m in [trainer.model, trainer.ref_model, value_model, reward_model]:
        if m is None:
            continue
        for param in m.parameters():
            if param.dtype == torch.float32:
                param.data = param.data.to(torch.bfloat16)

    trainer.train()
    trainer.save_model(str(output_dir / "final"))

    return run_id

def run_dpo(
    data: list[dict],
    model_id: str = DEFAULT_MODEL_ID,
    epochs: int = 1,
    output_dir: Path | None = None,
    lr: float = 5e-5,
    beta: float = 0.1,
    batch_size: int = 1,
    max_length: int = 1024,
    eval_size: float = 0.2,
    lora_r: int = 64,
) -> str:
    """Run QLoRA DPO training. Returns the training run ID."""
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    init_db()

    output_dir = output_dir or RUNS_DIR / "dpo"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create DB record
    with Session(get_engine()) as db:
        run = TrainingRun(
            method="dpo",
            base_model=model_id,
            config={
                "epochs": epochs, "lr": lr, "beta": beta,
                "batch_size": batch_size, "max_length": max_length,
                "num_examples": len(data), "eval_size": eval_size,
            },
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        run_id = run.id

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=_make_bnb_config(),
        device_map="auto",
    )

    dataset = Dataset.from_list(data)

    # Train/eval split
    eval_dataset = None
    if eval_size > 0 and len(data) >= 5:
        split = dataset.train_test_split(test_size=eval_size, seed=42)
        dataset = split["train"]
        eval_dataset = split["test"]

    eval_kwargs = {}
    if eval_dataset is not None:
        eval_kwargs["eval_strategy"] = "steps"
        eval_kwargs["eval_steps"] = 10
        eval_kwargs["per_device_eval_batch_size"] = 1

    training_args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        optim="paged_adamw_8bit",
        logging_dir=str(output_dir / "logs" / run_id[:8]),
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        report_to="tensorboard",
        beta=beta,
        max_length=max_length,
        **eval_kwargs,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=_make_lora_config(r=lora_r),
        callbacks=[_make_db_callback(run_id)],
    )

    trainer.train()
    trainer.save_model(str(output_dir / "final"))

    return run_id
