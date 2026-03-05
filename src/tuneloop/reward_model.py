"""Train and load a scalar reward model from DPO preference pairs."""

from __future__ import annotations

from pathlib import Path

from sqlmodel import Session

from tuneloop.config import DEFAULT_MODEL_ID, RUNS_DIR
from tuneloop.db import get_engine, init_db
from tuneloop.models import TrainingRun
from tuneloop.train import _make_bnb_config, _make_lora_config, _make_db_callback


def train_reward_model(
    data: list[dict],
    model_id: str = DEFAULT_MODEL_ID,
    epochs: int = 1,
    lr: float = 1e-5,
    output_dir: Path | None = None,
    max_length: int = 512,
) -> str:
    """Train a scalar reward model from DPO preference pairs. Returns run ID."""
    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from trl import RewardConfig, RewardTrainer

    init_db()

    output_dir = output_dir or RUNS_DIR / "reward_model"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create DB record
    with Session(get_engine()) as db:
        run = TrainingRun(
            method="reward_model",
            base_model=model_id,
            config={
                "epochs": epochs,
                "lr": lr,
                "max_length": max_length,
                "num_examples": len(data),
            },
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        run_id = run.id

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=1,
        quantization_config=_make_bnb_config(),
        device_map={"": 0},
    )

    # Convert DPO {prompt, chosen, rejected} → reward trainer format
    # RewardTrainer expects "chosen" and "rejected" columns as plain strings
    reward_data = []
    for ex in data:
        prompt_msgs = ex["prompt"]
        chosen_msgs = prompt_msgs + ex["chosen"]
        rejected_msgs = prompt_msgs + ex["rejected"]
        chosen_text = tokenizer.apply_chat_template(
            chosen_msgs, tokenize=False, add_generation_prompt=False
        )
        rejected_text = tokenizer.apply_chat_template(
            rejected_msgs, tokenize=False, add_generation_prompt=False
        )
        reward_data.append({"chosen": chosen_text, "rejected": rejected_text})

    dataset = Dataset.from_list(reward_data)

    # LoRA config for sequence classification
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        task_type=TaskType.SEQ_CLS,
    )

    training_args = RewardConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        optim="paged_adamw_8bit",
        logging_dir=str(output_dir / "logs" / run_id[:8]),
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        report_to="tensorboard",
        max_length=max_length,
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=[_make_db_callback(run_id)],
    )

    trainer.train()
    trainer.save_model(str(output_dir / "final"))

    return run_id


def load_reward_model(adapter_path: str | Path, model_id: str = DEFAULT_MODEL_ID):
    """Load a trained reward model for inference. Returns (model, tokenizer)."""
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=1,
        quantization_config=_make_bnb_config(),
        device_map={"": 0},
    )

    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()

    return model, tokenizer
