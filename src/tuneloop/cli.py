import json
import uuid
from pathlib import Path

import httpx
import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.table import Table
from sqlmodel import Session, select, func

from tuneloop.config import DEFAULT_MODEL_ID, PROXY_PORT
from tuneloop.db import get_engine, init_db
from tuneloop.models import Session as ChatSession, Message, TrainingRun

app = typer.Typer(help="tuneloop — RL fine-tuning workbench for local LLMs")
console = Console()


@app.command()
def serve(
    port: int = PROXY_PORT,
    debug: bool = typer.Option(False, help="Enable debug logging and auto-reload"),
):
    """Start the proxy server."""
    import os
    import uvicorn

    if debug:
        os.environ["TUNELOOP_DEBUG"] = "1"

    uvicorn.run(
        "tuneloop.proxy:app",
        host="0.0.0.0",
        port=port,
        reload=debug,
        log_level="debug" if debug else "info",
    )


@app.command()
def sessions():
    """List all chat sessions."""
    init_db()
    with Session(get_engine()) as db:
        rows = db.exec(
            select(
                ChatSession.id,
                ChatSession.created_at,
                func.count(Message.id).label("messages"),
            )
            .outerjoin(Message)
            .group_by(ChatSession.id)
            .order_by(ChatSession.created_at.desc())
        ).all()

    table = Table(title="Sessions")
    table.add_column("ID", style="cyan", max_width=36)
    table.add_column("Created", style="green")
    table.add_column("Messages", justify="right")

    for sid, created, count in rows:
        table.add_row(sid, str(created)[:19], str(count))

    console.print(table)


@app.command()
def messages(session_id: str):
    """Show messages for a session (supports prefix matching)."""
    init_db()
    with Session(get_engine()) as db:
        # Prefix match
        session = db.exec(
            select(ChatSession).where(ChatSession.id.startswith(session_id))
        ).first()

        if not session:
            console.print(f"[red]No session matching '{session_id}'[/red]")
            raise typer.Exit(1)

        msgs = db.exec(
            select(Message)
            .where(Message.session_id == session.id)
            .order_by(Message.created_at)
        ).all()

    console.print(f"[bold]Session:[/bold] {session.id}\n")
    for msg in msgs:
        color = "blue" if msg.role == "user" else "green"
        console.print(f"[{color}][{msg.role}][/{color}] {msg.content[:200]}")
        if msg.prompt_tokens or msg.completion_tokens:
            console.print(
                f"  tokens: {msg.prompt_tokens or 0} prompt, "
                f"{msg.completion_tokens or 0} completion",
                style="dim",
            )
        console.print()


@app.command()
def chat(
    model: str = typer.Option("qwen2.5:7b", help="Model name"),
    session_id: str = typer.Option("", help="Resume a session (prefix match)"),
    system: str = typer.Option("", help="System prompt"),
):
    """Interactive chat through the proxy (streams, manages session ID)."""
    init_db()
    history: list[dict] = []

    # Resolve or create session ID
    if session_id:
        with Session(get_engine()) as db:
            existing = db.exec(
                select(ChatSession).where(ChatSession.id.startswith(session_id))
            ).first()
            if existing:
                session_id = existing.id
                # Reload history from DB
                msgs = db.exec(
                    select(Message)
                    .where(Message.session_id == session_id)
                    .order_by(Message.created_at)
                ).all()
                for m in msgs:
                    history.append({"role": m.role, "content": m.content})
                console.print(f"[dim]Resuming session {session_id[:8]}... ({len(history)} messages loaded)[/dim]")
            else:
                console.print(f"[red]No session matching '{session_id}'[/red]")
                raise typer.Exit(1)
    else:
        session_id = str(uuid.uuid4())

    if system and not any(m["role"] == "system" for m in history):
        history.append({"role": "system", "content": system})

    console.print(f"[dim]Session {session_id[:8]}... | model={model} | Ctrl+C to quit[/dim]\n")

    proxy_url = f"http://localhost:{PROXY_PORT}/v1/chat/completions"

    while True:
        try:
            user_input = console.input("[blue]> [/blue]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye[/dim]")
            break

        if not user_input.strip():
            continue

        history.append({"role": "user", "content": user_input})

        payload = {"model": model, "messages": history, "stream": True}
        headers = {"x-session-id": session_id}

        chunks: list[str] = []
        buf = ""
        try:
            with httpx.Client() as client, client.stream(
                "POST", proxy_url, json=payload, headers=headers, timeout=120.0
            ) as resp:
                with Live("", console=console, refresh_per_second=15) as live:
                    for raw in resp.iter_bytes():
                        buf += raw.decode("utf-8", errors="replace")
                        while "\n" in buf:
                            line, buf = buf.split("\n", 1)
                            line = line.strip()
                            if not line or not line.startswith("data: "):
                                continue
                            if line == "data: [DONE]":
                                break
                            try:
                                data = json.loads(line[6:])
                                delta = (
                                    data.get("choices", [{}])[0]
                                    .get("delta", {})
                                    .get("content", "")
                                )
                                if delta:
                                    chunks.append(delta)
                                    live.update(Markdown("".join(chunks)))
                            except (json.JSONDecodeError, IndexError):
                                pass
        except httpx.ConnectError:
            console.print("[red]Cannot connect to proxy. Is `tuneloop serve` running?[/red]")
            history.pop()
            continue

        assistant_content = "".join(chunks)
        if assistant_content:
            history.append({"role": "assistant", "content": assistant_content})
        console.print()


@app.command()
def export(
    format: str = typer.Option("sft", help="Export format: sft or dpo"),
    output: Path = typer.Option(None, help="Output JSONL file path"),
    strategy: str = typer.Option("all", help="DPO strategy: first_last, consecutive, or all"),
):
    """Export conversations to SFT or DPO training format."""
    from tuneloop.export import get_conversations, format_sft, format_dpo, export_jsonl

    conversations = get_conversations()
    console.print(f"Found [bold]{len(conversations)}[/bold] conversations")

    if not conversations:
        console.print("[yellow]No conversations to export.[/yellow]")
        raise typer.Exit(1)

    if format == "sft":
        data = format_sft(conversations)
        out_path = output or Path("sft.jsonl")
    elif format == "dpo":
        data = format_dpo(conversations, strategy=strategy)
        out_path = output or Path("dpo.jsonl")
    else:
        console.print(f"[red]Unknown format: {format}[/red]")
        raise typer.Exit(1)

    if not data:
        console.print("[yellow]No examples produced (DPO needs multi-turn conversations).[/yellow]")
        raise typer.Exit(1)

    export_jsonl(data, out_path)
    console.print(f"Exported [bold]{len(data)}[/bold] examples to [cyan]{out_path}[/cyan]")


@app.command()
def train(
    method: str = typer.Option("sft", help="Training method: sft, dpo, or ppo"),
    data_file: Path = typer.Option(None, help="JSONL data file (auto-exports from DB if omitted)"),
    model: str = typer.Option(DEFAULT_MODEL_ID, help="Base model ID"),
    epochs: int = typer.Option(None, help="Number of training epochs"),
    lr: float = typer.Option(None, help="Learning rate"),
    output_dir: Path = typer.Option(None, help="Output directory for adapter"),
    eval_size: float = typer.Option(0.2, help="Eval split fraction (0 to disable)"),
    beta: float = typer.Option(None, help="DPO beta (KL penalty strength, default 0.1)"),
    reward_adapter_path: Path = typer.Option(None, help="Path to reward model adapter (required for PPO)"),
    response_length: int = typer.Option(256, help="Max response length for PPO generation"),
    kl_coef: float = typer.Option(0.05, help="KL penalty coefficient for PPO"),
    lora_r: int = typer.Option(64, help="LoRA rank (lower = less drift, e.g. 16)"),
    binary_reward: bool = typer.Option(False, help="Clamp reward to binary ±1 (PPO only)"),
    num_ppo_epochs: int = typer.Option(4, help="PPO optimization passes per batch"),
    gradient_accumulation_steps: int = typer.Option(4, help="Gradient accumulation steps"),
):
    """Run QLoRA fine-tuning (SFT, DPO, or PPO)."""
    from tuneloop.export import get_conversations, format_sft, format_dpo

    if method == "ppo" and reward_adapter_path is None:
        console.print("[red]--reward-adapter-path is required for PPO training.[/red]")
        raise typer.Exit(1)

    if data_file:
        data = [json.loads(line) for line in data_file.read_text().splitlines() if line.strip()]
        console.print(f"Loaded [bold]{len(data)}[/bold] examples from [cyan]{data_file}[/cyan]")
    else:
        conversations = get_conversations()
        console.print(f"Found [bold]{len(conversations)}[/bold] conversations in DB")
        if method == "sft":
            data = format_sft(conversations)
        elif method in ("dpo", "ppo"):
            data = format_dpo(conversations)
        else:
            console.print(f"[red]Unknown method: {method}[/red]")
            raise typer.Exit(1)

    if not data:
        console.print("[yellow]No training examples available.[/yellow]")
        raise typer.Exit(1)

    console.print(f"Training [bold]{method.upper()}[/bold] with [bold]{len(data)}[/bold] examples on [cyan]{model}[/cyan]")

    if method == "ppo":
        from tuneloop.train import run_ppo

        kwargs: dict = {"data": data, "model_id": model, "reward_adapter_path": reward_adapter_path}
        if output_dir:
            kwargs["output_dir"] = output_dir
        if epochs:
            kwargs["epochs"] = epochs
        if lr:
            kwargs["lr"] = lr
        kwargs["kl_coef"] = kl_coef
        kwargs["response_length"] = response_length
        kwargs["lora_r"] = lora_r
        kwargs["binary_reward"] = binary_reward
        kwargs["num_ppo_epochs"] = num_ppo_epochs
        kwargs["gradient_accumulation_steps"] = gradient_accumulation_steps
        run_id = run_ppo(**kwargs)
    else:
        from tuneloop.train import run_sft, run_dpo

        kwargs = {"data": data, "model_id": model, "eval_size": eval_size, "lora_r": lora_r}
        if output_dir:
            kwargs["output_dir"] = output_dir
        if epochs:
            kwargs["epochs"] = epochs
        if lr:
            kwargs["lr"] = lr
        if beta is not None:
            kwargs["beta"] = beta

        if method == "sft":
            run_id = run_sft(**kwargs)
        else:
            run_id = run_dpo(**kwargs)

    console.print(f"\n[green]Training complete![/green] Run ID: [cyan]{run_id}[/cyan]")


@app.command("train-reward-model")
def train_reward_model(
    data_file: Path = typer.Option(None, help="DPO JSONL data file (auto-exports from DB if omitted)"),
    model: str = typer.Option(DEFAULT_MODEL_ID, help="Base model ID"),
    epochs: int = typer.Option(1, help="Number of training epochs"),
    lr: float = typer.Option(1e-5, help="Learning rate"),
    output_dir: Path = typer.Option(None, help="Output directory for reward model adapter"),
):
    """Train a scalar reward model from DPO preference pairs."""
    from tuneloop.export import get_conversations, format_dpo

    if data_file:
        data = [json.loads(line) for line in data_file.read_text().splitlines() if line.strip()]
        console.print(f"Loaded [bold]{len(data)}[/bold] examples from [cyan]{data_file}[/cyan]")
    else:
        conversations = get_conversations()
        console.print(f"Found [bold]{len(conversations)}[/bold] conversations in DB")
        data = format_dpo(conversations)

    if not data:
        console.print("[yellow]No preference pairs available.[/yellow]")
        raise typer.Exit(1)

    console.print(f"Training reward model with [bold]{len(data)}[/bold] preference pairs on [cyan]{model}[/cyan]")

    from tuneloop.reward_model import train_reward_model as _train_rm

    kwargs: dict = {"data": data, "model_id": model, "epochs": epochs, "lr": lr}
    if output_dir:
        kwargs["output_dir"] = output_dir

    run_id = _train_rm(**kwargs)
    console.print(f"\n[green]Reward model training complete![/green] Run ID: [cyan]{run_id}[/cyan]")


@app.command()
def runs():
    """List training runs."""
    init_db()
    with Session(get_engine()) as db:
        rows = db.exec(
            select(TrainingRun).order_by(TrainingRun.created_at.desc())
        ).all()

    if not rows:
        console.print("[dim]No training runs yet.[/dim]")
        return

    table = Table(title="Training Runs")
    table.add_column("ID", style="cyan", max_width=8)
    table.add_column("Created", style="green")
    table.add_column("Method", style="bold")
    table.add_column("Model")
    table.add_column("Examples", justify="right")
    table.add_column("Status")

    for run in rows:
        status_style = {
            "completed": "green", "running": "yellow", "pending": "dim",
        }.get(run.status, "red")
        num_examples = str((run.config or {}).get("num_examples", "?"))
        table.add_row(
            run.id[:8],
            str(run.created_at)[:19],
            run.method.upper(),
            run.base_model or "?",
            num_examples,
            f"[{status_style}]{run.status}[/{status_style}]",
        )

    console.print(table)


@app.command()
def judge(
    base_model: str = typer.Option(..., help="Ollama model name for baseline"),
    finetuned_model: str = typer.Option(..., help="Ollama model name for fine-tuned"),
    judge_model: str = typer.Option(None, help="Ollama model for judging (defaults to base_model)"),
    limit: int = typer.Option(0, help="Max prompts to evaluate (0=all)"),
):
    """Run LLM-as-judge blind comparison between base and fine-tuned models."""
    from tuneloop.judge import run_judge, summarize_results

    results = run_judge(
        base_model=base_model,
        finetuned_model=finetuned_model,
        judge_model=judge_model,
        limit=limit,
    )
    summarize_results(results)


@app.command()
def publish(
    model_name: str = typer.Option(..., "--model-name", help="Ollama model name to create"),
    run_id: str = typer.Option("", "--run-id", help="Prefix-match a TrainingRun to find adapter"),
    adapter_path: Path = typer.Option(None, "--adapter-path", help="Direct path to adapter dir"),
    quant: str = typer.Option("q4_k_m", "--quant", help="GGUF quantization type (f16, q8_0, q5_k_m, q4_k_m, q4_0)"),
):
    """Merge adapter, convert to GGUF, and create an Ollama model."""
    from tuneloop.gguf import publish as run_publish

    if adapter_path is None:
        init_db()
        with Session(get_engine()) as db:
            if run_id:
                run = db.exec(
                    select(TrainingRun).where(TrainingRun.id.startswith(run_id))
                ).first()
            else:
                run = db.exec(
                    select(TrainingRun)
                    .where(TrainingRun.status == "completed")
                    .order_by(TrainingRun.created_at.desc())
                ).first()

            if not run:
                console.print("[red]No matching training run found.[/red]")
                raise typer.Exit(1)

            if not run.adapter_path:
                console.print(f"[red]Training run {run.id[:8]} has no adapter_path.[/red]")
                raise typer.Exit(1)

            adapter_path = Path(run.adapter_path)
            console.print(f"Using adapter from run [cyan]{run.id[:8]}[/cyan]: {adapter_path}")

    if not adapter_path.exists():
        console.print(f"[red]Adapter path does not exist: {adapter_path}[/red]")
        raise typer.Exit(1)

    run_publish(adapter_path=adapter_path, model_name=model_name, quant_type=quant)


@app.command()
def experiment(
    output_dir: Path = typer.Option(None, help="Base output directory for all experiment artifacts"),
    dpo_beta: float = typer.Option(0.1, help="DPO beta (KL penalty strength)"),
    ppo_kl_coef: float = typer.Option(0.05, help="PPO KL penalty coefficient"),
    judge_model: str = typer.Option(None, help="Ollama model for judging (defaults to qwen2.5:7b)"),
):
    """Run full PPO vs DPO experiment: export, train reward model, train both, publish, judge, compare."""
    from pathlib import Path as P

    from tuneloop.export import get_conversations, format_dpo

    base_dir = output_dir or P("runs/experiment")
    base_dir.mkdir(parents=True, exist_ok=True)
    judge_model = judge_model or "qwen2.5:7b"

    # Step 1: Export DPO data
    console.print("\n[bold]Step 1/6:[/bold] Exporting DPO preference data...")
    conversations = get_conversations()
    data = format_dpo(conversations)
    if not data:
        console.print("[red]No preference pairs available. Need multi-turn conversations.[/red]")
        raise typer.Exit(1)
    console.print(f"  {len(data)} preference pairs")

    # Step 2: Train reward model
    console.print("\n[bold]Step 2/6:[/bold] Training reward model...")
    from tuneloop.reward_model import train_reward_model as _train_rm

    rm_dir = base_dir / "reward_model"
    rm_run_id = _train_rm(data=data, output_dir=rm_dir)
    reward_adapter = rm_dir / "final"
    console.print(f"  Reward model done (run {rm_run_id[:8]})")

    # Step 3: Train DPO
    console.print("\n[bold]Step 3/6:[/bold] Training DPO...")
    from tuneloop.train import run_dpo

    dpo_dir = base_dir / "dpo"
    dpo_run_id = run_dpo(data=data, output_dir=dpo_dir, beta=dpo_beta)
    console.print(f"  DPO done (run {dpo_run_id[:8]})")

    # Step 4: Train PPO
    console.print("\n[bold]Step 4/6:[/bold] Training PPO...")
    from tuneloop.train import run_ppo

    ppo_dir = base_dir / "ppo"
    ppo_run_id = run_ppo(data=data, reward_adapter_path=reward_adapter, output_dir=ppo_dir, kl_coef=ppo_kl_coef)
    console.print(f"  PPO done (run {ppo_run_id[:8]})")

    # Step 5: Publish both to Ollama
    console.print("\n[bold]Step 5/6:[/bold] Publishing models to Ollama...")
    from tuneloop.gguf import publish as run_publish

    dpo_model_name = "qwen2.5:7b-dpo"
    ppo_model_name = "qwen2.5:7b-ppo"

    run_publish(adapter_path=dpo_dir / "final", model_name=dpo_model_name, quant_type="q4_k_m")
    console.print(f"  Published [cyan]{dpo_model_name}[/cyan]")

    run_publish(adapter_path=ppo_dir / "final", model_name=ppo_model_name, quant_type="q4_k_m")
    console.print(f"  Published [cyan]{ppo_model_name}[/cyan]")

    # Step 6: Judge both against baseline
    console.print("\n[bold]Step 6/6:[/bold] Running judge evaluations...")
    from tuneloop.judge import run_judge

    base_model_name = "qwen2.5:7b"

    console.print(f"\n  [dim]── DPO vs Base ──[/dim]")
    dpo_results = run_judge(
        base_model=base_model_name,
        finetuned_model=dpo_model_name,
        judge_model=judge_model,
    )

    console.print(f"\n  [dim]── PPO vs Base ──[/dim]")
    ppo_results = run_judge(
        base_model=base_model_name,
        finetuned_model=ppo_model_name,
        judge_model=judge_model,
    )

    # Print comparison table
    def _summarize(results):
        ft_wins = sum(1 for r in results if r.get("winner") == "finetuned")
        base_wins = sum(1 for r in results if r.get("winner") == "base")
        errors = sum(1 for r in results if r.get("error"))
        total = len(results)
        ft_scores = [r["finetuned_score"] for r in results if r.get("finetuned_score") is not None]
        base_scores = [r["base_score"] for r in results if r.get("base_score") is not None]
        return {
            "total": total,
            "ft_wins": ft_wins,
            "base_wins": base_wins,
            "ties": total - ft_wins - base_wins - errors,
            "errors": errors,
            "avg_ft": sum(ft_scores) / len(ft_scores) if ft_scores else 0,
            "avg_base": sum(base_scores) / len(base_scores) if base_scores else 0,
        }

    dpo_s = _summarize(dpo_results)
    ppo_s = _summarize(ppo_results)

    table = Table(title="Experiment Results: PPO vs DPO")
    table.add_column("Metric")
    table.add_column("DPO", justify="right")
    table.add_column("PPO", justify="right")
    table.add_row("Prompts evaluated", str(dpo_s["total"]), str(ppo_s["total"]))
    table.add_row("Wins vs base", f'[green]{dpo_s["ft_wins"]}[/green]', f'[green]{ppo_s["ft_wins"]}[/green]')
    table.add_row("Losses vs base", f'[red]{dpo_s["base_wins"]}[/red]', f'[red]{ppo_s["base_wins"]}[/red]')
    table.add_row("Ties", str(dpo_s["ties"]), str(ppo_s["ties"]))
    table.add_row("Avg fine-tuned score", f'{dpo_s["avg_ft"]:.2f}', f'{ppo_s["avg_ft"]:.2f}')
    table.add_row("Avg base score", f'{dpo_s["avg_base"]:.2f}', f'{ppo_s["avg_base"]:.2f}')

    console.print()
    console.print(table)


@app.command()
def stats():
    """Show database statistics."""
    init_db()
    with Session(get_engine()) as db:
        session_count = db.exec(select(func.count(ChatSession.id))).one()
        message_count = db.exec(select(func.count(Message.id))).one()
        prompt_total = db.exec(
            select(func.coalesce(func.sum(Message.prompt_tokens), 0))
        ).one()
        completion_total = db.exec(
            select(func.coalesce(func.sum(Message.completion_tokens), 0))
        ).one()

    table = Table(title="Stats")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Sessions", str(session_count))
    table.add_row("Messages", str(message_count))
    table.add_row("Total prompt tokens", str(prompt_total))
    table.add_row("Total completion tokens", str(completion_total))
    console.print(table)
