import json
import platform
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from rich.console import Console

from tuneloop.config import CACHE_DIR

console = Console()


def ensure_llama_cpp(cache_dir: Path = CACHE_DIR) -> Path:
    """Clone llama.cpp if needed and install its Python requirements."""
    repo_path = cache_dir / "llama.cpp"
    if repo_path.exists() and (repo_path / "convert_hf_to_gguf.py").exists():
        console.print("[dim]llama.cpp already present[/dim]")
        return repo_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    console.print("[bold]Cloning llama.cpp...[/bold]")
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp.git", str(repo_path)],
        check=True,
    )

    requirements = repo_path / "requirements.txt"
    if requirements.exists():
        console.print("[bold]Installing llama.cpp Python requirements...[/bold]")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements)],
            check=True,
        )

    return repo_path


def merge_adapter(adapter_path: Path, output_dir: Path) -> Path:
    """Merge a PEFT LoRA adapter into the base model and save."""
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    merged_path = output_dir / "merged"
    if merged_path.exists() and (merged_path / "config.json").exists():
        console.print("[dim]Merged model already exists, skipping merge[/dim]")
        return merged_path

    # Resolve to absolute path so PEFT doesn't treat it as a HF repo name
    adapter_path = adapter_path.resolve()

    # If adapter_config.json isn't at the top level, check final/ subdirectory
    if not (adapter_path / "adapter_config.json").exists():
        final_path = adapter_path / "final"
        if (final_path / "adapter_config.json").exists():
            adapter_path = final_path
        else:
            raise FileNotFoundError(
                f"No adapter_config.json found in {adapter_path} or {final_path}"
            )

    console.print("[bold]Loading adapter config...[/bold]")
    peft_config = PeftConfig.from_pretrained(str(adapter_path))
    base_model_name = peft_config.base_model_name_or_path

    console.print(f"[bold]Loading base model:[/bold] {base_model_name}")
    import torch

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    console.print("[bold]Loading and merging adapter...[/bold]")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model = model.merge_and_unload()

    merged_path.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold]Saving merged model to:[/bold] {merged_path}")
    model.save_pretrained(str(merged_path))
    tokenizer.save_pretrained(str(merged_path))

    return merged_path


def convert_to_gguf(merged_path: Path, output_path: Path, llama_cpp_path: Path) -> Path:
    """Convert a HuggingFace model to GGUF format (f16)."""
    if output_path.exists():
        console.print("[dim]GGUF f16 file already exists, skipping conversion[/dim]")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    console.print("[bold]Converting to GGUF (f16)...[/bold]")
    subprocess.run(
        [
            sys.executable, str(llama_cpp_path / "convert_hf_to_gguf.py"),
            str(merged_path),
            "--outfile", str(output_path),
            "--outtype", "f16",
        ],
        check=True,
    )
    return output_path


def _get_llama_cpp_release_asset_url() -> str:
    """Find the right pre-built llama.cpp release asset URL for this platform."""
    api_url = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
    with urllib.request.urlopen(api_url) as resp:
        release = json.loads(resp.read())

    machine = platform.machine().lower()
    system = platform.system().lower()

    # Map to llama.cpp asset naming conventions
    if system == "linux":
        if machine in ("x86_64", "amd64"):
            pattern = "ubuntu-x64"
        elif machine == "aarch64":
            pattern = "ubuntu-arm64"
        else:
            raise RuntimeError(f"No pre-built llama.cpp binary for linux/{machine}")
    elif system == "darwin":
        pattern = "macos-arm64" if machine == "arm64" else "macos-x64"
    else:
        raise RuntimeError(f"No pre-built llama.cpp binary for {system}/{machine}")

    for asset in release["assets"]:
        name = asset["name"]
        if pattern in name and (name.endswith(".tar.gz") or name.endswith(".zip")):
            # Skip specialized builds (rocm, vulkan, sycl, etc.)
            skip = ("rocm", "vulkan", "sycl", "opencl", "hip", "cuda", "adreno")
            if not any(s in name for s in skip):
                return asset["browser_download_url"]

    raise RuntimeError(f"Could not find llama.cpp release asset matching '{pattern}'")


def _ensure_quantize_bin(cache_dir: Path) -> Path:
    """Download pre-built llama-quantize binary if not already cached."""
    bin_dir = cache_dir / "llama-bin"
    quantize_bin = bin_dir / "llama-quantize"
    if quantize_bin.exists():
        return quantize_bin

    console.print("[bold]Downloading pre-built llama-quantize...[/bold]")
    url = _get_llama_cpp_release_asset_url()
    console.print(f"[dim]{url}[/dim]")

    archive_path = cache_dir / ("llama-bin.tar.gz" if url.endswith(".tar.gz") else "llama-bin.zip")
    cache_dir.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, archive_path)

    # Extract archive — keep all files (binary + shared libs like libllama.so)
    extract_dir = cache_dir / "llama-bin-extract"
    extract_dir.mkdir(parents=True, exist_ok=True)

    if url.endswith(".tar.gz"):
        with tarfile.open(archive_path) as tf:
            tf.extractall(extract_dir)
    else:
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(extract_dir)

    # Move the directory containing llama-quantize to bin_dir
    found = list(extract_dir.rglob("llama-quantize"))
    if not found:
        raise RuntimeError("llama-quantize not found in release archive")

    import shutil
    src_dir = found[0].parent
    if bin_dir.exists():
        shutil.rmtree(bin_dir)
    shutil.move(str(src_dir), str(bin_dir))

    quantize_bin.chmod(0o755)
    archive_path.unlink(missing_ok=True)
    shutil.rmtree(extract_dir, ignore_errors=True)
    console.print("[green]llama-quantize ready[/green]")
    return quantize_bin


def quantize_gguf(
    input_path: Path, output_path: Path, cache_dir: Path, quant_type: str = "q4_k_m"
) -> Path:
    """Quantize a GGUF file. Skips if quant_type is 'f16'."""
    if quant_type.lower() == "f16":
        console.print("[dim]Skipping quantization (already f16)[/dim]")
        return input_path

    if output_path.exists():
        console.print(f"[dim]Quantized GGUF ({quant_type}) already exists, skipping[/dim]")
        return output_path

    quantize_bin = _ensure_quantize_bin(cache_dir)
    lib_dir = quantize_bin.parent

    output_path.parent.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold]Quantizing to {quant_type}...[/bold]")
    env = {**subprocess.os.environ, "LD_LIBRARY_PATH": str(lib_dir)}
    subprocess.run(
        [str(quantize_bin), str(input_path), str(output_path), quant_type.upper()],
        check=True,
        env=env,
    )
    return output_path


def create_ollama_model(model_name: str, gguf_path: Path) -> None:
    """Create an Ollama model from a GGUF file."""
    console.print(f"[bold]Creating Ollama model:[/bold] {model_name}")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".Modelfile", delete=False) as f:
        f.write(f"FROM {gguf_path.resolve()}\n")
        modelfile_path = Path(f.name)

    try:
        subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile_path)],
            check=True,
        )
    finally:
        modelfile_path.unlink(missing_ok=True)

    console.print(f"[green bold]Model created:[/green bold] {model_name}")


def publish(
    adapter_path: Path,
    model_name: str,
    quant_type: str = "q4_k_m",
    cache_dir: Path = CACHE_DIR,
) -> None:
    """Orchestrate: merge adapter → convert to GGUF → quantize → register with Ollama."""
    publish_dir = adapter_path.parent / "publish"
    publish_dir.mkdir(parents=True, exist_ok=True)

    llama_cpp_path = ensure_llama_cpp(cache_dir)
    merged_path = merge_adapter(adapter_path, publish_dir)

    f16_gguf = publish_dir / "model-f16.gguf"
    convert_to_gguf(merged_path, f16_gguf, llama_cpp_path)

    final_gguf = quantize_gguf(
        f16_gguf,
        publish_dir / f"model-{quant_type}.gguf",
        cache_dir,
        quant_type,
    )

    create_ollama_model(model_name, final_gguf)
    console.print(f"\n[green bold]Done![/green bold] Run: ollama run {model_name}")
