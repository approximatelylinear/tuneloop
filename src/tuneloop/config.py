from pathlib import Path

OLLAMA_BASE_URL = "http://localhost:11434"
PROXY_PORT = 8000
DB_PATH = Path("tuneloop.db")
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
RUNS_DIR = Path("runs")
CACHE_DIR = Path.home() / ".cache" / "tuneloop"
