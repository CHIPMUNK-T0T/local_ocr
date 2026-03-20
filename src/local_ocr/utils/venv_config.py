from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_VENV_MAP = {
    "docling": PROJECT_ROOT / "ocr_venv",
    "glm": PROJECT_ROOT / "ocr_venv",
    "dots": PROJECT_ROOT / "venvs" / "dots_vllm_venv",
    "paddle": PROJECT_ROOT / "venvs" / "paddle_venv",
}


def get_model_venv_path(model_key: str) -> Path:
    return MODEL_VENV_MAP[model_key]


def get_model_venv_python(model_key: str) -> Path:
    return get_model_venv_path(model_key) / "bin" / "python"


def get_page_limited_worker_path() -> Path:
    return PROJECT_ROOT / "scripts" / "runners" / "venv_page_limited_worker.py"
