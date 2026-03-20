from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pathlib import Path
import subprocess
from PIL import Image
from src.local_ocr.utils.venv_config import get_model_venv_python, get_page_limited_worker_path

class BaseOCRModel(ABC):
    """
    Abstract base class for all OCR models to ensure easy swapping and DI.
    """
    def __init__(self, model_name: str, model_key: str):
        self.model_name = model_name
        self.model_key = model_key

    @abstractmethod
    def load(self):
        """Load the model weights into GPU memory."""
        pass

    @abstractmethod
    def unload(self):
        """Unload the model and clear VRAM."""
        pass

    @abstractmethod
    def infer_full_page(self, pdf_path: Path) -> Dict[str, Any]:
        """End-to-End full page processing (Step 1)."""
        pass

    @abstractmethod
    def infer_crop(self, image: Image.Image) -> str:
        """Local OCR for a single cropped image (Step 2)."""
        pass

    def build_page_result(
        self,
        page_number: int,
        text: str,
        elapsed_sec: float,
        status: str = "ok",
    ) -> Dict[str, Any]:
        return {
            "page_number": page_number,
            "text": text,
            "elapsed_sec": round(elapsed_sec, 3),
            "status": status,
        }

    def build_document_result(
        self,
        pdf_path: Path,
        pages: List[Dict[str, Any]],
        elapsed_sec: float,
        **extra: Any,
    ) -> Dict[str, Any]:
        combined_text = "\n\n".join(
            f"## Page {page['page_number']}\n\n{page['text']}" for page in pages
        )
        result = {
            "model": self.model_name,
            "source_pdf": str(pdf_path),
            "pages": pages,
            "page_count": len(pages),
            "text": combined_text,
            "time_sec": round(elapsed_sec, 3),
        }
        result.update(extra)
        return result

    @property
    def venv_python(self) -> Path:
        return get_model_venv_python(self.model_key)

    def run_page_limited_in_venv(
        self,
        pdf_path: Path,
        page_limit: int,
        dpi: int,
        max_new_tokens: int | None = None,
        max_length: int | None = None,
    ) -> int:
        worker_path = get_page_limited_worker_path()
        cmd = [
            str(self.venv_python),
            str(worker_path),
            "--model",
            self.model_key,
            "--pdf-path",
            str(pdf_path),
            "--page-limit",
            str(page_limit),
            "--dpi",
            str(dpi),
        ]
        if max_new_tokens is not None:
            cmd.extend(["--max-new-tokens", str(max_new_tokens)])
        if max_length is not None:
            cmd.extend(["--max-length", str(max_length)])
        completed = subprocess.run(cmd, check=False)
        return completed.returncode
