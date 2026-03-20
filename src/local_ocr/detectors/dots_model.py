import base64
import gc
import io
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

import requests
import torch
from PIL import Image

from src.local_ocr.detectors.base import BaseOCRModel


DOTS_OCR_PROMPT = (
    "Please extract all readable text from this document image in the original language "
    "and reading order. Return plain markdown text only. Do not translate."
)


class DotsOcrModel(BaseOCRModel):
    def __init__(
        self,
        base_url: str | None = None,
        served_model_name: str | None = None,
        max_new_tokens: int | None = None,
        max_length: int = 16384,
    ):
        super().__init__("DOTS-OCR-1.5", "dots")
        self.base_url = (base_url or os.environ.get("DOTS_VLLM_BASE_URL") or "http://127.0.0.1:8000").rstrip("/")
        self.served_model_name = served_model_name or os.environ.get("DOTS_VLLM_MODEL_NAME") or "model"
        self.max_new_tokens = max_new_tokens
        self.max_length = max_length
        self.session: requests.Session | None = None
        self.project_root = Path(__file__).resolve().parents[3]
        self.server_script = self.project_root / "scripts" / "runners" / "run_dots_vllm_server.sh"
        self.log_dir = self.project_root / "results" / "logs"
        self.server_log_path = self.log_dir / "dots_vllm_server.log"
        self.server_pid_path = self.log_dir / "dots_vllm_server.pid"

    def load(self):
        self.session = requests.Session()
        if not self._is_server_ready():
            self._start_server()
            self._wait_for_server()

        models_response = self.session.get(f"{self.base_url}/v1/models", timeout=10)
        models_response.raise_for_status()
        models = models_response.json().get("data", [])
        served_model_names = {model.get("id") for model in models}
        if self.served_model_name not in served_model_names:
            raise RuntimeError(
                f"DOTS vLLM server is up, but served model '{self.served_model_name}' was not found. "
                f"Available: {sorted(name for name in served_model_names if name)}"
            )

    def unload(self):
        if self.session is not None:
            self.session.close()
        self.session = None
        
        # Stop the vLLM server process if it was started
        if self.server_pid_path.exists():
            try:
                pid = int(self.server_pid_path.read_text().strip())
                print(f"[{self.model_name}] Stopping vLLM server (PID: {pid})...")
                # Kill the process group to ensure child processes are also stopped
                import signal
                os.kill(pid, signal.SIGTERM)
                # Wait a bit for it to shutdown gracefully
                time.sleep(5)
                # Check if still running and kill -9 if necessary
                try:
                    os.kill(pid, 0)
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
                self.server_pid_path.unlink(missing_ok=True)
            except Exception as e:
                print(f"[{self.model_name}] Warning: Failed to stop server: {e}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def infer_full_page(self, pdf_path: Path) -> Dict[str, Any]:
        from src.local_ocr.utils.pdf_utils import pdf_page_generator

        if self.session is None:
            self.load()

        pages = []
        start_time = time.time()

        for i, image in enumerate(pdf_page_generator(pdf_path)):
            print(f"  Processing page {i + 1}...")
            page_start_time = time.time()
            page_text = self.infer_crop(image)
            pages.append(
                self.build_page_result(
                    page_number=i + 1,
                    text=page_text,
                    elapsed_sec=time.time() - page_start_time,
                )
            )

        return self.build_document_result(
            pdf_path=pdf_path,
            pages=pages,
            elapsed_sec=time.time() - start_time,
        )

    def infer_crop(self, image: Image.Image) -> str:
        if self.session is None:
            self.load()

        payload = {
            "model": self.served_model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": DOTS_OCR_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": self._image_to_data_url(image.convert("RGB")),
                            },
                        },
                    ],
                }
            ],
            "temperature": 0,
        }
        if self.max_new_tokens is not None:
            payload["max_tokens"] = self.max_new_tokens

        response = self.session.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        response_json = response.json()
        choices = response_json.get("choices", [])
        if not choices:
            raise RuntimeError(f"DOTS vLLM returned no choices: {response_json}")

        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
            return "\n".join(part for part in text_parts if part).strip()
        return str(content).strip()

    @staticmethod
    def _image_to_data_url(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    def _is_server_ready(self) -> bool:
        if self.session is None:
            return False
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            return True
        except requests.RequestException:
            return False

    def _start_server(self) -> None:
        if not self.server_script.exists():
            raise RuntimeError(f"DOTS vLLM start script not found: {self.server_script}")

        self.log_dir.mkdir(parents=True, exist_ok=True)
        with self.server_log_path.open("ab") as log_file:
            process = subprocess.Popen(
                ["bash", str(self.server_script)],
                cwd=self.project_root,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.server_pid_path.write_text(str(process.pid), encoding="utf-8")

    def _wait_for_server(self, timeout_sec: int = 180) -> None:
        started_at = time.time()
        while time.time() - started_at < timeout_sec:
            if self._is_server_ready():
                return
            time.sleep(2)

        log_hint = ""
        if self.server_log_path.exists():
            log_hint = f" Check log: {self.server_log_path}"
        raise RuntimeError(
            f"DOTS vLLM server did not become ready within {timeout_sec} seconds.{log_hint}"
        )
