import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.local_ocr.detectors.docling_model import DoclingModel
from src.local_ocr.detectors.dots_model import DotsOcrModel
from src.local_ocr.detectors.glm_model import GlmOcrModel
from src.local_ocr.detectors.paddle_model import PaddleModel

INPUT_DIR = PROJECT_ROOT / "input"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["docling", "glm", "dots", "paddle"])
    parser.add_argument("--page-limit", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    return parser.parse_args()


def resolve_pdf_path() -> Path:
    pdf_path = next(INPUT_DIR.glob("*.pdf"), None)
    if pdf_path is None:
        raise FileNotFoundError("input/ に PDF がありません。")
    return pdf_path


def create_model(name: str, max_new_tokens: int | None, max_length: int | None):
    if name == "docling":
        return DoclingModel()
    if name == "glm":
        return GlmOcrModel(max_new_tokens=max_new_tokens, max_length=max_length or 16384)
    if name == "dots":
        return DotsOcrModel(max_new_tokens=max_new_tokens, max_length=max_length or 16384)
    if name == "paddle":
        return PaddleModel()
    raise ValueError(f"Unsupported model: {name}")

def main():
    args = parse_args()
    pdf_path = resolve_pdf_path()
    model = create_model(args.model, args.max_new_tokens, args.max_length)
    raise SystemExit(
        model.run_page_limited_in_venv(
            pdf_path=pdf_path,
            page_limit=args.page_limit,
            dpi=args.dpi,
            max_new_tokens=args.max_new_tokens,
            max_length=args.max_length,
        )
    )


if __name__ == "__main__":
    main()
