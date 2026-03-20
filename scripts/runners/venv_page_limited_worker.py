#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.local_ocr.utils.pdf_utils import pdf_page_generator

OUTPUT_DIR = PROJECT_ROOT / "results" / "page_limited_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["docling", "glm", "dots", "paddle"])
    parser.add_argument("--pdf-path", required=True)
    parser.add_argument("--page-limit", type=int, required=True)
    parser.add_argument("--dpi", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    return parser.parse_args()


def create_model(name: str, max_new_tokens: int | None, max_length: int | None):
    if name == "docling":
        from src.local_ocr.detectors.docling_model import DoclingModel

        return DoclingModel()
    if name == "glm":
        from src.local_ocr.detectors.glm_model import GlmOcrModel

        return GlmOcrModel(max_new_tokens=max_new_tokens, max_length=max_length or 16384)
    if name == "dots":
        from src.local_ocr.detectors.dots_model import DotsOcrModel

        return DotsOcrModel(max_new_tokens=max_new_tokens, max_length=max_length or 16384)
    if name == "paddle":
        from src.local_ocr.detectors.paddle_model import PaddleModel

        return PaddleModel()
    raise ValueError(f"Unsupported model: {name}")


def render_markdown(pages: list[dict]) -> str:
    return "\n\n".join(f"## Page {page['page_number']}\n\n{page['text']}" for page in pages)


def main():
    args = parse_args()
    pdf_path = Path(args.pdf_path)
    model = create_model(args.model, args.max_new_tokens, args.max_length)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n--- {model.model_name}: first {args.page_limit} pages ---")
    started_at = time.time()
    pages = []

    try:
        model.load()
        if args.model == "docling":
            result = model.infer_full_page(pdf_path)
            pages = result.get("pages", [])[: args.page_limit]
        else:
            for page_num, image in enumerate(pdf_page_generator(pdf_path, dpi=args.dpi), start=1):
                if page_num > args.page_limit:
                    break
                print(f"  Processing page {page_num}...")
                page_started_at = time.time()
                page_text = model.infer_crop(image)
                pages.append(
                    model.build_page_result(
                        page_number=page_num,
                        text=page_text,
                        elapsed_sec=time.time() - page_started_at,
                    )
                )
    finally:
        model.unload()

    output_path = OUTPUT_DIR / f"{pdf_path.stem}_{model.model_name.replace(' ', '_')}_{args.page_limit}pages.md"
    output_path.write_text(render_markdown(pages), encoding="utf-8")

    summary = {
        "model": model.model_name,
        "source_pdf": str(pdf_path),
        "page_limit": args.page_limit,
        "dpi": args.dpi,
        "pages": pages,
        "page_count": len(pages),
        "elapsed_sec": round(time.time() - started_at, 3),
        "output_file": str(output_path),
    }
    summary_path = OUTPUT_DIR / f"{pdf_path.stem}_{model.model_name.replace(' ', '_')}_{args.page_limit}pages.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Summary: {summary_path}")
    print(f"  Output : {output_path}")


if __name__ == "__main__":
    main()
