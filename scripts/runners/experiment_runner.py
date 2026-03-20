import json
from pathlib import Path
from typing import List
from src.local_ocr.detectors.base import BaseOCRModel
from src.local_ocr.detectors.docling_model import DoclingModel
from src.local_ocr.detectors.glm_model import GlmOcrModel
from src.local_ocr.detectors.dots_model import DotsOcrModel
from src.local_ocr.detectors.paddle_model import PaddleModel

class OCRExperimentRunner:
    """
    Runner that uses Dependency Injection to evaluate different models.
    """
    def __init__(self, models: List[BaseOCRModel], input_dir: Path, output_dir: Path):
        self.models = models
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

    @staticmethod
    def _render_markdown(e2e_res: dict) -> str:
        pages = e2e_res.get("pages")
        if pages:
            return "\n\n".join(
                f"## Page {page['page_number']}\n\n{page['text']}" for page in pages
            )
        return e2e_res.get("text", "")

    def run_all(self):
        pdf_files = list(self.input_dir.glob("*.pdf"))
        if not pdf_files:
            print("No PDF files found.")
            return

        for pdf in pdf_files:
            print(f"\n>>> Processing: {pdf.name}")
            for model in self.models:
                print(f"--- Testing Model: {model.model_name} ---")
                try:
                    # Step 1: E2E
                    e2e_res = model.infer_full_page(pdf)
                    rendered_text = self._render_markdown(e2e_res)
                    
                    # Save Full Text
                    output_path = self.output_dir / "detections" / f"{pdf.stem}_{model.model_name.replace(' ', '_')}.md"
                    output_path.parent.mkdir(exist_ok=True)
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(rendered_text)
                    
                    # Log Results
                    result_log = {
                        "filename": pdf.name,
                        "model": model.model_name,
                        "time_sec": e2e_res["time_sec"],
                        "page_count": e2e_res.get("page_count"),
                        "text_length": len(rendered_text),
                        "output_file": str(output_path)
                    }
                    print(f"  Result: {result_log}")
                    
                    # Save (Incremental)
                    with open(self.output_dir / "comparison_results.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps(result_log, ensure_ascii=False) + "\n")
                
                except Exception as e:
                    print(f"  Error with {model.model_name}: {e}")
                finally:
                    # Clean up VRAM after each model to avoid OOM
                    model.unload()

if __name__ == "__main__":
    # DI: Inject chosen models
    models_to_test = [
        DoclingModel(),
        GlmOcrModel(),
        DotsOcrModel(),
        PaddleModel()
    ]
    
    runner = OCRExperimentRunner(
        models=models_to_test,
        input_dir=Path("input"),
        output_dir=Path("results")
    )
    runner.run_all()
