import time
import torch
import gc
from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.layout_model_specs import DOCLING_LAYOUT_HERON_101

INPUT_DIR = Path("input")
TEST_PDF = next(INPUT_DIR.glob("*.pdf"), None)

def test_docling_heron_101():
    print("\n[Test] Loading Docling with HERON-101 (High Accuracy RT-DETRv2)...")
    try:
        # Layout Option with Heron-101
        pipeline_options = PdfPipelineOptions()
        pipeline_options.layout_options.model_spec = DOCLING_LAYOUT_HERON_101
        
        # Default OCR (RapidOCR)
        pipeline_options.do_ocr = True
        pipeline_options.ocr_options = RapidOcrOptions()
        
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
        if TEST_PDF:
            print(f"  Converting {TEST_PDF.name} with Heron-101...")
            start = time.time()
            result = converter.convert(TEST_PDF)
            print(f"  Success! Time: {time.time() - start:.2f}s")
            print(f"  Text length: {len(result.document.export_to_markdown())} chars.")
        else:
            print("  Skipping PDF test (No PDF in input/)")
    except Exception as e:
        print(f"  FAILED: {e}")
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    test_docling_heron_101()
