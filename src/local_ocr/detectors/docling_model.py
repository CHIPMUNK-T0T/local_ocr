import gc
import torch
import time
from pathlib import Path
from PIL import Image
from typing import Dict, Any, Optional
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.layout_model_specs import DOCLING_LAYOUT_HERON_101
from docling.utils.export import generate_multimodal_pages
from src.local_ocr.detectors.base import BaseOCRModel

class DoclingModel(BaseOCRModel):
    def __init__(self, model_name="Docling-RapidOCR", layout_spec=DOCLING_LAYOUT_HERON_101):
        super().__init__(model_name, "docling")
        self.layout_spec = layout_spec
        self.converter = None

    def load(self):
        print(f"[{self.model_name}] Initializing Docling Pipeline with Layout Spec...")
        pipeline_options = PdfPipelineOptions()
        pipeline_options.layout_options.model_spec = self.layout_spec
        pipeline_options.do_ocr = True
        pipeline_options.ocr_options = RapidOcrOptions()
        
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def unload(self):
        self.converter = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def infer_full_page(self, pdf_path: Path) -> Dict[str, Any]:
        if not self.converter:
            self.load()
        start = time.time()
        result = self.converter.convert(pdf_path)
        pages = []

        for page_number, (_, content_md, _, _, _, _) in enumerate(
            generate_multimodal_pages(result),
            start=1,
        ):
            pages.append(
                self.build_page_result(
                    page_number=page_number,
                    text=content_md.strip(),
                    elapsed_sec=0.0,
                )
            )

        if not pages:
            full_text = result.document.export_to_markdown().strip()
            if full_text:
                pages.append(
                    self.build_page_result(
                        page_number=1,
                        text=full_text,
                        elapsed_sec=0.0,
                        status="partial",
                    )
                )

        return self.build_document_result(
            pdf_path=pdf_path,
            pages=pages,
            elapsed_sec=time.time() - start,
            raw_doc=result.document,
        )

    def infer_crop(self, image: Image.Image) -> str:
        # RapidOCR direct call for crops (simplified placeholder)
        # In actual comparison, we might use RapidOCR object directly
        return "Docling-RapidOCR-Crop-Result"
