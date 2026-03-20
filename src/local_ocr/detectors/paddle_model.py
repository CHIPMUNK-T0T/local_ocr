import gc
import torch
import time
from pathlib import Path
from PIL import Image
from typing import Dict, Any
from transformers import AutoProcessor, AutoModelForCausalLM
from src.local_ocr.detectors.base import BaseOCRModel
from src.local_ocr.utils.hf_utils import configure_hf_caches, resolve_cached_hf_snapshot

class PaddleModel(BaseOCRModel):
    def __init__(self, model_id="PaddlePaddle/PaddleOCR-VL-1.5"):
        super().__init__("PaddleOCR-VL-1.5", "paddle")
        self.model_id = model_id
        self.model = None
        self.processor = None

    def load(self):
        if not torch.cuda.is_available():
            raise RuntimeError("PaddleOCR-VL requires CUDA. CPU offload is disabled by project policy.")

        configure_hf_caches()
        model_ref = resolve_cached_hf_snapshot(self.model_id) or self.model_id
        local_files_only = model_ref != self.model_id
        print(f"[{self.model_name}] Loading from {model_ref}...")
        # Note: Requires transformers>=4.48.0 and kernels package
        self.processor = AutoProcessor.from_pretrained(
            model_ref,
            trust_remote_code=True,
            local_files_only=local_files_only,
            use_fast=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_ref,
            trust_remote_code=True,
            local_files_only=local_files_only,
            dtype=torch.bfloat16,
            device_map={"": 0},
        )

    def unload(self):
        if self.model is not None:
            del self.model
        if self.processor is not None:
            del self.processor
        self.model = None
        self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def infer_full_page(self, pdf_path: Path) -> Dict[str, Any]:
        from src.local_ocr.utils.pdf_utils import pdf_page_generator
        
        if not self.model: self.load()
        
        pages = []
        start_time = time.time()
        
        for i, image in enumerate(pdf_page_generator(pdf_path)):
            print(f"  Processing page {i+1}...")
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
        if not self.model: self.load()
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image.convert("RGB")},
                    {"type": "text", "text": "OCR: "}
                ]
            }
        ]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=image.convert("RGB"), return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs, 
                max_new_tokens=512, 
                do_sample=False
            )
        
        result = self.processor.decode(output[0], skip_special_tokens=True)
        if "Assistant:" in result:
            result = result.split("Assistant:")[-1].strip()
        return result
