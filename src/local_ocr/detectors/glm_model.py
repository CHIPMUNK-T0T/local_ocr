import gc
import torch
import time
from pathlib import Path
from PIL import Image
from typing import Dict, Any
from packaging.version import Version
import transformers
from transformers import AutoModelForImageTextToText, AutoProcessor
from src.local_ocr.detectors.base import BaseOCRModel
from src.local_ocr.utils.hf_utils import configure_hf_caches, resolve_cached_hf_snapshot

try:
    from transformers import GlmOcrForConditionalGeneration
except ImportError:
    GlmOcrForConditionalGeneration = None

try:
    from transformers.models.glm46v.processing_glm46v import Glm46VProcessor
except ImportError:
    Glm46VProcessor = None


MIN_GLM_OCR_TRANSFORMERS_VERSION = Version("5.0.0rc0")

class GlmOcrModel(BaseOCRModel):
    def __init__(
        self,
        model_id="zai-org/GLM-OCR",
        max_new_tokens: int | None = None,
        max_length: int = 16384,
    ):
        super().__init__("GLM-OCR-0.9B", "glm")
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.max_length = max_length
        self.model = None
        self.processor = None

    def load(self):
        if Version(transformers.__version__) < MIN_GLM_OCR_TRANSFORMERS_VERSION:
            raise RuntimeError(
                "GLM-OCR requires transformers>=5.0.0rc0 per official docs. "
                f"Current version: {transformers.__version__}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("GLM-OCR requires CUDA. CPU offload is disabled by project policy.")

        configure_hf_caches()
        model_ref = resolve_cached_hf_snapshot(self.model_id) or self.model_id
        local_files_only = model_ref != self.model_id
        print(f"[{self.model_name}] Loading from {model_ref}...")
        model_cls = GlmOcrForConditionalGeneration or AutoModelForImageTextToText
        self.model = model_cls.from_pretrained(
            model_ref,
            dtype=torch.bfloat16,
            device_map={"": 0},
            local_files_only=local_files_only,
        )
        processor_cls = Glm46VProcessor or AutoProcessor
        self.processor = processor_cls.from_pretrained(
            model_ref,
            local_files_only=local_files_only,
            use_fast=False,
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
                    {"type": "text", "text": "Text Recognition:"},
                ],
            }
        ]
        
        inputs = self.processor.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_dict=True, 
            return_tensors="pt"
        ).to(self.model.device)

        generate_kwargs = dict(inputs)
        if self.max_new_tokens is not None:
            generate_kwargs["max_new_tokens"] = self.max_new_tokens
        else:
            generate_kwargs["max_length"] = self.max_length

        with torch.no_grad():
            output = self.model.generate(**generate_kwargs)

        generated_ids = output[:, inputs.input_ids.shape[1]:]
        result = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return result[0].strip()
