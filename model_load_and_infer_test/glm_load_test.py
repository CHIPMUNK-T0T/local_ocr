import time
import torch
import sys
from pathlib import Path
from packaging.version import Version
import transformers
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.local_ocr.utils.hf_utils import configure_hf_caches, resolve_cached_hf_snapshot

try:
    from transformers import GlmOcrForConditionalGeneration
except ImportError:
    GlmOcrForConditionalGeneration = None


MIN_GLM_OCR_TRANSFORMERS_VERSION = Version("5.0.0rc0")

def test_glm_ocr():
    print("\n[Test] Loading GLM-OCR (0.9B)...")
    model_id = "zai-org/GLM-OCR"
    try:
        if Version(transformers.__version__) < MIN_GLM_OCR_TRANSFORMERS_VERSION:
            raise RuntimeError(
                "GLM-OCR requires transformers>=5.0.0rc0 per official docs. "
                f"Current version: {transformers.__version__}"
            )

        configure_hf_caches()
        start = time.time()
        model_ref = resolve_cached_hf_snapshot(model_id) or model_id
        local_files_only = model_ref != model_id
        model_cls = GlmOcrForConditionalGeneration or AutoModelForImageTextToText
        model = model_cls.from_pretrained(
            model_ref,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            local_files_only=local_files_only,
        )
        processor = AutoProcessor.from_pretrained(
            model_ref,
            local_files_only=local_files_only,
        )
        print(f"  Loaded in {time.time() - start:.2f}s")

        image = Image.new("RGB", (224, 224), color="white")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Text Recognition:"},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        output = model.generate(**inputs, max_new_tokens=64)
        generated_ids = output[:, inputs.input_ids.shape[1]:]
        decoded = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        print(f"  Dummy inference call Success: {decoded[0]}")
        
    except Exception as e:
        print(f"  FAILED: {e}")

if __name__ == "__main__":
    test_glm_ocr()
