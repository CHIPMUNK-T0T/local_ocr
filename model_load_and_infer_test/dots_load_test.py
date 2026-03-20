import os
import torch
import time
import sys
from pathlib import Path
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.local_ocr.utils.hf_utils import configure_hf_caches, resolve_local_model_path

def create_dummy_image():
    DUMMY_IMAGE = "dummy_dots_test.jpg"
    img = Image.new('RGB', (224, 224), color='white')
    img.save(DUMMY_IMAGE)
    return DUMMY_IMAGE

def test_dots_ocr_15():
    print("\n[Test] Loading DOTS OCR 1.5 from local model path...")
    model_path = resolve_local_model_path("models/dots-ocr-1.5")
    DUMMY_IMAGE = create_dummy_image()
    try:
        configure_hf_caches()
        start = time.time()
        model_kwargs = {
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            "device_map": "auto" if torch.cuda.is_available() else "cpu",
            "trust_remote_code": True,
            "local_files_only": True,
        }
        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        if torch.cuda.is_available():
            model_kwargs["attn_implementation"] = "flash_attention_2"
        else:
            config.vision_config.attn_implementation = "eager"

        model = AutoModelForCausalLM.from_pretrained(model_path, config=config, **model_kwargs)
        if not torch.cuda.is_available():
            model = model.float()
            original_forward = model.vision_tower.forward

            def cpu_safe_forward(hidden_states, grid_thw, bf16=True):
                return original_forward(hidden_states, grid_thw, bf16=False)

            model.vision_tower.forward = cpu_safe_forward
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            use_fast=False,
        )
        
        # video_processor の初期化問題への対応
        if not hasattr(processor, "video_processor") or processor.video_processor is None:
            from transformers.video_processing_utils import BaseVideoProcessor
            class DummyVideoProcessor(BaseVideoProcessor):
                def __call__(self, *args, **kwargs): return None
            processor.video_processor = DummyVideoProcessor()

        print(f"  Loaded in {time.time() - start:.2f}s")

        print("  Running dummy inference...")
        image = Image.open(DUMMY_IMAGE).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "OCR the text in the image."}
                ]
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        ).to(model.device)
        for key, value in list(inputs.items()):
            if torch.is_tensor(value) and torch.is_floating_point(value):
                inputs[key] = value.to(dtype=model.dtype)
        
        # Qwen2-VL 系の修正
        if "mm_token_type_ids" in inputs:
            inputs.pop("mm_token_type_ids")
            
        output = model.generate(**inputs, max_new_tokens=100, use_cache=False)
        generated_ids = output[:, inputs.input_ids.shape[1]:]
        decoded = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        print(f"  Inference Success! Output: {decoded[0]}")

    except Exception as e:
        print(f"  FAILED: {e}")
    finally:
        if os.path.exists(DUMMY_IMAGE):
            os.remove(DUMMY_IMAGE)

if __name__ == "__main__":
    test_dots_ocr_15()
