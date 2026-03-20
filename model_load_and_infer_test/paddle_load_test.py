import os
import torch
import time
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

def create_dummy_image():
    DUMMY_IMAGE = "dummy_paddle_test.jpg"
    img = Image.new('RGB', (224, 224), color='white')
    img.save(DUMMY_IMAGE)
    return DUMMY_IMAGE

def test_paddle_ocr_vl_15():
    print("\n[Test] Loading PaddleOCR-VL-1.5 (0.9B)...")
    model_id = "PaddlePaddle/PaddleOCR-VL-1.5"
    DUMMY_IMAGE = create_dummy_image()
    
    try:
        start = time.time()
        # Try AutoModelForCausalLM as per auto_map in config.json
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"  Loaded in {time.time() - start:.2f}s")

        # Dummy inference test
        print("  Running dummy inference...")
        image = Image.open(DUMMY_IMAGE).convert("RGB")
        # Correct prompt formatting for VLM
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": DUMMY_IMAGE},
                    {"type": "text", "text": "OCR: "}
                ]
            }
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=image, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output = model.generate(
                **inputs, 
                max_new_tokens=50, 
                do_sample=False
            )
        
        result = processor.decode(output[0], skip_special_tokens=True)
        print(f"  Inference Success! Output sample: {result[:50]}...")

    except Exception as e:
        print(f"  FAILED: {e}")
    finally:
        if os.path.exists(DUMMY_IMAGE):
            os.remove(DUMMY_IMAGE)

if __name__ == "__main__":
    test_paddle_ocr_vl_15()
