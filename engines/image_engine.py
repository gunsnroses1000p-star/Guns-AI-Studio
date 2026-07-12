import os
import random
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import replicate
from huggingface_hub import InferenceClient, hf_hub_download

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")
hf_client = InferenceClient(token=HF_TOKEN)

def generate_with_provider(
    provider,
    prompt,
    negative_prompt,
    model,
    width,
    height,
    steps,
    seed,
    init_image,
    civitai_model_id,
):
    try:
        if not prompt or not prompt.strip():
            return None, "❌ Please enter a prompt."

        if seed is None or int(seed) == 0:
            seed = random.randint(1, 999999999)

        width = int(width)
        height = int(height)
        steps = int(steps)
        seed = int(seed)

        if provider == "Replicate":
            check_token()

            inputs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "num_inference_steps": steps,
                "seed": seed,
                "output_format": "png",
            }

            if init_image:
                inputs["image"] = Path(init_image)

            output = replicate.run(
                model or DEFAULT_IMAGE_MODEL,
                input=inputs,
            )

            return extract_output(output), f"✅ Replicate image generated. Seed: {seed}"
