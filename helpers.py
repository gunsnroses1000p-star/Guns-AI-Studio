import io
import os
import base64

import runpod
import torch

from PIL import Image
from diffusers import FluxPipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading FLUX model on {DEVICE}...")

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
)

pipe.to(DEVICE)

try:
    pipe.enable_model_cpu_offload()
except Exception:
    pass

print("FLUX model loaded successfully.")

def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def handler(job):
    job_input = job.get("input", {})

    prompt = job_input.get("prompt", "A beautiful landscape")
    negative_prompt = job_input.get("negative_prompt", "")
    width = int(job_input.get("width", 1024))
    height = int(job_input.get("height", 1024))
    steps = int(job_input.get("steps", 28))
    seed = int(job_input.get("seed", 0))

    generator = None
    if seed > 0:
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        generator=generator,
    ).images[0]

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    return {
        "image_base64": base64.b64encode(buffer.getvalue()).decode("utf-8")
    }
runpod.serverless.start({
    "handler": handler
})
