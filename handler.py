import io
import os
import sys
import base64
import traceback

import runpod
import torch
from diffusers import FluxPipeline


MODEL_ID = "black-forest-labs/FLUX.1-dev"

print("=== Guns AI Studio RunPod Worker Starting ===", flush=True)
print(f"Python: {sys.version}", flush=True)
print(f"PyTorch: {torch.__version__}", flush=True)
print(f"CUDA available: {torch.cuda.is_available()}", flush=True)

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
pipe = None


def load_model():
    global pipe

    if pipe is not None:
        return pipe

    print(f"Loading model: {MODEL_ID}", flush=True)

    try:
        dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32

        pipe = FluxPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
        )

        if DEVICE == "cuda":
            print("Enabling CPU offload...", flush=True)
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cpu")

        print("FLUX model loaded successfully.", flush=True)
        return pipe

    except Exception as e:
        print("=== MODEL LOADING FAILED ===", flush=True)
        print(f"Error type: {type(e).__name__}", flush=True)
        print(f"Error: {e}", flush=True)
        traceback.print_exc()
        raise


def handler(job):
    try:
        print("Received RunPod job.", flush=True)

        job_input = job.get("input", {})

        prompt = job_input.get(
            "prompt",
            "A beautiful photorealistic landscape"
        )

        width = int(job_input.get("width", 1024))
        height = int(job_input.get("height", 1024))
        steps = int(job_input.get("steps", 28))
        seed = int(job_input.get("seed", 0))

        model = load_model()

        generator = None

        if seed > 0:
            generator = torch.Generator(
                device="cpu"
            ).manual_seed(seed)

        print(
            f"Generating image: {width}x{height}, "
            f"steps={steps}, seed={seed}",
            flush=True,
        )

        image = model(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            generator=generator,
        ).images[0]

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        image_base64 = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        print("Image generated successfully.", flush=True)

        return {
            "image_base64": image_base64
        }

    except Exception as e:
        print("=== JOB FAILED ===", flush=True)
        print(f"Error type: {type(e).__name__}", flush=True)
        print(f"Error: {e}", flush=True)
        traceback.print_exc()

        return {
            "error": str(e),
            "error_type": type(e).__name__,
        }


print("Starting RunPod serverless handler...", flush=True)

runpod.serverless.start(
    {
        "handler": handler
    }
)
