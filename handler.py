import io
import os
import sys
import base64
import traceback

import runpod
import torch
from PIL import Image
from diffusers import FluxPipeline, FluxImg2ImgPipeline


MODEL_ID = "black-forest-labs/FLUX.1-dev"

print("=== Guns AI Studio RunPod Worker Starting ===", flush=True)
print(f"Python: {sys.version}", flush=True)
print(f"PyTorch: {torch.__version__}", flush=True)
print(f"CUDA available: {torch.cuda.is_available()}", flush=True)

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

txt2img_pipe = None
img2img_pipe = None


def load_txt2img_model():
    global txt2img_pipe

    if txt2img_pipe is not None:
        return txt2img_pipe

    print(f"Loading Text-to-Image model: {MODEL_ID}", flush=True)

    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32

    txt2img_pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    )

    if DEVICE == "cuda":
        print("Enabling CPU offload for Text-to-Image...", flush=True)
        txt2img_pipe.enable_model_cpu_offload()
    else:
        txt2img_pipe.to("cpu")

    print("FLUX Text-to-Image model loaded successfully.", flush=True)

    return txt2img_pipe


def load_img2img_model():
    global img2img_pipe

    if img2img_pipe is not None:
        return img2img_pipe

    print(f"Loading Img2Img model: {MODEL_ID}", flush=True)

    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32

    img2img_pipe = FluxImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    )

    if DEVICE == "cuda":
        print("Enabling CPU offload for Img2Img...", flush=True)
        img2img_pipe.enable_model_cpu_offload()
    else:
        img2img_pipe.to("cpu")

    print("FLUX Img2Img model loaded successfully.", flush=True)

    return img2img_pipe


def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


def base64_to_image(image_base64):
    # Also support data URLs if one is ever sent.
    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]

    image_bytes = base64.b64decode(image_base64)

    return Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")


def generate_txt2img(job_input):
    prompt = job_input.get(
        "prompt",
        "A beautiful photorealistic landscape"
    )

    width = int(job_input.get("width", 1024))
    height
