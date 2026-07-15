import io
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


# ==========================================
# MODEL LOADERS
# ==========================================

def load_txt2img_model():
    global txt2img_pipe

    if txt2img_pipe is not None:
        return txt2img_pipe

    print(
        f"Loading Text-to-Image model: {MODEL_ID}",
        flush=True,
    )

    dtype = (
        torch.bfloat16
        if DEVICE == "cuda"
        else torch.float32
    )

    txt2img_pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    )

    if DEVICE == "cuda":
        print(
            "Enabling CPU offload for Text-to-Image...",
            flush=True,
        )
        txt2img_pipe.enable_model_cpu_offload()
    else:
        txt2img_pipe.to("cpu")

    print(
        "FLUX Text-to-Image model loaded successfully.",
        flush=True,
    )

    return txt2img_pipe


def load_img2img_model():
    global img2img_pipe

    if img2img_pipe is not None:
        return img2img_pipe

    print(
        f"Loading Img2Img model: {MODEL_ID}",
        flush=True,
    )

    dtype = (
        torch.bfloat16
        if DEVICE == "cuda"
        else torch.float32
    )

    img2img_pipe = FluxImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    )

    if DEVICE == "cuda":
        print(
            "Enabling CPU offload for Img2Img...",
            flush=True,
        )
        img2img_pipe.enable_model_cpu_offload()
    else:
        img2img_pipe.to("cpu")

    print(
        "FLUX Img2Img model loaded successfully.",
        flush=True,
    )

    return img2img_pipe


# ==========================================
# IMAGE HELPERS
# ==========================================

def image_to_base64(image):
    buffer = io.BytesIO()

    image.convert("RGB").save(
        buffer,
        format="PNG",
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


def base64_to_image(image_base64):
    if not image_base64:
        raise ValueError(
            "No image_base64 was provided."
        )

    if "," in image_base64:
        image_base64 = image_base64.split(
            ",",
            1,
        )[1]

    image_bytes = base64.b64decode(
        image_base64
    )

    return Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")


# ==========================================
# TEXT TO IMAGE
# ==========================================

def generate_txt2img(job_input):
    prompt = job_input.get(
        "prompt",
        "A beautiful photorealistic landscape",
    )

    width = int(
        job_input.get("width", 1024)
    )

    height = int(
        job_input.get("height", 1024)
    )

    steps = int(
        job_input.get("steps", 28)
    )

    seed = int(
        job_input.get("seed", 0)
    )

    if seed <= 0:
        seed = torch.randint(
            1,
            2147483647,
            (1,),
        ).item()

    print(
        f"TXT2IMG: {width}x{height}, "
        f"steps={steps}, seed={seed}",
        flush=True,
    )

    model = load_txt2img_model()

    generator = torch.Generator(
        device="cpu"
    ).manual_seed(seed)

    image = model(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        generator=generator,
    ).images[0]

    return {
        "image_base64": image_to_base64(image),
        "seed": seed,
        "task": "txt2img",
    }


# ==========================================
# IMAGE TO IMAGE
# ==========================================

def generate_img2img(job_input):
    prompt = job_input.get(
        "prompt",
        "Enhance this image while preserving the subject.",
    )

    image_base64 = job_input.get(
        "image_base64"
    )

    if not image_base64:
        raise ValueError(
            "Img2Img requires image_base64."
        )

    strength = float(
        job_input.get("strength", 0.35)
    )

    guidance_scale = float(
        job_input.get("guidance_scale", 5.5)
    )

    steps = int(
        job_input.get("steps", 30)
    )

    seed = int(
        job_input.get("seed", 0)
    )

    if seed <= 0:
        seed = torch.randint(
            1,
            2147483647,
            (1,),
        ).item()

    input_image = base64_to_image(
        image_base64
    )

    # Keep image dimensions FLUX-friendly.
    original_width = input_image.width
    original_height = input_image.height

    max_dimension = 1024

    scale = min(
        1.0,
        max_dimension
        / max(
            original_width,
            original_height,
        ),
    )

    width = max(
        64,
        (int(original_width * scale) // 16) * 16,
    )

    height = max(
        64,
        (int(original_height * scale) // 16) * 16,
    )

    input_image = input_image.resize(
        (width, height),
        Image.LANCZOS,
    )

    print(
        f"IMG2IMG: {width}x{height}, "
        f"strength={strength}, "
        f"guidance={guidance_scale}, "
        f"steps={steps}, seed={seed}",
        flush=True,
    )

    model = load_img2img_model()

    generator = torch.Generator(
        device="cpu"
    ).manual_seed(seed)

    image = model(
        prompt=prompt,
        image=input_image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        generator=generator,
    ).images[0]

    # Return result at the original uploaded size.
    image = image.resize(
        (original_width, original_height),
        Image.LANCZOS,
    )

    return {
        "image_base64": image_to_base64(image),
        "seed": seed,
        "task": "img2img",
    }


# ==========================================
# RUNPOD HANDLER
# ==========================================

def handler(job):
    try:
        print(
            "Received RunPod job.",
            flush=True,
        )

        job_input = job.get(
            "input",
            {},
        )

        task = job_input.get(
            "task",
            "txt2img",
        )

        print(
            f"Task requested: {task}",
            flush=True,
        )

        if task == "img2img":
            result = generate_img2img(
                job_input
            )

        elif task == "txt2img":
            result = generate_txt2img(
                job_input
            )

        else:
            raise ValueError(
                f"Unsupported task: {task}"
            )

        print(
            f"{task} completed successfully.",
            flush=True,
        )

        return result

    except Exception as e:
        print(
            "=== JOB FAILED ===",
            flush=True,
        )

        print(
            f"Error type: {type(e).__name__}",
            flush=True,
        )

        print(
            f"Error: {e}",
            flush=True,
        )

        traceback.print_exc()

        return {
            "error": str(e),
            "error_type": type(e).__name__,
        }


# ==========================================
# START RUNPOD SERVERLESS
# ==========================================

print(
    "Starting RunPod serverless handler...",
    flush=True,
)

runpod.serverless.start(
    {
        "handler": handler
    }
)
