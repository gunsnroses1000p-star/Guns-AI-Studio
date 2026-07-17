import io
import os
import sys
import base64
import tempfile
import traceback

import runpod
import torch
import requests
import imageio.v2 as imageio

from PIL import Image

from diffusers import (
    FluxPipeline,
    FluxImg2ImgPipeline,
    LTXImageToVideoPipeline,
)


# ==========================================
# MODEL CONFIGURATION
# ==========================================

MODEL_ID = "black-forest-labs/FLUX.1-dev"
VIDEO_MODEL_ID = "Lightricks/LTX-Video"


print(
    "=== Guns AI Studio RunPod Worker Starting ===",
    flush=True,
)

print(
    f"Python: {sys.version}",
    flush=True,
)

print(
    f"PyTorch: {torch.__version__}",
    flush=True,
)

print(
    f"CUDA available: {torch.cuda.is_available()}",
    flush=True,
)

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}",
        flush=True,
    )


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


txt2img_pipe = None
img2img_pipe = None
img2video_pipe = None


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


def load_img2video_model():
    global img2video_pipe

    if img2video_pipe is not None:
        return img2video_pipe

    print(
        f"Loading Image-to-Video model: {VIDEO_MODEL_ID}",
        flush=True,
    )

    dtype = (
        torch.bfloat16
        if DEVICE == "cuda"
        else torch.float32
    )

    img2video_pipe = LTXImageToVideoPipeline.from_pretrained(
        VIDEO_MODEL_ID,
        torch_dtype=dtype,
    )

    if DEVICE == "cuda":
        print(
            "Enabling CPU offload for Image-to-Video...",
            flush=True,
        )

        img2video_pipe.enable_model_cpu_offload()

    else:
        img2video_pipe.to("cpu")

    if hasattr(
        img2video_pipe,
        "vae",
    ):
        try:
            img2video_pipe.vae.enable_tiling()

            print(
                "VAE tiling enabled.",
                flush=True,
            )

        except Exception as e:
            print(
                f"Could not enable VAE tiling: {e}",
                flush=True,
            )

    print(
        "LTX Image-to-Video model loaded successfully.",
        flush=True,
    )

    return img2video_pipe


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


def load_input_image(job_input):
    """
    Load an image from either image_base64 or image_url.
    """

    image_base64 = job_input.get(
        "image_base64"
    )

    image_url = job_input.get(
        "image_url"
    )

    if image_base64:
        print(
            "Loading input image from base64...",
            flush=True,
        )

        if "," in image_base64:
            image_base64 = image_base64.split(
                ",",
                1,
            )[1]

        image_bytes = base64.b64decode(
            image_base64
        )

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        print(
            "Base64 input image loaded successfully.",
            flush=True,
        )

        return image

    if image_url:
        print(
            f"Downloading input image from URL: {image_url}",
            flush=True,
        )

        response = requests.get(
            image_url,
            timeout=120,
        )

        response.raise_for_status()

        image = Image.open(
            io.BytesIO(response.content)
        ).convert("RGB")

        print(
            "URL input image loaded successfully.",
            flush=True,
        )

        return image

    raise ValueError(
        "No input image provided. "
        "Provide either image_base64 or image_url."
    )


def file_to_base64(file_path):
    with open(
        file_path,
        "rb",
    ) as file:
        return base64.b64encode(
            file.read()
        ).decode("utf-8")


# ==========================================
# TEXT TO IMAGE
# ==========================================

def generate_txt2img(job_input):
    prompt = job_input.get(
        "prompt",
        "A beautiful photorealistic landscape",
    )

    width = int(
        job_input.get(
            "width",
            1024,
        )
    )

    height = int(
        job_input.get(
            "height",
            1024,
        )
    )

    steps = int(
        job_input.get(
            "steps",
            28,
        )
    )

    seed = int(
        job_input.get(
            "seed",
            0,
        )
    )

    if seed <= 0:
        seed = torch.randint(
            1,
            2147483647,
            (1,),
        ).item()

    print(
        f"TXT2IMG: "
        f"{width}x{height}, "
        f"steps={steps}, "
        f"seed={seed}",
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
        "image_base64": image_to_base64(
            image
        ),
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

    strength = float(
        job_input.get(
            "strength",
            0.35,
        )
    )

    guidance_scale = float(
        job_input.get(
            "guidance_scale",
            5.5,
        )
    )

    steps = int(
        job_input.get(
            "steps",
            30,
        )
    )

    seed = int(
        job_input.get(
            "seed",
            0,
        )
    )

    if seed <= 0:
        seed = torch.randint(
            1,
            2147483647,
            (1,),
        ).item()

    input_image = load_input_image(
        job_input
    )

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
        (
            int(
                original_width
                * scale
            )
            // 16
        )
        * 16,
    )

    height = max(
        64,
        (
            int(
                original_height
                * scale
            )
            // 16
        )
        * 16,
    )

    input_image = input_image.resize(
        (
            width,
            height,
        ),
        Image.LANCZOS,
    )

    print(
        f"IMG2IMG: "
        f"{width}x{height}, "
        f"strength={strength}, "
        f"guidance={guidance_scale}, "
        f"steps={steps}, "
        f"seed={seed}",
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

    image = image.resize(
        (
            original_width,
            original_height,
        ),
        Image.LANCZOS,
    )

    return {
        "image_base64": image_to_base64(
            image
        ),
        "seed": seed,
        "task": "img2img",
    }


# ==========================================
# IMAGE TO VIDEO
# ==========================================

def generate_img2video(job_input):
    prompt = job_input.get(
        "prompt",
        (
            "Photorealistic natural subtle motion, "
            "same exact person and facial features as the input image, "
            "stable face, consistent identity, "
            "subtle natural blinking, "
            "very gentle head movement, "
            "realistic skin texture, "
            "physically realistic motion, "
            "static stable facial structure."
        ),
    )

    negative_prompt = job_input.get(
        "negative_prompt",
        (
            "different person, identity change, identity drift, "
            "face morphing, face warping, melting face, squished face, "
            "distorted face, deformed face, changing facial structure, "
            "asymmetrical eyes, deformed eyes, moving facial features, "
            "rubber face, elastic face, stretched face, "
            "blurry face, flickering face, unstable face, "
            "exaggerated expression, excessive head movement, "
            "jittery motion, unnatural motion, low quality"
        ),
    )

    steps = int(
        job_input.get(
            "steps",
            35,
        )
    )

    guidance_scale = float(
        job_input.get(
            "guidance_scale",
            4.0,
        )
    )

    num_frames = int(
        job_input.get(
            "num_frames",
            81,
        )
    )

    fps = int(
        job_input.get(
            "fps",
            24,
        )
    )

    seed = int(
        job_input.get(
            "seed",
            0,
        )
    )

    if seed <= 0:
        seed = torch.randint(
            1,
            2147483647,
            (1,),
        ).item()

    input_image = load_input_image(
        job_input
    )

    if input_image.width >= input_image.height:
        width = 704
        height = 480
    else:
        width = 480
        height = 704

    input_image = input_image.resize(
        (width, height),
        Image.LANCZOS,
    )

    print(
        f"IMG2VIDEO: "
        f"{width}x{height}, "
        f"frames={num_frames}, "
        f"fps={fps}, "
        f"steps={steps}, "
        f"guidance={guidance_scale}, "
        f"seed={seed}",
        flush=True,
    )

    model = load_img2video_model()

    generator_device = (
        "cuda"
        if DEVICE == "cuda"
        else "cpu"
    )

    generator = torch.Generator(
        device=generator_device
    ).manual_seed(seed)

    output = model(
        image=input_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        decode_timestep=0.03,
        decode_noise_scale=0.008,
    )

    frames = output.frames[0]

    print(
        f"Generated {len(frames)} video frames.",
        flush=True,
    )

    temp_file = tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False,
    )

    video_path = temp_file.name
    temp_file.close()

    try:
        imageio.mimsave(
            video_path,
            frames,
            fps=fps,
            codec="libx264",
        )

        print(
            "Video exported successfully.",
            flush=True,
        )

        video_base64 = file_to_base64(
            video_path
        )

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

    return {
        "video_base64": video_base64,
        "seed": seed,
        "fps": fps,
        "num_frames": num_frames,
        "task": "img2video",
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

        elif task == "img2video":
            result = generate_img2video(
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
            f"Error type: "
            f"{type(e).__name__}",
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
