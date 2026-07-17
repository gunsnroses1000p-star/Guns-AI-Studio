import io
import os
import sys
import base64
import tempfile
import subprocess
import random

import runpod
import torch

from PIL import Image
from diffusers import FluxPipeline
from huggingface_hub import snapshot_download


# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FLUX_MODEL_ID = "black-forest-labs/FLUX.1-dev"
WAN_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"

WAN_REPO_DIR = "/opt/Wan2.2"

# Use RunPod network volume if mounted.
# Otherwise fall back to container storage.
if os.path.isdir("/runpod-volume"):
    MODEL_ROOT = "/runpod-volume/models"
else:
    MODEL_ROOT = "/models"

WAN_MODEL_DIR = os.path.join(
    MODEL_ROOT,
    "Wan2.2-TI2V-5B",
)

_flux_pipe = None


# ============================================================
# HELPERS
# ============================================================

def file_to_base64(file_path):
    with open(file_path, "rb") as file:
        return base64.b64encode(
            file.read()
        ).decode("utf-8")


def image_to_base64(image):
    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


def load_input_image(job_input):
    image_base64 = job_input.get(
        "image_base64"
    )

    if not image_base64:
        raise ValueError(
            "Img2Video requires image_base64."
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


# ============================================================
# MODEL DOWNLOAD
# ============================================================

def ensure_wan_model():
    os.makedirs(
        MODEL_ROOT,
        exist_ok=True,
    )

    if os.path.isdir(WAN_MODEL_DIR):
        files = os.listdir(
            WAN_MODEL_DIR
        )

        if files:
            print(
                "Wan model already available.",
                flush=True,
            )

            return WAN_MODEL_DIR

    print(
        "Downloading Wan 2.2 TI2V-5B...",
        flush=True,
    )

    snapshot_download(
        repo_id=WAN_MODEL_ID,
        local_dir=WAN_MODEL_DIR,
        local_dir_use_symlinks=False,
    )

    print(
        "Wan model download complete.",
        flush=True,
    )

    return WAN_MODEL_DIR


# ============================================================
# FLUX
# ============================================================

def load_flux_model():
    global _flux_pipe

    if _flux_pipe is not None:
        return _flux_pipe

    print(
        f"Loading FLUX on {DEVICE}...",
        flush=True,
    )

    _flux_pipe = FluxPipeline.from_pretrained(
        FLUX_MODEL_ID,
        torch_dtype=(
            torch.bfloat16
            if DEVICE == "cuda"
            else torch.float32
        ),
    )

    if DEVICE == "cuda":
        _flux_pipe.enable_model_cpu_offload()
    else:
        _flux_pipe.to("cpu")

    print(
        "FLUX loaded.",
        flush=True,
    )

    return _flux_pipe


def generate_txt2img(job_input):
    prompt = job_input.get(
        "prompt",
        "A beautiful landscape",
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
        seed = random.randint(
            1,
            2147483647,
        )

    pipe = load_flux_model()

    generator = torch.Generator(
        device="cpu"
    ).manual_seed(seed)

    image = pipe(
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


# ============================================================
# WAN IMAGE TO VIDEO
# ============================================================

def generate_img2video(job_input):
    prompt = job_input.get(
        "prompt",
        (
            "Photorealistic natural subtle motion, "
            "stable facial identity, realistic movement."
        ),
    )

    seed = int(
        job_input.get(
            "seed",
            0,
        )
    )

    if seed <= 0:
        seed = random.randint(
            1,
            2147483647,
        )

    # Keep first tests short.
    num_frames = int(
        job_input.get(
            "num_frames",
            49,
        )
    )

    # Wan requires frame count in 4n+1 form.
    num_frames = max(
        5,
        ((num_frames - 1) // 4) * 4 + 1,
    )

    steps = int(
        job_input.get(
            "steps",
            30,
        )
    )

    input_image = load_input_image(
        job_input
    )

    model_dir = ensure_wan_model()

    image_file = tempfile.NamedTemporaryFile(
        suffix=".png",
        delete=False,
    )

    video_file = tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False,
    )

    image_path = image_file.name
    video_path = video_file.name

    image_file.close()
    video_file.close()

    input_image.save(
        image_path,
        format="PNG",
    )

    command = [
        sys.executable,
        os.path.join(
            WAN_REPO_DIR,
            "generate.py",
        ),

        "--task",
        "ti2v-5B",

        "--size",
        "1280*704",

        "--ckpt_dir",
        model_dir,

        "--offload_model",
        "True",

        "--convert_model_dtype",

        "--t5_cpu",

        "--image",
        image_path,

        "--prompt",
        prompt,

        "--frame_num",
        str(num_frames),

        "--sample_steps",
        str(steps),

        "--base_seed",
        str(seed),

        "--save_file",
        video_path,
    ]

    print(
        "Starting Wan Img2Video...",
        flush=True,
    )

    print(
        f"Frames={num_frames}, "
        f"steps={steps}, "
        f"seed={seed}",
        flush=True,
    )

    try:
        result = subprocess.run(
            command,
            cwd=WAN_REPO_DIR,
            check=True,
            capture_output=True,
            text=True,
        )

        if result.stdout:
            print(
                result.stdout,
                flush=True,
            )

        if not os.path.exists(
            video_path
        ):
            raise RuntimeError(
                "Wan completed but no video file was created."
            )

        video_base64 = file_to_base64(
            video_path
        )

    except subprocess.CalledProcessError as error:
        print(
            error.stdout,
            flush=True,
        )

        print(
            error.stderr,
            flush=True,
        )

        raise RuntimeError(
            "Wan Img2Video generation failed."
        )

    finally:
        if os.path.exists(
            image_path
        ):
            os.remove(
                image_path
            )

    try:
        return {
            "video_base64": video_base64,
            "seed": seed,
            "num_frames": num_frames,
            "task": "img2video",
        }

    finally:
        if os.path.exists(
            video_path
        ):
            os.remove(
                video_path
            )


# ============================================================
# RUNPOD HANDLER
# ============================================================

def handler(job):
    try:
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

        if task == "img2video":
            return generate_img2video(
                job_input
            )

        if task == "txt2img":
            return generate_txt2img(
                job_input
            )

        return {
            "error": (
                f"Unsupported task: {task}"
            )
        }

    except Exception as error:
        print(
            f"Worker error: {error}",
            flush=True,
        )

        return {
            "error": str(error)
        }


runpod.serverless.start(
    {
        "handler": handler
    }
    )
