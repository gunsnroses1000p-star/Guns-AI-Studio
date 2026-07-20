import io
import os
import sys
import base64
import tempfile
import subprocess
import random
import traceback
from pathlib import Path

import runpod
import torch
from PIL import Image
from diffusers import FluxPipeline
from huggingface_hub import snapshot_download


# ============================================================
# BOOT / DIAGNOSTICS
# ============================================================

print("BOOT: starting RunPod serverless handler", flush=True)
print("BOOT: python:", sys.executable, flush=True)
print("BOOT: version:", sys.version, flush=True)
print("BOOT: cwd:", os.getcwd(), flush=True)

try:
    print("BOOT: files:", os.listdir("."), flush=True)
except Exception as error:
    print(f"BOOT: could not list files: {error}", flush=True)

print(
    "BOOT: /runpod-volume exists:",
    os.path.isdir("/runpod-volume"),
    flush=True,
)


# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FLUX_MODEL_ID = "black-forest-labs/FLUX.1-dev"
WAN_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"

WAN_REPO_DIR = "/opt/Wan2.2"

# Use the RunPod network volume when mounted.
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

print(f"BOOT: device={DEVICE}", flush=True)
print(f"BOOT: WAN_REPO_DIR={WAN_REPO_DIR}", flush=True)
print(f"BOOT: WAN_MODEL_DIR={WAN_MODEL_DIR}", flush=True)


# ============================================================
# HELPERS
# ============================================================

def file_to_base64(file_path):
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def decode_base64_data(data):
    if "," in data:
        data = data.split(",", 1)[1]

    return base64.b64decode(data)


def load_input_image(job_input):
    image_url = job_input.get("image_url")
    image_base64 = job_input.get("image_base64")

    if image_url:
        print("Downloading input image from URL...", flush=True)

        import requests

        response = requests.get(
            image_url,
            timeout=60,
        )
        response.raise_for_status()
        image_bytes = response.content

    elif image_base64:
        print("Decoding base64 input image...", flush=True)

        image_bytes = decode_base64_data(
            image_base64
        )

    else:
        raise ValueError(
            "Img2Video requires either image_base64 or image_url."
        )

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    print(
        f"Input image loaded: {image.width}x{image.height}",
        flush=True,
    )

    return image


def random_seed(seed):
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        seed = 0

    if seed > 0:
        return seed

    return random.randint(
        1,
        2147483647,
    )


def sanitize_frames(frame_count):
    try:
        frame_count = int(frame_count)
    except (TypeError, ValueError):
        frame_count = 49

    frame_count = max(
        5,
        frame_count,
    )

    return ((frame_count - 1) // 4) * 4 + 1


# ============================================================
# STREAM SUBPROCESS LOGS
# ============================================================

def run_command_streaming(command, cwd=None):
    print(
        "COMMAND:",
        " ".join(command),
        flush=True,
    )

    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines = []

    try:
        if process.stdout is not None:
            for line in process.stdout:
                output_lines.append(line)
                print(
                    line.rstrip("\n"),
                    flush=True,
                )

        return_code = process.wait()

        return (
            return_code,
            "".join(output_lines),
        )

    finally:
        if process.stdout is not None:
            try:
                process.stdout.close()
            except Exception:
                pass


# ============================================================
# WAN MODEL DOWNLOAD / LOCATION
# ============================================================

def ensure_wan_model():
    os.makedirs(
        MODEL_ROOT,
        exist_ok=True,
    )

    if (
        os.path.isdir(WAN_MODEL_DIR)
        and any(Path(WAN_MODEL_DIR).iterdir())
    ):
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
# FLUX TEXT TO IMAGE
# ============================================================

def load_flux_model():
    global _flux_pipe

    if _flux_pipe is not None:
        return _flux_pipe

    print(
        f"Loading FLUX on {DEVICE}...",
        flush=True,
    )

    dtype = (
        torch.bfloat16
        if DEVICE == "cuda"
        else torch.float32
    )

    _flux_pipe = FluxPipeline.from_pretrained(
        FLUX_MODEL_ID,
        torch_dtype=dtype,
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

    seed = random_seed(
        job_input.get(
            "seed",
            0,
        )
    )

    print(
        "Starting FLUX generation...",
        flush=True,
    )

    print(
        f"Width={width}, height={height}, "
        f"steps={steps}, seed={seed}",
        flush=True,
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

    print(
        "FLUX generation complete.",
        flush=True,
    )

    return {
        "image_base64": image_to_base64(image),
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

    seed = random_seed(
        job_input.get(
            "seed",
            0,
        )
    )

    num_frames = sanitize_frames(
        job_input.get(
            "num_frames",
            81,
        )
    )

    steps = int(
        job_input.get(
            "steps",
            30,
        )
    )

    size = str(
        job_input.get(
            "size",
            "1280*704",
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

    try:
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
            size,
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
            f"seed={seed}, "
            f"size={size}",
            flush=True,
        )

        return_code, _ = run_command_streaming(
            command,
            cwd=WAN_REPO_DIR,
        )

        if return_code != 0:
            raise RuntimeError(
                "Wan generate.py exited "
                f"with code {return_code}."
            )

        if not os.path.exists(video_path):
            raise RuntimeError(
                "Wan completed but no video file was created."
            )

        video_size = os.path.getsize(
            video_path
        )

        if video_size <= 0:
            raise RuntimeError(
                "Wan created an empty video file."
            )

        print(
            "Wan video created successfully. "
            f"File size={video_size} bytes",
            flush=True,
        )

        video_base64 = file_to_base64(
            video_path
        )

        return {
            "video_base64": video_base64,
            "seed": seed,
            "num_frames": num_frames,
            "steps": steps,
            "size": size,
            "task": "img2video",
        }

    finally:
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
        except Exception as error:
            print(
                f"Could not remove temporary image: {error}",
                flush=True,
            )

        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except Exception as error:
            print(
                f"Could not remove temporary video: {error}",
                flush=True,
            )


# ============================================================
# RUNPOD HANDLER
# ============================================================

def handler(job):
    try:
        job_input = (
            job.get(
                "input",
                {},
            )
            or {}
        )

        task = job_input.get(
            "task",
            "txt2img",
        )

        print(
            "====================================",
            flush=True,
        )

        print(
            f"Job ID: {job.get('id')}",
            flush=True,
        )

        print(
            f"Task requested: {task}",
            flush=True,
        )

        print(
            "====================================",
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
            "error": f"Unsupported task: {task}",
            "supported": [
                "txt2img",
                "img2video",
            ],
        }

    except Exception as error:
        print(
            "HANDLER ERROR:",
            str(error),
            flush=True,
        )

        traceback.print_exc()

        return {
            "error": str(error),
            "traceback": traceback.format_exc(),
        }


# ============================================================
# START RUNPOD SERVERLESS WORKER
# ============================================================

runpod.serverless.start(
    {
        "handler": handler
    }
    )
