print("HANDLER.PY PROCESS STARTED", flush=True)

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

from huggingface_hub import snapshot_download
from diffusers import (
    FluxPipeline,
    StableDiffusionXLImg2ImgPipeline,
)

# ============================================================
# BOOT / DIAGNOSTICS
# ============================================================

print("BOOT: starting Runpod serverless handler", flush=True)
print("BOOT: python:", sys.executable, flush=True)
print("BOOT: version:", sys.version, flush=True)
print("BOOT: cwd:", os.getcwd(), flush=True)

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
print("BOOT: HF token present:", bool(HF_TOKEN), flush=True)

print("BOOT: cuda available:", torch.cuda.is_available(), flush=True)
print("BOOT: /runpod-volume exists:", os.path.isdir("/runpod-volume"), flush=True)

# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# FLUX (txt2img)
FLUX_MODEL_ID = "black-forest-labs/FLUX.1-dev"

# SDXL (img2img)
SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

# Wan2.2 (img2video)
WAN_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
WAN_REPO_DIR = "/opt/Wan2.2"

# Model cache root
MODEL_ROOT = "/runpod-volume/models" if os.path.isdir("/runpod-volume") else "/models"

WAN_MODEL_DIR = os.path.join(MODEL_ROOT, "Wan2.2-TI2V-5B")

print(f"BOOT: device={DEVICE}", flush=True)
print(f"BOOT: MODEL_ROOT={MODEL_ROOT}", flush=True)
print(f"BOOT: WAN_MODEL_DIR={WAN_MODEL_DIR}", flush=True)

# Global cached pipelines
_flux_pipe = None
_sdxl_img2img_pipe = None

# ============================================================
# HELPERS
# ============================================================

def file_to_base64(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def decode_base64_data(data: str) -> bytes:
    # Accept "data:image/png;base64,...." as well as raw base64
    if "," in data:
        data = data.split(",", 1)[1]
    return base64.b64decode(data)


def load_input_image(job_input: dict) -> Image.Image:
    image_url = job_input.get("image_url")
    image_base64 = job_input.get("image_base64")

    if image_url:
        print("Downloading input image from URL...", flush=True)
        import requests
        r = requests.get(image_url, timeout=60)
        r.raise_for_status()
        img_bytes = r.content

    elif image_base64:
        print("Decoding base64 input image...", flush=True)
        img_bytes = decode_base64_data(image_base64)

    else:
        raise ValueError("Expected either image_base64 or image_url.")

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    print(f"Input image loaded: {img.width}x{img.height}", flush=True)
    return img


def random_seed(seed):
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        seed = 0
    return seed if seed > 0 else random.randint(1, 2147483647)


def sanitize_frames(frame_count):
    try:
        frame_count = int(frame_count)
    except (TypeError, ValueError):
        frame_count = 49
    frame_count = max(5, frame_count)
    return ((frame_count - 1) // 4) * 4 + 1


# ============================================================
# WAN DOWNLOAD / LOCATION
# ============================================================

def ensure_wan_model() -> str:
    os.makedirs(MODEL_ROOT, exist_ok=True)

    if os.path.isdir(WAN_MODEL_DIR) and any(Path(WAN_MODEL_DIR).iterdir()):
        print("Wan model already available.", flush=True)
        return WAN_MODEL_DIR

    print("Downloading Wan 2.2 TI2V-5B...", flush=True)

    snapshot_download(
        repo_id=WAN_MODEL_ID,
        local_dir=WAN_MODEL_DIR,
        local_dir_use_symlinks=False,
        token=HF_TOKEN,
    )

    print("Wan model download complete.", flush=True)
    return WAN_MODEL_DIR


# ============================================================
# FLUX TXT2IMG
# ============================================================

def load_flux_model():
    global _flux_pipe
    if _flux_pipe is not None:
        return _flux_pipe

    print(f"Loading FLUX on {DEVICE}...", flush=True)
    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32

    _flux_pipe = FluxPipeline.from_pretrained(
        FLUX_MODEL_ID,
        torch_dtype=dtype,
        token=HF_TOKEN,
    )

    # Offload to reduce VRAM; adjust if you want max speed
    if DEVICE == "cuda":
        _flux_pipe.enable_model_cpu_offload()
    else:
        _flux_pipe.to("cpu")

    print("FLUX loaded.", flush=True)
    return _flux_pipe


def generate_txt2img(job_input: dict) -> dict:
    prompt = job_input.get("prompt", "A beautiful landscape")
    width = int(job_input.get("width", 1024))
    height = int(job_input.get("height", 1024))
    steps = int(job_input.get("steps", 28))
    seed = random_seed(job_input.get("seed", 0))

    print("Starting FLUX generation...", flush=True)
    print(f"Width={width}, height={height}, steps={steps}, seed={seed}", flush=True)

    pipe = load_flux_model()
    generator = torch.Generator(device="cpu").manual_seed(seed)

    image = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        generator=generator,
    ).images[0]

    print("FLUX generation complete.", flush=True)

    return {
        "task": "txt2img",
        "seed": seed,
        "image_base64": image_to_base64(image),
    }


# ============================================================
# SDXL IMG2IMG
# ============================================================

def load_sdxl_img2img():
    global _sdxl_img2img_pipe
    if _sdxl_img2img_pipe is not None:
        return _sdxl_img2img_pipe

    print(f"Loading SDXL Img2Img on {DEVICE}...", flush=True)

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    _sdxl_img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        SDXL_MODEL_ID,
        torch_dtype=dtype,
        token=HF_TOKEN,
    )

    if DEVICE == "cuda":
        _sdxl_img2img_pipe.to("cuda")
    else:
        _sdxl_img2img_pipe.to("cpu")

    print("SDXL Img2Img loaded.", flush=True)
    return _sdxl_img2img_pipe


def generate_img2img(job_input: dict) -> dict:
    prompt = job_input.get("prompt", "")
    if not prompt.strip():
        raise ValueError("img2img requires a non-empty prompt.")

    init_image = load_input_image(job_input)

    strength = float(job_input.get("strength", 0.35))
    strength = max(0.0, min(1.0, strength))

    guidance_scale = float(job_input.get("guidance_scale", job_input.get("guidance", 5.5)))
    steps = int(job_input.get("steps", 30))
    seed = random_seed(job_input.get("seed", 0))

    print("Starting SDXL img2img...", flush=True)
    print(
        f"strength={strength}, guidance_scale={guidance_scale}, steps={steps}, seed={seed}",
        flush=True,
    )

    pipe = load_sdxl_img2img()

    # SDXL uses CUDA generator if on GPU
    gen_device = "cuda" if DEVICE == "cuda" else "cpu"
    generator = torch.Generator(device=gen_device).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        image=init_image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        generator=generator,
    ).images[0]

    print("SDXL img2img complete.", flush=True)

    return {
        "task": "img2img",
        "seed": seed,
        "image_base64": image_to_base64(result),
    }


# ============================================================
# WAN IMG2VIDEO (subprocess)
# ============================================================

def run_command_streaming(command, cwd=None):
    print("COMMAND:", " ".join(command), flush=True)

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
                print(line.rstrip("\n"), flush=True)

        return_code = process.wait()
        return return_code, "".join(output_lines)

    finally:
        if process.stdout is not None:
            try:
                process.stdout.close()
            except Exception:
                pass


def generate_img2video(job_input: dict) -> dict:
    prompt = job_input.get(
        "prompt",
        "Photorealistic natural subtle motion, stable facial identity, realistic movement.",
    )

    seed = random_seed(job_input.get("seed", 0))
    num_frames = sanitize_frames(job_input.get("num_frames", 81))
    steps = int(job_input.get("steps", 30))
    size = str(job_input.get("size", "1280*704"))

    input_image = load_input_image(job_input)
    model_dir = ensure_wan_model()

    image_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    video_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    image_path = image_file.name
    video_path = video_file.name
    image_file.close()
    video_file.close()

    try:
        input_image.save(image_path, format="PNG")

        command = [
            sys.executable,
            os.path.join(WAN_REPO_DIR, "generate.py"),
            "--task", "ti2v-5B",
            "--size", size,
            "--ckpt_dir", model_dir,
            "--offload_model", "True",
            "--convert_model_dtype",
            "--t5_cpu",
            "--image", image_path,
            "--prompt", prompt,
            "--frame_num", str(num_frames),
            "--sample_steps", str(steps),
            "--base_seed", str(seed),
            "--save_file", video_path,
        ]

        print("Starting Wan Img2Video...", flush=True)
        print(f"Frames={num_frames}, steps={steps}, seed={seed}, size={size}", flush=True)

        return_code, _ = run_command_streaming(command, cwd=WAN_REPO_DIR)

        if return_code != 0:
            raise RuntimeError(f"Wan generate.py exited with code {return_code}.")

        if (not os.path.exists(video_path)) or os.path.getsize(video_path) <= 0:
            raise RuntimeError("Wan completed but produced no valid video file.")

        # Re-encode to 16 FPS
        fps_video_path = video_path + ".fps16.mp4"
        fps_command = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-r", "16",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            fps_video_path,
        ]

        print("Converting video to 16 FPS...", flush=True)
        subprocess.run(fps_command, check=True)
        os.replace(fps_video_path, video_path)
        print("Video converted to 16 FPS successfully.", flush=True)

        return {
            "task": "img2video",
            "seed": seed,
            "num_frames": num_frames,
            "steps": steps,
            "size": size,
            "video_base64": file_to_base64(video_path),
        }

    finally:
        for p in (image_path, video_path):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception as e:
                print(f"Could not remove temp file {p}: {e}", flush=True)


# ============================================================
# RUNPOD HANDLER
# ============================================================

def handler(job):
    try:
        job_input = job.get("input", {}) or {}
        task = job_input.get("task", "txt2img")

        print("====================================", flush=True)
        print(f"Job ID: {job.get('id')}", flush=True)
        print(f"Task requested: {task}", flush=True)
        print("====================================", flush=True)

        if task == "txt2img":
            return generate_txt2img(job_input)

        if task == "img2img":
            return generate_img2img(job_input)

        if task == "img2video":
            return generate_img2video(job_input)

        return {
            "error": f"Unsupported task: {task}",
            "supported": ["txt2img", "img2img", "img2video"],
        }

    except Exception as e:
        print("HANDLER ERROR:", str(e), flush=True)
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
