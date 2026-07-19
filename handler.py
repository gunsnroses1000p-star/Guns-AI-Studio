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


# ============================================================
# BOOT / DIAGNOSTICS
# ============================================================

print("BOOT: starting Runpod serverless handler", flush=True)
print("BOOT: python:", sys.executable, flush=True)
print("BOOT: version:", sys.version, flush=True)
print("BOOT: cwd:", os.getcwd(), flush=True)
print("BOOT: files:", os.listdir("."), flush=True)

# Helpful to know if your volume is present
print("BOOT: /runpod-volume exists:", os.path.isdir("/runpod-volume"), flush=True)

# Optional: quick pip introspection (safe, non-fatal)
def _pip_show(*pkgs):
    try:
        subprocess.run([sys.executable, "-m", "pip", "show", *pkgs], check=False)
    except Exception as e:
        print(f"BOOT: pip show failed: {e}", flush=True)

_pip_show("torch", "diffusers", "huggingface_hub", "runpod")


# ============================================================
# CONFIG
# ============================================================

WAN_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
WAN_REPO_DIR = "/opt/Wan2.2"

# Prefer persistent volume if mounted
MODEL_ROOT = "/runpod-volume/models" if os.path.isdir("/runpod-volume") else "/models"
WAN_MODEL_DIR = os.path.join(MODEL_ROOT, "Wan2.2-TI2V-5B")

# For Flux
FLUX_MODEL_ID = "black-forest-labs/FLUX.1-dev"
_flux_pipe = None


# ============================================================
# HELPERS
# ============================================================

def file_to_base64(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def image_to_base64(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _decode_base64_data(s: str) -> bytes:
    # allow "data:image/png;base64,...."
    if "," in s:
        s = s.split(",", 1)[1]
    return base64.b64decode(s)


def load_input_image(job_input):
    # Prefer URL if provided, else base64
    image_url = job_input.get("image_url")
    image_b64 = job_input.get("image_base64")

    if image_url:
        # Lazy import requests
        import requests
        r = requests.get(image_url, timeout=60)
        r.raise_for_status()
        data = r.content
    elif image_b64:
        data = _decode_base64_data(image_b64)
    else:
        raise ValueError("img2video requires either image_url or image_base64")

    # Lazy import PIL
    from PIL import Image
    return Image.open(io.BytesIO(data)).convert("RGB")


def _rand_seed(seed: int) -> int:
    if seed and int(seed) > 0:
        return int(seed)
    return random.randint(1, 2147483647)


# ============================================================
# MODEL DOWNLOAD
# ============================================================

def ensure_wan_model():
    os.makedirs(MODEL_ROOT, exist_ok=True)

    if os.path.isdir(WAN_MODEL_DIR) and any(Path(WAN_MODEL_DIR).iterdir()):
        print("Wan model already available.", flush=True)
        return WAN_MODEL_DIR

    print("Downloading Wan 2.2 TI2V-5B...", flush=True)

    # Lazy import to avoid import-time crashes
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=WAN_MODEL_ID,
        local_dir=WAN_MODEL_DIR,
        local_dir_use_symlinks=False,
    )

    print("Wan model download complete.", flush=True)
    return WAN_MODEL_DIR


# ============================================================
# FLUX (txt2img)
# ============================================================

def load_flux_model():
    global _flux_pipe
    if _flux_pipe is not None:
        return _flux_pipe

    print("Loading FLUX...", flush=True)

    import torch
    from diffusers import FluxPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    _flux_pipe = FluxPipeline.from_pretrained(FLUX_MODEL_ID, torch_dtype=dtype)

    if device == "cuda":
        _flux_pipe.enable_model_cpu_offload()
    else:
        _flux_pipe.to("cpu")

    print("FLUX loaded.", flush=True)
    return _flux_pipe


def generate_txt2img(job_input):
    import torch

    prompt = job_input.get("prompt", "A beautiful landscape")
    width = int(job_input.get("width", 1024))
    height = int(job_input.get("height", 1024))
    steps = int(job_input.get("steps", 28))
    seed = _rand_seed(int(job_input.get("seed", 0)))

    pipe = load_flux_model()
    generator = torch.Generator(device="cpu").manual_seed(seed)

    image = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        generator=generator,
    ).images[0]

    return {"task": "txt2img", "seed": seed, "image_base64": image_to_base64(image)}


# ============================================================
# WAN (img2video)
# ============================================================

def _sanitize_frames(n: int) -> int:
    # Wan requires 4n+1
    n = max(5, int(n))
    return ((n - 1) // 4) * 4 + 1


def _run_command_streaming(cmd, cwd=None, env=None, timeout=None):
    """
    Stream stdout/stderr to logs so you don't get silent failures.
    Returns (rc, combined_text).
    """
    print("CMD:", " ".join(cmd), flush=True)
    p = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    out_lines = []
    try:
        for line in p.stdout:
            out_lines.append(line)
            print(line.rstrip("\n"), flush=True)
        rc = p.wait(timeout=timeout)
        return rc, "".join(out_lines)
    finally:
        try:
            if p.stdout:
                p.stdout.close()
        except Exception:
            pass


def generate_img2video(job_input):
    prompt = job_input.get(
        "prompt",
        "Photorealistic natural subtle motion, stable identity, realistic movement.",
    )

    seed = _rand_seed(int(job_input.get("seed", 0)))
    num_frames = _sanitize_frames(int(job_input.get("num_frames", 49)))
    steps = int(job_input.get("steps", 30))

    # Optional controls
    size = job_input.get("size", "1280*704")          # Wan CLI expects WxH with *
    offload_model = str(job_input.get("offload_model", True))
    t5_cpu = str(job_input.get("t5_cpu", True))

    input_image = load_input_image(job_input)
    model_dir = ensure_wan_model()

    # write temp inputs/outputs
    image_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    video_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    image_path = image_file.name
    video_path = video_file.name
    image_file.close()
    video_file.close()

    try:
        input_image.save(image_path, format="PNG")

        cmd = [
            sys.executable,
            os.path.join(WAN_REPO_DIR, "generate.py"),
            "--task", "ti2v-5B",
            "--size", str(size),
            "--ckpt_dir", model_dir,
            "--offload_model", str(offload_model),
            "--convert_model_dtype",
            "--t5_cpu", str(t5_cpu),
            "--image", image_path,
            "--prompt", prompt,
            "--frame_num", str(num_frames),
            "--sample_steps", str(steps),
            "--base_seed", str(seed),
            "--save_file", video_path,
        ]

        print("Starting Wan Img2Video...", flush=True)
        print(f"Frames={num_frames}, steps={steps}, seed={seed}, size={size}", flush=True)

        rc, combined = _run_command_streaming(cmd, cwd=WAN_REPO_DIR, timeout=None)
        if rc != 0:
            raise RuntimeError(f"Wan generate.py exited with code {rc}")

        if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            raise RuntimeError("Wan completed but no video file was created (or file is empty).")

        # Warning: big responses for large videos
        video_base64 = file_to_base64(video_path)

        return {
            "task": "img2video",
            "seed": seed,
            "num_frames": num_frames,
            "steps": steps,
            "size": size,
            "video_base64": video_base64,
        }

    finally:
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
        except Exception:
            pass
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except Exception:
            pass


# ============================================================
# RUNPOD HANDLER
# ============================================================

def handler(job):
    """
    job = {"id": "...", "input": {...}}
    """
    try:
        job_input = job.get("input", {}) or {}
        task = job_input.get("task", "txt2img")

        print(f"Job {job.get('id')} task={task}", flush=True)

        if task == "img2video":
            return generate_img2video(job_input)

        if task == "txt2img":
            return generate_txt2img(job_input)

        return {"error": f"Unsupported task: {task}", "supported": ["txt2img", "img2video"]}

    except Exception as e:
        print("HANDLER ERROR:", str(e), flush=True)
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
