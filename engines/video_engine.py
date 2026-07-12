import random
import requests
import torch
import replicate
import spaces
import gradio as gr
from pathlib import Path
from PIL import Image
from diffusers import (
    LTXImageToVideoPipeline,
    CogVideoXImageToVideoPipeline,
)
from diffusers.utils import export_to_video

from io import BytesIO

from helpers import (
    check_token,
    extract_output,
    HF_TOKEN,
)

# =========================
# LOCAL HF LTX VIDEO PIPELINE
# =========================

_hf_video_pipe = None


def get_hf_video_pipe():
    global _hf_video_pipe

    if _hf_video_pipe is None:
        _hf_video_pipe = LTXImageToVideoPipeline.from_pretrained(
            "Lightricks/LTX-Video",
            torch_dtype=torch.bfloat16,
        )

        _hf_video_pipe.enable_model_cpu_offload()
        _hf_video_pipe.vae.enable_tiling()

    return _hf_video_pipe

@spaces.GPU(duration=300)
def generate_video_hf_local(image, prompt):
    if image is None:
        raise gr.Error("Please upload an image.")

    if not prompt or not prompt.strip():
        prompt = (
            "Natural realistic movement, subtle blinking, gentle head movement, "
            "real-time motion, preserve the same identity and facial features."
        )

    try:
        pipe = get_hf_video_pipe()

        if image.width >= image.height:
            width, height = 704, 480
        else:
            width, height = 480, 704

        input_image = image.convert("RGB").resize(
            (width, height),
            Image.LANCZOS,
        )

        seed = random.randint(1, 2147483647)
        generator = torch.Generator(device="cuda").manual_seed(seed)

        negative_prompt = (
            "different person, identity drift, distorted face, deformed face, "
            "asymmetrical eyes, blurry face, flickering, jittery motion, "
            "slow motion, low quality"
        )

        with torch.inference_mode():
            frames = pipe(
                image=input_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=121,
                num_inference_steps=30,
                guidance_scale=5.0,
                decode_timestep=0.05,
                decode_noise_scale=0.015,
                generator=generator,
            ).frames[0]

        output_path = "outputs/hf_ltx_video.mp4"
        export_to_video(frames, output_path, fps=24)

        return output_path, f"✅ HF LTX video generated locally. Seed: {seed}"

    except Exception as e:
        return None, f"❌ Hugging Face Local Video Error: {str(e)}"


