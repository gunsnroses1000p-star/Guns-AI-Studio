import random
import requests
import torch
import replicate
import spaces

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


