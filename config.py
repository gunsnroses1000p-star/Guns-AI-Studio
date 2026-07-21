import os
from pathlib import Path

# API Keys
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
FAL_KEY = os.getenv("FAL_KEY") 
CIVITAI_KEY = os.getenv("CIVITAI_API_KEY") 

# Ensure outputs directory exists
Path("outputs").mkdir(exist_ok=True)

# Model Defaults
DEFAULT_IMAGE_MODEL = "black-forest-labs/FLUX.1-dev"
DEFAULT_LORA_MODEL = "black-forest-labs/flux-dev-lora"
LORA_URL = "https://huggingface.co/spaces/Guns6996/guns-lora-app/resolve/main/flux-lora.safetensors"

DEFAULT_NEGATIVE = (
    "blurry, low quality, cartoon, anime, CGI, 3d render, digital painting, "
    "smooth plastic skin, perfect porcelain skin, doll skin, wax skin, plastic skin, overprocessed face, beauty filter, glamour makeup, "
    "thick eyebrows, blocky eyebrows, painted eyebrows, sharp eyebrows, black lipstick, "
    "dark lipstick, heavy lipstick, oversized lips, oversized eyes, distorted face, "
    "duplicate face, bad anatomy, extra fingers, malformed hands, octane render, unreal engine, "
    "subsurface scattering, ambient occlusion, perfectly symmetrical, airbrushed"
)

css = """
body, .gradio-container {
    max-width: 100% !important;
    overflow-x: hidden !important;
}
.tab-nav {
    justify-content: center !important;
    overflow-x: auto !important;
    white-space: nowrap !important;
}
.tab-nav button {
    min-width: fit-content !important;
}
button {
    background: linear-gradient(135deg, #8b3dff, #d946ef) !important;
    color: white !important;
    border-radius: 14px !important;
    font-weight: bold !important;
}
"""
