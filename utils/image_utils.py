
import gradio as gr
from PIL import Image
from pathlib import Path
import base64
from io import BytesIO
import requests
from config import REPLICATE_API_TOKEN

def check_token():
    if not REPLICATE_API_TOKEN:
        raise gr.Error("Missing REPLICATE_API_TOKEN in Hugging Face Secrets.")

def extract_output(output):
    if isinstance(output, list):
        output = output[0]
    return str(output)

def save_single_reference(image):
    path = "/tmp/reference_image.png"
    image.convert("RGB").save(path)
    return path

def save_combo_image(image1, image2):
    target_height = 512
    def resize_keep_aspect(img):
        img = img.convert("RGB")
        w, h = img.size
        scale = target_height / h
        new_w = (int(w * scale) // 8) * 8
        return img.resize((new_w, target_height))
    
    img1 = resize_keep_aspect(image1)
    img2 = resize_keep_aspect(image2)
    total_width = img1.width + img2.width
    combo = Image.new("RGB", (total_width, target_height))
    combo.paste(img1, (0, 0))
    combo.paste(img2, (img1.width, 0))
    path = "/tmp/ai_seamless_reference.png"
    combo.save(path)
    return path

def decode_runpod_output(output):
    try:
        if isinstance(output, str):
            if output.startswith("http"):
                response = requests.get(output, timeout=120)
                response.raise_for_status()
                return Image.open(BytesIO(response.content)).convert("RGB")
            
            image_data = output
            if "," in image_data:
                image_data = image_data.split(",", 1)[1]
            image_bytes = base64.b64decode(image_data)
            return Image.open(BytesIO(image_bytes)).convert("RGB")

        if isinstance(output, dict):
            image_url = (output.get("image_url") or output.get("url") or output.get("output_url"))
            if image_url:
                response = requests.get(image_url, timeout=120)
                response.raise_for_status()
                return Image.open(BytesIO(response.content)).convert("RGB")
            
            image_base64 = (output.get("image_base64") or output.get("image"))
            if image_base64:
                if "," in image_base64:
                    image_base64 = image_base64.split(",", 1)[1]
                image_bytes = base64.b64decode(image_base64)
                return Image.open(BytesIO(image_bytes)).convert("RGB")

        raise ValueError(f"Unsupported RunPod output format: {output}")
    except Exception as e:
        raise gr.Error(f"Could not decode RunPod output: {str(e)}")
