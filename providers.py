import os
import random
import time
import requests
import replicate
import fal_client
import gradio as gr
from pathlib import Path
from PIL import Image
from io import BytesIO
import base64

# Import from our new modular files
from config import (
    RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID, REPLICATE_API_TOKEN, 
    HF_TOKEN, FAL_KEY, CIVITAI_KEY, DEFAULT_IMAGE_MODEL, DEFAULT_NEGATIVE
)
from image_utils import check_token, extract_output, decode_runpod_output, save_single_reference, save_combo_image

def generate_with_runpod(prompt, negative_prompt, width, height, steps, seed):
    if not RUNPOD_API_KEY or not RUNPOD_ENDPOINT_ID:
        raise gr.Error("RunPod API Key or Endpoint ID is missing in environment.")

    endpoint_url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/run"
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "input": {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": int(width),
            "height": int(height),
            "steps": int(steps),
            "seed": int(seed) if int(seed) > 0 else -1
        }
    }

    response = requests.post(endpoint_url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    job_id = response.json().get("id")

    if not job_id:
        raise gr.Error("RunPod did not return a job ID.")

    status_url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status/{job_id}"
    for _ in range(180):
        status_response = requests.get(status_url, headers=headers, timeout=30)
        status_response.raise_for_status()
        result = status_response.json()
        status = result.get("status")

        if status == "COMPLETED":
            return decode_runpod_output(result.get("output", {}))
        if status in {"FAILED", "CANCELLED", "TIMED_OUT"}:
            raise gr.Error(result.get("error", "RunPod generation failed."))
        time.sleep(2)

    raise gr.Error("RunPod generation timed out.")

def call_fal_ai(prompt, image=None):
    try:
        image_url = None
        if image:
            path = image if isinstance(image, str) else "/tmp/fal_input.png"
            if not isinstance(image, str):
                image.convert("RGB").save(path)
            image_url = fal_client.upload_file(path)

        handler = fal_client.submit("fal-ai/flux/dev", arguments={"prompt": prompt, "image_url": image_url})
        result = handler.get()
        return result['images'][0]['url'], "✅ Generated via Fal.ai"
    except Exception as e:
        return None, f"❌ Fal.ai Error: {str(e)}"

def call_civitai(prompt, model_id="123456"): 
    try:
        url = "https://api.civitai.com/v1/model-runtime/image"
        headers = {"Authorization": f"Api-Key {CIVITAI_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": {"publishedModelId": model_id, "versionId": None},
            "parameters": {"prompt": prompt, "negativePrompt": DEFAULT_NEGATIVE, "steps": 25, "sampler": "Karras"}
        }
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 201:
            return response.json()['url'], "✅ Generated via Civitai"
        return None, f"❌ Civitai Error: {response.text}"
    except Exception as e:
        return None, f"❌ Connection Error: {str(e)}"

def generate_lora(prompt, negative_prompt, model, lora_url, lora_scale, width, height, steps, seed):
    check_token()
    if seed == 0: seed = random.randint(1, 999999999)
    output = replicate.run(model, input={
        "prompt": prompt, "negative_prompt": negative_prompt, "lora_weights": lora_url, 
        "lora_scale": float(lora_scale), "width": int(width), "height": int(height), 
        "num_inference_steps": int(steps), "seed": int(seed), "output_format": "png"
    })
    return extract_output(output), f"✅ LoRA image generated. Seed: {seed}"

def generate_ai_seamless_lora(image1, image2, prompt, negative_prompt, model, lora_url, lora_scale, width, height, steps, seed):
    check_token()
    if seed == 0: seed = random.randint(1, 999999999)
    combo_path = save_combo_image(image1, image2)
    final_prompt = f"Transform into one natural scene. Preserve identity. {prompt}"
    inputs = {
        "prompt": final_prompt, "input_image": Path(combo_path), "aspect_ratio": "match_input_image", 
        "guidance": 2.5, "num_inference_steps": int(steps), "output_format": "png", "seed": int(seed)
    }
    if lora_url:
        inputs["lora_weights"] = lora_url
        inputs["lora_strength"] = float(lora_scale)
    output = replicate.run(model or "black-forest-labs/flux-kontext-dev-lora", input=inputs)
    return extract_output(output), f"✅ AI Seamless Kontext generated. Seed: {seed}"

def generate_reference_lora(reference_image, prompt, negative_prompt, model, lora_url, lora_scale, width, height, steps, seed):
    check_token()
    if seed == 0: seed = random.randint(1, 999999999)
    ref_path = save_single_reference(reference_image)
    final_prompt = f"Photorealistic scene based on reference. {prompt}"
    inputs = {
        "prompt": final_prompt, "negative_prompt": negative_prompt, "image": Path(ref_path), 
        "width": int(width), "height": int(height), "num_inference_steps": int(steps), "seed": int(seed), "output_format": "png"
    }
    if lora_url:
        inputs["lora_weights"] = lora_url
        inputs["lora_scale"] = float(lora_scale)
    output = replicate.run(model, input=inputs)
    return extract_output(output), f"✅ Reference LoRA generated. Seed: {seed}"

def generate_with_provider(provider, prompt, negative_prompt, model, width, height, steps, seed, init_image, civitai_model_id):
    try:
        if not prompt or not prompt.strip(): return None, "❌ Please enter a prompt."
        if seed is None or int(seed) == 0: seed = random.randint(1, 999999999)
        
        if provider == "Replicate":
            check_token()
            inputs = {"prompt": prompt, "negative_prompt": negative_prompt, "width": int(width), 
                      "height": int(height), "num_inference_steps": int(steps), "seed": int(seed), "output_format": "png"}
            if init_image: inputs["image"] = Path(init_image)
            output = replicate.run(model or DEFAULT_IMAGE_MODEL, input=inputs)
            return extract_output(output), f"✅ Replicate image generated. Seed: {seed}"

        elif provider == "RunPod":
            image = generate_with_runpod(prompt, negative_prompt, width, height, steps, seed)
            output_path = "outputs/runpod_generated_image.png"
            image.save(output_path)
            return output_path, f"✅ RunPod image generated. Seed: {seed}"

        elif provider == "Fal.ai":
            return call_fal_ai(prompt, init_image)

        elif provider == "Civitai":
            return call_civitai(prompt, civitai_model_id if civitai_model_id else "123456")

        return None, "❌ Unknown image provider."
    except Exception as e:
        return None, f"❌ Image Generation Error: {str(e)}"
