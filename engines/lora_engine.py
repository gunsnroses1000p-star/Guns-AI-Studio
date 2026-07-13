import random
import gradio as gr
from pathlib import Path
import replicate

# Import modular components
from config import REPLICATE_API_TOKEN, LORA_URL
from image_utils import check_token, extract_output, save_single_reference, save_combo_image

def generate_lora(prompt, negative_prompt, model, lora_url, lora_scale, width, height, steps, seed):
    """
    Standard LoRA generation via Replicate.
    """
    check_token()
    if not lora_url: 
        lora_url = LORA_URL
    
    if seed == 0: 
        seed = random.randint(1, 999999999)
    
    output = replicate.run(
        model, 
        input={
            "prompt": prompt, 
            "negative_prompt": negative_prompt, 
            "lora_weights": lora_url, 
            "lora_scale": float(lora_scale), 
            "width": int(width), 
            "height": int(height), 
            "num_inference_steps": int(steps), 
            "seed": int(seed), 
            "output_format": "png"
        }
    )
    return extract_output(output), f"✅ LoRA image generated. Seed: {seed}"

def generate_ai_seamless_lora(image1, image2, prompt, negative_prompt, model, lora_url, lora_scale, width, height, steps, seed):
    """
    Specialized logic for combining two images into one scene using the Kontext LoRA.
    """
    check_token()
    if image1 is None or image2 is None: 
        raise gr.Error("Upload both reference images first.")
    if not prompt: 
        raise gr.Error("Please enter a prompt.")
    
    if seed == 0: 
        seed = random.randint(1, 999999999)
    
    # Uses our image_utils to create the horizontal collage
    combo_path = save_combo_image(image1, image2)
    final_prompt = f"Transform into one natural scene. Preserve identity. {prompt}"
    
    inputs = {
        "prompt": final_prompt, 
        "input_image": Path(combo_path), 
        "aspect_ratio": "match_input_image", 
        "guidance": 2.5, 
        "num_inference_steps": int(steps), 
        "output_format": "png", 
        "seed": int(seed)
    }
    
    if lora_url:
        inputs["lora_weights"] = lora_url
        inputs["lora_strength"] = float(lora_scale)
    
    output = replicate.run(
        model or "black-forest-labs/flux-kontext-dev-lora", 
        input=inputs
    )
    return extract_output(output), f"✅ AI Seamless Kontext generated. Seed: {seed}"

def generate_reference_lora(reference_image, prompt, negative_prompt, model, lora_url, lora_scale, width, height, steps, seed):
    """
    Generates a scene based on a single reference image via LoRA.
    """
    check_token()
    if not lora_url: 
        lora_url = LORA_URL
    
    if seed == 0: 
        seed = random.randint(1, 999999999)
    
    # Uses our image_utils to save a temporary ref image
    ref_path = save_single_reference(reference_image)
    final_prompt = f"Photorealistic scene based on reference. {prompt}"
    
    inputs = {
        "prompt": final_prompt, 
        "negative_prompt": negative_prompt, 
        "image": Path(ref_path), 
        "width": int(width), 
        "height": int(height), 
        "num_inference_steps": int(steps), 
        "seed": int(seed), 
        "output_format": "png"
    }
    
    if lora_url:
        inputs["lora_weights"] = lora_url
        inputs["lora_scale"] = float(lora_scale)
        
    output = replicate.run(model, input=inputs)
    return extract_output(output), f"✅ Reference LoRA generated. Seed: {seed}"
