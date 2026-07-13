import torch
import random
import spaces
import gradio as gr
from PIL import Image
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    DPMSolverMultistepScheduler,
)

# Import modular components
from face_engine import preserve_original_face

# Global pipeline cache to prevent reloading model from disk on every generation
_img2img_pipe = None
_current_pipe_repo = None

def _get_img2img_pipe(repo_id):
    """
    Internal helper to manage the pipeline singleton.
    Only reloads the model if the repo_id changes.
    """
    global _img2img_pipe, _current_pipe_repo
    if _img2img_pipe is not None and _current_pipe_repo == repo_id:
        return _img2img_pipe
    
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        repo_id, 
        torch_dtype=torch.float16, 
        safety_checker=None
    ).to("cuda")
    
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    _img2img_pipe = pipe
    _current_pipe_repo = repo_id
    return pipe

@spaces.GPU(duration=120)
def generate_img2img_local(prompt, init_image, init_image_2, model, strength, guidance, steps, seed, preserve_face, face_blend):
    """
    Local Img2Img logic: Resizes image for L4 GPU, runs the diffusion pipeline,
    and optionally applies the face preservation mask.
    """
    if not prompt: 
        raise gr.Error("Please enter a prompt.")
    if init_image is None: 
        raise gr.Error("Please upload an image.")
    
    # Fallback to SD v1.5 if no model specified
    repo_id = model if model else "runwayml/stable-diffusion-v1-5"
    pipe = _get_img2img_pipe(repo_id)
    
    # 1. Smart Resizing for VRAM Efficiency
    original_w, original_h = init_image.size
    max_dim = 768 
    scale = max_dim / max(original_w, original_h)
    new_w, new_h = (int(original_w * scale)//8)*8, (int(original_h * scale)//8)*8
    pil_image = init_image.convert("RGB").resize((new_w, new_h), Image.LANCZOS)
    
    # Handle Random Seed
    if seed == 0: 
        seed = random.randint(1, 2147483647)
    generator = torch.Generator(device="cuda").manual_seed(int(seed))
    
    # 2. Run Diffusion Pipeline
    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            image=pil_image,
            strength=float(strength), 
            guidance_scale=float(guidance),
            num_inference_steps=int(steps),
            generator=generator,
        ).images[0]
    
    # Restore to original resolution
    result = result.resize((original_w, original_h), Image.LANCZOS)
    
    # 3. Identity Preservation Bridge
    if preserve_face:
        # Connects to face_engine.py logic
        result = preserve_original_face(init_image, result, strength=float(face_blend))
        
    return result, f"✅ Img2Img complete. Seed: {seed}"
