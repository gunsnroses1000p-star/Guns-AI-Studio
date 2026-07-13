import torch
import random
import gc
import spaces
import numpy as np
from PIL import Image
from pathlib import Path
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionXLPipeline,
    LTXImageToVideoPipeline,
    CogVideoXImageToVideoPipeline,
)
from diffusers.utils import export_to_video
from huggingface_hub import hf_hub_download
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlusXL
import gradio as gr

# Import modular components
from config import DEFAULT_IMAGE_MODEL
from face_engine import load_face_analyzer, preserve_original_face

# Global pipeline caches to prevent reloading from Disk to GPU on every click
_img2img_pipe = None
_current_pipe_repo = None
_hf_video_pipe = None

def _get_img2img_pipe(repo_id):
    global _img2img_pipe, _current_pipe_repo
    if _img2img_pipe is not None and _current_pipe_repo == repo_id:
        return _img2img_pipe
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        repo_id, torch_dtype=torch.float16, safety_checker=None
    ).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    _img2img_pipe = pipe
    _current_pipe_repo = repo_id
    return pipe

@spaces.GPU(duration=120)
def generate_img2img_local(prompt, init_image, init_image_2, model, strength, guidance, steps, seed, preserve_face, face_blend):
    if not prompt: raise gr.Error("Please enter a prompt.")
    if init_image is None: raise gr.Error("Please upload an image.")
    
    repo_id = model if model else "runwayml/stable-diffusion-v1-5"
    pipe = _get_img2img_pipe(repo_id)
    
    original_w, original_h = init_image.size
    max_dim = 768 
    scale = max_dim / max(original_w, original_h)
    new_w, new_h = (int(original_w * scale)//8)*8, (int(original_h * scale)//8)*8
    pil_image = init_image.convert("RGB").resize((new_w, new_h), Image.LANCZOS)
    
    if seed == 0: seed = random.randint(1, 2147483647)
    generator = torch.Generator(device="cuda").manual_seed(int(seed))
    
    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            image=pil_image,
            strength=float(strength), 
            guidance_scale=float(guidance),
            num_inference_steps=int(steps),
            generator=generator,
        ).images[0]
    
    result = result.resize((original_w, original_h), Image.LANCZOS)
    if preserve_face:
        result = preserve_original_face(init_image, result, strength=float(face_blend))
    return result, f"✅ Img2Img complete. Seed: {seed}"

@spaces.GPU
def generate_ip_adapter_scene(face_image, prompt, negative_prompt, width, height, steps, seed, identity_strength):
    if face_image is None: return None, "❌ Please upload a reference face image."
    try:
        analyzer = load_face_analyzer()
        ref_img = Image.open(face_image).convert("RGB")
        faces = analyzer.get(np.array(ref_img))
        if not faces: return None, "❌ No face detected."
        
        face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
        faceid_embeds = torch.from_numpy(face.normed_embedding).unsqueeze(0)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        pipe = StableDiffusionXLPipeline.from_pretrained("SG161222/RealVisXL_V4.0", torch_dtype=dtype, variant="fp16").to(device)
        ip_ckpt = hf_hub_download(repo_id="h94/IP-Adapter-FaceID", filename="ip-adapter-faceid-plusv2_sdxl.bin")
        ip_model = IPAdapterFaceIDPlusXL(pipe, "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", ip_ckpt, device)
        
        image = ip_model.generate(
            prompt=prompt, negative_prompt=negative_prompt, face_image=ref_img, faceid_embeds=faceid_embeds,
            width=int(width), height=int(height), num_samples=1, num_inference_steps=int(steps),
            scale=float(identity_strength), s_scale=1.0, shortcut=True, seed=int(seed) if seed else None
        )[0]
        
        out_path = "outputs/ip_adapter_result.png"
        image.save(out_path)
        return out_path, "✅ FaceID Image generated"
    except Exception as e: return None, f"❌ FaceID error: {e}"

def get_hf_video_pipe():
    global _hf_video_pipe
    if _hf_video_pipe is None:
        _hf_video_pipe = LTXImageToVideoPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16)
        _hf_video_pipe.enable_model_cpu_offload()
        _hf_video_pipe.vae.enable_tiling()
    return _hf_video_pipe

@spaces.GPU(duration=300)
def generate_video_hf_local(image, prompt):
    if image is None: raise gr.Error("Please upload an image.")
    if not prompt or not prompt.strip():
        prompt = "Natural realistic movement, subtle blinking, gentle head movement."

    try:
        pipe = get_hf_video_pipe()
        width, height = (704, 480) if image.width >= image.height else (480, 704)
        input_image = image.convert("RGB").resize((width, height), Image.LANCZOS)
        
        seed = random.randint(1, 2147483647)
        generator = torch.Generator(device="cuda").manual_seed(seed)

        with torch.inference_mode():
            frames = pipe(
                image=input_image, prompt=prompt,
                width=width, height=height, num_frames=121,
                num_inference_steps=30, guidance_scale=5.0,
                generator=generator,
            ).frames[0]

        output_path = "outputs/hf_ltx_video.mp4"
        export_to_video(frames, output_path, fps=24)
        return output_path, f"✅ HF LTX video generated locally. Seed: {seed}"
    except Exception as e:
        return None, f"❌ HF Local Video Error: {str(e)}"

@spaces.GPU(duration=180)
def generate_hf_cogvideo(image, prompt):
    try:
        if image is None: return None, "❌ Please upload an image."
        image = image.convert("RGB")
        image.thumbnail((720, 480))

        pipe = CogVideoXImageToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()

        generator = torch.Generator(device="cuda").manual_seed(random.randint(1, 2_147_483_647))
        
        frames = pipe(
            image=image, prompt=prompt,
            num_frames=49, num_inference_steps=30,
            guidance_scale=6.0, generator=generator,
        ).frames[0]

        output_path = "outputs/cogvideo_high_motion.mp4"
        export_to_video(frames, output_path, fps=8)

        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        return output_path, "✅ HF CogVideoX high-motion video completed."
    except Exception as e:
        return None, f"❌ CogVideoX Error: {str(e)}"
