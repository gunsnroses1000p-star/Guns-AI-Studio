import torch
import numpy as np
import spaces
import gradio as gr
from PIL import Image
from huggingface_hub import hf_hub_download
from diffusers import StableDiffusionXLPipeline
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlusXL

# Import modular components from your existing files
from face_engine import load_face_analyzer

@spaces.GPU
def generate_ip_adapter_scene(face_image, prompt, negative_prompt, width, height, steps, seed, identity_strength):
    """
    Handles the IP-Adapter FaceID Plus XL pipeline to generate an 
    image based on a reference identity.
    """
    if face_image is None: 
        return None, "❌ Please upload a reference face image."
    
    try:
        # 1. Face Analysis & Embedding Extraction
        analyzer = load_face_analyzer()
        ref_img = Image.open(face_image).convert("RGB")
        faces = analyzer.get(np.array(ref_img))
        
        if not faces: 
            return None, "❌ No face detected."
        
        # Grab the most prominent face (largest bounding box)
        face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
        faceid_embeds = torch.from_numpy(face.normed_embedding).unsqueeze(0)
        
        # 2. Device & Pipeline Setup
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        # Using the high-quality RealVisXL model from your app.py
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "SG161222/RealVisXL_V4.0", 
            torch_dtype=dtype, 
            variant="fp16"
        ).to(device)
        
        # Download the specific IP-Adapter bin file
        ip_ckpt = hf_hub_download(
            repo_id="h94/IP-Adapter-FaceID", 
            filename="ip-adapter-faceid-plusv2_sdxl.bin"
        )
        
        # Initialize the IP-Adapter FaceID Plus XL model
        ip_model = IPAdapterFaceIDPlusXL(
            pipe, 
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", 
            ip_ckpt, 
            device
        )
        
        # 3. Generation
        image = ip_model.generate(
            prompt=prompt, 
            negative_prompt=negative_prompt, 
            face_image=ref_img, 
            faceid_embeds=faceid_embeds,
            width=int(width), 
            height=int(height), 
            num_samples=1, 
            num_inference_steps=int(steps),
            scale=float(identity_strength), 
            s_scale=1.0, 
            shortcut=True, 
            seed=int(seed) if seed else None
        )[0]
        
        # Save result to outputs directory
        out_path = "outputs/ip_adapter_result.png"
        image.save(out_path)
        
        return out_path, "✅ FaceID Image generated"
        
    except Exception as e: 
        return None, f"❌ FaceID error: {str(e)}"
