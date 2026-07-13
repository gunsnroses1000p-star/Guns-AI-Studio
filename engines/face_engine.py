import torch
import numpy as np
import insightface
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import gradio as gr
import spaces

# Global variable to avoid reloading the model on every request
face_analyzer = None

def load_face_analyzer():
    global face_analyzer
    if face_analyzer is None:
        # Uses CPUExecutionProvider to ensure compatibility across different HF Space tiers
        face_analyzer = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
    return face_analyzer

def preserve_original_face(original_image, edited_image, strength=0.65):
    original_image = original_image.convert("RGB")
    edited_image = edited_image.convert("RGB").resize(original_image.size)
    w, h = original_image.size
    
    # Region of interest (ROI) for the face based on your app.py logic
    x1, y1, x2, y2 = int(w * 0.28), int(h * 0.08), int(w * 0.72), int(h * 0.38)
    
    original_crop = original_image.crop((x1, y1, x2, y2))
    edited_crop = edited_image.crop((x1, y1, x2, y2))
    
    blended_face = Image.blend(edited_crop, original_crop, float(strength))
    
    # Create a soft circular mask for a natural blend
    mask = Image.new("L", blended_face.size, 0)
    mask_w, mask_h = blended_face.size
    for y in range(mask_h):
        for x in range(mask_w):
            dx, dy = (x - mask_w/2)/(mask_w/2), (y - mask_h/2)/(mask_h/2)
            value = int(max(0, min(255, 255 * (1 - (dx*dx + dy*dy)))))
            mask.putpixel((x, y), value)
            
    result = edited_image.copy()
    result.paste(blended_face, (x1, y1), mask)
    return result

def restore_face_eye_safe(image_path):
    if image_path is None: 
        return None
    try:
        img = Image.open(image_path).convert("RGB")
        # Sharpening and contrast enhancements specifically tuned for face clarity
        img = img.filter(ImageFilter.UnsharpMask(radius=0.4, percent=25, threshold=6))
        img = ImageEnhance.Contrast(img).enhance(1.02)
        img = ImageEnhance.Sharpness(img).enhance(1.02)
        
        fixed_path = "outputs/face_eye_safe.png"
        img.save(fixed_path)
        return fixed_path
    except Exception as e:
        print(f"Error in restore_face_eye_safe: {e}")
        return image_path
