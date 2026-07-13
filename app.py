import gradio as gr
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

# Import all modular logic
from config import *
from image_utils import *
from face_engine import *
from providers import *
from pipelines import *

# Necessary for some UI functions
import replicate
import requests

def make_seamless_image(image1, image2, direction, blend_size):
    image1, image2 = image1.convert("RGB"), image2.convert("RGB")
    blend_size = int(blend_size)
    if direction == "Side by Side":
        h = min(image1.height, image2.height)
        image1 = image1.resize((int(image1.width * h / image1.height), h))
        image2 = image2.resize((int(image2.width * h / image2.height), h))
        canvas = Image.new("RGB", (image1.width + image2.width - blend_size, h))
        canvas.paste(image1, (0, 0))
        for x in range(blend_size):
            alpha = x / blend_size
            col1 = image1.crop((image1.width - blend_size + x, 0, image1.width - blend_size + x + 1, h))
            col2 = image2.crop((x, 0, x + 1, h))
            canvas.paste(Image.blend(col1, col2, alpha), (image1.width - blend_size + x, 0))
        canvas.paste(image2.crop((blend_size, 0, image2.width, h)), (image1.width, 0))
        return canvas, "✅ Side-by-side stitch created."
    
    w = min(image1.width, image2.width)
    image1 = image1.resize((w, int(image1.height * w / image1.width)))
    image2 = image2.resize((w, int(image2.height * w / image2.width)))
    canvas = Image.new("RGB", (w, image1.height + image2.height - blend_size))
    canvas.paste(image1, (0, 0))
    for y in range(blend_size):
        alpha = y / blend_size
        row1 = image1.crop((0, image1.height - blend_size + y, w, image1.height - blend_size + y + 1))
        row2 = image2.crop((0, y, w, y + 1))
        canvas.paste(Image.blend(row1, row2, alpha), (0, image1.height - blend_size + y))
    canvas.paste(image2.crop((0, blend_size, w, image2.height)), (0, image1.height))
    return canvas, "✅ Top-bottom stitch created."

def face_swap_image(target_image, face_image, prompt):
    check_token()
    try:
        tp, fp, rp = "/tmp/target.png", "/tmp/face.png", "outputs/face_swap.png"
        target_image.convert("RGB").save(tp)
        face_image.convert("RGB").save(fp)
        output = replicate.run("kwaivgi/kling-v1.6-standard", input={"input_image": Path(tp), "swap_image": Path(fp), "prompt": prompt if prompt else ""})
        url = str(output[0]) if isinstance(output, list) else str(output)
        with open(rp, "wb") as f: f.write(requests.get(url).content)
        return restore_face_eye_safe(rp), "✅ Face swap complete"
    except Exception as e: return None, f"❌ Error: {e}"

def swap_selected_face(target_image, source_face, face_index, prompt):
    check_token()
    if target_image is None or source_face is None: raise gr.Error("Images missing.")
    try:
        target_image = target_image.convert("RGB")
        source_face = source_face.convert("RGB")
        analyzer = load_face_analyzer()
        detected_faces = analyzer.get(np.asarray(target_image))
        if not detected_faces: return None, "❌ No faces detected."
        detected_faces = sorted(detected_faces, key=lambda face: float(face.bbox[0]))
        if face_index < 0 or face_index >= len(detected_faces): return None, "❌ Face index out of range."
        
        selected_face = detected_faces[face_index]
        x1, y1, x2, y2 = selected_face.bbox.astype(int)
        padding_x, padding_top, padding_bottom = int((x2-x1)*0.65), int((y2-y1)*0.7), int((y2-y1)*0.55)
        target_crop = target_image.crop((max(0,x1-padding_x), max(0,y1-padding_top), min(target_image.width,x2+padding_x), min(target_image.height,y2+padding_bottom)))
        
        if not prompt or not prompt.strip(): prompt = "Replace selected person's face. Photorealistic."
        swapped_path, swap_status = face_swap_image(target_crop, source_face, prompt)
        if swapped_path is None: return None, swap_status
        
        swapped_crop = Image.open(swapped_path).convert("RGB").resize(target_crop.size, Image.LANCZOS)
        mask = Image.new("L", target_crop.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((int(target_crop.width*0.18), int(target_crop.height*0.1), int(target_crop.width*0.82), int(target_crop.height*0.9)), fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(max(12, int(min(target_crop.size)*0.1))))
        
        final_image = target_image.copy()
        final_image.paste(swapped_crop, (max(0,x1-padding_x), max(0,y1-padding_top)), mask)
        out_path = f"outputs/selected_face_{face_index}_swap.png"
        final_image.save(out_path)
        return final_image, f"✅ Swapped face {face_index}."
    except Exception as e: return None, f"❌ Error: {e}"

def face_to_video(face_image, motion_prompt):
    check_token()
    fp = "/tmp/f2v.png"
    face_image.convert("RGB").save(fp)
    output = replicate.run("kwaivgi/kling-v1.6-standard", input={"image": Path(fp), "prompt": motion_prompt, "duration": 5, "fps": 15})
    return str(output[0]) if isinstance(output, list) else str(output), "✅ Video generated."

def face_swap_video(face_image, target_video):
    check_token()
    fp = "/tmp/fsv.png"
    face_image.convert("RGB").save(fp)
    vp = target_video.name if hasattr(target_video, "name") else str(target_video)
    output = replicate.run("arabyai-replicate/roop_face_swap:11b6bf0f4e14d808f655e87e5448233cceff10a45f659d71539cafb7163b2e84", input={"swap_image": Path(fp), "target_video": Path(vp)})
    url = str(output[0]) if isinstance(output, list) else str(output)
    local_out = "/tmp/swap_out.mp4"
    with open(local_out, "wb") as f: f.write(requests.get(url).content)
    return local_out, "✅ Video swap complete."

def toggle_model_id(provider):
    return gr.update(visible=(provider == "Civitai"))

def generate_video_with_provider(provider, image, prompt, server_url):
    if image is None: return None, "❌ Please upload an image."
    if not prompt or not prompt.strip(): return None, "❌ Please enter a motion prompt."
    
    try:
        if provider == "Replicate (Filtered)":
            check_token()
            image_path = "/tmp/video_input.png"
            image.convert("RGB").save(image_path)
            output = replicate.run("minimax/video-01", input={"prompt": prompt, "first_frame_image": Path(image_path), "prompt_optimizer": True})
            return extract_output(output), "✅ Video generated via Replicate."
        elif provider == "Hugging Face LTX - Subtle Motion":
            return generate_video_hf_local(image, prompt)
        elif provider == "Hugging Face CogVideoX - High Motion":
            return generate_hf_cogvideo(image, prompt)
        elif provider == "Private Server (Unfiltered)":
            if not server_url: return None, "❌ Enter a Custom Server URL."
            image_path = "/tmp/video_input.png"
            image.convert("RGB").save(image_path)
            with open(image_path, "rb") as file: image_bytes = file.read()
            response = requests.post(server_url, headers={"Authorization": f"Bearer {HF_TOKEN}"}, json={"inputs": {"image": image_bytes, "prompt": prompt}})
            if response.status_code != 200: return None, f"❌ Server Error: {response.text}"
            output_path = "outputs/private_video.mp4"
            with open(output_path, "wb") as file: file.write(response.content)
            return output_path, "✅ Private Server video generated."
        return None, "❌ Unknown video provider."
    except Exception as e:
        return None, f"❌ Video Error: {str(e)}"

# =========================
# UI ASSEMBLY
# =========================

with gr.Blocks(title="Guns AI Studio", css=css) as demo:
    gr.Markdown("# ✨ Guns AI Studio ✨")

    with gr.Tabs():
        with gr.Tab("⚙️ Server Settings"):
            gr.Markdown("### Private Server Configuration")
            custom_server_url_display = gr.Textbox(label="Custom Server URL", placeholder="https://xxxxxxx.runpod.net")

        with gr.Tab("🖼️ Image Generation"):
            with gr.Row():
                with gr.Column():
                    provider_switch = gr.Dropdown(choices=["Replicate", "Fal.ai", "Civitai", "Hugging Face", "RunPod"])
                    model_id_input = gr.Textbox(label="Civitai Model ID (Optional)", visible=False)
                    prompt_input = gr.Textbox(label="Prompt", lines=4)
                    negative_input = gr.Textbox(label="Negative Prompt", value=DEFAULT_NEGATIVE)
                    model_input = gr.Textbox(label="Replicate/HF Model", value=DEFAULT_IMAGE_MODEL)
                    image_input = gr.Image(label="Init Image (Optional)", type="filepath")
                    with gr.Row():
                        width_in = gr.Number(label="Width", value=1024)
                        height_in = gr.Number(label="Height", value=1024)
                        steps_in = gr.Number(label="Steps", value=28)
                        seed_in = gr.Number(label="Seed (0=random)", value=0, precision=0)
                    gen_btn = gr.Button("🚀 Generate Image")
                with gr.Column():
                    output_img = gr.Image(label="Result")
                    status_text = gr.Textbox(label="Status")
            provider_switch.change(toggle_model_id, inputs=[provider_switch], outputs=[model_id_input])
            gen_btn.click(fn=generate_with_provider, inputs=[provider_switch, prompt_input, negative_input, model_input, width_in, height_in, steps_in, seed_in, image_input, model_id_input], outputs=[output_img, status_text])

        with gr.Tab("Img2Img"):
            img2img_model = gr.Textbox(label="HF Model ID", value="runwayml/stable-diffusion-v1-5")
            img2img_prompt = gr.Textbox(label="Prompt", lines=4)
            with gr.Row():
                img2img_input = gr.Image(label="Input Image 1", type="pil")
                img2img_input_2 = gr.Image(label="Input Image 2 (Optional)", type="pil")
                img2img_strength = gr.Slider(0.0, 1.0, value=0.35, step=0.05, label="Edit Strength")
                img2img_guidance = gr.Slider(1.0, 20.0, value=5.5, step=0.5, label="CFG")
                img2img_face_blend = gr.Slider(0.0, 1.0, value=0.95, step=0.05, label="Face Blend")
            with gr.Row():
                img2img_steps = gr.Number(label="Steps", value=30)
                img2img_seed = gr.Number(label="Seed", value=0, precision=0)
                img2img_preserve = gr.Checkbox(label="Preserve original face", value=True)
            img2img_btn = gr.Button("🎨 Generate Img2Img")
            img2img_output = gr.Image(label="Result")
            img2img_status = gr.Textbox(label="Status")
            img2img_btn.click(generate_img2img_local, [img2img_prompt, img2img_input, img2img_input_2, img2img_model, img2img_strength, img2img_guidance, img2img_steps, img2img_seed, img2img_preserve, img2img_face_blend], [img2img_output, img2img_status])

        with gr.Tab("LoRA Generator"):
            lora_prompt = gr.Textbox(label="LoRA Prompt", lines=4, value="gnrwoman01, candid DSLR portrait photo, high quality")
            lora_negative = gr.Textbox(label="Negative Prompt", value=DEFAULT_NEGATIVE)
            lora_model = gr.Textbox(label="LoRA Model", value=DEFAULT_LORA_MODEL)
            lora_url = gr.Textbox(label="LoRA Weights URL")
            lora_scale = gr.Slider(0.01, 1.5, value=0.8, step=0.05, label="LoRA Scale")
            with gr.Row():
                lora_width = gr.Number(label="Width", value=1024)
                lora_height = gr.Number(label="Height", value=1024)
                lora_steps = gr.Number(label="Steps", value=33)
                lora_seed = gr.Number(label="Seed", value=0, precision=0)
            lora_btn = gr.Button("✨ Generate LoRA Image")
            lora_output = gr.Image(label="LoRA Result")
            lora_status = gr.Textbox(label="Status")
            lora_btn.click(generate_lora, [lora_prompt, lora_negative, lora_model, lora_url, lora_scale, lora_width, lora_height, lora_steps, lora_seed], [lora_output, lora_status])

        with gr.Tab("Option C"):
            ref_image = gr.Image(label="Collage Image", type="pil")
            ref_prompt = gr.Textbox(label="Prompt", lines=4)
            ref_negative = gr.Textbox(label="Negative", value=DEFAULT_NEGATIVE)
            ref_model = gr.Textbox(label="Model", value=DEFAULT_LORA_MODEL)
            ref_lora_url = gr.Textbox(label="LoRA URL")
            ref_lora_scale = gr.Slider(0.0, 1.5, value=0.35, step=0.05, label="LoRA Scale")
            with gr.Row():
                ref_width = gr.Number(label="Width", value=1024)
                ref_height = gr.Number(label="Height", value=1024)
                ref_steps = gr.Number(label="Steps", value=35)
                ref_seed = gr.Number(label="Seed", value=0, precision=0)
            ref_btn = gr.Button("🧪 Generate Option C")
            ref_out = gr.Image(label="Result")
            ref_stat = gr.Textbox(label="Status")
            ref_btn.click(generate_reference_lora, [ref_image, ref_prompt, ref_negative, ref_model, ref_lora_url, ref_lora_scale, ref_width, ref_height, ref_steps, ref_seed], [ref_out, ref_stat])

        with gr.Tab("Seamless Stitcher"):
            s_img1 = gr.Image(label="Image 1", type="pil")
            s_img2 = gr.Image(label="Image 2", type="pil")
            s_dir = gr.Radio(["Side by Side", "Top and Bottom"], value="Side by Side", label="Direction")
            s_blend = gr.Slider(20, 300, value=100, step=10, label="Blend Size")
            s_btn = gr.Button("🪄 Stitch")
            s_out = gr.Image(label="Result")
            s_stat = gr.Textbox(label="Status")
            s_btn.click(make_seamless_image, [s_img1, s_img2, s_dir, s_blend], [s_out, s_stat])

        with gr.Tab("AI Seamless LoRA"):
            ai_img1 = gr.Image(label="Ref 1", type="pil")
            ai_img2 = gr.Image(label="Ref 2", type="pil")
            ai_prompt = gr.Textbox(label="Prompt", lines=4)
            ai_negative = gr.Textbox(label="Negative", value=DEFAULT_NEGATIVE)
            ai_model = gr.Textbox(label="Model", value="black-forest-labs/flux-kontext-dev-lora")
            ai_url = gr.Textbox(label="LoRA URL")
            ai_scale = gr.Slider(0.0, 1.5, value=0.25, step=0.05, label="LoRA Strength")
            with gr.Row():
                ai_w = gr.Number(label="Width", value=1024)
                ai_h = gr.Number(label="Height", value=1024)
                ai_s = gr.Number(label="Steps", value=28)
                ai_zd = gr.Number(label="Seed", value=0, precision=0)
            ai_btn = gr.Button("✨ Generate AI Seamless")
            ai_out = gr.Image(label="Result")
            ai_stat = gr.Textbox(label="Status")
            ai_btn.click(generate_ai_seamless_lora, [ai_img1, ai_img2, ai_prompt, ai_negative, ai_model, ai_url, ai_scale, ai_w, ai_h, ai_s, ai_zd], [ai_out, ai_stat])

        with gr.Tab("🎭 Face Swap"):
            f_target = gr.Image(label="Target", type="pil")
            f_source = gr.Image(label="Face Source", type="pil")
            f_prom = gr.Textbox(label="Prompt")
            f_btn = gr.Button("🎭 Swap")
            f_out = gr.Image(label="Result")
            f_stat = gr.Textbox(label="Status")
            f_btn.click(face_swap_image, [f_target, f_source, f_prom], [f_out, f_stat])
       
        with gr.Tab("🧪 Face Index Test"):
            index_target = gr.Image(label="Two-Person Target Image", type="pil")
            index_prompt = gr.Textbox(label="Face Swap Prompt", lines=3, value="Replace only the selected person's face. Photorealistic.")
            index_source = gr.Image(label="Replacement Face", type="pil")
            index_number = gr.Radio(choices=[0, 1], value=0, label="Target Face — 0 = Left, 1 = Right")
            index_button = gr.Button("🧬 Swap Selected Face")
            index_output = gr.Image(label="Selected-Face Result")
            index_status = gr.Textbox(label="Status")
            index_button.click(swap_selected_face, [index_target, index_source, index_number, index_prompt], [index_output, index_status])

        with gr.Tab("😊 Face → Video"):
            fv_img = gr.Image(label="Face Image", type="pil")
            fv_prom = gr.Textbox(label="Motion Prompt", value="Natural smile and blink")
            fv_btn = gr.Button("Generate Video 🎬")
            fv_out = gr.Video(label="Result")
            fv_stat = gr.Textbox(label="Status")
            fv_btn.click(face_to_video, [fv_img, fv_prom], [fv_out, fv_stat])

        with gr.Tab("🎥 Image to Video"):
            video_provider = gr.Dropdown(choices=["Replicate (Filtered)", "Hugging Face LTX - Subtle Motion", "Hugging Face CogVideoX - High Motion"], value="Hugging Face LTX - Subtle Motion", label="Video Engine")
            v_img = gr.Image(label="Upload Image", type="pil")
            v_prom = gr.Textbox(label="Prompt", value="Cinematic slow motion")
            v_btn = gr.Button("🎬 Generate Video")
            v_out = gr.Video(label="Result")
            v_stat = gr.Textbox(label="Status")
            v_btn.click(fn=generate_video_with_provider, inputs=[video_provider, v_img, v_prom, custom_server_url_display], outputs=[v_out, v_stat])

        with gr.Tab("🎭 Face Swap Video"):
            sv_img = gr.Image(label="Face Image", type="pil")
            sv_vid = gr.File(label="Target Video")
            sv_btn = gr.Button("🔄 Swap")
            sv_out = gr.File(label="Download Result")
            sv_stat = gr.Textbox(label="Status")
            sv_btn.click(face_swap_video, [sv_img, sv_vid], [sv_out, sv_stat])

        with gr.Tab("IP-Adapter Identity"):
            ip_face = gr.Image(label="Reference Face", type="filepath")
            ip_prompt = gr.Textbox(label="Prompt", lines=4)
            ip_neg = gr.Textbox(label="Negative", value=DEFAULT_NEGATIVE)
            with gr.Row():
                ip_w = gr.Number(label="Width", value=768)
                ip_h = gr.Number(label="Height", value=1024)
                ip_s = gr.Number(label="Steps", value=30)
                ip_zd = gr.Number(label="Seed", value=0)
                ip_str = gr.Slider(0.5, 1.2, value=0.7, step=0.05, label="Identity Strength")
            ip_btn = gr.Button("🧬 Generate IP-Adapter")
            ip_out = gr.Image(label="Result")
            ip_stat = gr.Markdown()
            ip_btn.click(generate_ip_adapter_scene, [ip_face, ip_prompt, ip_neg, ip_w, ip_h, ip_s, ip_zd, ip_str], [ip_out, ip_stat])

demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
