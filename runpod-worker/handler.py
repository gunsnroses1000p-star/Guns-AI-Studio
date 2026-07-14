import io
import base64

import runpod
from PIL import Image, ImageDraw


def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def handler(job):
    job_input = job.get("input", {})

    prompt = job_input.get("prompt", "No prompt provided")

    image = Image.new("RGB", (1024, 1024), (35, 35, 45))
    draw = ImageDraw.Draw(image)

    draw.text(
        (40, 40),
        f"RunPod Worker Online\n\nPrompt:\n{prompt}",
        fill="white",
    )

    return {
        "image_base64": image_to_base64(image)
    }


runpod.serverless.start({"handler": handler})
