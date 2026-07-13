import time
import requests
import gradio as gr
from PIL import Image
from io import BytesIO

# Import modular components
from config import RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID
from image_utils import decode_runpod_output

def generate_with_runpod(prompt, negative_prompt, width, height, steps, seed):
    """
    Handles the full RunPod lifecycle:
    1. POST request to submit the generation job.
    2. Polling the status endpoint until 'COMPLETED'.
    3. Decoding the response into a PIL Image.
    """
    if not RUNPOD_API_KEY:
        raise gr.Error("RUNPOD_API_KEY is missing in your environment secrets.")

    if not RUNPOD_ENDPOINT_ID:
        raise gr.Error("RUNPOD_ENDPOINT_ID is missing in your environment secrets.")

    # 1. Submit the Job
    endpoint_url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/run"
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }

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

    try:
        response = requests.post(endpoint_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        job_data = response.json()
        job_id = job_data.get("id")

        if not job_id:
            raise gr.Error(f"RunPod did not return a job ID: {job_data}")

        # 2. Poll for Result
        status_url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status/{job_id}"
        
        # Max poll for 180 seconds (3 minutes)
        for _ in range(180):
            status_response = requests.get(status_url, headers=headers, timeout=30)
            status_response.raise_for_status()
            result = status_response.json()
            status = result.get("status")

            if status == "COMPLETED":
                output = result.get("output", {})
                # Use the logic from image_utils to handle base64/URL strings
                return decode_runpod_output(output)

            if status in {"FAILED", "CANCELLED", "TIMED_OUT"}:
                error_message = result.get("error", "RunPod generation failed.")
                raise gr.Error(str(error_message))

            time.sleep(2)

        raise gr.Error("RunPod generation timed out.")
        
    except requests.exceptions.RequestException as e:
        raise gr.Error(f"Connection error with RunPod: {str(e)}")
