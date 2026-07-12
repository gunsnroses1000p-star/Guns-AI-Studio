
import gradio as gr

from config import DEFAULT_IMAGE_MODEL, DEFAULT_NEGATIVE
from engines.image_engine import generate_with_provider, toggle_model_id


def build_image_tab():
    with gr.Tab("🖼️ Image Generation"):
        with gr.Row():
            with gr.Column():
                provider_switch = gr.Dropdown(
                    choices=[
                        "Replicate",
                        "Fal.ai",
                        "Civitai",
                        "Hugging Face",
                        "RunPod",
                    ],
                    value="Replicate",
                    label="Image Provider",
                )

                model_id_input = gr.Textbox(
                    label="Civitai Model ID (Optional)",
                    placeholder="e.g. 123456",
                    visible=False,
                )

                prompt_input = gr.Textbox(
                    label="Prompt",
                    lines=4,
                )

                negative_input = gr.Textbox(
                    label="Negative Prompt",
                    value=DEFAULT_NEGATIVE,
                )

                model_input = gr.Textbox(
                    label="Replicate/HF Model",
                    value=DEFAULT_IMAGE_MODEL,
                )

                image_input = gr.Image(
                    label="Init Image (Optional)",
                    type="filepath",
                )

                with gr.Row():
                    width_in = gr.Number(
                        label="Width",
                        value=1024,
                    )

                    height_in = gr.Number(
                        label="Height",
                        value=1024,
                    )

                with gr.Row():
                    steps_in = gr.Number(
                        label="Steps",
                        value=28,
                    )

                    seed_in = gr.Number(
                        label="Seed (0 = random)",
                        value=0,
                        precision=0,
                    )

                generate_button = gr.Button(
                    "🚀 Generate Image"
                )

            with gr.Column():
                output_image = gr.Image(
                    label="Result"
                )

                status_text = gr.Textbox(
                    label="Status"
                )

        provider_switch.change(
            fn=toggle_model_id,
            inputs=[provider_switch],
            outputs=[model_id_input],
        )

        generate_button.click(
            fn=generate_with_provider,
            inputs=[
                provider_switch,
                prompt_input,
                negative_input,
                model_input,
                width_in,
                height_in,
                steps_in,
                seed_in,
                image_input,
                model_id_input,
            ],
            outputs=[
                output_image,
                status_text,
            ],
        )
