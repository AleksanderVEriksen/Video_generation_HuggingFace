import torch
import gradio as gr
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video
import webbrowser
import time
import sys
import os

# Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.to("cuda")

def generate_video(prompt, negative_prompt, video_name, progress=gr.Progress()):
    """
    Generate a video based on the provided prompt and video name (without extension).
    Returns the filename and the elapsed time as a string.
    """
    progress(0, desc="Starting video generation...")
    start_time = time.time()
    if video_name == "":
        filename = "output.mp4"
    # Ensure the filename ends with .mp4
    if not video_name.lower().endswith('.mp4'):
        filename = video_name + '.mp4'
    else:
        filename = video_name
    # If video name contains a space, replace it with an underscore
    filename = filename.replace(" ", "_")

    # Define the callback function for progress updates
    def on_step_end(step: int, total_steps: int, logs: dict):
        progress(step / total_steps, desc=f"Step {step}/{total_steps}")

    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=128,
        width=224,
        num_frames=9,
        guidance_scale=5.0,
        callback_on_step_end=on_step_end  # <-- Add the callback here
    ).frames[0]
    export_to_video(output, filename, fps=15)
    elapsed = time.time() - start_time
    progress(1, desc="Video generation completed.")
    print(f"Video generation took {elapsed:.2f} seconds.")
    return os.path.abspath(filename)

def shutdown():
    """Shutdown the Gradio app and Python process."""
    print("Shutting down...")
    # Close all Gradio interfaces
    if 'Video_app' in globals():
        Video_app.close()
    gr.close_all()
    sys.exit(0)

# Gradio interface for Video Generation
with gr.Blocks() as Video_app:
    gr.Markdown("# Video Generation\nDescribe a video. Enter a name for your video (no extension needed).")
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Video prompt", lines=2, placeholder="Description of a video to be generated", value="A frog on a bike drinking coffee")
            negative_prompt = gr.Textbox(label="Negative Prompt", lines=2, placeholder="Describe what to avoid in the video...", value="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards")
            video_name = gr.Textbox(label="Video name (without .mp4)", lines=1, placeholder="Name of the video to be generated, e.g. 'my video'")
            generate_btn = gr.Button("Generate Video")
            exit_btn = gr.Button("Exit", variant="stop")
        with gr.Column():
            video_output = gr.PlayableVideo(label="Output", format="mp4", show_download_button=True)
    examples = gr.Examples([
        ["Astronaut in a jungle, cold color palette, muted colors, detailed, 8k", "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards", "output"],
        ["A futuristic cityscape at night with neon lights", "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards", "city_night"],
        ["A serene beach at sunset with gentle waves", "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards", "beach_sunset"],
        ["A bustling market scene with vibrant colors", "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards", "market_scene"],
        ["A snowy mountain landscape with a clear blue sky", "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards", "mountain_snow"],
    ],
    inputs=[prompt, negative_prompt, video_name],
    outputs=video_output,
    )
    generate_btn.click(generate_video, inputs=[prompt, negative_prompt, video_name], outputs=video_output)
    exit_btn.click(shutdown, inputs=[], outputs=[])

if __name__ == "__main__":
    # Launch the app
    Video_app.launch(server_name="0.0.0.0", server_port=7860, debug=True, inbrowser=True, share=False)
    try:
        # Try to open Chrome, fallback to default browser if not found
        webbrowser.get('chrome').open_new('http://localhost:7860/')
    except webbrowser.Error:
        webbrowser.open_new('http://localhost:7860/')