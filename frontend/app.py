import gradio as gr
import requests
import json
import time

API_POLL_INTERVAL = 0.5  # in seconds
API_KEY = "your-api-key"
API_ENDPOINT = "https://your-api-endpoint.com/generate"

with open("sdxl.json", "r") as f:
    SDXL_TEMPLATE = json.load(f)

# Load flux template from flux.json
with open("flux.json", "r") as f:
    FLUX_TEMPLATE = json.load(f)

def generate_image_sdxl(prompt: str, instance_type: str):
    payload = {
        "prompt": prompt,
        "instance_type": instance_type
    }
    response = requests.post(API_ENDPOINT, json=payload)
    return response.json()

def generate_image_flux(prompt: str, instance_type: str):
    payload = {
        "prompt": prompt,
        "instance_type": instance_type
    }
    response = requests.post(API_ENDPOINT, json=payload)
    return response.json()


def get_image_status(task_id: str):
    response = requests.get(f"{API_ENDPOINT}/status/{task_id}")
    data = response.json()
    is_complete = data['status'] == 'complete'
    image_urls = {model: url for model, url in data.get('image_urls', {}).items()}
    return is_complete, image_urls

def generate_and_display_images(prompt: str, instance_type: str):
    yield None, None, "Generating...", "Generating..."

    # Generate image
    response_sdxl = generate_image_sdxl(prompt, instance_type)
    task_id_sdxl = response_sdxl['task_id']
    is_complete_sdxl = False

    response_flux = generate_image_flux(prompt, instance_type)
    task_id_flux = response_flux['task_id']
    is_complete_flux = False

    # Poll for result
    while True:
        if not(is_complete_sdxl):
            is_complete_sdxl, image_urls_sdxl, generation_time_sdxl = get_image_status(task_id_sdxl)
            if is_complete_sdxl:
                yield image_urls_sdxl, None, f"{generation_time_sdxl:.2f}s", None

        if not(is_complete_flux):
            is_complete_flux, image_urls_flux, generation_time_flux = get_image_status(task_id_flux)
            if is_complete_flux:
                yield None, image_urls_flux, None, f"{generation_time_flux:.2f}s"

        if is_complete_sdxl and is_complete_flux:
            break
        time.sleep(API_POLL_INTERVAL)

    return image_urls_sdxl, image_urls_flux, f"{generation_time_sdxl:.2f}s", f"{generation_time_flux:.2f}s"

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown(
    """
    # Stable Diffusion on EKS demo
    """)
    prompt = gr.Textbox(label="Prompt", value="cute anime girl with massive fluffy fennec ears and a big fluffy tail blonde messy long hair blue eyes wearing a maid outfit with a long black gold leaf pattern dress and a white apron mouth open holding a fancy black forest cake with candles on top in the kitchen of an old dark Victorian mansion lit by candlelight with a bright window to the foggy forest and very expensive stuff everywhere")
    instance_type = gr.Dropdown([["g5 with A10G GPU", "g5"], ["g6 with L4 GPU", "g6"], ["g6e with L40S GPU", "g6e"]], label="Backend Instance Type", value="g5")
    generate_btn = gr.Button("Generate Images")
    with gr.Row():
        with gr.Column():
            image1 = gr.Image(label="SDXL")
            time1 = gr.Textbox(label="SDXL Response Time")
        with gr.Column():
            image2 = gr.Image(label="Flux.1 Dev")
            time2 = gr.Textbox(label="Flux.1 Dev Response Time")
    generate_btn.click(generate_and_display_images,
                       inputs=[prompt, instance_type],
                       outputs=[image1, image2, time1, time2])

if __name__ == "__main__":
    demo.launch()
