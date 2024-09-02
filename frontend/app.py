import os
import gradio as gr
import requests
import time

API_POLL_INTERVAL = float(os.environ.get("API_POLL_INTERVAL", 1))  # in seconds
API_MAX_RETRY = int(os.environ.get("API_MAX_RETRY", 120))
API_ENDPOINT = os.environ.get("API_ENDPOINT", "http://127.0.0.1:20001")

LOADING_TEMPLATE = '<div class="ld ld-hourglass ld-spin-fast" style="font-size:64px;color:#8da;max-width:100%;display:block;margin-left:auto;margin-right:auto;"></div>'

def load_css(filename: str):
    with open(filename, 'r') as file:
        css_content = file.read()
    return css_content

def generate_image_same_instance_type(prompt: str, instance_type: str):
    payload = {
        "prompt": prompt,
        "instance_type": instance_type
    }
    response = requests.post(f"{API_ENDPOINT}/generate_with_same_instance_type", json=payload)
    return response.json()

def generate_image_same_model(prompt: str, model: str):
    payload = {
        "prompt": prompt,
        "model": model
    }
    response = requests.post(f"{API_ENDPOINT}/generate_with_same_model", json=payload)
    return response.json()

def get_image_status(task_id: str):
    response = requests.get(f"{API_ENDPOINT}/status/{task_id}").json()
    is_complete = False
    process_duration = None
    image_response = LOADING_TEMPLATE
    if (response.get("status") == "submitted"):
        status_output = "Queuing, please wait..."
    elif (response.get("status") == "running"):
        status_output = "Generating image.."
    elif (response.get("status") == "failed"):
        is_complete = True
        status_output = "Error, please retry"
    elif (response.get("status") == "completed"):
        is_complete = True
        image_response = f'<img src="{response.get("image_url")}" style="max-width:100%;">'
        process_duration = response.get("process_duration")
        status_output = f"Time usage: {process_duration}s"

    return status_output, is_complete, image_response

# def update_pricing():
#     response = requests.get(f"{API_ENDPOINT}/pricing").json()
#     return response


def generate_and_display_images_same_instance_type(prompt: str, instance_type: str):

    response = generate_image_same_instance_type(prompt, instance_type)

    task_id_sdxl = response['task_id_sdxl']
    task_id_flux = response['task_id_flux']
    is_complete_sdxl = False
    is_complete_flux = False
    count = 0

    # Poll for result
    while True:
        count = count + 1
        if (count > API_MAX_RETRY):
            return None, None, "Timeout", "Timeout"

        if not(is_complete_sdxl):
            status_output_sdxl, is_complete_sdxl, image_sdxl = get_image_status(task_id_sdxl)

        if not(is_complete_flux):
            status_output_flux, is_complete_flux, image_flux = get_image_status(task_id_flux)

        yield image_sdxl, image_flux, status_output_sdxl, status_output_flux

        if is_complete_sdxl and is_complete_flux:
            break
        time.sleep(API_POLL_INTERVAL)

    return image_sdxl, image_flux, status_output_sdxl, status_output_flux

def generate_and_display_images_same_model(prompt: str, model: str):
    # Generate image
    response = generate_image_same_model(prompt, model)

    task_id_g5 = response['task_id_g5']
    task_id_g6 = response['task_id_g6']
    task_id_g6e = response['task_id_g6e']
    is_complete_g5 = False
    is_complete_g6 = False
    is_complete_g6e = False
    count = 0

    # Poll for result
    while True:
        count = count + 1
        if (count > API_MAX_RETRY):
            return None, None, "Timeout", "Timeout"

        if not(is_complete_g5):
            status_output_g5, is_complete_g5, image_g5 = get_image_status(task_id_g5)

        if not(is_complete_g6):
            status_output_g6, is_complete_g6, image_g6 = get_image_status(task_id_g6)

        if not(is_complete_g6e):
            status_output_g6e, is_complete_g6e, image_g6e = get_image_status(task_id_g6e)

        yield image_g5, image_g6, image_g6e, status_output_g5, status_output_g6, status_output_g6e

        if is_complete_g5 and is_complete_g6 and is_complete_g6e:
            break
        time.sleep(API_POLL_INTERVAL)

    return image_g5, image_g6, image_g6e, status_output_g5, status_output_g6, status_output_g6e

# Create Gradio interface
with gr.Blocks(css=load_css("loading.min.css")) as demo:
    gr.Markdown(
    """
    # Stable Diffusion on EKS demo
    """)
    prompt = gr.Textbox(label="Prompt", value="cute anime girl with massive fluffy fennec ears and a big fluffy tail blonde messy long hair blue eyes wearing a maid outfit with a long black gold leaf pattern dress and a white apron mouth open holding a fancy black forest cake with candles on top in the kitchen of an old dark Victorian mansion lit by candlelight with a bright window to the foggy forest and very expensive stuff everywhere")
    with gr.Tab("Same instance type, Model Comparison"):
        instance_type = gr.Dropdown([["g5 with A10G GPU", "g5"], ["g6 with L4 GPU", "g6"], ["g6e with L40S GPU", "g6e"]], label="Backend Instance Type", value="g6e")
        generate_btn_i = gr.Button("Generate Images")
#        with gr.Row():
#            with gr.Column():
#                spot_pricing_i = gr.Textbox(label="Current Spot Pricing (us-west-2)")
#            with gr.Column():
#                on_demand_pricing_i = gr.Textbox(label="Current On-Demand Pricing (us-west-2)")
#            with gr.Column():
#                refresh_pricing_i = gr.Button("Refresh Pricing")
        with gr.Row():
            with gr.Column():
                image_i_1 = gr.HTML(label="SDXL", elem_id="image-sdxl")
                time_i_1 = gr.Textbox(label="SDXL Response Time")
#                cost_i_1 = gr.Textbox(label="Estimated Cost (us-west-2)")
            with gr.Column():
                image_i_2 = gr.HTML(label="Flux.1 Dev", elem_id="image-flux")
                time_i_2 = gr.Textbox(label="Flux.1 Dev Response Time")
                # cost_i_2 = gr.Textbox(label="Estimated Cost (us-west-2)")

        generate_btn_i.click(generate_and_display_images_same_instance_type,
                        inputs=[prompt, instance_type],
                        outputs=[image_i_1, image_i_2, time_i_1, time_i_2])

        # refresh_pricing_i.click(update_pricing, inputs=[spot_pricing_i, on_demand_pricing_i], outputs=[spot_pricing_i, on_demand_pricing_i])


    with gr.Tab("Same model, Instance Type Comparison"):
        model = gr.Dropdown([["Stable Diffusions XL", "sdxl"], ["Flux.1 Dev", "flux"]], label="Model", value="flux")
        generate_btn_m = gr.Button("Generate Images")
        with gr.Row():
            with gr.Column():
                image_m_1 = gr.HTML(label="g5 Instance Type", elem_id="image-g5")
                time_m_1 = gr.Textbox(label="g5 Instance Type Response Time")
            with gr.Column():
                image_m_2 = gr.HTML(label="g6 Instance Type", elem_id="image-g6")
                time_m_2 = gr.Textbox(label="g6 Instance Type Response Time")
            with gr.Column():
                image_m_3 = gr.HTML(label="g6e Instance Type", elem_id="image-g6e")
                time_m_3 = gr.Textbox(label="g6e Instance Type Response Time")
        generate_btn_m.click(generate_and_display_images_same_model,
                        inputs=[prompt, model],
                        outputs=[image_m_1, image_m_2, image_m_3, time_m_1, time_m_2, time_m_3])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7890, show_error=True, debug=False)
