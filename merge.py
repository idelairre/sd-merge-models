import gradio as gr
import os
import torch
import safetensors.torch
import json
from safetensors.torch import save_file
from tqdm import tqdm


chckpoint_dict_replacements = {
    "cond_stage_model.transformer.embeddings.": "cond_stage_model.transformer.text_model.embeddings.",
    "cond_stage_model.transformer.encoder.": "cond_stage_model.transformer.text_model.encoder.",
    "cond_stage_model.transformer.final_layer_norm.": "cond_stage_model.transformer.text_model.final_layer_norm.",
}


def read_state_dict(checkpoint_file, print_global_state=False, map_location=None):
    _, extension = os.path.splitext(checkpoint_file)
    if extension.lower() == ".safetensors":
        pl_sd = safetensors.torch.load_file(
            checkpoint_file, device=map_location)
    else:
        pl_sd = torch.load(checkpoint_file, map_location=map_location)

    if print_global_state and "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")

    sd = get_state_dict_from_checkpoint(pl_sd)
    return sd


def transform_checkpoint_dict_key(k):
    for text, replacement in chckpoint_dict_replacements.items():
        if k.startswith(text):
            k = replacement + k[len(text):]

    return k


def get_state_dict_from_checkpoint(pl_sd):
    pl_sd = pl_sd.pop("state_dict", pl_sd)
    pl_sd.pop("state_dict", None)

    sd = {}
    for k, v in pl_sd.items():
        new_key = transform_checkpoint_dict_key(k)

        if new_key is not None:
            sd[new_key] = v

    pl_sd.clear()
    pl_sd.update(sd)

    return pl_sd


def merge(model1, model2, model3, alpha, without_vae, device, output, safetensor, merge_type):
    path="C:\\Users\\idela\\Documents\\Github\\stable-diffusion-webui\\models\\Stable-diffusion\\"
    model0 = path + model1
    model1 = path + model2
    if model3:
        model2 = path + model3

    theta_0 = read_state_dict(model0, map_location=device)
    theta_1 = read_state_dict(model1, map_location=device)

    if model3:
        theta_2 = read_state_dict(model2, map_location=device)

    output_file = f"{output}-{str(alpha)[2:] + '0'}.ckpt"

    if safetensor:
        output_file = f"{output_file.replace('.ckpt', '')}.safetensors"

    if merge_type == "Sigmoid":
        for key in tqdm(theta_0.keys(), desc="Stage 1/2"):
            if without_vae and "first_stage_model" in key:
                continue

            if "model" in key and key in theta_1:
                theta_0[key] = (1 / (1 + torch.exp(-4 * alpha))) * (theta_0[key] + theta_1[key]) - \
                    (1 / (1 + torch.exp(-alpha))) * theta_0[key]

    elif merge_type == "Linear":
        for key in tqdm(theta_0.keys(), desc="Stage 1/2"):
            if without_vae and "first_stage_model" in key:
                continue

            if "model" in key and key in theta_1:
                theta_0[key] = (1 - alpha) * theta_0[key] + \
                    alpha * theta_1[key]

    elif merge_type == "Geometric":
        for key in tqdm(theta_0.keys(), desc="Stage 1/2"):
            if without_vae and "first_stage_model" in key:
                continue

            if "model" in key and key in theta_1:
                theta_0[key] = torch.pow(
                    theta_0[key], 1 - alpha) * torch.pow(theta_1[key], alpha)

    elif merge_type == "Max":
        for key in tqdm(theta_0.keys(), desc="Stage 1/2"):
            if without_vae and "first_stage_model" in key:
                continue

            if "model" in key and key in theta_1:
                theta_0[key] = torch.max(theta_0[key], theta_1[key])

    elif merge_type == "Difference":
        if not model3:
            raise ValueError(
                "A third model must be provided for Difference merge")

        for key in tqdm(theta_0.keys(), desc="Stage 1/3"):
            if without_vae and "first_stage_model" in key:
                continue

            if "model" in key and key in theta_1 and key in theta_2:
                theta_0[key] = theta_0[key] + \
                    (theta_1[key] - theta_2[key]) * alpha

    else:
        raise ValueError("Invalid merge type selected")

    for key in tqdm(theta_1.keys(), desc="Stage 2/2"):
        if "model" in key and key not in theta_0:
            theta_0[key] = theta_1[key]

    print("Saving...")

    if safetensor:
        with torch.no_grad():
            save_file(theta_0, output_file, metadata={"format": "pt"})
    else:
        torch.save({"state_dict": theta_0}, output_file)

    print("Done!")


def update_model_list(path):
    global model_list
    model_list = [f for f in os.listdir(path) if f.endswith(
        '.ckpt') or f.endswith('.safetensors')]
    return model_list


def app():
    model_list = update_model_list(
        "C:\\Users\\idela\\Documents\\Github\\stable-diffusion-webui\\models\\Stable-diffusion\\")
    model1 = gr.inputs.Dropdown(model_list, label="Select the first model")
    model2 = gr.inputs.Dropdown(model_list, label="Select the second model")
    model3 = gr.inputs.Dropdown(model_list, label="Select the third model")
    alpha = gr.inputs.Slider(minimum=0, maximum=1,
                             step=0.01, default=0.5, label="Alpha value")
    without_vae = gr.inputs.Radio(
        ["yes", "no"], label="Do not merge VAE", default="yes")
    device = gr.inputs.Radio(
        ["cpu", "gpu"], label="Device to use", default="cpu")
    output = gr.inputs.Textbox(
        default="merged", label="Output file name, without extension")
    safetensor = gr.inputs.Radio(
        ["yes", "no"], label="Convert to safetensor format", default="yes")
    merge_type = gr.inputs.Dropdown(
        ["Sigmoid", "Linear", "Geometric", "Max", "Difference"], label="Select merge type")

    iface = gr.Interface(
        fn=merge,
        inputs=[model1, model2, model3, alpha, without_vae,
                device, output, safetensor, merge_type],
        outputs="text",
        title="Model Merger",
        description="Merge two models with the given parameters.",
    ).launch()


app()
