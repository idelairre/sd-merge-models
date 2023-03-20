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


def merge(answers, path="C:\\Users\\idela\\Documents\\Github\\stable-diffusion-webui\\models\\Stable-diffusion\\"):
    model0 = path + answers.get("model1")
    model1 = path + answers.get("model2")
    device = answers.get("device")
    without_vae = answers.get("without_vae") == "yes"
    alpha = float(answers.get("alpha"))
    output = path + answers.get("output")
    safetensor = answers.get("safetensor") == "yes"

    theta_0 = read_state_dict(model0, map_location=device)
    theta_1 = read_state_dict(model1, map_location=device)

    output_file = f"{output}-{str(alpha)[2:] + '0'}.ckpt"

    if safetensor:
        output_file = f"{output_file.replace('.ckpt', '')}.safetensors"

    for key in tqdm(theta_0.keys(), desc="Stage 1/2"):
        # skip VAE model parameters to get better results(tested for anime models)
        # for anime model，with merging VAE model, the result will be worse (dark and blurry)
        if without_vae and "first_stage_model" in key:
            continue

        if "model" in key and key in theta_1:
            theta_0[key] = (1 - alpha) * theta_0[key] + alpha * theta_1[key]

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


def save_answers(answers):
    with open('options.json', 'w', encoding='utf8') as json_file:
        json.dump(answers, json_file, ensure_ascii=True)


def merge_and_save(model1, model2, alpha, without_vae, device, output, safetensor):
    answers = {
        "model1": model1,
        "model2": model2,
        "alpha": alpha,
        "without_vae": without_vae,
        "device": device,
        "output": output,
        "safetensor": safetensor,
    }

    merge(answers)
    return "Done! Models merged and saved."


def update_model_list(path):
    global model_list
    model_list = [f for f in os.listdir(path) if f.endswith(
        '.ckpt') or f.endswith('.safetensors')]
    return model_list


def app():
    # file_path = gr.inputs.Textbox(
    #     default="C:\\Users\\idela\\Documents\\Github\\stable-diffusion-webui\\models\\Stable-diffusion\\",
    #     label="Enter the file path",
    # )
    model_list = update_model_list(
        "C:\\Users\\idela\\Documents\\Github\\stable-diffusion-webui\\models\\Stable-diffusion\\")
    model1 = gr.inputs.Dropdown(model_list, label="Select the first model")
    model2 = gr.inputs.Dropdown(model_list, label="Select the second model")
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

    iface = gr.Interface(
        fn=merge_and_save,
        inputs=[model1, model2, alpha, without_vae,
                device, output, safetensor],
        outputs="text",
        title="Model Merger",
        description="Merge two models with the given parameters.",
    ).launch()


app()
