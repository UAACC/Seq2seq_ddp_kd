import torch
from safetensors import safe_open
from transformers import AutoConfig, T5ForConditionalGeneration, AutoModel, AutoModelForSeq2SeqLM

def check_model_class_and_load_safetensors(model_dir):
    try:
        # Paths to the necessary files
        config_path = f"{model_dir}/config.json"
        safetensors_path = f"{model_dir}/model.safetensors"

        # Load the config file to get the model architecture
        config = AutoConfig.from_pretrained(config_path)
        
        # Load the model with the correct class
        model = T5ForConditionalGeneration(config)

        # Load the weights from the safetensors file
        with safe_open(safetensors_path, framework="pt") as f:
            state_dict = {}
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        
        # Load state dict into model
        model.load_state_dict(state_dict, strict=False)
        

        # Print the class of the loaded model
        print(f"Model class loaded successfully: {model.__class__.__name__}")
        return model

    except Exception as e:
        # Print out any errors encountered during the loading process
        print(f"Error loading model from {model_dir}: {e}")

# Example usage:
model_directory = "/home/dongheng/LLMR/accelerate/debug_dailyDialog"
check_model_class_and_load_safetensors(model_directory)
