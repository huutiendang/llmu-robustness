import json
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
random.seed(0)
import tqdm
from torch.autograd import grad
################################
##### Activation functions #####
################################

def forward_with_cache(model, inputs, module, no_grad=True):
    # define a tensor with the size of our cached activations
    cache = []
    def hook(module, input, output):
        if isinstance(output, tuple):
            cache.append(output[0])
        else:
            cache.append(output)
        return None 
    
    hook_handle = module.register_forward_hook(hook)
    
    if no_grad:
        with torch.no_grad():
            _ = model(**inputs)
    else:
        _ = model(**inputs)
        
    hook_handle.remove()

    return cache[0]
    
#######################################
##### Model and data loading code #####
#######################################


def get_params(model, layer_ids, param_ids):
    params = []
    for layer_id in layer_ids:
        for i, p in enumerate(model.model.layers[layer_id].parameters()):
            if i in param_ids:
                params.append(p)
    return params


def load_model(model_name_or_path):
    torch_dtype = "auto" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True, 
        use_fast=False
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.mask_token_id = tokenizer.eos_token_id
    tokenizer.sep_token_id = tokenizer.eos_token_id
    tokenizer.cls_token_id = tokenizer.eos_token_id

    return model, tokenizer

def get_random_vector(model, tokenizer, module, keyword="helper", dtype=torch.bfloat16):
    inputs = tokenizer(keyword, return_tensors="pt", padding=True).to(model.device)
    activations = forward_with_cache(model, inputs, module)
    # return activations
    gaussian_noise = torch.randn([1, 1, activations.shape[-1]])
    gaussian_noise = gaussian_noise.to(device=model.device, dtype=dtype)
    gaussian_noise = gaussian_noise / gaussian_noise.norm(dim=-1, keepdim=True)
    return gaussian_noise

def get_steering_vec(model, tokenizer, keyword, module, dtype=torch.bfloat16):

    # p_novice = f"You are a novice in {keyword} who often makes mistakes."
    # p_expert = f"You are a world-class expert in {keyword}."
    inputs = tokenizer(keyword, return_tensors="pt", padding=True).to(model.device)
    activations = forward_with_cache(model, inputs, module)

    # novice - expert
    # direction = activations[0:1, token_pos:, :] - activations[1:, token_pos:, :]
    direction = activations.mean(dim=1, keepdim=True)
    direction = direction.to(device=model.device, dtype=dtype)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    return direction

def get_data(forget_corpora, retain_corpora, min_len=0, max_len=2000, batch_size=4):
    def get_dataset(name):
        data = []
        if name == "wikitext":
            from datasets import load_dataset
            raw_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            for x in raw_data:
                if len(x['text']) > min_len:
                    data.append(str(x['text']))
        else:
            for line in open(f"data/{name}.jsonl", "r"):
                if "bio-forget-corpus" in name:
                    raw_text = json.loads(line)['text']
                else:
                    raw_text = line
                if len(raw_text) > min_len:
                    data.append(str(raw_text))
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        return data

    return (
        [get_dataset(c) for c in forget_corpora],
        [get_dataset(c) for c in retain_corpora]
    )

class RandomizedModel:
    def __init__(self, model_name_or_path: str, target_layers: List[int]):
        self.model, self.tokenizer = load_model(model_name_or_path=model_name_or_path)
        self.target_layers = target_layers
    
    
    def forward_with_injected_noise(self, input_ids, nu, no_grad=True):
        def hook_fn(module, input, output):

            modified_output = list(output)
            temp_hidden_states = modified_output[0]
            noise = torch.randn_like(temp_hidden_states) * nu
            modified_output[0] = temp_hidden_states + noise 
            if isinstance(output, tuple):
                return tuple(modified_output)
            else:
                return modified_output[0]
                
        hooks = []
        
        for layer_ids in self.target_layers:
            if layer_ids < 0 or layer_ids >= len(self.model.model.layers):
                raise ValueError(f"Layer {layer_ids} is invalid.")
            else:
                layer = self.model.model.layers[layer_ids]
                hooks.append(layer.register_forward_hook(hook_fn))
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, labels=input_ids)
            
        for hook in hooks:
            hook.remove()
            
        return outputs

    # def generate(self, prompt, max_length=15, nu=0.01, device="cuda"):
    #     input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
    #     generated_ids = input_ids.clone()
        
    #     for _ in range(max_length):
    #         context_length = min(len(generated_ids[0]), self.model.config.max_position_embeddings)
    #         input_context = generated_ids[:, -context_length:]
            
    #         with torch.no_grad():
    #             outputs = self.forward_with_injected_noise(input_context, nu=nu)
    #             next_token_logits = outputs.logits[:, -1, :]
    #             next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

    #         generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
            
    #         if next_token_id[0, 0].item() == self.tokenizer.eos_token_id:
    #             break
    #     generated_text = self.tokenizer.decode(generated_ids[0])
    #     return generated_text[len(prompt):]