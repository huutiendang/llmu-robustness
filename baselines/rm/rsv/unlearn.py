import numpy as np
import torch
from transformers import AdamW 
from tqdm import tqdm
import pandas as pd
import json
from transformers import AutoTokenizer, AutoModelForCausalLM 
from datasets import load_dataset
from baselines.utils import get_random_vector, get_params, forward_with_cache, load_model

def get_data(forget_corpora, retain_corpora, min_len=0, max_len=2000, batch_size=4):
    def get_dataset(name):
        data = []
        if name == "wikitext":
            raw_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            for x in raw_data:
                if len(x['text']) > min_len:
                    data.append(str(x['text'][:max_len]))
        else:
            for line in open(f"data/{name}.jsonl", "r"):
                raw_text = json.loads(line)
                if isinstance(raw_text, dict):
                    raw_text = raw_text.get("text", "")
                raw_text = raw_text.strip()
                if len(raw_text) > min_len:
                    data.append(str(raw_text[:max_len]))
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        return data 
        
    def corpus_to_keywords(corpus):
        if 'bio' in corpus:
            return 'bio'
        elif 'cyber' in corpus:
            return 'cyber'
        else:
            raise NotImplementedError(f"Add your keywords for {corpus} to the keywords.json file.")

    with open("data/keywords.json", "r") as f:
        keywords = json.load(f)

    return (
        [keywords[corpus_to_keywords(c)] for c in forget_corpora],
        [get_dataset(c) for c in forget_corpora],
        [get_dataset(c) for c in retain_corpora]
    )

def run_rsv(
    updated_model,
    frozen_model,
    tokenizer,
    keywords_list,
    forget_data_list,
    retain_data_list,
    args,
):
    assert len(keywords_list) == len(forget_data_list) == len(retain_data_list)
    
    updated_model = updated_model.train()
    params = get_params(updated_model, args.layer_ids, args.param_ids)
    optimizer = AdamW(params, lr=args.lr)
    
    frozen_module = eval(args.module_str.format(model_name="frozen_model", layer_id=args.layer_id))
    updated_module = eval(args.module_str.format(model_name="updated_model", layer_id=args.layer_id))

    # Get steering vectors
    steering_vectors_list = [[] for _ in range(len(keywords_list))]
    for i in range(len(steering_vectors_list)):
        steering_vectors_list[i].append(
            get_random_vector(
                model=frozen_model,
                tokenizer=tokenizer,
                module=frozen_module
            )
        )

    num_batches = args.max_num_batches
    truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side="right"

    with tqdm(total=num_batches) as pbar:
        for idx in range(num_batches):
            topic_idx = idx % len(keywords_list)
            batch_idx = idx // len(keywords_list)

            steering_vecs = steering_vectors_list[topic_idx]
            steering_vec_idx = np.random.choice(len(steering_vecs))
            steering_vec = steering_vecs[steering_vec_idx]

            unlearn_batch = forget_data_list[topic_idx][batch_idx]
            retain_batch = retain_data_list[topic_idx][batch_idx]

            max_length = 512 if topic_idx == 0 else 768
            # Unlearning loss
            unlearn_inputs = tokenizer(unlearn_batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(updated_model.device)

            updated_activations = forward_with_cache(
                updated_model, unlearn_inputs, module=updated_module, no_grad=False
            ).to(updated_model.device)

            frozen_activations = forward_with_cache(
                frozen_model, unlearn_inputs, module=frozen_module, no_grad=True
            ).to(updated_model.device)

            frozen_activations += args.steering_coeff_list[topic_idx] * steering_vec

            frozen_activations.to(updated_model.device)

            unlearn_loss = torch.nn.functional.mse_loss(
                updated_activations, frozen_activations
            )

            # Retain loss
            retain_inputs = tokenizer(
                retain_batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
            ).to(updated_model.device)
            
            updated_retain_activations = forward_with_cache(
                updated_model, retain_inputs, module=updated_module, no_grad=False
            ).to(updated_model.device)
            
            frozen_retain_activations = forward_with_cache(
                frozen_model, retain_inputs, module=frozen_module, no_grad=True
            ).to(updated_model.device)

            # add random noise
            noise = torch.randn_like(frozen_retain_activations) * args.nu
            noise_frozen_retain_activations = frozen_retain_activations + noise

            retain_loss = torch.nn.functional.mse_loss(
                updated_retain_activations, noise_frozen_retain_activations
            )

            retain_loss *= args.alpha

            # Update model
            loss = unlearn_loss + retain_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"loss: {loss.item():.4g} | unlearn_loss: {unlearn_loss.item():.4g} | retain_loss: {retain_loss.item():.4g} | param_change: {params[0].grad.abs().mean().item():.4g}")
            pbar.update(1)
        
    tokenizer.truncation_side = truncation_side
    # Save model
    path = f"checkpoints/rm/rsv/{args.model_name_or_path}_alpha-{int(args.alpha)}-{int(args.alpha)}_coeffs-{float(args.steering_coeff_list[0])}-{float(args.steering_coeff_list[1])}_batches-{num_batches}_layer-{args.layer_id}_nu-{args.nu}"
    updated_model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Saved model to {path}")


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    ### Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="HuggingFaceH4/zephyr-7b-beta")
    parser.add_argument("--module_str", type=str, default="{model_name}.model.layers[{layer_id}]")
    parser.add_argument("--retain_corpora",type=str, default="wikitext,wikitext", help="comma-separated list of corpora to retain",)
    parser.add_argument("--forget_corpora",type=str, default="bio-forget-corpus,cyber-forget-corpus", help="comma-separated list of corpora to forget",)
    ### CUT hyperparameters
    parser.add_argument("--alpha", type=float, default=5000, help="unlearning weight")
    parser.add_argument("--nu", type=float, default=0.03, help="unlearning weight")
    parser.add_argument("--steering_coeffs",type=str,default="10,10", help="Steer vector weight in order of topic",)
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_num_batches", type=int, default=500)
    parser.add_argument("--layer_id", type=int, default=7, help="layer to unlearn")
    parser.add_argument("--layer_ids", type=str, default="5,6,7", help="update layers")
    parser.add_argument("--param_ids", type=str, default="6", help="update params")
    parser.add_argument("--seed", type=int, default=42, help="Seed")


    args = parser.parse_args()
    args.retain_corpora = args.retain_corpora.split(",")
    args.forget_corpora = args.forget_corpora.split(",")
    args.steering_coeff_list = [float(c) for c in args.steering_coeffs.split(",")]
    args.layer_ids = [int(layer_id) for layer_id in args.layer_ids.split(",")]
    args.param_ids = [int(param_id) for param_id in args.param_ids.split(",")]
    return args


if __name__ == "__main__":
    args = get_args()
    SEED = args.seed
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    frozen_model, tokenizer = load_model(args.model_name_or_path)
    updated_model, tokenizer = load_model(args.model_name_or_path)
    
    keywords_list, forget_data_list, retain_data_list = get_data(
        args.forget_corpora,
        args.retain_corpora,
        args.batch_size,
    )
    run_rsv(
        updated_model,
        frozen_model,
        tokenizer,
        keywords_list,
        forget_data_list,
        retain_data_list,
        args,
    )
