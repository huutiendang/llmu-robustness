import numpy as np
import torch
from transformers import AdamW 
import tqdm as tqdm
import json
import pandas as pd
import random
from typing import List
from baselines.utils import load_model, get_params, get_data, RandomizedModel
from baselines.losses import *

def run_dpo(
    updated_model,
    ref_model,
    tokenizer,
    keyword_list,
    forget_data_list,
    retain_data_list,
    args,
):
    updated_model = updated_model.train()
    params = get_params(updated_model, args.layer_ids, args.param_ids)

    optimizer = AdamW(params, lr=args.lr)

    truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side="right"

    with tqdm.tqdm(total=args.max_num_batches) as pbar:
        for idx in range(args.max_num_batches):
            topic_idx = idx % len(forget_data_list)
            batch_idx = idx // len(forget_data_list)

            unlearn_batch = forget_data_list[topic_idx][batch_idx]
            retain_batch = retain_data_list[topic_idx][batch_idx]
           
            max_length = 512 if topic_idx == 0 else 768

            idk_inputs = tokenizer(keyword_list[0: args.batch_size], return_tensors="pt", padding=True,  truncation=True, max_length=max_length).to(updated_model.device)

            unlearn_inputs = tokenizer(
                unlearn_batch, return_tensors="pt", padding=True,  truncation=True, max_length=max_length
            ).to(updated_model.device)
            # Unlearn loss
            unlearn_loss = dpo(updated_model=updated_model, ref_model=ref_model, unlearn_inputs=unlearn_inputs, idk_inputs=idk_inputs, beta=args.beta)

            retain_inputs = tokenizer(
                retain_batch, return_tensors="pt", padding=True,  truncation=True, max_length=max_length
            ).to(updated_model.device)

            # Retain loss
            if args.retain_loss_fn == "kl":
                retain_loss = kl(updated_model=updated_model, ref_model=ref_model, retain_inputs=retain_inputs, nu=args.nu)
            elif args.retain_loss_fn == "mse":         
                retain_loss = mse(updated_model=updated_model, ref_model=ref_model, retain_inputs=retain_inputs, nu=args.nu)
            else:
                raise ValueError(f"The loss is not support.")

            retain_loss *= args.alpha[topic_idx]

            loss = unlearn_loss + retain_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"loss topic {topic_idx}: {loss.item():.4g} | unlearn_loss: {unlearn_loss.item():.4g} | retain_loss: {retain_loss.item():.4g} | param_change: {params[0].grad.abs().mean().item():.4g}")
            pbar.update(1)
    
    tokenizer.truncation_side = truncation_side
    path = f"checkpoints/dpo/dpo_{args.retain_loss_fn}/{args.model_name_or_path}_batches-{args.max_num_batches}_alpha-{args.alpha[0]}-{args.alpha[1]}_lr-{args.lr}_target-layer-{args.target_layers[0]}_nu-{args.nu}"
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

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nu", type=float, default=0.05, help="noise magnitude")
    parser.add_argument("--alpha", type=str, default="50.0,50.0", help="unlearning weight")
    parser.add_argument("--beta", type=float, default=0.1, help="unlearning weight")
    parser.add_argument("--retain_loss_fn", type=str, choices=["mse", "kl"])
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_num_batches", type=int, default=500)
    parser.add_argument("--target_layers", type=str, default="7", help="layer to unlearn")
    parser.add_argument("--layer_ids", type=str, default="5,6,7", help="update layers")
    parser.add_argument("--param_ids", type=str, default="6", help="update params")
    
    args = parser.parse_args()
    args.retain_corpora = args.retain_corpora.split(",")
    args.forget_corpora = args.forget_corpora.split(",")
    args.alpha = [float(c) for c in args.alpha.split(",")]
    args.target_layers = [int(layer) for layer in args.target_layers.split(",")]
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

    ref_model = RandomizedModel(model_name_or_path=args.model_name_or_path, target_layers=args.target_layers)
    updated_model, tokenizer = load_model(args.model_name_or_path)

    with open("data/idontknow.json", "r") as f:
        keyword_list = json.load(f)

    forget_data_list, retain_data_list = get_data(
        forget_corpora=args.forget_corpora,
        retain_corpora=args.retain_corpora,
        batch_size=args.batch_size,
    )
    run_dpo(
        updated_model=updated_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        keyword_list=keyword_list["idk"],
        forget_data_list=forget_data_list,
        retain_data_list=retain_data_list,
        args=args,
    )
