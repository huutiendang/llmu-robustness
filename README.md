# llmu-robustness

Official implementation for the paper:  **Improving LLM Unlearning Robustness via Random Perturbations**, 

Dang Huu-Tien and Hoang Thanh-Tung and Anh Tuan Bui and Phuong Minh Nguyen and Le-Minh Nguyen and Naoya Inoue, 

**Transactions on Machine Learning Research**, 2026.

## Installation
---

## Installation

**Create environment:**
```bash
conda create -n llmu-robustness
conda activate llmu-robustness
pip install -r requirements.txt
```

## Evaluation Framework

We use the lm-evaluation-harness for evaluation.

```bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```
## Dataset 
Download the required datasets from the WMDP repository and place them in the data/ directory.

## Unlearning
Run the unlearning process using one of the following methods:
For example:  RMU and SimNPO
```
python -m baselines.rm.rmu.unlearn \
    --model_name_or_path "HuggingFaceH4/zephyr-7b-beta" \
    --max_num_batches 500 \
    --alpha "1200,1200" \
    --steering_coeffs "6.5,6.5" \
    --seed 42 \
    --batch_size 4 \
    --nu 0.0 \
    --layer_id 7 \
    --layer_ids "5,6,7";
```
```
python -m baselines.po.simnpo.unlearn \
    --model_name_or_path "HuggingFaceH4/zephyr-7b-beta" \
    --max_num_batches $BATCH \
    --alpha "20,50" \
    --retain_loss_fn "kl" \
    --seed 42 \
    --beta 0.1 \
    --nu 0.0 \
    --batch_size 4 \
    --target_layers "7" \
    --layer_ids "5,6,7";
```
To perform a grid search over unlearning methods:
For RM in [rmu, adaptive_rmu, rsv]
```
bash scripts/rm/$RM.sh
```
For PO in [dpo, npo, simnpo]
```
bash scripts/po/$PO.sh
```
Trained models will be saved at ```checkpoints/```
## Evaluation

```
!lm-eval --model hf \
    --model_args pretrained=$CHECK_POINT \
    --tasks mmlu,wmdp \
    --batch_size=16
```

## Citations

```
@article{
huu-tien2026improving,
title={Improving {LLM} Unlearning Robustness via Random Perturbations},
author={Dang Huu-Tien and Hoang Thanh-Tung and Anh Tuan Bui and Phuong Minh Nguyen and Le-Minh Nguyen and Naoya Inoue},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2026},
url={https://openreview.net/forum?id=QYw192hTdH},
note={}
}
```
