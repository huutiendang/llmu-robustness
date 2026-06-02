# llmu-robustness

Official implementation for the paper:  **Improving LLM Unlearning Robustness via Random Perturbations**, Dang Huu-Tien and Hoang Thanh-Tung and Anh Tuan Bui and Phuong Minh Nguyen and Le-Minh Nguyen and Naoya Inoue, **TMLR**, 2026.

## Abstract
Here, we show that current LLM unlearning methods inherently reduce models' robustness, causing them to misbehave even when a single non-adversarial forget-token is present in the retain-query. Toward understanding underlying causes, we propose a novel theoretical framework that reframes the unlearning process as a backdoor attack and defense problem: we formulate how the forgetting process inadvertently learns to align forget-tokens (backdoor triggers) with the target-representations (target labels). As a result, forget-tokens act as backdoor triggers that, when activated in retain-queries, cause disruptions in unlearned models' behaviors, similar to successful backdoor attacks. The sense that, LLM unlearning methods themselves poison the model, make it more vulnerable to forget-tokens, and hide rather than erase target knowledge, describes their true mechanism. To mitigate the vulnerability caused by the forgetting process, we reinterpret the retaining process as a backdoor defense and propose Random Noise Augmentation (RNA), a lightweight, model and method-agnostic approach with theoretical guarantees for improving the robustness of unlearned models. Extensive experiments demonstrate that RNA significantly improves the robustness of unlearned models while preserving forget and retain performances. This backdoor attack-defense framework offers insights into the mechanism of unlearning that can shed light on future research directions for improving unlearning robustness.

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
