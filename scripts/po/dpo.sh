for SEED in 42
do
    for BATCH in 500
    do
        for ID in 7
        do
            for NU in 0.0 #0.01 0.012 0.014 0.016 0.018 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1
            do
                python -m baselines.po.dpo.unlearn \
                    --model_name_or_path "HuggingFaceH4/zephyr-7b-beta" \
                    --max_num_batches $BATCH \
                    --alpha "30,50" \
                    --retain_loss_fn "kl" \
                    --seed $SEED \
                    --beta 0.1 \
                    --nu $NU \
                    --batch_size 4 \
                    --target_layers "$ID" \
                    --layer_ids "$((ID - 2)),$((ID - 1)),$ID";
                
                python -m baselines.po.dpo.unlearn \
                    --model_name_or_path "HuggingFaceH4/zephyr-7b-beta" \
                    --max_num_batches $BATCH \
                    --alpha "5,20" \
                    --retain_loss_fn "mse" \
                    --seed $SEED \
                    --beta 0.1 \
                    --nu $NU \
                    --batch_size 4 \
                    --target_layers "$ID" \
                    --layer_ids "$((ID - 2)),$((ID - 1)),$ID";
            done
        done
    done
done