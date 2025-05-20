for SEED in 42
do
    for ALPHA in 1200
    do
        for ID in 7
        do
            for NU in 0.0 #0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1
            do
                for BATCH in 500
                do
                    python -m baselines.rm.adaptive_rmu.unlearn \
                    --model_name_or_path "HuggingFaceH4/zephyr-7b-beta" \
                    --max_num_batches $BATCH \
                    --alpha "${ALPHA},${ALPHA}" \
                    --batch_size 4 \
                    --nu $NU \
                    --seed $SEED \
                    --scale "3.0,3.0" \
                    --layer_id $ID \
                    --layer_ids "$((ID - 2)),$((ID - 1)),$ID";
                done
            done
        done
    done
done