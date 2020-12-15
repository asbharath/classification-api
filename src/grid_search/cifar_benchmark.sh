#!/bin/bash 

CLASSES=10
BATCH_SIZE=128
MODEL_NAME="MobileNetV3LargeCIFAR"

for RUN in 1 2; do
    for SGD_MOM in  0.80 0.85; do 
        for FRAMEWORK in "tensorflow" "upstride_type2"; do 
            if [[ "$FRAMEWORK" == "upstride_type2" ]]; then 
                FACTOR=4
            else
                FACTOR=1
            fi
            python train.py --model_name $MODEL_NAME \
                --num_epochs 400 \
                --early_stopping 40 \
                --factor $FACTOR \
                --num_classes $CLASSES \
                --framework $FRAMEWORK \
                --use_wandb false \
                --run_name "${RUN}_${MODEL_NAME}_${FRAMEWORK}_MOM_${SGD_MOM}_BS_${BATCH_SIZE}" \
                --log_dir "/results/${RUN}_${MODEL_NAME}_${FRAMEWORK}_MOM_${SGD_MOM}_BS_${BATCH_SIZE}" \
                --checkpoint_dir "/checkpoints/${RUN}_${MODEL_NAME}_${FRAMEWORK}_MOM_${SGD_MOM}_BS_${BATCH_SIZE}" \
                --configuration.with_mixed_precision True \
                --configuration.profiler True \
                --input_size 32 32 3 \
                --dataloader.train_list Normalize RandomHorizontalFlip Cutout Translate \
                --dataloader.Normalize.scale_in_zero_to_one True \
                --dataloader.Normalize.only_subtract_mean True \
                --dataloader.Cutout.length 4 \
                --dataloader.Translate.width_shift_range 0.25 \
                --dataloader.Translate.height_shift_range 0.25 \
                --dataloader.val_list Normalize \
                --dataloader.name cifar10 \
                --dataloader.train_split_id train \
                --dataloader.val_split_id test \
                --dataloader.batch_size $BATCH_SIZE \
                --optimizer.name sgd_momentum \
                --optimizer.momentum $SGD_MOM \
                --optimizer.lr 0.1 \
                --optimizer.lr_decay_strategy.activate True \
                --optimizer.lr_decay_strategy.lr_params.patience 20 \
                --optimizer.lr_decay_strategy.lr_params.strategy lr_reduce_on_plateau \
                --optimizer.lr_decay_strategy.lr_params.decay_rate 0.3 \
                    > "${RUN}_${MODEL_NAME}_${FRAMEWORK}_MOM_${SGD_MOM}_BS_${BATCH_SIZE}.log" 
        done
    done
done