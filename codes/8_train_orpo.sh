#!/bin/bash
python 8_train_orpo.py \
    --model_name_or_path=microsoft/Phi-3-mini-4k-instruct \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --num_train_epochs 10 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 4 \
    --beta 2.5 \
    --logging_steps 50 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 2 \
    --output_dir="color-phi-orpo-b8-lr1e6-beta2.5" \
    --resume_from_checkpoint=True \
    --load_best_model_at_end \
    --warmup_steps 100 \
    --report_to wandb \
    --logging_first_step \
    --no_remove_unused_columns