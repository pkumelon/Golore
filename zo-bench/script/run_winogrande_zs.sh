#!/bin/bash
lr=1e-4
momentum=0.9
dampening=0.0
optimizer=sgd
task_name=WinoGrande
num_train_epochs=30
time=$(date "+%Y%m%d%H%M%S")
python run.py --model_name=$model_name \
    --task_name=$task_name \
    --output_dir=result/$task_name-ft-$optimizer-$time \
    --num_train_epochs=30 \
    --per_device_train_batch_size=16 \
    --load_best_model_at_end \
    --evaluation_strategy=steps \
    --save_strategy=steps \
    --save_total_limit=1 \
    --eval_steps=500 \
    --max_steps=-1 \
    --logging_steps=10 \
    --num_eval=1000 \
    --num_train=1000 \
    --num_dev=100 \
    --train_as_classification=False \
    --trainer=zeroshot \
    --train_set_seed=0 \
    --lr_scheduler_type=constant \
    --save_steps=1000 \
    --load_bfloat16 \
    --bf16 \
    --optimizer=$optimizer \
    --learning_rate=$lr \
    --momentum=$momentum \
    --dampening=$dampening \
    --weight_decay=0.0 \
