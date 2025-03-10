export DATA_PATH="./preprocessed_data/c4_en_t5-base_512"
torchrun --nproc-per-node 1 --master_addr 127.0.0.1 --master_port 10050 torchrun_main.py \
    --model_config configs/llama_60m.json \
    --dataset_path $DATA_PATH \
    --base_dir /data/datasets/c4_en \
    --autoresume True \
    --batch_size 128 \
    --total_batch_size 512 \
    --lr 0.001 \
    --max_length 256 \
    --rank 128 \
    --update_proj_gap 200 \
    --cycle_length 10000 \
    --warmup_steps 1000 \
    --num_training_steps 10000 \
    --save_every 10000 \
    --eval_every 500 \
    --save_dir /data/pretrained_models/Llama60M \
    --optimizer adamw \
    --tags Relora_60M \
    --use_peft True\
    --Relora \
    # --rand_ratio 0.8 \