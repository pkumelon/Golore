export DATA_PATH="./preprocessed_data/c4_en_t5-base_512"
export CUDA_VISIBLE_DEVICES=2,3
torchrun --nproc-per-node 2 --master_addr 127.0.0.2 --master_port 10028 torchrun_main.py \
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
    --rand_ratio 0.8 \
    --tags GoLore_60M_rand0.8 \
    --use_peft True \
    --Golore \