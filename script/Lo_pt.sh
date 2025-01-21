export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc-per-node 2 --master_addr 127.0.0.1 --master_port 10002 torchrun_main.py \
    --model_config configs/llama_60m.json \
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
    --save_every 2000 \
    --eval_every 500 \
    --save_dir /data/pretrained_models/Llama60M \
    --optimizer galore_adamw \
    --tags GaLore_60M \
    --num_extra_training_steps 5000 \
    --with_tracking \
    # --dtype float32 \