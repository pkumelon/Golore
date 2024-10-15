#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export model_name="/data/pretrained_models/Llama-2-7b-hf"
# cd zo-bench
# bash ./script/run_sst2_galore.sh
# bash ./script/run_sst2_msgd.sh
# bash ./script/run_sst2_galore_adamw.sh
# bash ./script/run_sst2_golore_adamw.sh

# bash ./script/run_winogrande_msgd.sh
# bash ./script/run_winogrande_adamw.sh
# bash ./script/run_winogrande_zs.sh
bash ./script/run_winogrande_msgd_all.sh

