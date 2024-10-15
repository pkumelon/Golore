#!/bin/bash
#SBATCH -J GoLore
#SBATCH -o output/job.%j.out
#SBATCH -e output/job.%j.err
#SBATCH -p GPUA800
#SBATCH --time=5-00:00:00
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# network
export http_proxy="http://172.17.6.6:7890"
export https_proxy="http://172.17.6.6:7890"
curl -I google.com

# environment
source /gpfs/share/home/2301110044/miniconda3/bin/activate zobench # 载入 conda 环境

# cuda 
# export CUDA_VISIBLE_DEVICES=0

# wandb
export WANDB_API_KEY="YOUR_WANDB_API_KEY"
wandb login

# run and location
export model_name="/gpfs/share/home/2301110044/pretrained_models/Llama-2-7b-hf"
cd /gpfs/share/home/2301110044/ccy/code/LLM-repo/zo-bench
# module add gcc/13.2.0   # 导入gcc
# bash ./script/run_sst2_galore.sh
# bash ./script/run_sst2_msgd.sh
# bash ./script/run_sst2_adam.sh
# bash ./script/run_sst2_adamw.sh
# bash ./script/run_sst2_galore_adamw.sh
# bash ./script/run_sst2_golore_adamw.sh
# bash ./script/run_winogrande_msgd_all.sh
# bash ./script/run_winogrande_adamw_all.sh
# bash ./script/run_winogrande_galore_all.sh
bash ./script/run_winogrande_golore_all.sh
# bash ./script/run_winogrande_galore_adamw_all.sh
# bash ./script/run_winogrande_golore_adamw_all.sh



# for slurm interactive
# salloc -p GPUA800 -N1 --gres=gpu:1 --qos normal -t 50:00
  
