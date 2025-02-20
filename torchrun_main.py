"""
Distributed training code for ReLoRA.
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import peft_pretraining.GoLore
import peft_pretraining.relora
import sys
import yaml
import time
import json
import random
import argparse
from typing import Union

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
)

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    default_data_collator,
)
from tokenizers import Tokenizer

import datasets
import datasets.distributed
import wandb

from tqdm import tqdm
from loguru import logger

from peft_pretraining import training_utils, args_utils
from peft_pretraining.dataloader import SkipDataLoader, PreprocessedIterableDataset
from peft_pretraining.modeling_llama import LlamaForCausalLM
from peft_pretraining.modeling_pythia import GPTNeoXForCausalLM

transformers.logging.set_verbosity_error()

import peft_pretraining



# The code is based on https://github.com/Guitaricet/relora/blob/main/torchrun_main.py
def parse_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--training_config", type=str, default=None,
                        help="Alternative to providing the parameters. Overrides all parameters. Path to a yaml file with training run config")

    parser.add_argument("--model_config", type=str, default=None)
    parser.add_argument("--model_name_or_path", type=str, default=None, help="Huggingface model identifier, alternative to --model_config")
    parser.add_argument("--model_revision", type=str, default=None, help="Tag name, branch name, or commit hash of the model from HuggingFace Hub. E.g., v2.0.1 or step1000")
    parser.add_argument("--warmed_up_model", type=str, default=None, help="Start with warmed-up model weights. Does not restore optimizer and scheduler.")
    parser.add_argument("--resume_from", type=str, default=None, help="Continue training with ReLoRA, loading optimizer and scheduler from the checkpoint.")
    parser.add_argument("--load_optimizer_state_on_resume", default=True, type=lambda x: x.lower() == "true",
                        help="Load optimizer state from the checkpoint when resuming training. "
                             "If False, optimizer state will be initialized from scratch. Setting it to False is useful for some very specific experiments.")
    parser.add_argument("--base_dir", type=str, default=None, help="Path to a local dataset directory")
    parser.add_argument("--max_length", type=int, default=512)

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)

    parser.add_argument("--use_peft", default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--train_scaling", default=False, action="store_true")
    parser.add_argument("--force_keep_original", default=False, type=lambda x: x.lower() == "true",
                        help=("Keep original model parameters even if relora is None. "
                              "Useful for making sure that full-LoRa model is equivalent to model+LoRa."))

    parser.add_argument("--optimizer", default="Adam", help="Could be adam (for AdamW) or adam_zero for ZeroRedundancyOptimizer(AdamW)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restarts"])
    parser.add_argument("--cycle_length", type=int, default=None, help="Number of steps per cycle for cosine scheduler")
    parser.add_argument("--restart_warmup_steps", type=int, default=None, help="Number of steps for cosine restarts (only used for cosine_restarts)")
    parser.add_argument("--adjust_step", type=int, default=0, help="Number of steps to adjust the scheduler by. "
                            f"Useful when you want to sync ReLoRA resets with the scheduler for a warmed up model. "
                            f"You need to use it, when your warmup_step % relora_resets != 0")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)

    parser.add_argument("--eval_every", type=int, default=1_000)

    parser.add_argument("--num_training_steps", type=int, default=10_000,
                        help="Number of **update steps** to train for. "
                             "Notice that gradient accumulation is taken into account.")
    parser.add_argument("--max_train_tokens", type=training_utils.max_train_tokens_to_number, default=None,
                        help="Number of tokens to train on. Overwrites num_training_steps. "
                             "You can use M and B suffixes, e.g. 100M or 1B.")
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--keep_checkpoints", type=int, default=None,
                        help="Number of checkpoints to keep. By default, keep all checkpoints.")
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--workers", type=int, default=8)

    parser.add_argument("--quantize", default=None, type=str, choices=[None, "4bit", "8bit"])
    parser.add_argument("--use_double_quant", default=True, type=lambda x: x.lower() == "true")


    parser.add_argument("--distributed_type", type=str, default="ddp", choices=["ddp"])
    parser.add_argument("--autoresume", default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--comment", type=str, default=None, help="Wandb notes")
    parser.add_argument("--wandb_watch", default=False, type=lambda x: x.lower() == "true",
                        help="Enable wandb.watch (may make training unstable, but might be good for understanding gradients)")
    parser.add_argument("--skip_batches", default=None, type=str, help="Batch numbers to skip, separated by comma. E.g., 2003,2990,12309. Specifically, update_step numbers.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--proj_type", type=str, default="std")
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--update_proj_gap", type=int, default=None)
    parser.add_argument("--Golore", default=False, action="store_true")
    parser.add_argument("--Relora", default=False, action="store_true")
    parser.add_argument("--rand_ratio", type=float, default=2)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--dampening", type=float, default=0.9)
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument("--num_extra_training_steps", type=int, default=0,
                        help="Number of **extra update steps** to train for. "
                             "Notice that gradient accumulation is taken into account.")

    parser.add_argument("--single_gpu", default=False, action="store_true")
    
    args = parser.parse_args(args)

    args = args_utils.check_args_torchrun_main(args)
    return args


@torch.no_grad()
def evaluate_model(model: nn.Module, eval_dataloader, device, pad_idx, target_eval_tokens=10_000_000):
    _time = time.time()
    was_training = model.train
    model.eval()

    ddp_loss_info = torch.zeros(3).to(device)  # [loss, n_batches, n_tokens]
    tokens_in_batch_info = torch.zeros(1).to(device)

    rank = dist.get_rank()
    for i, batch in enumerate(eval_dataloader):
        if i == 0:
            # this way of estiming the number of eval steps
            # is needed to avoid a deadlock when using FSDP
            batch["input_ids"]: torch.Tensor
            tokens_in_batch_info[0] += batch["input_ids"].numel()
            dist.all_reduce(tokens_in_batch_info, op=dist.ReduceOp.SUM)
            n_eval_iters = int(target_eval_tokens / tokens_in_batch_info[0])

        if target_eval_tokens != -1 and i > n_eval_iters: break

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        loss = model(**batch, labels=labels).loss
        if torch.isnan(ddp_loss_info[0]):
            print(f"Rank {dist.get_rank()} got nan loss. This is probably a bug.")

        tokens_in_batch = batch["input_ids"].numel()
        assert tokens_in_batch > 0, "Batch size is zero"
        ddp_loss_info[0] += loss.detach()
        ddp_loss_info[1] += 1
        ddp_loss_info[2] += tokens_in_batch

    # check if loss is nan
    if torch.isnan(ddp_loss_info[0]):
        raise RuntimeError(f"Rank {rank} got nan loss. This is probably a bug.")

    # Gather losses across all GPUs
    dist.all_reduce(ddp_loss_info, op=dist.ReduceOp.SUM)
    eval_loss = ddp_loss_info[0] / ddp_loss_info[1]
    evaluated_on_tokens = ddp_loss_info[2].item()
    logger.info(f"Evaluated on {evaluated_on_tokens} tokens, eval loss: {eval_loss:.4f}")

    logger.info(f"Evaluation took {time.time() - _time:.2f} seconds")

    if was_training: model.train()
    return eval_loss, evaluated_on_tokens


def save_model_ddp(model, optimizer, scheduler, training_state_checkpoint, run_config, save_dir):
    global_rank = dist.get_rank()
    _time = time.time()

    if global_rank == 0:
        update_step = training_state_checkpoint["update_step"]
        scheduler_step = training_state_checkpoint["scheduler_step"]
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)

        _model = model.module
        _model.save_pretrained(save_dir)

    dist.barrier()
    if isinstance(optimizer, ZeroRedundancyOptimizer):
        logger.info("Started consolidating optimizer state dict")
        optimizer.consolidate_state_dict()
        logger.info(f"Consolidating optimizer state dict took {time.time() - _time:.2f} seconds")

    if global_rank == 0:
        optimizer_checkpoint = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "update_step": update_step,
            "scheduler_step": scheduler_step,
            "global_step": training_state_checkpoint["global_step"],
            "config": run_config,
            "dtype": args.dtype,
        }
        torch.save(optimizer_checkpoint, f"{save_dir}/optimizer.pt")

        if args.with_tracking: training_state_checkpoint["wandb_id"] = wandb.run.id
        with open(f"{save_dir}/training_state.json", "w") as f:
            json.dump(training_state_checkpoint, f, indent=4)

    logger.info(f"Saving took {time.time() - _time:.2f} seconds")
    dist.barrier()

def save_model_fsdp(model, optimizer, scheduler, training_state_checkpoint, run_config, save_dir):
    raise RuntimeError("FSDP is not supported anymore. There were a lot of isses with ReLoRA and FSDP and no speed or memory improvements.")
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        global_rank = dist.get_rank()
        update_step = training_state_checkpoint["update_step"]

        if global_rank == 0:
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)

        _model = model.module
        _model.save_pretrained(save_dir)

        if global_rank == 0:
            optimizer_checkpoint = {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "update_step": update_step,
                "global_step": training_state_checkpoint["global_step"],
                "config": run_config,
                "wandb": wandb.run.dir,
                "dtype": args.dtype,
            }
            torch.save(optimizer_checkpoint, f"{save_dir}/optimizer.pt")

            training_state_checkpoint["wandb_id"] = wandb.run.id
            with open(f"{save_dir}/training_state.json", "w") as f:
                json.dump(training_state_checkpoint, f, indent=4)


def save_model(model, *, optimizer, scheduler, training_state_checkpoint, run_config, distributed_type, save_dir):
    """
    Args:
        training_state_checkpoint: dict with keys:
            global_step: int
            update_step: int
            tokens_seen: int
            tokens_seen_before: int
            n_lora_restarts: int
            update_time: float
        run_config: 
    """
    if distributed_type == "ddp":
        save_model_ddp(model, optimizer, scheduler, training_state_checkpoint, run_config, save_dir)
    else:
        raise ValueError(f"Unknown distributed type {distributed_type}")

def main(args):
    # seed all
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    global_rank = int(os.environ['RANK'])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    logger.info(f"Global rank {global_rank}, local rank {local_rank}, device: {torch.cuda.current_device()}")

    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)

    logger.info("Process group initialized")
    device = f"cuda:{local_rank}"


    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert args.total_batch_size % world_size == 0, "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (args.batch_size * world_size)
            assert args.gradient_accumulation > 0, "gradient_accumulation must be greater than 0"

    assert args.gradient_accumulation * args.batch_size * world_size == args.total_batch_size, \
        "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"

    if args.max_train_tokens is not None:
        args.num_training_steps = args.max_train_tokens // args.total_batch_size
        logger.info(f"Setting num_training_steps to {args.num_training_steps} based on max_train_tokens")

    # turn off logger
    if global_rank != 0: logger.remove()

    wandb_id = None

    dist.barrier()  # guarantees none of the workers will read save_dir above here before it's created by rank 0

    # initialize wandb without config (it is passed later)
    if global_rank == 0 and args.with_tracking:
        wandb.init(project="GoLore", tags=args.tags, id=wandb_id, resume="allow", notes=args.comment)
        args.run_name = wandb.run.name
        if args.save_dir is None:
            args.save_dir = f"checkpoints/{wandb.run.name}"

        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, "training_config.yaml"), "w") as f:
            yaml.dump(vars(args), f)

    dist.barrier()  # guarantees that save_dir exists and wand initialized on rank 0

    # synchronize run name and save dir across all ranks
    run_name = [wandb.run.name] if global_rank == 0 and args.with_tracking else [""]
    dist.broadcast_object_list(run_name, src=0)
    run_name = run_name[0]
    args.run_name = run_name
    if args.save_dir is None:
        args.save_dir = f"checkpoints/{args.run_name}"

    logger.info(f"Using dist with rank {global_rank} (only rank 0 will log)")
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    logger.info("All good! Loading tokenizer now")
    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=args.max_length)
    pad_idx = tokenizer.pad_token_id
    logger.info("Tokenizer loaded")

    logger.info("Loading Huggingface dataset from directory")
    seed_for_shuffle = 42 
    base_dir = args.base_dir
    dataset_dict = datasets.load_dataset('json', 
                            data_files = {'train' : [f'c4-train.0{idx:04d}-of-01024.json.gz' for idx in range(1024)],
                                            'validation' : [f'c4-validation.0{idx:04d}-of-00008.json.gz' for idx in range(8)]},
                        data_dir = base_dir, streaming = True)
    train_dataset: datasets.Dataset = dataset_dict['train'].shuffle(seed=seed_for_shuffle)
    eval_dataset: datasets.Dataset = dataset_dict['validation'].shuffle(seed=seed_for_shuffle)
    if not args.single_gpu:
        train_dataset = datasets.distributed.split_dataset_by_node(train_dataset, rank=global_rank, world_size=world_size)
        eval_dataset = datasets.distributed.split_dataset_by_node(eval_dataset, rank=global_rank, world_size=world_size)
    train_dataset = PreprocessedIterableDataset(train_dataset, tokenizer, args.batch_size, args.max_length)
    eval_dataset = PreprocessedIterableDataset(eval_dataset, tokenizer, args.batch_size, args.max_length)

    if args.model_config is not None:
        model_config = AutoConfig.from_pretrained(args.model_config)
        t_vocab_size = tokenizer.get_vocab_size() if isinstance(tokenizer, Tokenizer) else tokenizer.vocab_size
        
        if model_config.vocab_size != t_vocab_size:
            logger.warning(f"Model config vocab size ({model_config.vocab_size}) does not match tokenizer vocab size ({t_vocab_size})")
            if model_config.vocab_size == 32000 and t_vocab_size == 32100:
                logger.warning("You are most likely reusing old checkpoints. This is alright, but not recommended.")
            else:
                raise ValueError(f"Model config vocab size ({model_config.vocab_size}) does not match tokenizer vocab size ({t_vocab_size})")

        if not isinstance(model_config, LlamaConfig):
            raise NotImplementedError(f"Unknown model config type {type(model_config)}, only LLaMA is supported")

        logger.info("Using local version of LLaMA")
        model = LlamaForCausalLM(model_config)
    else:
        logger.info(f"Using HuggingFace model {args.model_name_or_path} revision {args.model_revision}")
        model = GPTNeoXForCausalLM.from_pretrained(args.model_name_or_path, revision=args.model_revision)
        model_config = model.config

    global_step = 0
    update_step = 0
    scheduler_step = 0
    tokens_seen = 0
    tokens_seen_before = 0
    n_lora_restarts = 0
    n_optimizer_resets = 0

    if args.warmed_up_model is not None:
        logger.info("*" * 40)
        logger.info(f"Loading a warmed-up model from {args.warmed_up_model}")
        checkpoint_path = os.path.join(args.warmed_up_model, "pytorch_model.bin")  # !! won't work with sharded models
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
        logger.info(f"Model successfully loaded (strict=True policy)")

        if os.path.exists(os.path.join(args.warmed_up_model, "training_state.json")):
            logger.info(f"Loading training state variables like global_step, update_step, and tokens_seen from {args.warmed_up_model} (not optimizer state)")
            with open(os.path.join(args.warmed_up_model, "training_state.json")) as f:
                _old_state = json.load(f)
            global_step = _old_state["global_step"]
            update_step = _old_state["update_step"]
            scheduler_step = _old_state["scheduler_step"]
            tokens_seen = _old_state["tokens_seen"]
            tokens_seen_before = _old_state["tokens_seen_before"]
            logger.info(f"global_step       : {global_step}")
            logger.info(f"update_step       : {update_step}")
            logger.info(f"scheduler_step    : {scheduler_step}")
            logger.info(f"tokens_seen       : {tokens_seen}")
            logger.info(f"tokens_seen_before: {tokens_seen_before}")
            logger.info(f"Will train for {args.num_training_steps + args.num_extra_training_steps - update_step} update steps")
        else:
            logger.warning(f"Did not find training state in {args.warmed_up_model}, global step will start from zero")
        logger.info("*" * 40)

    params_before = sum(p.numel() for p in model.parameters())

    if args.use_peft:
        need_linear_weight = (
            args.rank is not None
            or args.force_keep_original
            or args.warmed_up_model is not None
        )
        logger.info(f"Wrapping model with LoRA ({need_linear_weight=})")

        # target modules should define all linear layers from transformer block
        # "attn" and "mlp" are used in LLaMA
        # "attention" and "mlp" are used in Pythia
        if args.Golore:
            from peft_pretraining.GoLore import ReLoRaModel, ReLoRaLinear
        if args.Relora:
            from peft_pretraining.relora import ReLoRaModel, ReLoRaLinear
        model = ReLoRaModel(
            model,
            r=args.rank,
            lora_dropout=0,
            target_modules=["attn", "attention", "mlp"],
            scale = args.scale,
            keep_original_weights=True,
            lora_only=not need_linear_weight,
            quantize=args.quantize,
            use_double_quant=args.use_double_quant,
        )


    if args.resume_from:
        logger.info(f"Loading model from {args.resume_from}")
        checkpoint_path = os.path.join(args.resume_from, "pytorch_model.bin")
        if isinstance(model, ReLoRaModel):
            model.wrapped_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
        else:
            model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)

        logger.info(f"Model successfully loaded (strict=True policy)")

        logger.info(f"Loading training state like global_step, update_step, and tokens_seen from {args.resume_from}")
        with open(os.path.join(args.resume_from, "training_state.json")) as f:
            _old_state = json.load(f)

        global_step = _old_state["global_step"]
        # We do overwrite update_step here to correctly initialize the scheduler
        # which should start from warmed_up_model's update step or zero
        _update_step = _old_state["update_step"]
        _scheduler_step = _old_state["scheduler_step"]
        tokens_seen = _old_state["tokens_seen"]
        tokens_seen_before = _old_state["tokens_seen_before"]
        n_lora_restarts = _old_state["n_lora_restarts"]
        logger.info(f"global_step       : {global_step}")
        logger.info(f"update_step       : {update_step}")
        logger.info(f"scheduler_step    : {scheduler_step}")
        logger.info(f"tokens_seen       : {tokens_seen}")
        logger.info(f"tokens_seen_before: {tokens_seen_before}")
        logger.info(f"Will train for {args.num_training_steps + args.num_extra_training_steps - _update_step} update steps")

    params_after = sum(p.numel() for p in model.parameters())

    added_floats = params_after - params_before

    # print params and trainable params
    logger.info(f"\n{model}\n")
    logger.info(f"Total params  before LoRA: {params_before / 1_000_000:.2f}M")
    logger.info(f"Total params  after  LoRA: {params_after / 1_000_000:.2f}M")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")
    logger.info(f"In total, added {added_floats / 1_000_000:.2f}M parameters to the model")

    logger.info(f"Saving model to {args.save_dir} every {args.save_every} update steps")

    if args.dtype in ["bf16", "bfloat16"]:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)

    n_total_params = sum(p.numel() for p in model.parameters())
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    p_trainable_params = n_trainable_params / n_total_params

    # ##############################
    # Distributed wrapping
    if args.distributed_type == "ddp":
        logger.info("Wrapping model with DDP")
        model: Union[ReLoRaModel, LlamaForCausalLM] = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters = True,
        )
    # ##############################
    if args.wandb_watch and args.with_tracking and global_rank == 0:
        _log_freq = 500
        logger.info(f"Tracking model gradients with wandb every {_log_freq} update steps")
        wandb.watch(model, log_freq=_log_freq)

    # Computing the number of parameters is done before wrapping the model with FSDP
    # but gettint the parameters for optimization is done after. This is intentional and doing it other way causes errors.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    trainable_params_names = [name for name, p in model.named_parameters() if p.requires_grad]
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    if args.use_peft and len(lora_params) == 0:
        raise ValueError("No LoRA parameters found")

    # Initialize wandb
    run_config = dict(vars(args))
    run_config.update({
        "max_lr": run_config.pop("lr"),  # rename lr to max_lr to avoid conflicts with scheduler
        "total_params_M": n_total_params / 1_000_000,
        "trainable_params_M": n_trainable_params / 1_000_000,
        "equivalent_params_M": params_before / 1_000_000,
        "percent_trainable_params": p_trainable_params,
        "name_trainable_params": trainable_params_names,
        "model": model_config.to_dict(),
        "world_size": world_size,
        "device": str(device),
    })

    if global_rank == 0 and args.with_tracking:
        wandb.config.update(run_config, allow_val_change=True)
        wandb.save(os.path.abspath(__file__), policy="now") # save current script

    optimizer_state_keys = None
    optimizer_kwargs = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "betas": (args.adam_beta1, args.adam_beta2),
    }
    if 'lore' in args.optimizer.lower() or 'fira' in args.optimizer.lower():
        # make parameters with "rank" to a single group, if param_name has "mlp" or "attn"
        galore_params = []
        target_modules_list = ["attn", "attention", "mlp"]
        for module_name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue
            
            print('enable Lore for weights in module: ', module_name)
            galore_params.append(module.weight)
        id_galore_params = [id(p) for p in galore_params]
        # make parameters without "rank" to another group
        regular_params = [p for p in model.parameters() if id(p) not in id_galore_params]
        param_groups = [{'params': regular_params}, 
                        {'params': galore_params, 'rank': args.rank, 'update_proj_gap': args.update_proj_gap, 'scale': args.scale, 'proj_type': args.proj_type, 'rand_epoch': args.rand_ratio * args.num_training_steps}]
        
    momentum = args.momentum
    dampening = args.dampening
    if args.optimizer.lower() == "sgd":
        from torch.optim import SGD
        optimizer = SGD(optimizer_grouped_parameters, lr = args.lr, momentum = momentum, dampening = dampening)
    elif args.optimizer.lower() == "adamw":
        from peft_pretraining.adamw import AdamW
        logger.info("Using Adam optimizer")
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.weight_decay)
        optimizer_state_keys = ["exp_avg", "exp_avg_sq"]
    elif args.optimizer.lower() == "adamw8bit":
        from peft_pretraining.adamw8bit import AdamW8bit
        logger.info("Using Adam optimizer")
        optimizer = AdamW8bit(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.weight_decay)
        optimizer_state_keys = ["exp_avg", "exp_avg_sq"]
    elif args.optimizer.lower() == "adam_zero":
        logger.info("Using Adam optimizer with ZeRO")
        optimizer = ZeroRedundancyOptimizer(
            trainable_params,
            optimizer_class=torch.optim.AdamW,
            **optimizer_kwargs,
        )
        optimizer_state_keys = ["exp_avg", "exp_avg_sq"]
    elif args.optimizer.lower() == "galore_adamw":
        from peft_pretraining.galore_torch.adamw import AdamW
        optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "galore_adamw8bit":
        from peft_pretraining.galore_torch.adamw8bit import AdamW8bit
        optimizer = AdamW8bit(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "galore_sgd":
        from peft_pretraining.galore_torch.sgd import SGD
        # from torch.optim import SGD
        optimizer = SGD(param_groups, lr=args.lr, weight_decay=args.weight_decay, momentum = momentum, dampening = dampening)
    elif args.optimizer.lower() == "golore_adamw":
        from peft_pretraining.golore_torch.adamw import AdamW
        optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "golore_adamw8bit":
        from peft_pretraining.golore_torch.adamw8bit import AdamW8bit
        optimizer = AdamW8bit(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "golore_sgd":
        from peft_pretraining.golore_torch.sgd import SGD
        optimizer = SGD(param_groups, lr=args.lr, weight_decay=args.weight_decay, momentum = momentum, dampening = dampening)
    elif args.optimizer.lower() == "fira_adamw":
        from peft_pretraining.fira.fira_adamw import AdamW
        optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

    logger.info(f"Scheduler will run for {args.num_training_steps} update steps")
    scheduler = training_utils.get_scheduler(
        optimizer=optimizer,
        scheduler_type=args.scheduler,
        num_training_steps=args.num_training_steps,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        cycle_length=args.cycle_length,
        restart_warmup_steps=args.restart_warmup_steps,
        adjust_step=args.adjust_step,
    )

    if args.resume_from:
        if args.load_optimizer_state_on_resume:
            _optimizer_dir = args.resume_from
            optimizer_checkpoint = torch.load(os.path.join(_optimizer_dir, "optimizer.pt"), map_location="cpu")
            optimizer.load_state_dict(optimizer_checkpoint["optimizer"])
            scheduler.load_state_dict(optimizer_checkpoint["scheduler"])
            update_step = optimizer_checkpoint["update_step"]
            scheduler_step = optimizer_checkpoint["scheduler_step"]
            global_step = optimizer_checkpoint["global_step"]
            logger.info(f"Optimizer and scheduler restored from {_optimizer_dir}")

        # check that batch_size did not change or dataloader rewinding won't work
        _training_config_path = os.path.join(args.resume_from, "training_config.yaml")
        if os.path.exists(_training_config_path):
            with open(_training_config_path) as f:
                _old_training_config = yaml.safe_load(f)
            if args.batch_size != _old_training_config["batch_size"]:
                raise RuntimeError("Cannot resume from a checkpoint with a different batch size.")

        logger.info("Setting scheduler to the same state as in the checkpoint")
        
    for _ in range(scheduler_step):
        scheduler.step()
    logger.info(f"Scheduler state restored from {args.resume_from}")
    # current lr
    logger.info(f"Current lr is {optimizer.param_groups[0]['lr']}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=None, num_workers=args.workers)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=None, num_workers=args.workers)

    _skip_batches = update_step * args.gradient_accumulation

    # global steps and others are defined above
    update_time = time.time()
    local_step = 0  # when warmed_up_model is used, local_step != global_step
    loss_info = torch.tensor([0.0, 0.0, 0.0], device=device)  # loss, n_batches, n_NaNs
    n_skipped_batches = 0

    # ##############################
    # TRAINING LOOP
    # we assert above that the dataset is large enough to train for num_training_steps, so no need for epochs
    # ##############################

    unwrapped_model = model
    if args.distributed_type == "ddp": unwrapped_model = model.module

    logger.info(f"Starting training at update step {update_step} with {args.num_training_steps - update_step} update steps")
    if global_rank == 0:
        # fix tqdm visual length to 80 so that the progress bar
        # doesn't jump around when changing from external display to laptop
        pbar = tqdm(total=args.num_training_steps + args.num_extra_training_steps - update_step, desc="Update steps", ncols=80)
    logger.info(f"Performing evaluation at step {update_step}")
    total_loss, evaluated_on_tokens = evaluate_model(model, eval_loader, device, pad_idx = pad_idx)

    if global_rank == 0 and args.with_tracking:
        wandb.log({
            "final_eval_loss": total_loss,
            "final_eval_tokens": evaluated_on_tokens,
            },
            step=global_step,
        )
    logger.info(f"Eval loss at step {update_step}: {total_loss}")

    for batch in train_loader:
        global_step += 1
        local_step += 1

        if update_step in args.skip_batches:
            if global_step % args.gradient_accumulation == 0:
                update_step += 1
            continue

        if local_step == 1: logger.info(f"Starting first step")
        if update_step >= args.num_training_steps + args.num_extra_training_steps:
            logger.info(f"Reached max number of update steps (f{args.num_training_steps + args.num_extra_training_steps}). Stopping training.")
            print(f"Rank {global_rank} stopping training.")
            break
        
        if args.use_peft and isinstance(unwrapped_model, ReLoRaModel):
            reset_relora = update_step % args.update_proj_gap == 0
            if isinstance(unwrapped_model, peft_pretraining.GoLore.ReLoRaModel):
                unwrapped_model._config.forward_type = reset_relora

        batch = {k: v.to(device) for k, v in batch.items()}
        tokens_seen += batch["input_ids"].numel() * world_size
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        loss = model(**batch, labels=labels).loss

        loss_info[0] += loss.detach()
        loss_info[1] += 1
        loss_info[2] += torch.isnan(loss).float()

        if global_step == 0 and global_rank == 0 and args.with_tracking:
            # log loss without any optimization
            wandb.log({"loss": loss.item(), "update_step": 0}, step=0)

        scaled_loss = loss / args.gradient_accumulation
        scaled_loss.backward()

        if global_step % args.gradient_accumulation != 0:
            continue

        # The below code is only executed during the update step

        if args.clip_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.clip_grad_norm, error_if_nonfinite=True)
            if global_rank == 0 and args.with_tracking:
                wandb.log({"grad_norm": grad_norm.item()}, step=global_step)

        # ##############################
        # MERGE AND REINIT

        # restart model after we modify the learning rate, so on the next step after the relora frequency
        # print(update_step, reset_relora)
        if args.use_peft and isinstance(unwrapped_model, ReLoRaModel) and reset_relora:
            _lora_reset_time = time.time()
            # logger.info(f"{args.resume_from=}, {local_step=}, {args.relora=}, thresh: {local_step // args.gradient_accumulation}")
            logger.info(f"Performing lora reset at update step {update_step}. Current lr is {optimizer.param_groups[0]['lr']}")
            n_lora_restarts += 1

            if isinstance(unwrapped_model, peft_pretraining.GoLore.ReLoRaModel):
                use_rand = update_step / args.num_training_steps >= args.rand_ratio
                unwrapped_model.merge_and_reinit(optimizer, use_rand)
            else:
                unwrapped_model.merge_and_reinit()

            
            _lora_reset_time = time.time() - _lora_reset_time
            logger.info(f"LoRA reset took {_lora_reset_time:.2f}s")

        dist.all_reduce(loss_info, op=dist.ReduceOp.SUM)
        _loss = loss_info[0] / loss_info[1]  # loss to log in wandb below

        if loss_info[2] == 0:  # no NaNs, update model
            optimizer.step()
            if scheduler_step < args.num_training_steps - 1:
                scheduler.step()
                scheduler_step += 1
        else:
            logger.error(f"Nan detected in loss_info, {_loss=}, skipping update")
            n_skipped_batches += 1

            if n_skipped_batches > 0.05 * args.num_training_steps:
                logger.error(f"More than 5% of batches skipped due to NaNs, stopping training.")
                break

        if global_rank == 0: pbar.update(1)
        optimizer.zero_grad()
        update_step += 1
        update_time = time.time() - update_time

        loss_info = torch.zeros_like(loss_info)

        if local_step > args.gradient_accumulation and update_step % args.save_every == 0:
            current_model_directory = f"{args.save_dir}/model_{update_step}_use_peft({args.use_peft})_optimizer({args.optimizer})_Golore({args.Golore})"
            logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
            training_state_checkpoint = {
                "global_step": global_step,
                "update_step": update_step,
                "scheduler_step": scheduler_step,
                "tokens_seen": tokens_seen,
                "tokens_seen_before": tokens_seen_before,
                "n_lora_restarts": n_lora_restarts,
                "n_optimizer_resets": n_optimizer_resets,
                "update_time": update_time,
            }
            save_model(
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                training_state_checkpoint=training_state_checkpoint,
                run_config=run_config,
                distributed_type=args.distributed_type,
                save_dir=current_model_directory,
            )
            if args.keep_checkpoints is not None:
                training_utils.delete_old_checkpoints(args.save_dir, keep=args.keep_checkpoints)

        # ##############################
        # EVALUATION
        if update_step % args.eval_every == 0:
            logger.info(f"Performing evaluation at step {update_step}")
            total_loss, evaluated_on_tokens = evaluate_model(model, eval_loader, device, pad_idx = pad_idx)

            if global_rank == 0 and args.with_tracking:
                wandb.log({
                    "final_eval_loss": total_loss,
                    "final_eval_tokens": evaluated_on_tokens,
                    },
                    step=global_step,
                )
            logger.info(f"Eval loss at step {update_step}: {total_loss}")
        # ##############################

        lr = optimizer.param_groups[0]["lr"]
        tokens_in_update = tokens_seen - tokens_seen_before
        tokens_seen_before = tokens_seen
        batches_in_update = args.gradient_accumulation * world_size

        if global_rank == 0 and args.with_tracking:
            wandb.log({
                "loss": loss.item(),
                "lr": lr,
                "update_step": update_step,
                "scheduler_step": scheduler_step,
                "tokens_seen": tokens_seen,
                "throughput_tokens": tokens_in_update / update_time,
                "throughput_examples": args.total_batch_size / update_time,
                "throughput_batches": batches_in_update / update_time,
                "n_lora_restarts": n_lora_restarts,
                "n_optimizer_resets": n_optimizer_resets,
                },
                step=global_step,
            )
            if args.train_scaling:
                all_scaling_factors = []
                for module in model.modules():
                    if isinstance(module, ReLoRaLinear):
                        all_scaling_factors.append(module.scaling.data.item())
                if args.with_tracking: wandb.log({"lora_scaling": torch.tensor(all_scaling_factors)}, step=global_step)
        update_time = time.time()
    else: # for-else statement
        print(f"Warning: reached the end of the dataset. Training stopped, {global_rank=}, {update_step=}")
        logger.warning("Reached the end of the dataset. Training stopped")

    # ##############################
    # END of training loop
    # ##############################
    logger.info("Training finished")
    if global_rank == 0: pbar.close()

    current_model_directory = f"{args.save_dir}/model_{update_step}_use_peft({args.use_peft})_optimizer({args.optimizer})_Golore({args.Golore})"
    if not os.path.exists(current_model_directory):
        logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
        training_state_checkpoint = {
            "global_step": global_step,
            "update_step": update_step,
            "scheduler_step": scheduler_step,
            "tokens_seen": tokens_seen,
            "tokens_seen_before": tokens_seen_before,
            "n_lora_restarts": n_lora_restarts,
            "update_time": update_time,
        }
        save_model(
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            training_state_checkpoint=training_state_checkpoint,
            run_config=run_config,
            distributed_type=args.distributed_type,
            save_dir=current_model_directory,
        )

    # Final evaluation
    logger.info("Running final evaluation")
    model.eval()
    del loss, optimizer, scheduler
    import gc; gc.collect()
    torch.cuda.empty_cache()

    total_loss, evaluated_on_tokens = evaluate_model(
        model, eval_loader, device, pad_idx = pad_idx,
        target_eval_tokens=100_000_000,
    )
    
    if global_rank == 0 and args.with_tracking:
        wandb.log({
            "final_eval_loss": total_loss,
            "final_eval_tokens": evaluated_on_tokens,
            },
            step=global_step,
        )
        logger.info(f"Final eval loss: {total_loss}")

    if eval_loader is not None:
        logger.info("Running test evaluation (full test set!)")
        total_loss, evaluated_on_tokens = evaluate_model(
            model, eval_loader, device, pad_idx = pad_idx,
            target_eval_tokens=-1,
        )

        if global_rank == 0 and args.with_tracking:
            wandb.log({
                "final_test_loss": total_loss,
                "final_test_tokens": evaluated_on_tokens,
                },
                step=global_step,
            )
            logger.info(f"Test loss: {total_loss}")

    if global_rank == 0 and args.with_tracking:
        wandb.finish()

    logger.info("Script finished successfully")
    print(f"Rank {global_rank} finished successfully")


if __name__ == "__main__":
    args = parse_args()
    main(args)
