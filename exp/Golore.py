import os
import math
import json
from typing import List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb
import bitsandbytes.functional as bnbF

from transformers import AutoModelForCausalLM, AutoConfig

from loguru import logger

from peft_pretraining.projector import get_orthogonal_matrix, get_random_orthogonal_matrix
from peft_pretraining.adamw import AdamW

@dataclass
class ReLoRaConfig:
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: List[str]
    keep_original_weights: bool
    lora_only: bool = False
    scale: float = 1.0
    quantize: str = None
    use_double_quant: bool = False
    forward_type: int = 0

class ReLoRaModel(torch.nn.Module):
    def __init__(
        self,
        model,
        *,
        target_modules=["attn", "attention", "mlp"],
        r=128,
        lora_alpha=32,
        lora_dropout=0.1,
        keep_original_weights=True,
        lora_only=False,
        scale: float = 1.0,
        quantize=None,
        use_double_quant=False,
    ):
        if r <= 0:
            raise ValueError("r must be positive. If you want r == 0, use the original model.")

        super().__init__()
        self.wrapped_model: nn.Module = model
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules
        self.keep_original_weights = keep_original_weights
        self.lora_only = lora_only
        self.scale = scale

        self._config = ReLoRaConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            keep_original_weights=keep_original_weights,
            quantize=quantize,
            use_double_quant=use_double_quant,
            forward_type = 0
        )

        print('enable golore')
        self.X = ReLoRaLinear(
            model.X.shape[0],
            model.X.shape[0],
            bias=False,
            r=self.r,
            scale=self.scale,
            relora_config=self._config,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            lora_only=self.lora_only,
            quantize=quantize,
            weight_data=model.X,
            bias_data=None,
            bnb_4bit_use_double_quant=use_double_quant,
        )

        torch.cuda.empty_cache()

    def forward(self, *args, **kwargs):
        # print()
        return self.X(*args, **kwargs)


    def _get_parent(self, module_name):
        module_names_list = module_name.split(".")
        parent_name = ".".join(module_names_list[:-1])
        parent = self.wrapped_model.get_submodule(parent_name)
        return parent

    def merge_and_reinit(self, optimizer = None, rand = False):
        for module in self.modules():
            if isinstance(module, ReLoRaLinear):
                module.merge_and_reinit(optimizer, rand)

    def save_pretrained(self, path):
        self.wrapped_model.save_pretrained(path)
        with open(os.path.join(path, "relora_config.json"), "w") as f:
            json.dump(self._config.__dict__, f, indent=4)

    @classmethod
    def from_pretrained(cls, path):
        with open(os.path.join(path, "relora_config.json"), "r") as f:
            relora_config = json.load(f)

        config = AutoConfig.from_pretrained(path)

        base_model = AutoModelForCausalLM.from_config(config)
        if "keep_original" in relora_config:
            print("WARNING: keep_original is deprecated. Use lora_only instead.")
            print(f"keep_original: {relora_config['keep_original']}")
            relora_config["lora_only"] = not relora_config.pop("keep_original")
            relora_config["keep_original_weights"] = not relora_config["lora_only"]

        if "trainable_scaling" not in relora_config:
            relora_config["trainable_scaling"] = False

        model = cls(base_model, **relora_config)

        with open(os.path.join(path, "pytorch_model.bin"), "rb") as f:
            state_dict = torch.load(f, map_location="cpu")

        model.wrapped_model.load_state_dict(state_dict, strict=True)
        return model


# The code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
class ReLoRaLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        relora_config: ReLoRaConfig = None,
        *,
        scale = 1.0,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_only: bool = False,
        weight_data=None,
        bias_data=None,
        bias=True,
        device=None,
        dtype=None,
        quantize=False,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
    ):
        nn.Module.__init__(self)
        if r <= 0:
            raise ValueError("r must be positive. If you want r == 0, use the original model.")

        if lora_only:
            self.weight = None
            self.bias = None
        else:
            # if full model weight + lora weight
            if bias_data is None:
                bias_data = torch.zeros(out_features, device=device, dtype=dtype, requires_grad=True) if bias else None
            self.bias = nn.Parameter(bias_data) if bias else None

            if weight_data is None:
                # note that our trainable weight are W_a and W_b
                weight_data = torch.zeros(out_features, in_features, device=device, dtype=dtype, requires_grad=False)

            if quantize is None:
                self.weight = nn.Parameter(weight_data, requires_grad=False)

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.relora_config = relora_config
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        self.lora_only = lora_only
        self.quantize = quantize
        self.proj_type = weight_data.shape[0] >= weight_data.shape[1]
        self.scale = scale
        self.proj_type = False

        if r > 0:
            self.lora_B = nn.Linear(in_features, r, bias=False)
            self.lora_B.requires_grad_(not self.proj_type)
            self.lora_A = nn.Linear(r, out_features, bias=False)
            self.lora_A.requires_grad_(self.proj_type)
            nn.init.zeros_(self.lora_A.weight)
            nn.init.zeros_(self.lora_B.weight)
            if not self.lora_only:
                self.weight.requires_grad = False


    @torch.no_grad()
    def merge_and_reinit(self, optimizer = None, rand = False):
        if self.lora_only:
            print("WARNING: Skipping merge and reinit, because only lora parameters are used")
            return

        WG = self.weight.grad.detach().clone()
        self.weight.grad.zero_()
        self.weight.requires_grad = False
        self.weight.addmm_(self.lora_A.weight, self.lora_B.weight, alpha = -self.scale)

        if self.proj_type:
            B0 = self.lora_B.weight.detach().clone()
            if not rand: self.lora_B.weight.copy_(get_orthogonal_matrix(WG, self.relora_config.r, 'right'))
            else: self.lora_B.weight.copy_(get_random_orthogonal_matrix(self.relora_config.r, WG.shape[1], dtype = WG.dtype))
            if isinstance(optimizer, torch.optim.SGD) and 'momentum_buffer' in optimizer.state[self.lora_A.weight] \
                and optimizer.state[self.lora_A.weight]['momentum_buffer'] is not None:
                M = optimizer.state[self.lora_A.weight]['momentum_buffer'].detach().clone()
                optimizer.state[self.lora_A.weight]['momentum_buffer'] =  (M @ B0) @ self.lora_B.weight.T
            elif isinstance(optimizer, AdamW) and 'exp_avg' in optimizer.state[self.lora_A.weight]:
                exp_avg = optimizer.state[self.lora_A.weight]['exp_avg'].detach().clone()
                optimizer.state[self.lora_A.weight]['exp_avg'] = (exp_avg @ B0) @ self.lora_B.weight.T
                # exp_avg_sq = optimizer.state[self.lora_A.weight]['exp_avg_sq'].detach().clone()
                # optimizer.state[self.lora_A.weight]['exp_avg_sq'] = (exp_avg_sq @ B0) @ self.lora_B.weight.T

            self.lora_A.weight.zero_()
            self.lora_A.requires_grad_(True)
            self.lora_A.weight.grad = torch.zeros_like(self.lora_A.weight)
            self.lora_A.weight.grad.copy_(-self.scale * torch.mm(WG, self.lora_B.weight.T))
        else:
            A0 = self.lora_A.weight.detach().clone()
            if not rand: self.lora_A.weight.copy_(get_orthogonal_matrix(WG, self.relora_config.r, 'left'))
            else: self.lora_A.weight.copy_(get_random_orthogonal_matrix(WG.shape[0], self.relora_config.r, dtype = WG.dtype))
            if isinstance(optimizer, torch.optim.SGD) and 'momentum_buffer' in optimizer.state[self.lora_B.weight] \
                and optimizer.state[self.lora_B.weight]['momentum_buffer'] is not None:
                M = optimizer.state[self.lora_B.weight]['momentum_buffer'].detach().clone()
                optimizer.state[self.lora_B.weight]['momentum_buffer'] =   self.lora_A.weight.T @ (A0 @ M)
            elif isinstance(optimizer, AdamW) and 'exp_avg' in optimizer.state[self.lora_A.weight]:
                exp_avg = optimizer.state[self.lora_A.weight]['exp_avg'].detach().clone()
                optimizer.state[self.lora_A.weight]['exp_avg'] = self.lora_A.weight.T @ (A0 @ exp_avg)
                # exp_avg_sq = optimizer.state[self.lora_A.weight]['exp_avg_sq'].detach().clone()
                # optimizer.state[self.lora_A.weight]['exp_avg_sq'] = self.lora_A.weight.T @ (A0 @ exp_avg_sq)
            
            self.lora_B.weight.zero_()
            self.lora_B.requires_grad_(True)
            self.lora_B.weight.grad = torch.zeros_like(self.lora_B.weight)
            self.lora_B.weight.grad.copy_(-self.scale * torch.mm(self.lora_A.weight.T, WG))

    def forward(self, A, B):
        if self.relora_config.forward_type == 1:
            if not self.weight.requires_grad:
                self.weight.requires_grad = True
                if self.proj_type: self.lora_A.requires_grad_(False)
                if not self.proj_type: self.lora_B.requires_grad_(False)
        # print(id(self.weight))
        weight = self.weight - self.scale * (self.lora_A.weight @ self.lora_B.weight)
        tmp = A @ weight
        return 0.5 * (tmp * tmp).sum() + (B * weight).sum()