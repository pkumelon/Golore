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

from .projector import get_orthogonal_matrix, get_random_orthogonal_matrix
from .adamw import AdamW

# The code is based on https://github.com/Guitaricet/relora/blob/main/peft_pretraining/relora.py
@dataclass
class ReLoRaConfig:
    r: int
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
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules
        self.keep_original_weights = keep_original_weights
        self.lora_only = lora_only
        self.scale = scale

        self._config = ReLoRaConfig(
            r=r,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            keep_original_weights=keep_original_weights,
            quantize=quantize,
            use_double_quant=use_double_quant,
            forward_type = 0
        )

        # patch methods
        # self.forward = self.wrapped_model.forward

        target_modules_list = target_modules
        if isinstance(target_modules_list, str):
            target_modules_list = [target_modules_list]

        for module_name, module in self.wrapped_model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue
            
            print('enable GaCoRe for weights in module: ', module_name)

            weight_data = module.weight.data if keep_original_weights else None
            bias_data = None
            if module.bias is not None:
                bias_data = module.bias.data if keep_original_weights else None

            new_module = ReLoRaLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                r=self.r,
                scale=self.scale,
                relora_config=self._config,
                lora_dropout=self.lora_dropout,
                lora_only=self.lora_only,
                quantize=quantize,
                weight_data=weight_data,
                bias_data=bias_data,
                bnb_4bit_use_double_quant=use_double_quant,
            )
            # print(module_name, new_module)
            if self.keep_original_weights:
                # make lora'ed network to be exacty the same as the original network at initialization
                # nn.init.zeros_(new_module.lora_A.weight)
                assert new_module.lora_A.bias is None
                assert new_module.lora_B.bias is None

            if self.lora_only:
                assert not self.keep_original_weights
                module.weight = None

            del module

            parent = self._get_parent(module_name)
            module_suffix = module_name.split(".")[-1]
            setattr(parent, module_suffix, new_module)

        torch.cuda.empty_cache()

    def forward(self, *args, **kwargs):
        return self.wrapped_model(*args, **kwargs)


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
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        self.lora_only = lora_only
        self.quantize = quantize
        self.proj_type = weight_data.shape[0] >= weight_data.shape[1]
        self.scale = scale
        # self.proj_type = True

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
            if isinstance(optimizer, torch.optim.SGD) and 'momentum_buffer' in optimizer.state[self.lora_A.weight]:
                M = optimizer.state[self.lora_A.weight]['momentum_buffer'].detach().clone()
                optimizer.state[self.lora_A.weight]['momentum_buffer'].copy_((M @ B0) @ self.lora_B.weight.T)
            elif isinstance(optimizer, AdamW) and 'exp_avg' in optimizer.state[self.lora_A.weight]:
                exp_avg = optimizer.state[self.lora_A.weight]['exp_avg'].detach().clone()
                optimizer.state[self.lora_A.weight]['exp_avg'].copy_((exp_avg @ B0) @ self.lora_B.weight.T)
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
            if isinstance(optimizer, torch.optim.SGD) and 'momentum_buffer' in optimizer.state[self.lora_B.weight]:
                M = optimizer.state[self.lora_B.weight]['momentum_buffer'].detach().clone()
                optimizer.state[self.lora_B.weight]['momentum_buffer'].copy_(self.lora_A.weight.T @ (A0 @ M))
            elif isinstance(optimizer, AdamW) and 'exp_avg' in optimizer.state[self.lora_B.weight]:
                exp_avg = optimizer.state[self.lora_B.weight]['exp_avg'].detach().clone()
                optimizer.state[self.lora_B.weight]['exp_avg'].copy_(self.lora_A.weight.T @ (A0 @ exp_avg))
                # exp_avg_sq = optimizer.state[self.lora_A.weight]['exp_avg_sq'].detach().clone()
                # optimizer.state[self.lora_A.weight]['exp_avg_sq'] = self.lora_A.weight.T @ (A0 @ exp_avg_sq)
            
            self.lora_B.weight.zero_()
            self.lora_B.requires_grad_(True)
            self.lora_B.weight.grad = torch.zeros_like(self.lora_B.weight)
            self.lora_B.weight.grad.copy_(-self.scale * torch.mm(self.lora_A.weight.T, WG))

    def forward(self, x: torch.Tensor):
        if self.relora_config.forward_type == True:
            if not self.weight.requires_grad:
                self.weight.requires_grad = True
                if self.proj_type: self.lora_A.requires_grad_(False)
                if not self.proj_type: self.lora_B.requires_grad_(False)

        if self.lora_only:
            # just lora
            return self.lora_A(self.lora_B(self.lora_dropout(x))) * self.scale

        result = F.linear(x, self.weight, bias=self.bias)

        if self.r > 0:
            # result -= F.linear(x, self.lora_A.weight @ self.lora_B.weight)
            result -= self.lora_A(self.lora_B(self.lora_dropout(x))) * self.scale
        return result