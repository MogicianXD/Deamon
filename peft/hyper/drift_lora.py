#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import peft

def add_lora_(parent_module: nn.Module, module_name: str, r: int, lora_alpha: int, freeze_weight: bool,
              merge_weights=True, load_weights=True, **kwargs):
    old_module = getattr(parent_module, module_name)
    if isinstance(old_module, nn.Linear):
        new_module = Linear(in_features=old_module.in_features, out_features=old_module.out_features,
                            bias=old_module.bias is not None, r=r, lora_alpha=lora_alpha,
                            device=old_module.weight.device, dtype=old_module.weight.dtype,
                            merge_weights=merge_weights, freeze_weight=freeze_weight, **kwargs)
    elif isinstance(old_module, transformers.Conv1D) and 'c_attn' in module_name:
        new_module = AttnConv1D(nx=old_module.weight.shape[0], nf=old_module.nf,
                                bias=old_module.bias is not None, r=r, lora_alpha=lora_alpha,
                                merge_weights=merge_weights, freeze_weight=freeze_weight, **kwargs)
    elif isinstance(old_module, nn.Conv1d):
        new_module = Conv1d(old_module.in_channels, old_module.out_channels,
                            kernel_size=old_module.kernel_size, stride=old_module.stride, padding=old_module.padding,
                            dilation=old_module.dilation, groups=old_module.groups, bias=old_module.bias is not None,
                            padding_mode=old_module.padding_mode,
                            device=old_module.weight.device, dtype=old_module.weight.dtype,
                            r=r, lora_alpha=lora_alpha,
                            merge_weights=merge_weights, freeze_weight=freeze_weight, **kwargs)
    else:
        raise NotImplementedError

    if load_weights:
        new_module.load_state_dict(old_module.state_dict(), strict=False)
    setattr(parent_module, module_name, new_module)


class LoRA_external:
    def __init__(self, r: int = 0):
        self.r = r
        self.merged = False

    def assign_lora(self, delta_w):
        self.merge_AB = delta_w


class Linear(peft.hyper.lora_up.Linear, LoRA_external):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            merge_weights: bool = True, freeze_weight: bool = True,
            r: int = 0,
            **kwargs
    ):
        LoRA_external.__init__(self, r=r)
        peft.hyper.lora_up.Linear.__init__(self, in_features=in_features, out_features=out_features, bias=bias,
                                           merge_weights=merge_weights, freeze_weight=freeze_weight, **kwargs)

    def forward(self, x: torch.Tensor):
        if self.scale is None:
            return F.linear(x, self.weight + self.merge_AB, bias=self.bias)
        weight, bias = self.weight, self.bias
        if self.scale.shape[0] == 1:
            weight, bias = self._merge(weight, bias)
            return F.linear(x, weight + self.merge_AB, bias=bias)
        else:
            res = F.linear(x, weight, bias=None)
            return self._ssf(res) + F.linear(x, self.merge_AB, bias=None)
        # weight, bias = self.weight, self.bias
        # if self.r > 0 and not self.merged:
        #     weight = weight + self.merge_AB
        # if self.scale.shape[0] == 1 and self.merge_weights:
        #     weight, bias = self._merge(weight, bias)
        #     return F.linear(x, weight, bias=bias)
        # else:
        #     return self._ssf(F.linear(x, weight, bias=None))

class AttnConv1D(peft.hyper.lora_up.TFMConv1D, LoRA_external):
    def __init__(
            self,
            nx: int,
            nf: int,
            r: int,
            merge_weights: bool = True, freeze_weight: bool = True,
            **kwargs
    ):
        LoRA_external.__init__(self, r=r)
        transformers.Conv1D.__init__(self, nx=nx, nf=nf)
        peft.hyper.lora_up.LoRA_Up.__init__(self, out_features=nf, flag_adapt_bias=True, fan_in_fan_out=True,
                            merge_weights=merge_weights, freeze_weight=freeze_weight, **kwargs)
        # self.register_buffer('zeros', torch.zeros(1, 1, self.nf // 3))
        # self.register_buffer('ones', torch.zeros(1, 1, self.nf // 3))

    # def assign_adaptation(self, adaptation):
    #     if adaptation is None:
    #         self.scale, self.shift = None, None
    #     else:
    #         out_features = adaptation.shape[-1] // 2
    #         self.scale = adaptation[..., :out_features] + 1
    #         self.scale = self.scale.view(self.scale.shape[0], 1, self.scale.shape[1])
    #         dim = out_features // 2
    #         if self.flag_adapt_bias:
    #             self.shift = adaptation[..., -out_features:]
    #             self.shift = self.shift.view_as(self.scale)
    #             zeros = self.zeros.expand(len(self.scale), -1, -1) if self.scale.shape[0] > 1 else self.zeros
    #             self.shift = torch.cat([self.shift[..., :dim], zeros, self.shift[..., dim:]], -1)
    #         ones = self.ones.expand(len(self.scale), -1, -1) if self.scale.shape[0] > 1 else self.ones
    #         self.scale = torch.cat([self.scale[..., :dim], ones, self.scale[..., dim:]], -1)

    def forward(self, x: torch.Tensor):
        weight, bias = self.weight, self.bias
        if self.scale.shape[0] == 1:
            weight, bias = self._merge(weight, bias)
            res = F.linear(x, (weight + self.merge_AB).transpose(-1, -2), bias=bias)
        else:
            res = x @ weight
            res = self._ssf(res) + x @ self.merge_AB
        return res

class Conv1d(peft.hyper.lora_up.Conv1d, LoRA_external):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 r=0, merge_weights: bool = True, freeze_weight: bool = True, **kwargs):
        LoRA_external.__init__(self, r=r)
        peft.hyper.lora_up.Conv1d.__init__(self, in_channels, out_channels, kernel_size,
                                           merge_weights=merge_weights, freeze_weight=freeze_weight, **kwargs)

    def assign_lora(self, delta_w):
        self.merge_AB = delta_w.view(self.weight.shape)

    def forward(self, x):
        if self.scale is None:
            return self._conv_forward(x, self.weight + self.merge_AB, bias=self.bias)
        weight, bias = self.weight, self.bias
        if self.scale.shape[0] == 1:
            weight, bias = self._merge(weight, bias)
            return self._conv_forward(x, weight + self.merge_AB, bias=bias)
        else:
            res = self._conv_forward(x, weight, bias=None)
            return self._ssf(res) + self._conv_forward(x, self.merge_AB, bias=None)
        # weight, bias = self.weight, self.bias
        # if self.r > 0 and not self.merged:
        #     weight = weight + self.merge_AB
        # if self.scale.shape[0] == 1 and self.merge_weights:
        #     weight, bias = self._merge(weight, bias)
        #     return self._conv_forward(x, weight, bias=bias)
        # else:
        #     return self._ssf(self._conv_forward(x, weight, bias=None))