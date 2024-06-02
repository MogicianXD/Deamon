#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import typing
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List

import transformers
from torch import Tensor

from peft.hyper import lora_up
from peft.hyper.base import Adaptation


def add_lora_(parent_module: nn.Module, module_name: str, r: int, lora_alpha: int, freeze_weight: bool,
              merge_weights=True, load_weights=True, **kwargs):
    old_module = getattr(parent_module, module_name)
    if isinstance(old_module, nn.Linear):
        new_module = Linear(in_features=old_module.in_features, out_features=old_module.out_features,
                            bias=old_module.bias is not None, r=r, lora_alpha=lora_alpha,
                            device=old_module.weight.device, dtype=old_module.weight.dtype,
                            merge_weights=merge_weights, freeze_weight=freeze_weight, **kwargs)
    elif isinstance(old_module, transformers.Conv1D):
        new_module = MergedLinear(in_features=old_module.weight.shape[0], out_features=old_module.nf,
                                  r=r, lora_alpha=lora_alpha, enable_lora=[True, False, True],
                                  fan_in_fan_out=True,
                                  merge_weights=merge_weights, freeze_weight=freeze_weight, **kwargs)
    elif isinstance(old_module, nn.Conv1d):
        new_module = Conv1d_shift(old_module.in_channels, old_module.out_channels,
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


class LoRALayer(Adaptation):
    def __init__(self, in_features: int, out_features: int, r: int, lora_alpha: int = 1, flag_adapt_lora_A: bool = True,
                 flag_adapt_bias: bool = True, merge_weights: bool = True,
                 freeze_weight: bool = True, freeze_bias: bool = False, **kwargs):
        super().__init__(flag_adapt_bias=flag_adapt_bias, merge_weights=merge_weights,
                         freeze_weight=freeze_weight, freeze_bias=freeze_bias)
        self.in_dim = in_features
        self.out_dim = out_features
        self.r = r
        self.scaling = lora_alpha / self.r
        self.flag_adapt_lora_A = flag_adapt_lora_A
        self.merged = False
        self.lora_A = None
        self.lora_B = None
        self.shift = None

    def assign_adaptation(self, adaptation):
        A_size = self.in_dim * self.r if self.flag_adapt_lora_A else 0
        if self.flag_adapt_lora_A:
            self.lora_A = adaptation[..., :A_size].view(-1, self.r, self.in_dim)
        if self.flag_adapt_bias:
            self.lora_B = adaptation[..., A_size:-self.out_dim].view(-1, self.out_dim, self.r)
            self.shift = adaptation[..., -self.out_dim:]
        else:
            self.lora_B = adaptation[..., A_size:].view(-1, self.out_dim, self.r)

    def merge_AB(self):
        return (self.lora_B @ self.lora_A).squeeze(0) * self.scaling


class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            device=None, dtype=None,
            r: int = 0, lora_alpha: int = 1,
            flag_adapt_lora_A: bool = True,
            merge_weights: bool = True,
            freeze_weight: bool = True,
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, bias=bias, device=device, dtype=dtype)
        LoRALayer.__init__(self, in_features, out_features, r=r, lora_alpha=lora_alpha, flag_adapt_lora_A=flag_adapt_lora_A,
                           flag_adapt_bias=bias, merge_weights=merge_weights, freeze_weight=freeze_weight, **kwargs)
        if not flag_adapt_lora_A:
            self.lora_A = nn.Parameter(torch.empty(r, in_features))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor):
        if self.merged:
            return F.linear(x, self.weight, bias=self.bias)
        elif x.shape[0] == 1 and self.merge_weights:
            return F.linear(x, self.weight + self.merge_AB(),
                            bias=(self.bias + self.shift.squeeze()) if self.flag_adapt_bias else self.bias)
        else:
            delta_w = self.merge_AB()
            if x.dim() > 2:
                delta_w = delta_w.view(delta_w.shape[:-2] + (1, ) * (x.dim() - self.shift.dim()) + delta_w.shape[-2:])
            res = ((self.weight + delta_w) @ x.unsqueeze(-1)).squeeze(-1)
            if self.shift is None:
                return res
            else:
                shift = self.shift
                if x.dim() > 2:
                    shift = shift.view(self.shift.shape[:-1] + (1, ) * (x.dim() - self.shift.dim()) + self.shift.shape[-1:])
                return res + shift + self.bias


class MergedLinear(transformers.Conv1D, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            enable_lora: List[bool] = [False],
            fan_in_fan_out: bool = False,
            merge_weights: bool = True,
            **kwargs
    ):
        transformers.Conv1D.__init__(self, nx=in_features, nf=out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights, **kwargs)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        assert int(out_features / in_features) == 3
        self.enable_lora = enable_lora
        self.fan_in_fan_out = True
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            )  # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features,), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        # if fan_in_fan_out:
        #     self.weight.data = self.weight.data.transpose(0, 1)

    def zero_pad(self, x):
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))
        result[self.lora_ind] = x
        return result

    def merge_AB(self):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        delta_w = F.conv1d(
            self.lora_A.unsqueeze(0),
            self.lora_B.unsqueeze(-1),
            groups=sum(self.enable_lora)
        ).squeeze(0)
        return T(self.zero_pad(delta_w))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        if not self.merged and self.r > 0:
            result = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight + self.merge_AB() * self.scaling)
        else:
            result = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        result = result.view(size_out)
        return result


class Conv1d(nn.Conv1d, LoRALayer):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups: int = 1, bias: bool = True,
                 padding_mode: str = 'zeros', device=None, dtype=None, flag_adapt_lora_A: bool = True,
                 r=0, lora_alpha=1, merge_weights=True, freeze_weights=True, **kwargs):
        nn.Conv1d.__init__(self, in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                           dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode,
                           device=device, dtype=dtype)
        LoRALayer.__init__(self, in_features=in_channels, out_features=out_channels, r=r, lora_alpha=lora_alpha,
                           flag_adapt_lora_A=flag_adapt_lora_A,
                           flag_adapt_bias=bias, merge_weights=merge_weights, freeze_weight=freeze_weights, **kwargs)
        self._kernel_size = self.kernel_size[0]
        if not flag_adapt_lora_A:
            self.lora_A = nn.Parameter(torch.empty(kernel_size[0], r, in_channels))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def assign_adaptation(self, adaptation):
        A_size = self.in_dim * self.r * self._kernel_size if self.flag_adapt_lora_A else 0
        if self.flag_adapt_lora_A:
            self.lora_A = adaptation[..., :A_size].view(-1, self._kernel_size, self.r, self.in_dim)
        if self.flag_adapt_bias:
            self.lora_B = adaptation[..., A_size:-self.out_dim].view(-1, self._kernel_size, self.out_dim, self.r)
            self.shift = adaptation[..., -self.out_dim:]
        else:
            self.lora_B = adaptation[..., A_size:].view(-1, self._kernel_size, self.out_dim, self.r)

    def merge_AB(self):
        w = (self.lora_B @ self.lora_A) * self.scaling  # [bs, k, o, i]
        return w.permute(0, 2, 3, 1).reshape(-1, self.in_channels, self._kernel_size)

    def forward(self, x: torch.Tensor):
        if self.merged:
            return super()._conv_forward(x, self.weight, bias=self.bias)
        elif x.shape[0] == 1 and self.merge_weights:
            return super()._conv_forward(x, self.weight + self.merge_AB(),
                                         bias=(self.bias + self.shift.squeeze()) if self.flag_adapt_bias else self.bias)
        else:
            delta_w = self.merge_AB()
            shift = self.shift.unsqueeze(-1)
            batch_shape = x.shape[:-2]
            if len(batch_shape) > 1:
                x = x.reshape(len(x), -1, *x.shape[-2:]).transpose(0, 1).reshape(-1, len(x) * self.in_channels, x.shape[-1])
            else:
                x = x.reshape(1, len(x) * self.in_channels, x.shape[-1])
                shift = shift.view(len(shift), *((1,) * (len(batch_shape) - 1)), -1, 1)
            weight = self.weight.unsqueeze(0).expand(batch_shape[0], -1, -1, -1).reshape(-1, *self.weight.shape[-2:]) + delta_w
            res = self._conv_forward(x, weight, None, groups=batch_shape[0] * self.groups)
            res = res.view(len(res), batch_shape[0], self.out_dim, -1)
            if len(batch_shape) > 1:
                res = res.transpose(0, 1).reshape(*batch_shape, self.out_dim, res.shape[-1])
            else:
                res = res.squeeze(0)
            if self.bias is None:
                return res
            else:
                return res + (self.bias.unsqueeze(-1) + shift)


    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor], groups: int):
        if self.padding_mode != 'zeros':
            return F.conv1d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            nn.Conv1d._single(0), self.dilation, groups)
        return F.conv1d(input, weight, bias, self.stride,
                        self.padding, self.dilation, groups)


class Conv1d_shift(nn.Conv1d, LoRALayer):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups: int = 1, bias: bool = True,
                 padding_mode: str = 'zeros', device=None, dtype=None, flag_adapt_lora_A: bool = True,
                 r=0, lora_alpha=1, merge_weights=True, freeze_weights=True, **kwargs):
        nn.Conv1d.__init__(self, in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                           dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode,
                           device=device, dtype=dtype)
        LoRALayer.__init__(self, in_features=in_channels, out_features=out_channels, r=r, lora_alpha=lora_alpha,
                           flag_adapt_lora_A=flag_adapt_lora_A,
                           flag_adapt_bias=True, merge_weights=merge_weights, freeze_weight=freeze_weights, **kwargs)
        self._kernel_size = self.kernel_size[0]
        self.lora_A = nn.Parameter(torch.empty(self._kernel_size, r, in_channels))
        self.lora_B = nn.Parameter(torch.zeros(self._kernel_size, out_channels, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def assign_adaptation(self, adaptation):
        self.shift = adaptation

    def merge_AB(self):
        w = (self.lora_B @ self.lora_A) * self.scaling  # [k, o, i]
        return w.permute(1, 2, 0).reshape(-1, self.in_channels, self._kernel_size)

    def forward(self, x: torch.Tensor):
        if self.merged:
            return super()._conv_forward(x, self.weight, bias=self.bias)
        elif x.shape[0] == 1 and self.merge_weights:
            return super()._conv_forward(x, self.weight + self.merge_AB(),
                                         bias=(self.bias + self.shift.squeeze()) if self.flag_adapt_bias else self.bias)
        else:
            res = super()._conv_forward(x, self.weight + self.merge_AB(), bias=None)
            if self.bias is None:
                return res
            else:
                return res + (self.bias + self.shift).unsqueeze(-1)


class Conv1d_shared(nn.Conv1d, LoRALayer):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups: int = 1, bias: bool = True,
                 padding_mode: str = 'zeros', device=None, dtype=None, flag_adapt_lora_A: bool = True,
                 r=0, lora_alpha=1, merge_weights=True, freeze_weights=True, **kwargs):
        nn.Conv1d.__init__(self, in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                           dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode,
                           device=device, dtype=dtype)
        LoRALayer.__init__(self, in_features=in_channels, out_features=out_channels, r=r, lora_alpha=lora_alpha,
                           flag_adapt_lora_A=flag_adapt_lora_A,
                           flag_adapt_bias=bias, merge_weights=merge_weights, freeze_weight=freeze_weights, **kwargs)
        self._kernel_size = self.kernel_size[0]
        if not flag_adapt_lora_A:
            self.lora_A = nn.Parameter(torch.empty(self._kernel_size, r, in_channels))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def merge_AB(self):
        if self.flag_adapt_lora_A:
            w = (self.lora_B @ self.lora_A) * self.scaling  # [bs, o, i]
            return w.unsqueeze(-1).expand(-1, -1, -1, self._kernel_size).reshape(-1, self.in_channels, self._kernel_size)
        else:
            w = (self.lora_B.unsqueeze(-3) @ self.lora_A) * self.scaling  # [bs, k, o, i]
            return w.permute(0, 2, 3, 1).reshape(-1, self.in_channels, self._kernel_size)

    def forward(self, x: torch.Tensor):
        if self.merged:
            return super()._conv_forward(x, self.weight, bias=self.bias)
        elif x.shape[0] == 1 and self.merge_weights:
            return super()._conv_forward(x, self.weight + self.merge_AB(),
                                         bias=(self.bias + self.shift.squeeze()) if self.flag_adapt_bias else self.bias)
        else:
            delta_w = self.merge_AB()
            shift = self.shift.unsqueeze(-1)
            batch_shape = x.shape[:-2]
            if len(batch_shape) > 1:
                x = x.reshape(len(x), -1, *x.shape[-2:]).transpose(0, 1).reshape(-1, len(x) * self.in_channels, x.shape[-1])
            else:
                x = x.reshape(1, len(x) * self.in_channels, x.shape[-1])
                shift = shift.view(len(shift), *((1,) * (len(batch_shape) - 1)), -1, 1)
            weight = self.weight.unsqueeze(0).expand(batch_shape[0], -1, -1, -1).reshape(-1, *self.weight.shape[-2:]) + delta_w
            res = self._conv_forward(x, weight, None, groups=batch_shape[0] * self.groups)
            res = res.view(len(res), batch_shape[0], self.out_dim, -1)
            if len(batch_shape) > 1:
                res = res.transpose(0, 1).reshape(*batch_shape, self.out_dim, res.shape[-1])
            else:
                res = res.squeeze(0)
            if self.bias is None:
                return res
            else:
                return res + (self.bias.unsqueeze(-1) + shift)


    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor], groups: int):
        if self.padding_mode != 'zeros':
            return F.conv1d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            nn.Conv1d._single(0), self.dilation, groups)
        return F.conv1d(input, weight, bias, self.stride,
                        self.padding, self.dilation, groups)