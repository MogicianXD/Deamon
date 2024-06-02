import collections
import typing

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Adaptation(object):
    def __init__(self, flag_adapt_bias: bool, flag_adapt_weight: bool = True,
                 merge_weights: bool = True, freeze_weight: bool = True, freeze_bias: bool = True):
        assert isinstance(self, nn.Module)
        self.flag_adapt_bias = flag_adapt_bias
        self.flag_adapt_weight = flag_adapt_weight
        self.weight.requires_grad = not freeze_weight
        if self.bias is not None:
            self.bias.requires_grad = not freeze_bias
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

    def assign_adaptation(self, adaptation):
        raise NotImplementedError


def normalize(W, max_norm=1, dim=-1):
    W_norm = torch.norm(W, dim=dim, keepdim=True)
    scale = torch.clip(max_norm / W_norm, max=1)
    return W * scale


class Adapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_layers: int,
                 hid_dims: typing.Union[typing.List[int], int], activation=nn.LeakyReLU, need_bias: bool = False,
                 store_recent: bool = True, shared=False, rand_init: dict = None, need_norm: bool = False):
        super().__init__()
        self.out_dim = out_dim
        if hid_dims is None:
            hid_dims = []
        elif isinstance(hid_dims, int):
            hid_dims = [hid_dims]
        hid_dims = [in_dim] + hid_dims + [self.out_dim]
        self.activation = activation()
        self.shared = shared
        self.need_bias = need_bias
        self.need_norm = need_norm
        self.biases = nn.ParameterList([nn.Parameter(torch.empty(n_layers, 1, hid_dims[i])) for i in range(1, len(hid_dims) - 1)])
        if shared and len(hid_dims) > 2:
            self.weights = nn.ParameterList([nn.Linear(hid_dims[0], hid_dims[1], bias=False)])
            # self.weights.append(nn.Parameter(torch.empty(n_layers, 1, hid_dims[-1], hid_dims[-2])))
            # nn.init.zeros_(self.weights[-1])
        else:
            # self.biases = nn.ParameterList([nn.Parameter(torch.empty(n_layers, 1, hid_dims[i])) for i in range(1, len(hid_dims) - 1)])
            self.weights = nn.ParameterList([nn.Parameter(torch.empty(n_layers, 1, hid_dims[i], hid_dims[i-1])) for i in range(1, len(hid_dims)-1)])
        if self.need_bias:
            self.biases.append(nn.Parameter(torch.zeros(n_layers, 1, hid_dims[-1])))
        # if len(hid_dims) > 2:
        #     self.weights.append(nn.Parameter(torch.empty(1, hid_dims[-1], hid_dims[-2])))
        # else:
        self.weights.append(nn.Parameter(torch.empty(n_layers, 1, hid_dims[-1], hid_dims[-2])))
        nn.init.zeros_(self.weights[-1])
        for i in range(0, len(self.weights) - 1):
            for j in range(n_layers):
                if not shared and j < len(self.weights[i]):
                    nn.init.kaiming_uniform_(self.weights[i][j, 0], a=math.sqrt(5))
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights[i].weight if shared else self.weights[i][j, 0])
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.biases[i][j], -bound, bound)
        if self.need_bias:
            nn.init.zeros_(self.biases[-1])
            if rand_init is not None:
                for pos, (r, in_fea) in rand_init.items():
                    # template = torch.empty(r, in_fea)
                    # nn.init.kaiming_uniform_(template, a=math.sqrt(5))
                    # self.biases[-1][pos, :, :r * in_fea].data = template.view_as(self.biases[-1][pos, :, :r * in_fea]).to(self.biases[-1].device)
                    nn.init.kaiming_uniform_(self.biases[-1][pos, :, :r * in_fea], a=math.sqrt(5))

        if store_recent:
            self.previous_adaptation = nn.Parameter(torch.zeros(n_layers, 1, self.out_dim))
        self.memories = None
        self.last_adaptation = None

    def forward(self, x, ema=0, requires_grad=False, sim_K=None, indices=None, resume=False, save=False, mask=None, training=False):
        if self.shared:
            for i in range(len(self.weights) - 1):
                x = self.activation(self.weights[i](x) + self.biases[i])
            # x = self.weights[-1](x)
            x = x.unsqueeze(-1)
        else:
            x = x.unsqueeze(-1)
            for i in range(len(self.weights) - 1):
                x = self.activation(self.weights[i] @ x + self.biases[i].unsqueeze(-1))
        if self.need_norm:
            x = normalize(x, dim=-2)
        # if not training and mask is not None and (mask == 0).any:
        #     x = self.weights[-1][mask] @ x
        # else:
        x = self.weights[-1] @ x
        x = x.squeeze(-1)
        # if not training and mask is not None and (mask == 0).any:
        #     if self.need_bias:
        #         x = x + self.biases[-1][mask]
        #     return [x[i] if mask[i] == 1 else self.biases[-1][i] for i in range(len(mask))]
        if training and mask is not None:
            x = x * mask.unsqueeze(-1)
        # x = x.view(2, x.shape[0] // 2, *x.shape[1:]).permute(1, 2, 3, 0).reshape(x.shape[0] // 2, *x.shape[1:-1], -1)
        if self.need_bias:
            x = x + self.biases[-1]
        return x

    def _refine(self, x, ema=0, requires_grad=False, sim_K=None, indices=None, resume=False):

        if indices is not None:
            if sim_K.shape[-1] == indices.shape[-1]:
                # adaptations = torch.cat([self.memories[k][indices], getattr(self, k)], 0)
                sim_K = sim_K.view((-1,) + (1,) * (x.dim() - 1))
                x = (self.memories[indices] * sim_K).sum(0, keepdims=True)
            elif sim_K.shape[-1] == indices.shape[-1] + 1:
                sim_K, self_portion = sim_K[:-1], sim_K[-1]
                # adaptations = torch.cat([self.memories[k][indices], getattr(self, k)], 0)
                sim_K = sim_K.view((-1,) + (1,) * (x.dim() - 1))
                x = (self.memories[indices] * sim_K).sum(0) + x * self_portion
            elif sim_K.shape[-1] == indices.shape[-1] + 1:
                sim_K, last_portion, self_portion = sim_K[:-2], sim_K[-2], sim_K[-1]
                # adaptations = torch.cat([self.memories[k][indices], getattr(self, k)], 0)
                sim_K = sim_K.view((-1,) + (1,) * (x.dim() - 1))
                x = (self.memories[indices] * sim_K).sum(0) + self.last_adaptation * last_portion + x * self_portion

        # if requires_grad:
        #     self.adaptation = x
        #     self.adaptation.retain_grad()

        # if scaling is not None:
        #     for k in self.adaptation_names:
        #         # adaptations = torch.cat([self.memories[k][indices], getattr(self, k)], 0)
        #         adaptations = self.memories[k] * scaling + getattr(self, k) * (1 - scaling)
        #         setattr(self, k, adaptations)
        return x

    def update_ema(self, x, ema):
        self.previous_adaptation.data = self.previous_adaptation.data * ema + x.mean(-2, keepdim=True).detach() * (1 - ema)

    def memorize(self):
        if self.memories is None:
            self.memories = self.last_adaptation.unsqueeze(0)
        else:
            self.memories = torch.cat([self.memories, self.last_adaptation.unsqueeze(0)], 0)

class LoRA(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_layers: int, lora_rank: int = 1, scaling: int = 1, mode='default'):
        super().__init__()
        self.out_dim = out_dim
        self.scaling = scaling
        if mode == 'attn':
            self.lora_A = nn.Parameter(torch.empty(n_layers, 2 * lora_rank, in_dim))
            self.lora_B = nn.Parameter(torch.zeros(n_layers, 2, out_dim, lora_rank))
        elif mode[:4] == 'conv' and int(mode[5:]) > 0:
            kernel_size = int(mode[5:])
            self.lora_A = nn.Parameter(torch.empty(n_layers, kernel_size, lora_rank, in_dim))
            self.lora_B = nn.Parameter(torch.zeros(n_layers, kernel_size, out_dim, lora_rank))
        else:
            self.lora_A = nn.Parameter(torch.empty(n_layers, lora_rank, in_dim))
            self.lora_B = nn.Parameter(torch.zeros(n_layers, out_dim, lora_rank))
        for i in range(n_layers):
            nn.init.kaiming_uniform_(self.lora_A[i], a=math.sqrt(5))
        if mode == 'attn':
            self.lora_A.data = self.lora_A.reshape(n_layers, 2, lora_rank, in_dim)
        self.memories_A = None
        self.memories_B = None
        self.last_A = None
        self.last_B = None
        self.mode = mode
        if mode == 'attn':
            self.register_buffer('pads', torch.zeros(n_layers, out_dim, in_dim))

    def forward(self, requires_grad=False, sim_K=None, indices=None, resume=False):
        if self.mode == 'conv' and self.lora_A.dim() == 4:
            delta_weights = (self.lora_B @ self.lora_A).permute(0, 2, 3, 1) * self.scaling
        else:
            delta_weights = (self.lora_B @ self.lora_A) * self.scaling
            if self.mode == 'attn':
                delta_weights = torch.cat([delta_weights[:, 0], self.pads, delta_weights[:, 1]], 1).transpose(-2, -1)
        delta_weights = self._refine(delta_weights, requires_grad, sim_K, indices, resume)
        return delta_weights

    def _refine(self, x, requires_grad=False, sim_K=None, indices=None, resume=False):
        if indices is not None:
            if sim_K.shape[-1] == indices.shape[-1]:
                # adaptations = torch.cat([self.memories[k][indices], getattr(self, k)], 0)
                sim_K = sim_K.view((-1,) + (1,) * (x.dim() - 1))
                past_A = (self.memories_A[indices] * sim_K).sum(0, keepdims=True)
                past_B = (self.memories_B[indices] * sim_K).sum(0, keepdims=True)
                x = past_B @ past_A
            elif sim_K.shape[-1] == indices.shape[-1] + 1:
                sim_K, self_portion = sim_K[:-1], sim_K[-1]
                # adaptations = torch.cat([self.memories[k][indices], getattr(self, k)], 0)
                sim_K = sim_K.view((-1,) + (1,) * (x.dim() - 1))
                past_x = ((self.memories_B[indices] @ self.memories_A[indices]) * sim_K).sum(0, keepdims=True)
                x = past_x + x * self_portion
            elif sim_K.shape[-1] == indices.shape[-1] + 2:
                sim_K, last_portion, self_portion = sim_K[:-1], sim_K[-2], sim_K[-1]
                # adaptations = torch.cat([self.memories[k][indices], getattr(self, k)], 0)
                sim_K = sim_K.view((-1,) + (1,) * (x.dim() - 1))
                past_x = ((self.memories_B[indices] @ self.memories_A[indices]) * sim_K).sum(0, keepdims=True)
                x = past_x + (self.last_B @ self.last_A) * last_portion + x * self_portion

        if requires_grad:
            self.adaptation = x
            self.adaptation.retain_grad()

        return x

    def memorize(self):
        if self.memories_A is None:
            self.memories_A = self.last_A.unsqueeze(0)
            self.memories_B = self.last_B.unsqueeze(0)
        else:
            self.memories_A = torch.cat([self.memories_A, self.last_A.unsqueeze(0)], 0)
            self.memories_B = torch.cat([self.memories_B, self.last_B.unsqueeze(0)], 0)
