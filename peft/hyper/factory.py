import collections
import typing

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from peft.hyper import lora, drift_lora
from peft.hyper.base import Adaptation, Adapter, LoRA


def normalize(W, max_norm=1):
    W_norm = torch.norm(W, dim=-1, keepdim=True)
    scale = torch.clip(max_norm / W_norm, max=1)
    return W * scale


def _gumbel_sigmoid(
    logits, tau=1, hard=False, eps=1e-10, training = True, threshold = 0.5
):
    if training :
        # ~Gumbel(0,1)`
        gumbels1 = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        gumbels2 = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        # Difference of two` gumbels because we apply a sigmoid
        gumbels1 = (logits + gumbels1 - gumbels2) / tau
        y_soft = gumbels1.sigmoid()
    else :
        y_soft = logits.sigmoid()

    if hard:
        # Straight through.
        y_hard = y_soft > threshold
        # y_hard = torch.zeros_like(
        #     logits, memory_format=torch.legacy_contiguous_format
        # ).masked_fill(y_soft > threshold, 1.0)
        if training:
            ret = y_hard.long() - y_soft.detach() + y_soft
        else:
            ret = y_hard
    else:
        if training:
            ret = y_soft
        else:
            ret = y_soft > threshold
    return ret


class Adapters(nn.Module):
    def __init__(self, backbone: nn.Module, concept_features: int,
                 hid_dims: typing.Union[typing.List[int], int], activation=nn.LeakyReLU,
                 shared: bool = False, need_bias: bool = True, adaptive_dim: bool = False, diff_ensemble: bool = False,
                 r: int = 1, store_recent: bool = True, flag_adapt_lora_A: bool = True, prune: bool = False,
                 scaling: int = 1, need_norm: bool = False, bottleneck_condition=None, mid_dim: int = None, temperature=1.0):
        super().__init__()
        self.dim_name_dict = collections.defaultdict(list)
        self.dim_name_dict_lora = collections.defaultdict(list)
        self.adapters = nn.ModuleDict()
        self.loras = nn.ModuleDict()
        self.memories = None
        self.memories_lora = None
        pos_rand_init_size = collections.defaultdict(dict)
        for name, module in backbone.named_modules():
            if isinstance(module, Adaptation):
                out_features = module.weight.shape[1 if isinstance(module, transformers.Conv1D) else 0]
                if module.weight.dim() == 1:
                    in_features = 1
                else:
                    in_features = module.weight.shape[0 if isinstance(module, transformers.Conv1D) else 1]
                if isinstance(module, drift_lora.LoRA_external) and r != 0:
                    if isinstance(module, drift_lora.Conv1d):
                        key = "_".join(str(dim) for dim in [in_features * module.kernel_size[0],
                                                            r * module.kernel_size[0],
                                                            out_features // module.groups, 'default'])
                        # key = "_".join(str(dim) for dim in [in_features,
                        #                                     r,
                        #                                     out_features // module.groups, f'conv@{module.kernel_size[0]}'])
                    elif isinstance(module, drift_lora.AttnConv1D):
                        key = "_".join(str(dim) for dim in [in_features, r, out_features // 3, 'attn'])
                        # out_features = out_features // 3 * 2
                    else:
                        key = "_".join(str(dim) for dim in [in_features, r, out_features, 'default'])
                    self.dim_name_dict_lora[key].append(name)
                # if isinstance(module, lora.LoRALayer):
                    # if flag_adapt_lora_A:
                    #     out_dim = (module.out_dim + in_features) * r
                    # else:
                    #     out_dim = module.out_dim * r
                    # if isinstance(module, nn.Conv1d) and not isinstance(module, lora.Conv1d_shared):
                    #     out_dim *= module.kernel_size[0]
                out_dim = out_features
                if module.bias is not None:
                    out_dim += out_features
                if shared:
                    if diff_ensemble and 'encoder_time' in name:
                        self.dim_name_dict[name.split('.')[-1] + '_time_' + str(out_dim)].append(name)
                    else:
                        self.dim_name_dict[name.split('.')[-1] + '_' + str(out_dim)].append(name)
                else:
                    self.dim_name_dict[str(out_dim)].append(name)
                # if isinstance(module, lora.LoRALayer) and flag_adapt_lora_A:
                #     pos_rand_init_size[out_dim][len(self.dim_name_dict[out_dim]) - 1] = (r, in_features)

        for key, names in self.dim_name_dict.items():
            # rand_init = pos_rand_init_size.get(out_dim)
            out_dim = int(key.split('_')[-1])
            # _hid_dims = min(mid_dim, max(concept_features, out_dim) // 4) if adaptive_dim else mid_dim
            _hid_dims = min(mid_dim, out_dim // 4) if adaptive_dim else mid_dim

            # if _hid_dims * (concept_features + out_dim) >= concept_features * out_dim:
            #     _hid_dims = None
            # if hid_dims is None and bottleneck_condition is not None and bottleneck_condition(out_dim):
            #     _hid_dims = mid_dim
            # else:
            #     _hid_dims = None
            self.adapters[key] = Adapter(concept_features, out_dim, len(names), _hid_dims,
                                                  activation,
                                                  store_recent=store_recent, shared=shared,
                                                  need_norm=need_norm and _hid_dims is not None, need_bias=need_bias)
        for key, names in self.dim_name_dict_lora.items():
            in_dim, r, out_dim, mode = [int(s) if i <= 2 else s for i, s in enumerate(key.split('_'))]
            self.loras[key] = LoRA(in_dim, out_dim, len(names), lora_rank=r if r >= 0 else (min(in_dim, out_dim) // -r),
                                   scaling=scaling, mode=mode)

        self.prune = prune
        if prune:
            # self.classifier = nn.Sequential(nn.Linear(concept_features, sum(len(names) for names in self.dim_name_dict.values())))
            # self.classifier = nn.Linear(concept_features, len(self.dim_name_dict))
            # self.classifier = nn.Sequential(nn.Linear(concept_features, mid_dim // 2),
            #                                 nn.Sigmoid(),
            #                                 nn.Linear(mid_dim // 2, len(self.dim_name_dict)))
            self.classifier = nn.Sequential(nn.Linear(concept_features, mid_dim),
                                            nn.Sigmoid(),
                                            # nn.Linear(mid_dim, sum(len(names) for names in self.dim_name_dict.values())))
                                            nn.Linear(mid_dim, len(self.dim_name_dict.values())))
            self.classifier[-1].bias.data += 5
            self.register_buffer('temperature', torch.tensor(temperature), persistent=True)

    def forward(self, x, ema=0, requires_grad=False, sim_K=None, indices=None, resume=False, save=False, update_lora_cache=True, prune=False):
        deltas = {key: net(requires_grad, sim_K, indices, resume) for key, net in self.loras.items()} if update_lora_cache else {}
        adapters = {}
        if prune:
            is_training = self.classifier[-1].weight.requires_grad
            prune_flag = _gumbel_sigmoid(self.classifier(x), hard=True, tau=self.temperature, training=is_training)
            for i, (k, adapter) in enumerate(self.adapters.items()):
                # mask = prune_flag[..., i: i + len(self.dim_name_dict[k])]
                mask = prune_flag[..., [i]]
                if not is_training and mask.sum() == 0:
                    adapters[k] = adapter.biases[-1] if adapter.need_bias else [None] * len(adapter.weights[-1])
                else:
                    adapters[k] = adapter(x, ema, requires_grad, sim_K, indices, resume, save,
                                          mask=mask.transpose(0, 1), training=is_training)
        else:
            adapters = {k: adapter(x, ema, requires_grad, sim_K, indices, resume, save)
                for k, adapter in self.adapters.items()}
        return adapters, deltas

    def memorize(self):
        for adapter in self.adapters.values():
            adapter.memorize()
        for lora in self.loras.values():
            lora.memorize()

