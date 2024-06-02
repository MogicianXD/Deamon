#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import collections
import typing
from types import MethodType

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.autograd import Variable

from peft.hyper.factory import Adapters
from peft.hyper import ssf, lora_up, drift_lora
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP, GPT2Attention


def normalize(W, max_norm=1):
    W_norm = torch.norm(W, dim=-1, keepdim=True)
    scale = torch.clip(max_norm / W_norm, max=1)
    return W * scale

class ConceptEncoder(nn.Module):
    def __init__(self, dim_x, new_dim_x, dim_y, new_dim_y, dim_other=0, new_dim_other=48, final_dim=None):
        super().__init__()
        self.dim_x, self.dim_y = dim_x, dim_y
        self.dim_other = dim_other
        self.linear_X = nn.Linear(dim_x, new_dim_x, bias=False)
        self.linear_Y = nn.Linear(dim_y, new_dim_y, bias=False)
        if dim_other > 60:
            self.linear_other = nn.Linear(dim_other, new_dim_other)
        self.out_dim = new_dim_y * new_dim_x + (new_dim_other if dim_other > 60 else dim_other)
        if final_dim is not None:
            self.final_enc = nn.Linear(self.out_dim, final_dim)
            self.out_dim = final_dim

    def forward(self, x):
        xy = x[..., :self.dim_x * self.dim_y].view(*x.shape[:-1], self.dim_y, self.dim_x)
        xy = self.linear_Y(self.linear_X(xy).transpose(-2, -1)).view(*x.shape[:-1], -1)
        if self.dim_other > 0:
            other = x[..., self.dim_x * self.dim_y:]
            if hasattr(self, 'linear_other'):
                other = self.linear_other(other)
            res = torch.cat([xy, other], -1)
        else:
            res = xy
        if hasattr(self, 'final_enc'):
            return self.final_enc(res)
        return res


class DriftTune(nn.Module):
    def __init__(self, backbone, args):
        super().__init__()
        self.args = args
        backbone.requires_grad_(not args.freeze)
        self.backbone = add_adapters_(backbone, args)
        self.concept_dim = args.pred_len * (args.seq_len + args.concept_bias)
        if args.use_mean:
            self.concept_dim += (args.seq_len if args.general_stat else args.enc_in)
        # self.concept_dim = args.pred_len + (args.seq_len + args.concept_bias) + (args.seq_len if args.general_stat else args.enc_in)

        # self.encoder_Y = nn.Sequential(nn.Linear(args.pred_len, args.new_y_dim))
        # self.encoder_X = nn.Sequential(nn.Linear(args.seq_len + args.concept_bias, args.new_x_dim))
        # self.concept_dim = args.new_y_dim * args.new_x_dim + 48
        dim_mean = args.seq_len if args.general_stat else args.enc_in
        if dim_mean > 60:
            dim_mean = args.new_x_dim
        self.concept_encoder = ConceptEncoder(args.seq_len + args.concept_bias, args.new_x_dim,
                                              args.pred_len, args.new_y_dim,
                                              (args.seq_len if args.general_stat else args.enc_in) if args.use_mean else 0,
                                              args.new_x_dim,
                                              args.concept_dim)
        self.concept_encoder2 = ConceptEncoder(args.seq_len + args.concept_bias, args.new_x_dim,
                                               args.pred_len, args.new_y_dim,
                                               (args.seq_len if args.general_stat else args.enc_in) if args.use_mean else 0,
                                               args.new_x_dim,
                                               args.concept_dim)
        self.concept_encoder2.load_state_dict(self.concept_encoder.state_dict())
        self.concept_encoder.linear_Y.bias = None
        if hasattr(self.concept_encoder, 'final_enc'):
            self.concept_encoder.final_enc.bias = None
        elif hasattr(self.concept_encoder, 'linear_other'):
            self.concept_encoder.linear_other.bias = None
        # if args.dataset == 'ETTh2':
        concept_dim = self.concept_encoder.out_dim
        bottleneck_condition = lambda out_dim: args.bottleneck_dim * (concept_dim + out_dim) < concept_dim * out_dim
        # else:
        #     bottleneck_condition = lambda out_dim: out_dim > args.bottleneck_dim * 2
        # if args.dataset == 'Traffic' and args.pred_len == 192 and 'TCN' in args.model:
        #     mid_dim = args.bottleneck_dim // 8
        # else:
        #     mid_dim = args.bottleneck_dim // 4
        self.prune = args.prune
        self.adapters = Adapters(backbone, concept_dim, hid_dims=None, activation=nn.Sigmoid, prune=self.prune,
                                 adaptive_dim=args.model != 'GPT4TS', diff_ensemble=args.diff_ensemble,
                                 r=args.lora_rank, store_recent=True, need_bias=args.more_bias, need_norm=False,
                                 flag_adapt_lora_A=args.adapt_lora_A, scaling=args.lora_alpha / (args.lora_rank + 1e-5),
                                 shared=args.shared_encoding, bottleneck_condition=bottleneck_condition,
                                 mid_dim=args.bottleneck_dim, temperature=args.temperature)
        # self.concept_encoder.weight.data[:, :self.concept_dim] = -self.concept_encoder.weight.data[:, self.concept_dim:]

        self.threshold = args.trigger_threshold
        self.concept_pool = None
        self.register_buffer('recent_concept', torch.zeros(1, self.concept_dim), persistent=True)
        # self.register_buffer('drift', torch.zeros(1, self.concept_dim), persistent=True)
        self.gamma = 0
        self.ema = args.ema
        self.max_norm = 1
        self.reuse_num = args.reuse
        self.flag_reuse = False
        self.flag_online_learning = False
        self.flag_update = False
        self.register_buffer('recent_c2', None, persistent=True)

    def generate_adaptation(self, concept_pred, resume=False, save=False, update_adaptation=True, update_lora_cache=True):
        # drift = self.w1 * drift + self.w2 * self.recent_concept
        # drift_YX = self.encoder_Y(drift_YX.transpose(-2, -1)).view(len(drift), -1)
        # drift_emb = F.normalize(drift_emb, dim=-1)
        if self.ema > 0:
            if self.recent_c2 is not None:
                recent_concept = self.recent_c2 * self.ema + self.recent_concept * (1 - self.ema)
            else:
                recent_concept = self.recent_concept
            if self.flag_update or self.flag_online_learning:
                self.recent_c2 = recent_concept.detach()
        else:
            recent_concept = self.recent_concept
        recent_concept = self.concept_encoder2(recent_concept)
        drift = self.concept_encoder(concept_pred) - recent_concept
        # drift = normalize(drift, self.max_norm)
        if self.flag_reuse and not self.flag_online_learning:
            sim_K, indices, is_new_concept = self.eval_concept(self.recent_concept)
        else:
            sim_K, indices, is_new_concept = None, None, False
        res = self.adapters(drift, sim_K=sim_K, indices=indices,
                            ema=0, resume=resume, save=save,
                            update_lora_cache=update_lora_cache, prune=self.prune)
                                        # learning_rate=self.model_optim.param_groups[-1]['lr'])
        return res

    def forward(self, *x):
        concept_pred = x[-1].to(x[0].device)
        adaptations, loras = self.generate_adaptation(concept_pred, save=False, resume=False)
        for key, delta_w in loras.items():
            for i in range(len(delta_w)):
                name = self.adapters.dim_name_dict_lora[key][i]
                self.backbone.get_submodule(name).assign_lora(delta_w[i])
        for out_dim, adaptation in adaptations.items():
            for i in range(len(adaptation)):
                name = self.adapters.dim_name_dict[out_dim][i]
                self.backbone.get_submodule(name).assign_adaptation(adaptation[i])
        return self.backbone(*x[:-2])

    def freeze_encoder(self, freeze=True, include_prune=True):
        if self.args.freeze_delta:
            # self.ema.requires_grad = not freeze
            # if freeze:
            #     self.ema.grad = None
            for module_name in ['concept_encoder', 'concept_encoder2']:
                if hasattr(self, module_name):
                    getattr(self, module_name).requires_grad_(not freeze)
                    getattr(self, module_name).zero_grad(set_to_none=True)
            if include_prune and self.args.prune:
                self.adapters.classifier.requires_grad_(not freeze)
                self.adapters.classifier.zero_grad(set_to_none=True)
            for adapter in self.adapters.adapters.values():
                if self.args.freeze_w2:
                    adapter.weights.requires_grad_(not freeze)
                    adapter.weights.zero_grad(set_to_none=True)
                else:
                    adapter.weights[:-1].requires_grad_(not freeze)
                    adapter.weights[:-1].zero_grad(set_to_none=True)
                # if self.args.more_bias:
                # if adapter.more_bias:
                adapter.biases[:len(adapter.weights) - 1].requires_grad_(not freeze)
                adapter.biases[:len(adapter.weights) - 1].zero_grad(set_to_none=True)
                # else:
                #     if len(adapter.biases) > 0:
                #         adapter.biases.requires_grad_(not freeze)
                #         adapter.biases.zero_grad(set_to_none=True)


def add_adapters_(parent_module: nn.Module, args, top_level=True):
    for name, module in parent_module.named_children():
        # if isinstance(module, nn.Linear) and 'regressor' in name or isinstance(module, nn.Conv1d):
        # if isinstance(module, nn.Conv1d):
        #     peft.lora.add_lora_(parent_module, name, r=args.lora_rank, lora_alpha=args.lora_alpha,
        #                    merge_weights=args.merge_weights,)
        # if isinstance(module, nn.Linear):
        # if isinstance(module, nn.Conv1d) and 'PadConv' in type(parent_module).__name__ or isinstance(module, nn.Linear):
        # if isinstance(module, nn.Conv1d):
        #     if args.tune_mode == 'scale_shift':
        #         ssf.add_ssf_(parent_module, name, freeze_weight=args.freeze, merge_weights=args.merge_weights)
        #     else:
        #         lora_up.add_lora_up_(parent_module, name, freeze_weight=args.freeze, merge_weights=args.merge_weights)
        # el
        if args.tune_mode == 'ssf' and isinstance(module, (nn.Conv1d, nn.Linear, transformers.Conv1D)):
            ssf.add_ssf_(parent_module, name, freeze_weight=args.freeze, merge_weights=args.merge_weights,)
        elif args.tune_mode == 'ssfln' and isinstance(module, (nn.Conv1d, nn.Linear, transformers.Conv1D, nn.LayerNorm)):
            ssf.add_ssf_(parent_module, name, freeze_weight=args.freeze, merge_weights=args.merge_weights,)
        elif 'adapter' in args.tune_mode and isinstance(module, (nn.Conv1d, nn.Linear, transformers.Conv1D, nn.LayerNorm))\
            and isinstance(parent_module, (GPT2MLP, GPT2Attention)):
            lora_up.add_lora_up_(parent_module, name, freeze_weight=args.freeze, merge_weights=args.merge_weights,)
        elif 'lora' in args.tune_mode and args.model == 'GPT4TS' and isinstance(module, transformers.Conv1D) and 'c_attn' in name:
            drift_lora.add_lora_(parent_module, name, r=args.lora_rank, lora_alpha=args.lora_alpha,
                                 freeze_weight=args.freeze, freeze_bias=args.more_bias,
                                 merge_weights=args.merge_weights, flag_adapt_lora_A=args.adapt_lora_A)
        elif args.tune_mode == 'up' and isinstance(module, (nn.Conv1d, nn.Linear, transformers.Conv1D)):
            lora_up.add_lora_up_(parent_module, name, freeze_weight=args.freeze, merge_weights=args.merge_weights,)
        elif args.tune_mode == 'mix' and isinstance(module, nn.Conv1d):
                lora_up.add_lora_up_(parent_module, name, freeze_weight=args.freeze, merge_weights=args.merge_weights,)
        elif args.tune_mode == 'only_up' and isinstance(module, (nn.Conv1d, nn.Linear, transformers.Conv1D, nn.LayerNorm)):
            lora_up.add_lora_up_(parent_module, name, freeze_weight=args.freeze, merge_weights=args.merge_weights,)
        elif args.tune_mode == 'ln' and isinstance(module, (nn.Linear, nn.LayerNorm)):
            lora_up.add_lora_up_(parent_module, name, freeze_weight=args.freeze, merge_weights=args.merge_weights,)
        # elif args.model == 'TCN' and isinstance(module, nn.Conv1d):
        #     drift_lora.add_lora_(parent_module, name, r=args.lora_rank, lora_alpha=args.lora_alpha,
        #                    freeze_weight=args.freeze, merge_weights=args.merge_weights,
        #                    flag_adapt_lora_A=args.adapt_lora_A)
        # elif args.model == 'TCN' and isinstance(module, (nn.Conv1d, nn.Linear)):
        #     peft.lora.add_lora_(parent_module, name, r=args.lora_rank, lora_alpha=args.lora_alpha,
        #                         freeze_weight=args.freeze, merge_weights=args.merge_weights, )
        elif args.tune_mode == 'lora_up' and (args.model != 'GPT4TS' and (isinstance(module, nn.Linear) or
             isinstance(module, nn.Conv1d)) or isinstance(module, transformers.Conv1D) and 'c_attn' in name):
            drift_lora.add_lora_(parent_module, name, r=args.lora_rank, lora_alpha=args.lora_alpha,
                                 freeze_weight=args.freeze, freeze_bias=args.more_bias,
                                 merge_weights=args.merge_weights, flag_adapt_lora_A=args.adapt_lora_A)
        else:
            add_adapters_(module, args, False)
    return parent_module
