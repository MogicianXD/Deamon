import copy
from typing import Optional, Any, Union, Callable

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import *
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
import numpy as np
from torch import nn
from torch.nn import BatchNorm2d, Conv2d, ModuleList


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_nodes = args.enc_in
        self.in_dim = args.in_dim
        self.pred_len = args.pred_len
        self.model = ttnet(num_nodes=self.num_nodes, in_dim=self.in_dim, out_dim=self.pred_len * self.in_dim,
                           hid_dim=48, layers=args.gc_layers)

    def forward(self, x):
        B, L, N = x.shape
        x = x.reshape(B, L, self.num_nodes, self.in_dim).permute(0, 3, 2, 1)
        pred = self.model(x)
        if self.in_dim == 1:
            return pred.squeeze(-1)
        else:
            return pred.reshape(B, self.pred_len, self.in_dim, self.num_nodes).transpose(-2, -1).reshape(B, self.pred_len, self.num_nodes * self.in_dim)


class ttnet(nn.Module):
    def __init__(self, num_nodes, dropout=0.1, supports=None, in_dim=2, out_dim=12, hid_dim=32, layers=8, cnn_layers=4, group=4):
        super(ttnet, self).__init__()
        self.start_conv = Conv2d(in_channels=in_dim,
                                    out_channels=hid_dim,
                                    kernel_size=(1, 1))
        self.cnn_layers = cnn_layers
        self.filter_convs = ModuleList()
        self.gate_convs = ModuleList()
        self.group=group
        D = [1, 2, 4, 8, 16, 32, 48, 64]
        additional_scope = 1
        receptive_field = 1
        for i in range(self.cnn_layers):
            self.filter_convs.append(Conv2d(hid_dim, hid_dim, (1, 2), dilation=D[i], groups=group))
            self.gate_convs.append(Conv2d(hid_dim, hid_dim, (1, 2), dilation=D[i], groups=group))
            receptive_field += additional_scope
            additional_scope *= 2
        self.receptive_field=receptive_field
        depth = list(range(self.cnn_layers))
        self.bn = ModuleList([BatchNorm2d(hid_dim) for _ in depth])

        self.end_conv1 = nn.Linear(hid_dim, hid_dim*4)
        self.end_conv2 = nn.Linear(hid_dim*4, out_dim)

        self.network = MyTransformer(num_nodes, hid_dim, layers=layers, heads=8)

        self.se=SELayer(hid_dim)


    def forward(self, input, epoch=0):

        in_len = input.size(3)
        if in_len < self.receptive_field:
            input = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        x = self.start_conv(input)
        skip = 0
        for i in range(self.cnn_layers):
            residual = x
            filter = torch.tanh(self.filter_convs[i](residual))
            gate = torch.sigmoid(self.gate_convs[i](residual))
            x = filter * gate
            if self.group !=1:
                x = channel_shuffle(x,self.group)
            try:
                skip += x[:, :, :, -1:]
            except:
                skip = 0
            if i == self.cnn_layers-1:
                break
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)
        x = torch.squeeze(skip, dim=-1)
        x=self.se(x)
        x = x.transpose(1, 2)
        x_residual = x
        x = self.network(x)
        x += x_residual
        x = F.relu(self.end_conv1(x))
        x = self.end_conv2(x)
        output=x.transpose(1, 2).unsqueeze(-1)
        return output



def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x

class MyTransformer(nn.Module):
    def __init__(self, num_nodes, hid_dim, layers, heads=8):
        super().__init__()
        self.heads = heads
        self.layers = layers
        self.hid_dim = hid_dim
        self.trans = Transformer(num_nodes, hid_dim, heads, layers)

    def forward(self, x):
        x = self.trans(x)
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class LearnedPositionalEncoding(nn.Embedding):
    def __init__(self,d_model, dropout = 0.1,max_len = 207):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p = dropout)
    def forward(self, x):
        weight = self.weight.data.unsqueeze(1)
        x = x + weight[:x.size(0),:]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(
        self,
        num_nodes,
        d_model=32,
        n_heads=8,
        layers=6
    ):
        super(Transformer, self).__init__()
        self.num_nodes = num_nodes
        self.layers = 4
        self.hid_dim =d_model
        self.heads = n_heads

        self.attention_layer = LightformerLayer(self.hid_dim, self.heads, self.hid_dim * 4)
        self.attention_norm = nn.LayerNorm(self.hid_dim)
        self.attention = Lightformer(self.attention_layer, self.layers, self.attention_norm)
        self.lpos = LearnedPositionalEncoding(self.hid_dim, max_len=self.num_nodes)

    def forward(self,input):
        x = input.permute(1,0,2)
        x = self.lpos(x)
        output = self.attention(x)

        return output.permute(1,0,2)


class LScaledDotProductAttention(Module):

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, groups=2):

        super(LScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model // groups, h * d_k // groups)
        self.fc_k = nn.Linear(d_model // groups, h * d_k // groups)
        self.fc_v = nn.Linear(d_model // groups, h * d_v // groups)
        self.fc_o = nn.Linear(h * d_v // groups, d_model // groups)  # nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.groups = groups

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        xavier_uniform_(self.fc_q.weight)
        xavier_uniform_(self.fc_k.weight)
        xavier_uniform_(self.fc_v.weight)
        xavier_uniform_(self.fc_o.weight)
        constant_(self.fc_q.bias, 0)
        constant_(self.fc_k.bias, 0)
        constant_(self.fc_v.bias, 0)
        constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        queries = queries.permute(1, 0, 2)
        keys = keys.permute(1, 0, 2)
        values = values.permute(1, 0, 2)
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries.view(b_s, nq, self.groups, -1)).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys.view(b_s, nk, self.groups, -1)).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values.view(b_s, nk, self.groups, -1)).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out.view(b_s, nq, self.groups, -1)).view(b_s, nq, -1)  # (b_s, nq, d_model)
        return out.permute(1, 0, 2)


class LMultiHeadAttention(Module):
    def __init__(self, d_model, h, dropout=.1, batch_first=False, groups=2, device=None, dtype=None):
        super(LMultiHeadAttention, self).__init__()

        self.attention = LScaledDotProductAttention(d_model=d_model, groups=groups, d_k=d_model // h, d_v=d_model // h,
                                                    h=h, dropout=dropout)

    def forward(self, queries, keys, values, attn_mask=None, key_padding_mask=None,need_weights=False,attention_weights=None):
        out = self.attention(queries, keys, values, attn_mask, attention_weights)
        return out, out



class Lightformer(Module):

    __constants__ = ['norm']

    def __init__(self, attention_layer, num_layers, norm=None):
        super(Lightformer, self).__init__()
        self.layers = _get_clones(attention_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        output = src
        for i, mod in enumerate(self.layers):
            if i % 2 ==0:
                output = mod(output)
            else:
                output = mod(output, src_mask=mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class LightformerLayer(Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.gelu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LightformerLayer, self).__init__()
        # self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
        #                                     **factory_kwargs)

        self.self_attn = LMultiHeadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                             **factory_kwargs)

        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        # self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        #group fnn =2 自己改的
        self.linear2 = Linear(dim_feedforward // 2, d_model // 2, **factory_kwargs)  ###

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.gelu
        super(LightformerLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = src
        x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
        x = self.norm2(x + self._ff_block(x))
        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    # def _ff_block(self, x: Tensor) -> Tensor:
    #     x = self.linear2(self.dropout(self.activation(self.linear1(x))))
    #     return self.dropout2(x)

    # 自己改的group
    def _ff_block(self, x: Tensor) -> Tensor:
        b, l, d = x.size() ######
        x = self.linear2(self.dropout(self.activation(self.linear1(x))).view(b, l, 2, d*4 // 2)) ###
        x= x.view(b, l, d)  ####
        return self.dropout2(x)



def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    return F.gelu
