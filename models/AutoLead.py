import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.RevIN import RevIN


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.univariate = configs.univariate

        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = configs.enc_in
        self.K = configs.leader_num
        self.linear = nn.Linear(self.seq_len, self.pred_len, bias=False)
        self.leaders = None     # [N, K]
        self.shifts = None      # [N, K]
        self.leader_weight = nn.Parameter(torch.ones(self.channels, 1 + self.K) / (1 + self.K))
        self.revIN = RevIN(self.channels, affine=False)

    def predefine_leaders(self, leaders):
        # self.leaders = torch.cat([torch.arange(self.channels).unsqueeze(-1), leaders]).view(-1)
        # self.shifts = torch.cat([torch.zeros(self.channels).unsqueeze(-1), shifts])
        self.leaders = leaders.view(-1)
        # self.shifts = nn.Parameter(shifts.view(-1).float())

    def forward(self, x):
        x = self.revIN(x, 'norm')   # [B, L, N]
        y_hat = self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)      # [B, H, N]
        if not self.univariate:
            seq = torch.cat([x, y_hat], -2)[:, :, self.leaders]    # [B, L+H, N*K)]
            indices = torch.arange(x.shape[1], x.shape[1] + y_hat.shape[1], device=x.device).unsqueeze(0).unsqueeze(-1)
            shifts = F.sigmoid(self.shifts) * self.seq_len
            shift_floor, shift_ceil = shifts.floor().long(), shifts.ceil().long()
            indices_floor = (indices - shift_floor).repeat((len(x), 1, 1))    # [1, H, N*K]
            indices_ceil = (indices - shift_ceil).repeat((len(x), 1, 1))  # [1, H, N*K]
            seq_shifted = seq.gather(1, indices_floor) * (shifts - shift_floor) + \
                          seq.gather(1, indices_ceil) * (shifts + 1 - shift_ceil)

            seq_shifted = seq_shifted.view(seq_shifted.shape[0], seq_shifted.shape[1], self.channels, self.K)
            seq_shifted = torch.cat([y_hat.unsqueeze(-1), seq_shifted], -1)
            w = torch.softmax(self.leader_weight, -1)
            y_hat = (seq_shifted * w).sum(-1)
        y_hat = self.revIN(y_hat, 'denorm')
        if not self.univariate:
            return y_hat, seq_shifted
        return y_hat