import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hid_dim = args.rnn_dim
        if hasattr(args, 'in_dim'):
            self.in_dim = args.in_dim
            self.label_num = 1
        else:
            self.in_dim = args.seq_len
            self.label_num = args.pred_len
        self.y_emb = nn.Parameter(torch.randn(1, self.label_num, self.hid_dim))
        self.proj = nn.Sequential(nn.Linear(self.in_dim + args.concept_bias, self.hid_dim), nn.LeakyReLU())
        self.rnn = nn.LSTM(self.hid_dim, self.hid_dim, batch_first=True)
        self.predictor = nn.Linear(self.hid_dim, self.in_dim + args.concept_bias)
        self.zero = nn.Parameter(torch.ones(1))
        # self.rnn_X = nn.GRU(self.in_dim * 2, self.hid_dim * 2, batch_first=True)
        # self.predictor_mean = nn.Linear(self.hid_dim * 2, self.in_dim)
        # self.predictor_std = nn.Linear(self.hid_dim * 2, self.in_dim)

    def forward(self, concept_YX):
        if self.label_num == 1:
            # concept_YX = (self.proj(concept_YX) + self.y_emb)
            h1 = self.rnn(self.proj(concept_YX) + self.y_emb)[1][0].squeeze(0)
            concept_YX_pred = self.predictor(h1)
            return concept_YX_pred
        else:
            bs, past_len, label_num, input_dim = concept_YX.shape
            # y_emb = self.y_emb.repeat(bs, past_len, 1, 1)
            # concept_YX = torch.cat([concept_YX, y_emb], -1).transpose(1, 2).reshape(bs * label_num, past_len, -1)
            concept_YX = (self.proj(concept_YX) + self.y_emb).transpose(1, 2).reshape(bs * label_num, past_len, -1)
            h1 = self.rnn(concept_YX)[1][0].reshape(bs, label_num, -1)
            concept_YX_pred = self.predictor(h1)
            return concept_YX_pred
