import torch
from torch import nn

class OneNet(nn.Module):
    def __init__(self, backbone, args):
        super().__init__()
        self.backbone = backbone
        self.decision = MLP(n_inputs=args.pred_len * 3, n_outputs=1, mlp_width=32, mlp_depth=3, mlp_dropout=0.1, act=nn.Tanh())
        self.weight = nn.Parameter(torch.zeros(args.enc_in))
        # self.bias = nn.Parameter(torch.zeros(args.enc_in))

    def forward(self, *inp):
        y1, y2 = self.backbone.forward_individual(*inp[:-2])
        return y1.detach() * inp[-2] + y2.detach() * inp[-1], y1, y2

    def store_grad(self):
        self.backbone.store_grad()
1
class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, mlp_width, mlp_depth, mlp_dropout, act=nn.ReLU()):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, mlp_width)
        self.dropout = nn.Dropout(mlp_dropout)
        self.hiddens = nn.ModuleList([
            nn.Linear(mlp_width, mlp_width)
            for _ in range(mlp_depth-2)])
        self.output = nn.Linear(mlp_width, n_outputs)
        self.n_outputs = n_outputs
        self.act = act

    def forward(self, x, train=True):
        x = self.input(x)
        if train:
            x = self.dropout(x)
        x = self.act(x)
        for hidden in self.hiddens:
            x = hidden(x)
            if train:
                x = self.dropout(x)
            x = self.act(x)
        x = self.output(x)
        # x = F.sigmoid(x)
        return x
