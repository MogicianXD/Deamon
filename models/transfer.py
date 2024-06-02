import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine(x1, x2, eps=1e-9):
    x1 = x1 / (torch.norm(x1, p=2, dim=-1, keepdim=True) + eps)
    x2 = x2 / (torch.norm(x2, p=2, dim=-1, keepdim=True) + eps)
    return x1 @ x2.transpose(0, 1)


def l2_sim(x1, x2):
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    return - torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)



class Transfer(nn.Module):
    def __init__(self, num_features: int, eps=1e-8, affine=True, seq_len=None, memory_size=32, **kwargs):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(Transfer, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = False
        if self.affine:
            self._init_params()
        self.memory = None
        self.memory_cnt = None
        self.memory_size = memory_size
        self.concept_extractor = nn.Parameter(torch.ones(seq_len, num_features) / seq_len, requires_grad=False)

    def forward(self, x, mode:str):
        if mode == 'forward':
            x = self._transfer(x)
        elif mode == 'inverse':
            x = self._recover(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def init_centroids(self, means):
        self.memory = torch.tensor(means).float().cuda()
        return
        self.means = torch.tensor(means).float().cuda()
        means = self.means.view(len(means), -1, self.num_features)
        concept = means.mean(1, keepdims=True)
        self.memory = torch.cat([concept.squeeze(-2), torch.sqrt(((means - concept) ** 2).mean(-2) + self.eps)], -1)

    def _get_concept(self, x):
        concept = (self.concept_extractor * x).sum(-2, keepdim=True)
        return torch.cat([concept.squeeze(-2), torch.sqrt(((x - concept) ** 2).mean(-2) + self.eps)], -1)

    def _transfer(self, x):
        concept = self._get_concept(x)
        sim = l2_sim(concept, self.memory)
        indices = sim.max(-1)[1]
        memorized_concept = self.memory[indices]

        # if self.memory is None:
        #     memorized_concept = torch.cat([torch.zeros(1, self.num_features, device=x.device),
        #                                      torch.ones(1, self.num_features, device=x.device)], -1)
        #     self.memory = concept.clone().detach()
        #     self.memory_cnt = torch.ones(self.memory_size, device=x.device)
        # else:
        #     # sim = cosine(concept, self.memory)
        #     sim = l2_sim(concept, self.memory)
        #     indices = sim.max(-1)[1]
        #     memorized_concept = self.memory[indices]
        #
        #     cnt = torch.bincount(indices, minlength=self.memory_size)
        #     self.memory_cnt += cnt
        #     update_rate = 1 / self.memory_cnt
        #     c = F.one_hot(indices, self.memory_size).transpose(0, 1).float() @ concept
        #     self.memory[:, :self.num_features] = (1 - cnt * update_rate).unsqueeze(-1) * self.memory[:, :self.num_features] + \
        #                   update_rate.unsqueeze(-1) * (c[:, :self.num_features])
        #     self.memory[:, self.num_features:] = torch.sqrt((1 - cnt * update_rate).unsqueeze(-1) * self.memory[:, self.num_features:] ** 2 + \
        #                                       update_rate.unsqueeze(-1) * (c[:, self.num_features:] ** 2))

        # self.bias = (memorized_concept[:, :self.num_features] - concept[:, :self.num_features]).unsqueeze(1)
        # self.scale = (memorized_concept[:, self.num_features:] / concept[:, self.num_features:]).unsqueeze(1)
        self.scale = (memorized_concept[:, self.num_features:] / concept[:, self.num_features:]).unsqueeze(1)
        self.bias = memorized_concept[:, :self.num_features].unsqueeze(1) - concept[:, :self.num_features].unsqueeze(1) * self.scale

        x = x * self.scale
        x = x + self.bias
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _recover(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x - self.bias
        x = x / self.scale
        return x