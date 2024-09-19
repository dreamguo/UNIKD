import os
import sys
import ipdb
import torch
import torch.nn as nn


class MAS(object):
    def __init__(self, model: nn.Module):
        self.model = model
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self.omega_matrices = {n: p.clone().detach().fill_(0) for n, p in self.params.items()}
        self._means = {n: p.clone().detach() for n, p in self.params.items()}
        self.data_num = 0

    def update_omega(self, outputs):
        self.outputs = outputs  # (N_rand, 3)
        self.model.eval()
        self.model.zero_grad()
        self.outputs.pow_(2)
        loss = torch.sum(self.outputs,dim=1)
        loss = loss.mean()
        loss.backward()
        for n, p in self.model.named_parameters():
            if 'uncertain' not in n:
                self.omega_matrices[n].data += p.grad.abs()
        self.data_num += 1

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            if 'uncertain' not in n:
                _loss = self.omega_matrices[n] / self.data_num * (p - self._means[n]) ** 2
                loss += _loss.sum()
        return loss
