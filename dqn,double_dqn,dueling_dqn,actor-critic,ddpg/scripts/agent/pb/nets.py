import torch
import torch.nn as nn
import torch.nn.functional as F


class DDPGActor(nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.layer_0 = torch.nn.Linear(state_dim, 128)
        self.layer_1 = torch.nn.Linear(128, 64)
        self.layer_2 = torch.nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.tanh(self.layer_0(x))
        x = torch.tanh(self.layer_1(x))
        x = self.layer_2(x)
        return x


class DDPGCritic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.layer_0 = torch.nn.Linear(state_dim + action_dim, 256)
        self.layer_1 = torch.nn.Linear(256, 128)
        self.layer_2 = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = torch.tanh(self.layer_0(x))
        x = torch.tanh(self.layer_1(x))
        x = self.layer_2(x)
        return x


class ACActor(nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.layer_0 = torch.nn.Linear(state_dim, 128)
        self.layer_1 = torch.nn.Linear(128, 64)
        self.mu = torch.nn.Linear(64, action_dim)
        self.sigma = torch.nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.tanh(self.layer_0(x))
        x = torch.tanh(self.layer_1(x))
        mean = self.mu(x)
        std = F.elu(self.sigma(x)) + 1.01
        return mean, std


class ACCritic(nn.Module):

    def __init__(self, state_dim):
        super().__init__()
        self.layer_0 = torch.nn.Linear(state_dim, 256)
        self.layer_1 = torch.nn.Linear(256, 128)
        self.layer_2 = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = torch.tanh(self.layer_0(x))
        x = torch.tanh(self.layer_1(x))
        x = self.layer_2(x)
        return x

