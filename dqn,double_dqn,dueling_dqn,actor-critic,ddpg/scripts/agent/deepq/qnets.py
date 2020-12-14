import torch
import torch.nn as nn
import torch.nn.functional as F


class QNet(nn.Module):

    def __init__(self, state_dim, action_count):
        super().__init__()
        self.layer_0 = torch.nn.Linear(state_dim, 200)
        self.layer_1 = torch.nn.Linear(200, 100)
        self.layer_2 = torch.nn.Linear(100, 50)
        self.layer_3 = torch.nn.Linear(50, action_count)

    def forward(self, x):
        x = torch.tanh(self.layer_0(x))
        x = torch.tanh(self.layer_1(x))
        x = torch.tanh(self.layer_2(x))
        x = self.layer_3(x)
        return x


class DuelingQNet(nn.Module):

    def __init__(self, state_dim, action_count):
        super().__init__()
        self.layer_0 = torch.nn.Linear(state_dim, 200)
        self.layer_1 = torch.nn.Linear(200, 100)

        self.v_layer_0 = torch.nn.Linear(100, 50)
        self.v_layer_1 = torch.nn.Linear(50, 1)
        self.a_layer_0 = torch.nn.Linear(100, 50)
        self.a_layer_1 = torch.nn.Linear(50, action_count)

    def forward(self, x):
        x = torch.tanh(self.layer_0(x))
        x = torch.tanh(self.layer_1(x))
        v = self.v_layer_1(torch.tanh(self.v_layer_0(x)))
        a = self.a_layer_1(torch.tanh(self.a_layer_0(x)))

        a_mean = torch.mean(a) if len(a.shape) == 1 else torch.mean(a, dim=1, keepdim=True)
        x = v + (a - a_mean)

        return x
