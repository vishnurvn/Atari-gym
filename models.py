import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.categorical import Categorical


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.inp = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3)
        )
        self.pool = nn.MaxPool2d((2, 2))
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.inp(x)
        x = self.pool(x)
        x = self.act(x)
        return x


class SimplePolicy(nn.Module):
    def __init__(self, action_size) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(3, 8), ConvBlock(8, 16), ConvBlock(16, 32), ConvBlock(32, 64)
        )
        self.lp = nn.LazyLinear(32)
        self.action_head = nn.LazyLinear(action_size)

    def forward(self, x):
        x = self.net(x)
        x = f.relu(torch.mean(self.lp(x), dim=(-1, -2)))
        x = self.action_head(x)
        return Categorical(logits=x)

    @torch.no_grad()
    def get_action(self, state):
        logits = self(state)
        return logits
