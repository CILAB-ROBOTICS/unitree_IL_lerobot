import torch
import torch.nn as nn

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)