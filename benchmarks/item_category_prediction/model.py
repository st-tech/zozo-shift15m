import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, n_outputs) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, n_outputs),
        )

    def forward(self, x):
        y = F.log_softmax(self.mlp(x), dim=1)
        return y
