import torch.nn as nn


class Net(nn.Module):
    def __init__(self, n_outputs) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, n_outputs),
        )

    def forward(self, x):
        y = self.mlp(x)
        return y
