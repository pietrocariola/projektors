import torch.nn as nn

class PrePostProjCls(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 2000),
            nn.ReLU(),
            nn.Linear(2000, 3600),
            nn.ReLU(),
            nn.Linear(3600, 1024),
            nn.ReLU(),
            nn.Linear(1024, 600),
            nn.ReLU(),
            nn.Linear(600, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        return self.classifier(x)