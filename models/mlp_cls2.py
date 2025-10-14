import torch.nn as nn

class PrePostProjCls2(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1536),
            nn.ReLU(),
            nn.Linear(1536, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        return self.classifier(x)