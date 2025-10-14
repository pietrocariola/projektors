import torch.nn as nn
import torch

class PrePostProjCls(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=input_size,
                    nhead=1,
                    batch_first=True
                ),
                num_layers=1
            ),
            nn.Linear(input_size, output_size),
        )
        self.soft_max = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.classifier(x)
        x = self.soft_max(x)
        x = torch.mean(x, dim=1)
        x = torch.log(x)
        return x