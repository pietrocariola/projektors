import torch.nn as nn
import torch

class PrePostProjCls2(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=1,
                    batch_first=True
                ),
                num_layers=1
            ),
            nn.Linear(hidden_size, output_size),
        )
        self.soft_max = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.classifier(x)
        x = self.soft_max(x)
        x = torch.mean(x, dim=1)
        x = torch.log(x)
        return x