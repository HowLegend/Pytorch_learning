import torch
from torch import nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input +1
        return output



myModel = Model()
x = torch.tensor(1.0)
output = myModel(x)
print(output)