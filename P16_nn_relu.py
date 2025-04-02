import torch
from torch import nn

input = torch.tensor([[1, -0.5],
                      [-1, 3]])

class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu1 = nn.ReLU()
    
    def forward(self, input):
        output = self.relu1(input)
        return output
    
tudui = Tudui()
output = tudui(input)
print(output)