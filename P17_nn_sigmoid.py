import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch import nn

dataset = torchvision.datasets.CIFAR10("dataset/CIFAR10", train= False, transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset, 64)

class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output
    

writer = SummaryWriter("logs/sigmoid")
tudui = Tudui()
step = 0
for data_batch in dataloader:
    imgs, targets = data_batch
    writer.add_images("input", imgs, step)
    output = tudui(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()