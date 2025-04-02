# 池化
# 池化的作用：压缩特征，比如把1080p的视频压缩为480p，也能看，而且更轻量了，那么训练、传输会更快

import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("dataset/CIFAR10", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, 64)

# input = torch.tensor([[1,2,0,3,1],
#                       [0,1,2,3,1],
#                       [1,2,1,0,0],
#                       [5,2,3,1,1],
#                       [2,1,0,1,1]])

# input = torch.reshape(input,(-1, 1, 5, 5))
# print(input.shape)

class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,ceil_mode=True)
    
    def forward(self, input):
        output = self.maxpool1(input)
        return output
    

tudui = Tudui()
# output = tudui(input)
# print(output)

writer = SummaryWriter("logs/maxpool") 
step = 0
for data_batch in dataloader:
    imgs, target = data_batch
    writer.add_images("input", imgs, step)
    output = tudui(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()

