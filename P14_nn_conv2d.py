import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("dataset/CIFAR10",train=False,transform=torchvision.transforms.ToTensor(),download=True)

dataloader = DataLoader(dataset,batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x
    

tudui = Tudui()

writer = SummaryWriter("logs/conv2d")
step = 0
for data_batch in dataloader:
    imgs, targets = data_batch
    output = tudui(imgs)
    # print(imgs.shape)
    # print(output.shape)
    # torch.Size([64, 3, 32, 32])
    writer.add_images("input", imgs, step)

    # output: torch.Size([64, 6, 30, 30])
    # output 通道数为6，tensorboard不知道怎么显示，不严谨的解决方法
    output = torch.reshape(output,(-1,3,30,30))   # 第一个数写-1，会自动根据后面的数来计算出第一个数的值
    writer.add_images("output", output, step)

    step += 1


writer.close()