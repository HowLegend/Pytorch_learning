from torch.nn import Module, Conv2d, MaxPool2d, Flatten, Linear, Sequential
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn

# 准备测试数据集
dataset = torchvision.datasets.CIFAR10("dataset/CIFAR10",train=True,transform=torchvision.transforms.ToTensor(),download=True)

dataloader = DataLoader(dataset, 1)

# 构建神经网络
class Tudui(Module):
    def __init__(self):
        super().__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding="same"),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding="same"),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10),
        )

    def forward(self, x):
        x = self.model1(x)
        return x

tudui = Tudui()
loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(tudui.parameters(), lr=0.01)


for epoch in range(20):
    running_loss = 0.0
    for data_batch in dataloader:
        imgs, targets = data_batch
        outputs = tudui(imgs)
        # print(outputs)   # tensor([[ 0.0409,  0.0220,  0.0892, -0.0120,  0.0903,  0.1029, -0.1118,  0.0838, -0.1454, -0.0706]], grad_fn=<AddmmBackward0>)
        # print(targets)   # tensor([5])
        result_loss = loss(outputs, targets)        # loss 作用1：计算出实际输出和目标之间的差距
        # print(result_loss)        # tensor(2.2919, grad_fn=<NllLossBackward0>)

        # result_loss.backward()                      # loss 作用2：为我们更新输出提供一定的依据，反向传播-->为网络中的weight计算梯度grad          
        # ！！！注意：是计算出来的 result 采用 backward，不是 loss 采用 backward！！！

        optim.zero_grad()
        result_loss.backward()
        optim.step()
        # print(result_loss)
        running_loss += result_loss
    print(running_loss)


