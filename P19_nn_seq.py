from torch.nn import Module, Conv2d, MaxPool2d, Flatten, Linear, Sequential
import torch
from torch.utils.tensorboard import SummaryWriter


class Tudui(Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = Conv2d(3, 32, 5, padding=2)
        # self.maxpool1 = MaxPool2d(2)
        # self.conv2 = Conv2d(32, 32, 5, padding="same")
        # self.maxpool2 = MaxPool2d(2)
        # self.conv3 = Conv2d(32, 64, 5, padding="same")
        # self.maxpool3 = MaxPool2d(2)
        # self.flatten = Flatten()
        # self.linear1 = Linear(1024, 64)
        # self.linear2 = Linear(64, 10)
        
        # Sequential 的作用，与上面等价
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
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x)
        # x = self.linear1(x)
        # x = self.linear2(x)

        x = self.model1(x)
        return x
    
tudui = Tudui()
print(tudui)

# 一般写完网络后，会去检查准确性   eg看看 linear1 中的 1024 会不会写成 10240
input = torch.ones((64, 3, 32, 32))
output = tudui(input)
print(output.shape)


writer = SummaryWriter("logs/seq")
writer.add_graph(tudui, input)
writer.close()