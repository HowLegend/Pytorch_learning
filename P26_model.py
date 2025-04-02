from torch import nn
import torch


# 搭建神经网络
class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, (5,5), 1, "same"),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, (5,5), 1, "same"),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, (5,5), 1, "same"),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# 测试网络
if __name__ == '__main__':
    tudui = Tudui()
    # 创建一个输入尺寸来测试
    input = torch.ones((64, 3, 32, 32))   # 输入64张（3，32，32）的图片
    output = tudui(input)
    print(output.shape)                   # torch.Size([64, 10])   返回64个数据，每个数据都有10个概率，对应10种分类的概率