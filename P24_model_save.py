import torch
import torchvision
from torch import nn


vgg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)

# 模型和参数都被保存下来了
# 模型保存方式1: 保存模型结构+模型参数
torch.save(vgg16, "vgg16_method1.pth")

# 模型保存方式2: 保存模型参数（官方推荐）       ----------对于较大的模型，这个的文件大小更小    ------把模型的“状态”保存成“字典”（python的一种数据格式）
torch.save(vgg16.state_dict(), "vgg16_method2.pth")

# 方式1 可能会遇到的陷阱
class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x
    
tudui = Tudui()
torch.save(tudui, "tudui_method1.pth")