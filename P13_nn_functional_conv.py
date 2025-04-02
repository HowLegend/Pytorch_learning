import torch
import torch.nn.functional as F

input = torch.tensor([[1,2,0,3,1],     # 开头有几个中括号，就是几维矩阵
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]])

kernel = torch.tensor([[1,2,1],
                       [0,1,0],
                       [2,1,0]])

print(input.shape)
print(kernel.shape)     # 阅读官网文档，发现输入尺寸不对

input = torch.reshape(input, (1,1,5,5))
kernel = torch.reshape(kernel,(1,1,3,3))

print(input.shape)
print(kernel.shape)


output = F.conv2d(input, kernel, stride=1)
print(output)

# stride
output2 = F.conv2d(input=input,weight=kernel,stride=2)
print(output2)

# padding
output3 = F.conv2d(input=input,weight=kernel,stride=1,padding=1)
print(output3)
