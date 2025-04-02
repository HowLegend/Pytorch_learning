''' 使用 gpu 训练的方法1：
在以下三个地方：
1. 网络模型
2. 损失函数
3. 数据（输入、标注）

添加代码：
if torch.cuda.is_available():
    实例.cuda()             ---网络模型/损失函数
    实例 = 实例.cuda()       ---数据（输入、标注）     网络模型/损失函数 也可以
'''

import torchvision
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

train_data = torchvision.datasets.CIFAR10("dataset/CIFAR10", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10("dataset/CIFAR10", train=False, transform=torchvision.transforms.ToTensor(), download=True)

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, 64)
test_dataloader = DataLoader(test_data, 64)

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

# 创建网络模型
tudui = Tudui()
if torch.cuda.is_available():
    tutui = tudui.cuda()  # 1. 网络模型：将网络模型放到cuda上，使用gpu训练

# 损失函数
loss_fn = nn.CrossEntropyLoss()  # CrossEntropyLoss: 交叉熵 损失
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()  # 2. 损失函数：将损失函数放到cuda上，使用gpu训练

# 优化器
learning_rate = 1e-2  # 1e-2 = 1 × (10)^(-2) = 0.01
optimizer = torch.optim.SGD(params=tudui.parameters(), lr=learning_rate)  # SGD: 随机梯度下降

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加 tensorboard
writer = SummaryWriter("logs/train")

for i in range(epoch):
    print("--------第 {} 轮训练开始-------".format(i+1))

    # 训练步骤开始
    tudui.train()  # 大部分情况，不写也可以，不过小部分网络层需要这个
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()  # 3. 数据（输入、标注）：将数据（输入、标注）放到cuda上，使用gpu训练
            targets = targets.cuda()  # 3. 数据（输入、标注）：将数据（输入、标注）放到cuda上，使用gpu训练

        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 也算是训练了一次
        total_train_step += 1
        if total_train_step%100 == 0:          # 防止打印太多，从而让更有价值的信息被掩藏
            print("训练次数：{}，Loss：{}".format(total_train_step, loss.item()))   
            # 小知识：xxx.item() 的作用是把xxx转化为真实的数据，比如a = torch.tensor(5);print(a)-->tensor(5)，而print(a.item())-->5
            writer.add_scalar("train_loss", loss.item(), total_train_step)
        
    # 测试步骤开始
    tudui.eval()  # 大部分情况，不写也可以，不过小部分网络层需要这个
    total_test_loss = 0
    total_correct = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()  # 3. 数据（输入、标注）：将数据（输入、标注）放到cuda上，使用gpu训练
                targets = targets.cuda()  # 3. 数据（输入、标注）：将数据（输入、标注）放到cuda上，使用gpu训练
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            # 计算正确的个数
            correct = (outputs.argmax(1) == targets).sum()
            total_correct += correct

    total_test_step += 1
    print("整体测试集上的Loss：{}".format(total_test_loss))
    test_accuracy = total_correct/test_data_size
    print("整体测试集上的正确率：{}".format(test_accuracy))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", test_accuracy, total_test_step)

    # 保存每一轮训练的模型
    torch.save(tudui, "model/train/tudui{}.pth".format(i))
    # torch.save(tudui.state_dict(), "model/train/tudui{}.pth".format(i))
    print("模型 {} 已保存至 model/train/tudui{}.pth".format(i,i))

writer.close()