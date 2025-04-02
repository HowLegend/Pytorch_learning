import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备测试数据集
test_data = torchvision.datasets.CIFAR10("dataset/CIFAR10",train=True,transform=torchvision.transforms.ToTensor(),download=True)

test_loader = DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=True)

# 测试数据集中的第一张图片及target
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("logs/dataloader")

for epoch in range(2):   # 重复两次，生成两组图
    step = 0
    for data in test_loader:
        imgs,targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images("Epoch: {}".format(epoch),imgs,step)   # 注意这里的image + s，是复数的                    # {}是占位符，“.format()”方法会按顺序将传递给它的变量插入到字符串中
        step = step + 1

writer.close()