import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

# 下载比较慢的话，用迅雷下载，然后把压缩文件放到dataset文件夹里，运行时他会自动解压
# 下载链接，查看CIFAR10定义中的源代码，其中的url就是下载链接
train_set = torchvision.datasets.CIFAR10(root="dataset/CIFAR10",train=True,transform=dataset_transform,download=True)
test_set = torchvision.datasets.CIFAR10(root="dataset/CIFAR10",train=False,transform=dataset_transform,download=True)

# print(test_set[0]) 
# print(test_set.classes)

# img,target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])

print(train_set[0])

writer = SummaryWriter("logs/tool")
for i in range(10):
    img, target = train_set[i]
    writer.add_image("train_set",img,i)

writer.close()
