import torchvision
from torch import nn

# train_data = torchvision.datasets.ImageNet("dataset/ImageNet", split="train", transform=torchvision.transforms.ToTensor())

vgg16_false = torchvision.models.vgg16()
vgg16_true = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)

train_data = torchvision.datasets.CIFAR10("dataset/CIFAR10", True, torchvision.transforms.ToTensor(), download=True)

# vgg16_true.add_module('add_linear', nn.Linear(1000,10))
vgg16_true.classifier.add_module("add_linear", nn.Linear(1000,10))
print(vgg16_true)

vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)