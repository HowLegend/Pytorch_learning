from PIL import Image
from torchvision import transforms
from torch import nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 调整图片格式、大小
image_path = "dataset/image/plane.png"
image = Image.open(image_path)
print(image)
image = image.convert("RGB")  # png格式是四个通道，除了RGB的三个通道外，还有一个透明度通道。所以，我们调用 image = image.convert("RGB")，保留其颜色通道
print(image)

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor()
])

image = transform(image)
print(image.shape)

image = torch.reshape(image, (1,3,32,32))  # 注意！网络在输入图片时是需要 batch_size 的，也就是(1,3,32,32)中的第一位，之前用DataLoader设置的 batch_size=64，这里用 torch.reshape()
image = image.to(device) # 应为之前训练的模型是放在gpu上的，所以这里的图片也要放进gpu，和模型放在一起才能正常运行

# 加载模型
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
    

model = torch.load("model/train/tudui9.pth", weights_only=False)  # 因为这个模型是在gpu上训练的，如果你要用cpu来运行，需要使用参数 “map_location=torch.device("cpu")” 完成gpu到cpu的映射
print(model)

# 开始测试
model.eval()
with torch.no_grad():
    output = model(image)
print(output)               # tensor([[-1.2996, -1.3062,  0.5614,  0.9986,  1.1198,  1.2148,  3.3134, -1.3477, -2.5502, -0.6553]], device='cuda:0')
print(output.argmax(1))     # tensor([6], device='cuda:0')