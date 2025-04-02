"""
经验总结：
1. 关注输入输出
2. 多看官方文档，安装ctrl点击函数即可
3. 关注方法【输入】需要什么参数，关注【输出】，若没告诉你输出是什么，可以使用 print()、print(type())、进行Debug查看过程变量、上网查
4. 可以将图片转化为 tensor 后，去 tensorboard 里面看一下
"""
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter("logs/tool")
img = Image.open("dataset/image/paragliding_firstFrame.png")
print(img)

# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("Totensor",img_tensor,0)

# Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([1,2,3],[4,5,6])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Nomalize", img_norm, 1)

# Resize
print(img.size)
trans_resize_1 = transforms.Resize([512,200])   # [512,200] --> 200
img_resize = trans_resize_1(img_tensor)
print(img_resize)
writer.add_image("Resize", img_resize, 0)

trans_resize_2 = transforms.Resize(200)
img_resize = trans_resize_2(img_tensor)
writer.add_image("Resize", img_resize, 1)

# Compose
trans_compose = transforms.Compose([
    trans_totensor,
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    trans_resize_1,
])
img_compose = trans_compose(img)
writer.add_image("Compose",img_compose, 0)

# RandomCrop
trans_rCrop = transforms.RandomCrop(100)     # [50,100] 也可以
trans_compose_2 = transforms.Compose([
    trans_totensor,
    trans_rCrop,
])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)

writer.close()