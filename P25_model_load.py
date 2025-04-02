import torch
import torchvision
# from P24_model_save import Tudui    # 这里报错的话，不用管他，是应为当前文件默认的根目录不是该文件所处的文件夹。具体原因看：https://blog.csdn.net/bigData1994pb/article/details/124911913?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522670a8eccc19c2f18f074b142b32e3dbd%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=670a8eccc19c2f18f074b142b32e3dbd&biz_id=0

# （与P24的“模型保存方式 n ”对应）

# 模型和参数都被加载进来了
# 模型加载方式1
model = torch.load("vgg16_method1.pth", weights_only=False)
print(model)

# 模型加载方式2
vgg_16 = torchvision.models.vgg16()   # 没有参数 weights=torchvision.models.VGG16_Weights.DEFAULT ，所以导入的只有模型，没有权重
vgg_16.load_state_dict(torch.load("vgg16_method2.pth"))        # 加载参数 weight = torch.load("vgg16_method2.pth")
print(vgg_16)



# 方式1 可能会遇到的陷阱
# 陷阱1
model = torch.load("tudui_method1.pth", weights_only=False)
print(model)
# 报错：AttributeError: Can't get attribute 'Tudui' on <module '__main__' from '/home/zhihongyan/lijunhao/Pytorch_learning/P25_model_load.py'>
# 因此需要先把 Tudui类 写在前面，或者在开头写 from P24_model_save import Tudui