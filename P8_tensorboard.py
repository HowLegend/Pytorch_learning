from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
import os


dataset_dir = "dataset"
img_dir = "image/paragliding_firstFrame.png"
img_dir = os.path.join(dataset_dir, img_dir)
print(img_dir)

writer = SummaryWriter("logs/tool")

img = Image.open(img_dir)   
print(type(img))         # class 'PIL.PngImagePlugin.PngImageFile'
img = np.array(img)
print(type(img))      # class 'numpy.ndarray'
print(img.shape)

writer.add_image("train",img,0,dataformats='HWC')

for i in range(100):
    writer.add_scalar("y=x^3",i*i*i,i+i)


writer.close()