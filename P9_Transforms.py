from torchvision import transforms
from PIL import Image
import cv2 as cv
from torch.utils.tensorboard import SummaryWriter



writer = SummaryWriter("logs/tool")


img_path = "dataset/image/cat_firstFrame.png"
# img = Image.open(img_path)           # PIL
img = cv.imread(img_path)              # ndarray
print(type(img))
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
print(type(tensor_img))                # tensor
print(tensor_img.shape)

writer.add_image("Tensor_img", tensor_img, 0)

writer.close()

