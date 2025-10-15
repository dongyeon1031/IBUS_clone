# @Time : 2022-04-24 20:07
# @Author : Wang Zhen
# @Email : wangzhen@stu.xpu.edu.cn
# @File : GeoTransforms.py
# @Project : UAVGeoMymodel0424
from torchvision import transforms

def BuildTransforms(size=256,Norm=True):
    BasetransformsList=[
        transforms.ToTensor(),
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.Resize([size, size]),

    ]
    if Norm:
        BasetransformsList.append(transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]))
    T = transforms.Compose(BasetransformsList)
    return T