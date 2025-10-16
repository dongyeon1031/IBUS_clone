# @Time : 2022-05-10 19:46
# @Author : Wang Zhen
# @Email : wangzhen@stu.xpu.edu.cn
# @File : dataloader.py
# @Project : GPR
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import cv2
from torchvision import transforms
import csv
import glob
import numpy as np
from .imagepreprocess import satellite_image_preprocess,uav_image_preprocess


class HelicopterUAV(Dataset):
    def __init__(self, root_dir,rotate, transform):
        self.root_dir = root_dir
        self.query_images_path=os.path.join(root_dir,"query_images")
        self.query_images = os.listdir(self.query_images_path)
        self.query_images.sort()
        self.transform = transform
        self.rotate=rotate
        # print("self.rotate",self.rotate)
    def __getitem__(self, idx):
        query_name=self.query_images[idx]
        query_img_item_path = os.path.join(self.root_dir, self.query_images_path, query_name)
        query_img=cv2.imread(query_img_item_path)
        if self.rotate:
            query_img=uav_image_preprocess(query_img)
        query_img = cv2.cvtColor(query_img,cv2.COLOR_BGR2RGB)
        query_img = self.transform(query_img)
        return query_img,query_name

    def __len__(self):
        return len(self.query_images)

class HelicopterSatellite(Dataset):
    def __init__(self, root_dir,rotate, transform):
        self.root_dir = root_dir
        self.query_images_path=os.path.join(root_dir,"reference_images")
        self.query_images_path=self.load_satellite(self.query_images_path)
        self.query_images_path.sort()
        self.transform = transform
        self.rotate=rotate
        # print("self.rotate",self.rotate)
    def load_satellite(self,path):
        # query_images_path = glob.glob(os.path.join(path, "offset_0_None", "*.png"))
        image_extensions = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
        query_images_path = []
        for ext in image_extensions:
            query_images_path.extend(glob.glob(os.path.join(path, "offset_0_None", ext)))
            
        relative_path=[]
        for p in query_images_path:
            relative_path.append(p[len(path)+1:])
        return relative_path
    def __getitem__(self, idx):
        query_name=self.query_images_path[idx]
        query_img_item_path = os.path.join(self.root_dir,"reference_images", query_name)
        query_img=cv2.imread(query_img_item_path)
        if self.rotate:
            query_img = satellite_image_preprocess(query_img)
        query_img = cv2.cvtColor(query_img,cv2.COLOR_BGR2RGB)
        query_img = self.transform(query_img)
        return query_img,query_name

    def __len__(self):
        return len(self.query_images_path)


if __name__ == '__main__':
    root_dir="/root/data/DATASET/GPR/HelicopterRound2/"
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = Helicopter(root_dir, transform=transform)
    print(train_dataset.reference_id_num)
    train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True,num_workers=32,prefetch_factor=4)

    print(len(train_dataset))
    for batch_idx, (query_img,reference_img,reference_img_2,query_ind) in enumerate(train_loader):
        print(query_ind)
        # break
        # print(batch_idx)