#!/usr/bin/env python
# coding: utf-8

# In[21]:


import torch
import numpy as np
import cv2
import os
from dataset import HelicopterUAV,HelicopterSatellite,BuildTransforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import build_model
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from PIL import Image

# # Load satellite images

# In[31]:


# root_dir="./data/round2/Val"
root_dir="./data/jeju"
uav_images=os.listdir(os.path.join(root_dir,"query_images"))
satellite_images=os.listdir(os.path.join(root_dir,"reference_images/offset_0_None"))
batch_size=8
transform=BuildTransforms(256)
# uav_dataset=HelicopterUAV(root_dir,False, transform)
satellite_dataset=HelicopterSatellite(root_dir,False, transform)
#
# uav_dataloader= DataLoader(uav_dataset,batch_size=batch_size)
satellite_dataloader= DataLoader(satellite_dataset,batch_size=batch_size)

gt=np.loadtxt(os.path.join(root_dir,"gt_matches.csv"),delimiter=',',dtype=str)[1:,:]

# ## Build model
# Alexnet

# In[5]:

device_ids = [0, 1]
device = torch.device(f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu')

model_name="alexnet_triplet"
# model=build_model(model_name,dropout_p=False).cuda()
model = build_model(model_name, dropout_p=False)
model = torch.nn.DataParallel(model, device_ids=device_ids).to(device)

# # Extract features

# In[6]:


def ExtractFeature(model, Dataloader):
    model.eval()
    drone_name = []
    with torch.no_grad():
        for batch_idx, (image, name), in enumerate(tqdm(Dataloader)):
            image = image.cuda()
            v1 = model(image)
            # v1= nn.functional.normalize(v1, p=2, dim=1)
            if batch_idx == 0:
                drone_feature = v1
            else:
                drone_feature = torch.cat([drone_feature, v1], dim=0)
            drone_name.extend(name)
    return drone_name,drone_feature

# ### Save satellite features

# In[8]:


save_path="features"
satellite_feature_file='satellite_feature.npy'

satellite_feature_path=os.path.join(save_path, satellite_feature_file)

if os.path.exists(satellite_feature_path):
    print("Existing...")
    satellite_feature =np.load(satellite_feature_path)
else:
    print("ExtractFeature。。。")
    satellite_name,satellite_feature =ExtractFeature(model,satellite_dataloader)
    satellite_feature=satellite_feature.cpu().numpy()
    np.save(os.path.join(save_path, satellite_feature_file), satellite_feature)

# # Image sort

# In[9]:


def manifold(feature,n_neighbors=5):
    isomap = Isomap(n_neighbors=n_neighbors, n_components=1, p=2)
    result = isomap.fit_transform(feature)
    return result

# In[13]:


satellite_result=manifold(satellite_feature)#Dimension-reduced features
print(satellite_result.shape)

# # Visualization

# In[16]:


satellite_rank=np.argsort(satellite_result[:,0])#Satellite sort result
satellite_rank

# In[24]:


Vis_10_images=[]
for i in range(10):
    satellite_index=satellite_rank[i]
    satellite_bgr=cv2.imread(os.path.join(root_dir,"reference_images/offset_0_None",satellite_images[satellite_index]))
    Vis_10_images.append(satellite_bgr)
Vis_10_images=np.hstack(Vis_10_images)
Image.fromarray(cv2.cvtColor(Vis_10_images,cv2.COLOR_BGR2RGB))

# In[14]:


satellite_rank_true=range(satellite_result.shape[0])
plt.scatter(satellite_rank_true,satellite_result, c=satellite_rank_true, cmap='brg')
plt.show()

erro=satellite_rank-satellite_rank_true#Sorting error
num_bins = 10
plt.hist(erro, num_bins)#Error histogram
plt.show()
print("Sorting error:",np.mean(abs(erro)))

# # Image matching
# LoFTR

# In[25]:


from match.src.loftr import LoFTR, default_cfg

# In[26]:


#https://github.com/zju3dv/LoFTR
matcher = LoFTR(config=default_cfg)
matcher.load_state_dict(torch.load("match/weights/outdoor_ds.ckpt")['state_dict'])
matcher = matcher.eval().cuda()

# In[32]:


def eq(m, n):#平均距离
    return np.sqrt(np.sum((m - n) ** 2))
def frame2tensor(frame):
    return torch.from_numpy(frame/255.).float()[None, None].cuda()

Pointer=0#指针,指向下一次可能出现的位置
results_info=[]#实验结果保存
history_predict=[]#历史定位信息

for uav_index in range(len(uav_images)):#对于每个无人机图像
    info=[]
    matchinfo=[]
    uav_path=uav_images[uav_index]
    #gt
    _,_,true_id,_,_=gt[uav_index]
    #read UAV images
    uav_gray=cv2.imread(os.path.join(root_dir,"query_images",uav_path),0)
    uav_gray=cv2.resize(uav_gray,(256,256))
    uav_image_tensor=frame2tensor(uav_gray)
    #取局部卫星图像
    if Pointer>=10:
        local_search=satellite_rank[Pointer-10:Pointer+10]
    else:
        local_search=satellite_rank[0:Pointer+10]

    local_distance=[]
    for satellite_index in local_search:
        satellite_gray=cv2.imread(os.path.join(root_dir,"reference_images/offset_0_None",satellite_images[satellite_index]),0)
        satellite_gray=cv2.resize(satellite_gray,(256,256))
        satellite_image_tensor=frame2tensor(satellite_gray)

        batch={}
        batch['image0']=uav_image_tensor
        batch['image1']=satellite_image_tensor

        # Inference
        with torch.no_grad():
            matcher(batch)    # batch = {'image0': img0, 'image1': img1}
            mkpts0 = batch['mkpts0_f'].cpu().numpy()
            mkpts1 = batch['mkpts1_f'].cpu().numpy()
        M, mask = cv2.findHomography(mkpts0,mkpts1, cv2.RANSAC, 3)#Homography
        mkpts0,mkpts1=mkpts0[np.where(mask[:,0]==1)],mkpts1[np.where(mask[:,0]==1)]

        img1_dims = np.float32([[128, 128]]).reshape(-1, 1, 2)
        # img2_dims = np.float32([[178, 178]]).reshape(-1, 1, 2)
        img1_transform = cv2.perspectiveTransform(img1_dims, M)[0][0]
        # img2_transform = cv2.perspectiveTransform(img2_dims, M)[0][0]
        distance=eq(img1_transform,[128, 128])#offset
        local_distance.append(distance)
    pridict_local_id=np.argmin(local_distance)

    info="UAV ID:{},Ture ID:{},Global ID:{},Distance:{}".format(uav_index,true_id,local_search[pridict_local_id],local_distance[pridict_local_id])
    print(info)
    Pointer=local_search[pridict_local_id]+1
    history_predict.append(Pointer)
    results_info.append([uav_index,true_id,local_search[pridict_local_id],local_distance[pridict_local_id]])

# # Save results

# In[39]:

# 결과 저장
results_dir = "./outputs"
os.makedirs(results_dir, exist_ok=True)

base_name = "results_out"
ext = ".csv"
results_path = os.path.join(results_dir, base_name + ext)

i = 1
while os.path.exists(results_path):
    results_path = os.path.join(results_dir, f"{base_name}_{i}{ext}")
    i += 1

results_out = [["UAV ID", "True ID", "Predict ID", "Distance"]]
results_out.extend(results_info)
results_out = np.array(results_out)

np.savetxt(results_path, results_out, delimiter=',', fmt="%s")
print(f"✅ Results saved to: {results_path}")

# 