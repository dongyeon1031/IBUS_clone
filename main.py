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
import argparse

# # Load satellite images
parser = argparse.ArgumentParser()
parser.add_argument('--eval-n', type=int, default=0, help="앞 n개만 평가, 0이면 전체 평가")
args = parser.parse_args()

# In[31]:


# root_dir="./data/round2/Val"
# root_dir="./data/NewYorkFly/Val"
root_dir="./data/jeju"
uav_images=os.listdir(os.path.join(root_dir,"query_images"))
uav_images = sorted(uav_images)

# 이미지를 오름차순으로 불러와야하는거 아닐까?
satellite_images=os.listdir(os.path.join(root_dir,"reference_images/offset_0_None"))
satellite_images = sorted(satellite_images)

batch_size=32
transform=BuildTransforms(256)
# uav_dataset=HelicopterUAV(root_dir,False, transform)
satellite_dataset=HelicopterSatellite(root_dir,False, transform)
#
# uav_dataloader= DataLoader(uav_dataset,batch_size=batch_size)
satellite_dataloader= DataLoader(satellite_dataset,batch_size=batch_size)

gt=np.loadtxt(os.path.join(root_dir,"gt_matches.csv"),delimiter=',',dtype=str)[1:,:]

if args.eval_n and args.eval_n > 0:
    max_n = min(args.eval_n, len(uav_images), gt.shape[0])
    uav_images = uav_images[:max_n]
    gt = gt[:max_n, :]
    print(f"[Eval] 앞에서 {max_n}개 프레임만 평가합니다.")
else:
    print(f"[Eval] 전체 {len(uav_images)}개 프레임 평가합니다.")
# ## Build model
# Alexnet

# In[5]:

device_ids = [0, 1]
device = torch.device(f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu')

model_name="alexnet_triplet"
model=build_model(model_name,dropout_p=False).cuda()
# model = build_model(model_name, dropout_p=False)
model = torch.nn.DataParallel(model, device_ids=device_ids).to(device)

# # Extract features

# In[6]:


def visualize_match(uav_gray, sat_gray, mkpts0, mkpts1, mask, save_path, max_draw=300):
    """LoFTR 매칭 결과를 이미지로 저장.
    - uav_gray, sat_gray: (H,W) 그레이스케일, 이미 256x256으로 리사이즈된 것 사용
    - mkpts0, mkpts1: (N,2) float32
    - mask: (N,1) uint8 (RANSAC inlier=1), 없으면 None 가능
    """
    # 컬러로 변환
    img0 = cv2.cvtColor(uav_gray, cv2.COLOR_GRAY2BGR)
    img1 = cv2.cvtColor(sat_gray, cv2.COLOR_GRAY2BGR)

    N = mkpts0.shape[0]
    if N == 0:
        # 매칭 없음 표시만 저장
        vis = cv2.hconcat([img0, img1])
        cv2.putText(vis, "No matches", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, vis)
        return

    # 그릴 인덱스 선택 (최대 max_draw개)
    idxs = np.arange(N)
    if N > max_draw:
        idxs = np.random.choice(N, max_draw, replace=False)

    # KeyPoint / DMatch 리스트로 변환 (drawMatches 사용)
    kp0 = [cv2.KeyPoint(float(mkpts0[i,0]), float(mkpts0[i,1]), 1) for i in idxs]
    kp1 = [cv2.KeyPoint(float(mkpts1[i,0]), float(mkpts1[i,1]), 1) for i in idxs]
    matches = [cv2.DMatch(_queryIdx=j, _trainIdx=j, _distance=0) for j in range(len(idxs))]

    # 전체 매칭 그리기
    vis_all = cv2.drawMatches(img0, kp0, img1, kp1, matches, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # inlier만 별도로 강조(초록), outlier는 빨강 선으로 덧그림
    if mask is not None and mask.size == N:
        inlier_mask = mask.ravel().astype(bool)
        in_idx = idxs[inlier_mask[idxs]] if N > max_draw else idxs[inlier_mask]
        out_idx = idxs[~inlier_mask[idxs]] if N > max_draw else idxs[~inlier_mask]

        # 선 직접 그리기 (좌측 폭)
        w = img0.shape[1]
        def draw_lines(canvas, pick_idx, color, thickness=1):
            for i in pick_idx:
                p0 = (int(mkpts0[i,0]), int(mkpts0[i,1]))
                p1 = (int(mkpts1[i,0]) + w, int(mkpts1[i,1]))
                cv2.line(canvas, p0, p1, color, thickness, cv2.LINE_AA)

        def draw_lines(canvas, pick_idx, color, alpha=0.35):
            """매칭 선을 반투명하게 얇게 표시"""
            overlay = canvas.copy()
            for i in pick_idx:
                p0 = (int(mkpts0[i,0]), int(mkpts0[i,1]))
                p1 = (int(mkpts1[i,0]) + w, int(mkpts1[i,1]))
                cv2.line(overlay, p0, p1, color, 1, cv2.LINE_AA)
            # 반투명하게 합성 (alpha 작을수록 더 얇게)
            cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
        draw_lines(vis_all, out_idx, (0,0,255), 0.35) # outliers (red)
        draw_lines(vis_all, in_idx,  (0,255,0), 0.8) # inliers  (green)

        inlier_ratio = float(inlier_mask.sum()) / float(N)
        text = f"matches={N}, inliers={inlier_mask.sum()} ({inlier_ratio*100:.1f}%)"
        cv2.putText(vis_all, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 4)    # 검정 테두리
        cv2.putText(vis_all, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)  # 흰색 본문

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, vis_all)


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
    isomap = Isomap(n_neighbors=n_neighbors, n_components=1, p=2)   # n_components = 반환할 피처의 차원
    result = isomap.fit_transform(feature)
    return result

# In[13]:


satellite_result=manifold(satellite_feature)#Dimension-reduced features
print("shape: ",satellite_result.shape)
# print("result: ",satellite_result[np.argsort(satellite_result[:,0])])

emb = satellite_result[:, 0]
rank = np.argsort(emb)
print("embedding min/max:", float(emb.min()), float(emb.max()))
print("rank head:", rank[:10])

# ====================== 정렬 순서 시각화 ======================
plt.figure(figsize=(6, 4))
plt.scatter(np.arange(len(emb)), emb,
            c=np.arange(len(emb)), cmap='plasma', s=6)
plt.title("Isomap Embedding vs Original Index")
plt.xlabel("Original Image Index")
plt.ylabel("Isomap 1D Embedding Value")
plt.colorbar(label="Image index (color = order)")
plt.tight_layout()
plt.savefig("./outputs/isomap_scatter.png", dpi=200)
plt.close()
print("✅ Saved: ./outputs/isomap_scatter.png")

# In[16]:

# ===================== 랭크 강제 정렬 =====================
# satellite_rank=np.argsort(satellite_result[:,0])#Satellite sort result
satellite_rank = np.arange(len(satellite_images), dtype=int)

# print("rank: ",satellite_rank)
np.savetxt("./outputs/rank_debug.txt", satellite_rank, fmt="%d")
satellite_images = np.array(satellite_images)[satellite_rank]
# print("image index: ",satellite_images)



# In[24]:

# 시각화
# Vis_10_images=[]
# for i in range(10):
#     satellite_index=satellite_rank[i]
#     satellite_bgr=cv2.imread(os.path.join(root_dir,"reference_images/offset_0_None",satellite_images[satellite_index]))
#     Vis_10_images.append(satellite_bgr)
# Vis_10_images=np.hstack(Vis_10_images)
# Image.fromarray(cv2.cvtColor(Vis_10_images,cv2.COLOR_BGR2RGB))

# In[14]:


satellite_rank_true=range(satellite_result.shape[0])
plt.scatter(satellite_rank_true,satellite_result, c=satellite_rank_true, cmap='brg')
plt.show()
print("rank true: ",satellite_rank_true[0],satellite_rank_true[1],satellite_rank_true[2],satellite_rank_true[3],satellite_rank_true[4])

erro=satellite_rank-satellite_rank_true#Sorting error
num_bins = 10
plt.hist(erro, num_bins)#Error histogram
plt.show()
print("Sorting error:",np.mean(abs(erro)))

# # Image matching
# LoFTR

from match.src.loftr import LoFTR, default_cfg

#https://github.com/zju3dv/LoFTR
matcher = LoFTR(config=default_cfg)
matcher.load_state_dict(torch.load("match/weights/outdoor_ds.ckpt")['state_dict'])
matcher = matcher.eval().cuda()

# In[32]:


def eq(m, n):#平均距离
    return np.sqrt(np.sum((m - n) ** 2))
def frame2tensor(frame):
    return torch.from_numpy(frame/255.).float()[None, None].cuda()

# Pointer=0
last_chosen_idx = 0   # 첫 프레임 이전 상태라고 가정(0-based)
pointer_pos = 0 # satellite_rank의 위치(0-based)
L = 10 # 좌우 창 크기 → 총 2L+1개
results_info=[]
history_predict=[]
top_n = 5

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
    lo = max(0, pointer_pos - L)
    hi = min(len(satellite_rank), pointer_pos + L + 1)  # +1 해야 포괄적 슬라이스

    local_pos  = np.arange(lo, hi) # 창의 '위치'들
    local_inds = satellite_rank[local_pos] # 실제 이미지 인덱스(값)

    # 윈도우 어떻게 구성하는지 찍기
    # win_inds  = np.array(local_inds, dtype=int)
    # win_files = [satellite_images[i] for i in win_inds]
    # print(f"[UAV {uav_index:04d}] pos={pointer_pos:4d} | window_pos={local_pos.tolist()} | window_idx={win_inds.tolist()}")
    

    local_distance=[]
    for satellite_index in local_inds:
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

        # # ==== 시각화: 매칭 시각화 저장 ====
        # # uav_gray / satellite_gray는 이미 256x256으로 리사이즈한 그레이 이미지
        # save_dir = "./outputs/match_viz"
        # save_name = f"uav{uav_index:04d}_cand{int(satellite_index):04d}.png"
        # visualize_match(uav_gray, satellite_gray, mkpts0, mkpts1, mask, os.path.join(save_dir, save_name))
        # # ===================================== 

        mkpts0,mkpts1=mkpts0[np.where(mask[:,0]==1)],mkpts1[np.where(mask[:,0]==1)]

        img1_dims = np.float32([[128, 128]]).reshape(-1, 1, 2)
        # img2_dims = np.float32([[178, 178]]).reshape(-1, 1, 2)
        img1_transform = cv2.perspectiveTransform(img1_dims, M)[0][0]
        # img2_transform = cv2.perspectiveTransform(img2_dims, M)[0][0]
        distance=eq(img1_transform,[128, 128])#offset
        local_distance.append(distance)
    # 거리 오름차순 정렬로 Top-n 후보 선정
    # order = np.argsort(local_distance)[:top_n]
    # ===== 관측비용 정규화 =====
    obs = np.array(local_distance, dtype=np.float32)
    finite_mask = np.isfinite(obs)
    if not finite_mask.any():
        # 전부 실패면 큰 값으로 처리
        obs_norm = np.ones_like(obs, dtype=np.float32)
    else:
        m = float(np.nanmin(obs[finite_mask]))
        M = float(np.nanmax(obs[finite_mask]))
        obs_norm = (obs - m) / (M - m + 1e-6)
        obs_norm[~finite_mask] = 1.0  # 실패는 최악 점수

    # ===== 방향성(전이) 패널티 =====
    # prev_idx: 직전 선택된 실제 인덱스(0-based)
    prev_idx = 0 if uav_index == 0 else last_chosen_idx

    # 하이퍼파라미터 (원하는 방향성 강도에 맞춰 조정)
    exp_step = 1   # 기대 전진량(보통 1)
    tol      = 2   # 허용 여유 (±tol 이내는 패널티 작게)
    alpha    = 0.2 # 과대/과소 점프 패널티 가중치
    beta     = 0.8 # 역진(backward) 패널티 가중치 (뒤로 가면 크게)

    scores = []
    for j, cand_idx in enumerate(local_inds):
        delta    = int(cand_idx) - int(prev_idx)          # 얼마나 전/후진했는가
        back_pen = max(0, -delta)                          # 뒤로 가면 양수
        jump_pen = max(0, abs(delta - exp_step) - tol)     # 기대 step에서 벗어난 점프
        trans    = alpha * jump_pen + beta * back_pen
        score    = float(obs_norm[j]) + float(trans)       # 관측비용 + 방향성 패널티
        scores.append(score)
    scores = np.array(scores, dtype=np.float32)

    # ===== Top-k 및 최종 선택 =====
    order = np.argsort(scores)[:top_n]

    # 실제 이미지 인덱스(값)들에서 top-k 뽑기
    top_n_idx   = [int(local_inds[j])         for j in order] # 값(파일 인덱스)
    top_n_dists = [float(local_distance[j])   for j in order]

    # 최종 선택
    chosen_idx = int(local_inds[order[0]])    # 실제 인덱스(값)
    chosen_fn  = satellite_images[chosen_idx]

    info = f"UAV ID:{uav_index},True ID:{true_id},Global ID:{chosen_idx},file Name:{chosen_fn},Distance:{local_distance[order[0]]}"
    print(info)
    # Pointer=local_search[pridict_local_id]+1
    pointer_pos = int(np.where(satellite_rank == chosen_idx)[0][0]) + 1
    pointer_pos = min(pointer_pos, len(satellite_rank) - 1)  # 경계 보호
    last_chosen_idx = chosen_idx
    history_predict.append(pointer_pos)

    top_n_ids_str   = ";".join(map(str, top_n_idx))
    top_n_dists_str = ";".join(f"{d:.6f}" for d in top_n_dists)

    results_info.append([
        uav_index,
        true_id,
        chosen_idx,                       # Predict ID
        float(local_distance[order[0]]),  # Distance
        top_n_ids_str,
        top_n_dists_str
    ])


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

results_out = [["UAV ID", "True ID", "Predict ID", "Distance", "Top5 IDs", "Top5 Distances"]]
results_out.extend(results_info)
results_out = np.array(results_out)

np.savetxt(results_path, results_out, delimiter=',', fmt="%s")
print(f"✅ Results saved to: {results_path}")

# 

