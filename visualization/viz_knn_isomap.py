# viz_knn_isomap.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import Isomap

# --- 사용자 파라미터 ---
features_path = "./features/satellite_feature.npy"  # 위성 임베딩 캐시
k = 15                  # k-NN 갯수
highlight_temporal_gap = 50  # 시간적으로 멀리 떨어진 연결만 강조하고 싶으면 임계값 (프레임 차이)

# --- 데이터 로드 ---
F = np.load(features_path)           # shape: (N, D)
N = F.shape[0]
print(f"features: {F.shape}")

# --- Isomap 2D로 레이아웃(시각화용 좌표) ---
iso = Isomap(n_neighbors=max(5, k), n_components=2)
Y = iso.fit_transform(F)             # shape: (N, 2)

# --- k-NN 그래프 (feature 공간 기준) ---
nn = NearestNeighbors(n_neighbors=k+1, metric='euclidean')  # 자기 자신 포함 → +1
nn.fit(F)
dists, idxs = nn.kneighbors(F)       # idxs[i, 0] == i

# --- 노드 색상: 0(초반) → 1(후반) ---
t = np.linspace(0, 1, N)

# --- 플롯 준비 ---
plt.figure(figsize=(10, 8))
ax = plt.gca()
ax.set_title(f"k-NN graph on Isomap(2D)\nN={N}, k={k}")

# 1) 에지 그리기 (연한 회색)
for i in range(N):
    xi, yi = Y[i]
    for j in idxs[i, 1:]:  # 자기 자신 제외
        xj, yj = Y[j]
        # 기본 에지
        ax.plot([xi, xj], [yi, yj], lw=0.5, c=(0, 0, 0, 0.05))

# 2) 시간적으로 먼 연결 강조(빨간색)
if highlight_temporal_gap is not None and highlight_temporal_gap > 0:
    for i in range(N):
        xi, yi = Y[i]
        for j in idxs[i, 1:]:
            if abs(i - j) >= highlight_temporal_gap:
                xj, yj = Y[j]
                ax.plot([xi, xj], [yi, yj], lw=0.9, c=(1, 0, 0, 0.25))

# 3) 노드 산점도: 프레임 진행도 컬러맵(viridis)
sc = ax.scatter(Y[:, 0], Y[:, 1], c=t, s=12, cmap='viridis')
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Frame progress (early → late)")

ax.set_xlabel("Isomap-1")
ax.set_ylabel("Isomap-2")
ax.set_aspect("equal", adjustable="datalim")
plt.tight_layout()

os.makedirs("./outputs", exist_ok=True)
out_path = "./outputs/knn_isomap2d.png"
plt.savefig(out_path, dpi=200)
print(f"Saved: {out_path}")
plt.show()