import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def kendall_tau_slow(true_idx, pred_idx):
    '''
    Permutation Plot
    - x(실제 인덱스)와 y(예측 인덱스)를 y=x 그래프를 기준으로 벗어난 정도를 통해 예측 오차를 확인하는 그래프를 그림
    '''
    # n이 수천이면 O(n^2)도 충분. 더 크면 scipy.stats.kendalltau 쓰면 됨.
    n = len(true_idx)
    inv = 0
    for i in range(n):
        for j in range(i+1, n):
            a = true_idx[i] - true_idx[j]
            b = pred_idx[i] - pred_idx[j]
            if a*b < 0:
                inv += 1
    total = n*(n-1)//2
    tau = 1 - 2*inv/total if total>0 else 1.0
    return tau

def plot_permutation(true_idx, pred_idx, save_path):
    true_idx = np.asarray(true_idx, dtype=int)
    pred_idx = np.asarray(pred_idx, dtype=int)

    plt.figure(figsize=(6,6))
    sc = plt.scatter(true_idx, pred_idx, c=np.arange(len(true_idx)), s=8, cmap='viridis')
    mx = max(true_idx.max(), pred_idx.max()) + 1
    plt.plot([0, mx], [0, mx], 'r--', linewidth=1, label='y=x')
    tau = kendall_tau_slow(true_idx, pred_idx)
    plt.title(f'Permutation Plot  (Kendall τ={tau:.3f})')
    plt.xlabel('True index')
    plt.ylabel('Predicted index')
    plt.colorbar(sc, label='frame order')
    plt.legend(loc='lower right')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# ===== Top-k 후보 요약 패널 저장 =====
def save_topk_panel(root_dir,
                    uav_bgr,
                    cand_indices,                # list[int] (global index)
                    cand_files,                  # list[str] (filename)
                    dists,                       # list[float]
                    scores,                      # list[float]
                    inliers,                     # list[int]
                    totals,                      # list[int]
                    chosen_idx,                  # int (global index)
                    gt_idx0,                     # int or None (0-based GT), 모르면 None
                    save_path,                   # str
                    tile_size=256):
    """Top-k 후보 요약 패널을 저장"""

    pad = 8
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 왼쪽 UAV 타일
    uav_tile = cv2.resize(uav_bgr, (tile_size, tile_size))
    cv2.putText(uav_tile, f"UAV (GT={gt_idx0})", (10, 24), font, 0.7, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(uav_tile, f"UAV (GT={gt_idx0})", (10, 24), font, 0.7, (255,255,255), 2, cv2.LINE_AA)

    # 오른쪽 후보 타일들
    tiles = []

    order_idx = np.argsort(cand_indices)  # ID 오름차순
    cand_indices = [cand_indices[i] for i in order_idx]
    cand_files   = [cand_files[i]   for i in order_idx]
    dists        = [dists[i]        for i in order_idx]
    scores       = [scores[i]       for i in order_idx]
    inliers      = [inliers[i]      for i in order_idx]
    totals       = [totals[i]       for i in order_idx]
    for idx, f, dist, sc, inl, tot in zip(cand_indices, cand_files, dists, scores, inliers, totals):
        sat_bgr = cv2.imread(os.path.join(root_dir, "reference_images/offset_0_None", f))
        if sat_bgr is None:
            sat_bgr = np.zeros((tile_size, tile_size, 3), np.uint8)
        sat_bgr = cv2.resize(sat_bgr, (tile_size, tile_size))

        # 텍스트 오버레이
        label1 = f"ID:{idx}  score:{sc:.3f}"
        label2 = f"dist:{dist:.2f}  n_inl:{inl}/{tot}"
        cv2.putText(sat_bgr, label1, (10, 24), font, 0.55, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(sat_bgr, label1, (10, 24), font, 0.55, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(sat_bgr, label2, (10, 48), font, 0.55, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(sat_bgr, label2, (10, 48), font, 0.55, (255,255,255), 2, cv2.LINE_AA)

        # 테두리(선택/GT 강조)
        border_color = (200,200,200)
        thick = 2
        if idx == chosen_idx:
            border_color = (255,0,0)     # 선택: 초록
            thick = 3
        if gt_idx0 is not None and idx == gt_idx0:
            border_color = (255,128,0) if idx==chosen_idx else (0,255,0)  # GT: 파랑(동시에 선택이면 청록/주황 적당히)
            thick = 3

        cv2.rectangle(sat_bgr, (2,2), (tile_size-3, tile_size-3), border_color, thick, cv2.LINE_AA)
        tiles.append(sat_bgr)

    # 그리드 구성(Top-k를 가로 한 줄로)
    # [UAV | cand1 | cand2 | ...]
    panel_w = tile_size * (1 + len(tiles)) + pad * (len(tiles))  # UAV 1장 + 후보 n장, 타일 사이 pad
    panel_h = tile_size
    panel = np.full((panel_h, panel_w, 3), 20, np.uint8)

    # UAV 배치
    x = 0
    panel[0:tile_size, x:x+tile_size] = uav_tile
    x += tile_size
    # 후보 배치
    for t in tiles:
        x += pad
        panel[0:tile_size, x:x+tile_size] = t
        x += tile_size

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, panel)



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