import pandas as pd
import os

# === 경로 설정 ===
root_dir = "./data/jeju"  # 상황에 맞게 수정
gt_path   = os.path.join(root_dir, "gt_matches.csv")
meta_path = os.path.join(root_dir, "metadata.csv")
out_path  = os.path.join(root_dir, "gt_matches_with_coords.csv")

# === 1. CSV 불러오기 ===
gt   = pd.read_csv(gt_path)         # query_ind, query_name, ref_ind, ref_name, distance
meta = pd.read_csv(meta_path)       # frame_id, image_name, latitude, longitude, ...

# === 2. 타입 정리 ===
gt["query_ind"]   = gt["query_ind"].astype(int)
gt["ref_ind"]     = gt["ref_ind"].astype(int)
meta["frame_id"]  = meta["frame_id"].astype(int)

# === 3. 조인: query_ind == frame_id ===
merged = gt.merge(
    meta[["frame_id", "latitude", "longitude"]],
    left_on="query_ind",
    right_on="frame_id",
    how="left",
)

# === 6. 불필요한 열 정리 ===
merged = merged.drop(columns=["frame_id"])

# === 7. 저장 ===
merged.to_csv(out_path, index=False)
print(f"✅ Saved merged GT file with coordinates to: {out_path}")
print(merged.head())