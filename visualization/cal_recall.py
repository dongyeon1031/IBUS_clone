import pandas as pd
import numpy as np
import math
from pathlib import Path

# ===== 파일 경로 =====
RESULT_CSV = Path("../outputs/results_out_카카오.csv")
METADATA_CSV = Path("./metadata_kakao.csv")

# ===== 유틸 =====
def haversine_m(lat1, lon1, lat2, lon2):
    """위경도(도 단위) → 거리(m) 계산"""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def find_col(df, candidates):
    norm = {c.strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in norm:
            return norm[key]
    raise KeyError(f"필요한 컬럼을 찾지 못했어요: {candidates}")

# ===== 메타데이터 로드 =====
meta = pd.read_csv(METADATA_CSV)

id_col  = find_col(meta, ["id", "frame_id", "uid"])
lat_col = find_col(meta, ["latitude", "lat"])
lon_col = find_col(meta, ["longitude", "lon", "lng"])

meta_map = (
    meta.dropna(subset=[id_col, lat_col, lon_col])
        .assign(_id=lambda d: d[id_col].astype(str))
        .set_index("_id")[[lat_col, lon_col]]
        .to_dict(orient="index")
)

# ===== 결과 로드 =====
df = pd.read_csv(RESULT_CSV)

true_id_col = find_col(df, ["True ID", "true id", "true_id", "gt_id"])
pred_id_col = find_col(df, ["Predict ID", "predict id", "pred_id"])
top5_col    = find_col(df, ["Top5 IDs", "top5 ids", "top5_ids"])

# ===== Top5 파싱 =====
def parse_top5(s):
    return [int(x) for x in str(s).split(";") if x.isdigit()]

df["Top5_List"] = df[top5_col].apply(parse_top5)

# ===== R@1 / R@5 계산 =====
r1 = (df[true_id_col].astype(int) == df[pred_id_col].astype(int)).mean()
r5 = df.apply(lambda row: int(row[true_id_col]) in row["Top5_List"], axis=1).mean()

# ===== 좌표 거리 계산 =====
def id_to_coords(_id):
    if pd.isna(_id):
        return None
    rec = meta_map.get(str(int(_id)) if str(_id).isdigit() else str(_id))
    if rec is None:
        return None
    return rec[lat_col], rec[lon_col]

true_coords = df[true_id_col].apply(id_to_coords)
pred_coords = df[pred_id_col].apply(id_to_coords)

def row_distance(tp):
    tc, pc = tp
    if (tc is None) or (pc is None):
        return np.nan
    (tlat, tlon), (plat, plon) = tc, pc
    return haversine_m(tlat, tlon, plat, plon)

df["distance_m"] = list(map(row_distance, zip(true_coords, pred_coords)))

# ===== 거리 통계 =====
mean_distance = df["distance_m"].mean(skipna=True)
within_10m_ratio = (df["distance_m"] <= 10).mean(skipna=True)

# ===== 결과 저장 및 출력 =====
print(f"🔹 R@1\t\t\t= {r1 * 100:.2f}%")
print(f"🔹 R@5\t\t\t= {r5 * 100:.2f}%")
print(f"🔹 평균 거리 차이\t= {mean_distance:.3f} m")
print(f"🔹 10m 이내 비율\t= {within_10m_ratio * 100:.2f}%")
print(f"🔹 샘플 수\t\t= {len(df)}")