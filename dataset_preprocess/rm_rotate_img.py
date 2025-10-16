import os
import re
import glob
import csv
from pathlib import Path

# === 설정 ===
root_dir = "../data/jeju"   # 작업할 루트 폴더
query_dir = Path(root_dir) / "query_images"
ref_dir   = Path(root_dir) / "reference_images" / "offset_0_None"
gt_path   = Path(root_dir) / "gt_matches.csv"

# 패딩 자릿수(라운드2 스타일이면 보통 6자리: 000000.png)
QUERY_PAD = 6
REF_PAD   = 6

# 실제 적용 여부 (True면 rename/CSV 덮어씀, False면 드라이런)
APPLY = True

# === 유틸 ===
def numeric_stem(p: Path):
    """파일명에서 숫자부분 추출 (없으면 None)"""
    m = re.search(r'(\d+)$', p.stem)
    return int(m.group(1)) if m else None

def collect_images(folder: Path):
    exts = ("*.jpg", "*.jpeg", "*.png")
    files = []
    for pat in exts:
        files.extend(sorted(Path(folder).glob(pat)))
    # 숫자 인덱스가 있는 파일만 남기고 숫자로 정렬
    files = [p for p in files if numeric_stem(p) is not None]
    files.sort(key=lambda p: numeric_stem(p))
    return files

def two_phase_rename(rename_pairs):
    """충돌 방지용 두 단계 rename"""
    temp_suffix = "__tmp__"
    # 1단계: 임시 이름으로
    for src, dst in rename_pairs:
        tmp = src.with_name(src.name + temp_suffix)
        if APPLY:
            src.rename(tmp)
    # 2단계: 최종 이름으로
    for src, dst in rename_pairs:
        tmp = src.with_name(src.name + temp_suffix)
        if APPLY:
            tmp.rename(dst)

# === 1) 남은 이미지 수집(이미 삭제된 상태라고 가정) ===
q_files = collect_images(query_dir)
r_files = collect_images(ref_dir)

if len(q_files) != len(r_files):
    print(f"[경고] 쿼리({len(q_files)})와 레퍼런스({len(r_files)}) 개수가 다릅니다. 1:1 매칭 전제라면 확인하세요.")

N = min(len(q_files), len(r_files))
print(f"[INFO] 남은 파일 수(최소 기준): {N}")

# === 2) 새로운 연속 인덱스 부여: 1..N ===
#    - 확장자는 기존 파일 확장자 유지
#    - 패딩만 설정값에 맞게 적용
rename_q = []
rename_r = []
new_query_names = []  # CSV용
new_ref_names   = []  # CSV용 (상대경로: offset_0_None/XXXXX.ext)

for new_idx in range(1, N + 1):
    q_old = q_files[new_idx - 1]
    r_old = r_files[new_idx - 1]

    q_ext = q_old.suffix.lower()
    r_ext = r_old.suffix.lower()

    q_new = q_old.with_name(f"{str(new_idx).zfill(QUERY_PAD)}{q_ext}")
    r_new = r_old.with_name(f"{str(new_idx).zfill(REF_PAD)}{r_ext}")

    rename_q.append((q_old, q_new))
    rename_r.append((r_old, r_new))

    new_query_names.append(q_new.name)                         # 예: 000001.jpg
    new_ref_names.append(f"offset_0_None/{r_new.name}")        # 예: offset_0_None/000001.jpg

# === 3) 파일명 실제 변경 ===
print("[DRY-RUN]" if not APPLY else "[APPLY]", "쿼리/레퍼런스 이름 땡기기 & 패딩 통일")
print(f"  예시: {rename_q[0][0].name} -> {rename_q[0][1].name}")
print(f"  예시: {rename_r[0][0].name} -> {rename_r[0][1].name}")

two_phase_rename(rename_q)
two_phase_rename(rename_r)

# === 4) CSV 재작성 ===
# 형식: query_ind,query_name,ref_ind,ref_name,distance
# 인덱스는 1부터 시작 (원래 round2 스타일과 맞춤)
gt_rows = []
gt_rows.append(["query_ind","query_name","ref_ind","ref_name","distance"])
for i in range(1, N + 1):
    gt_rows.append([i, new_query_names[i-1], i, new_ref_names[i-1], 0.0])

if APPLY:
    with open(gt_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(gt_rows)
print(f"[INFO] gt_matches.csv 갱신 완료: {gt_path}")

# === 5) 캐시 무효화 안내 ===
print("[TIP] features/satellite_feature.npy 가 있다면 삭제하세요. (이름/순서가 바뀌었기 때문)")