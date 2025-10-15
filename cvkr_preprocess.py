import os
import csv

# 경로 설정
root_dir = "./data/jeju"
query_dir = os.path.join(root_dir, "query_images")
ref_dir = os.path.join(root_dir, "reference_images/offset_0_None")

output_path = os.path.join(root_dir, "gt_matches.csv")

# 파일 목록 읽기 (정렬 포함)
query_files = sorted([f for f in os.listdir(query_dir) if f.lower().endswith(".jpg")])
ref_files = sorted([f for f in os.listdir(ref_dir) if f.lower().endswith(".jpg")])

# 개수 확인
num_queries = len(query_files)
num_refs = len(ref_files)
print(f"Found {num_queries} UAV images and {num_refs} satellite images")

# CSV 저장
with open(output_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["query_ind", "query_name", "ref_ind", "ref_name", "distance"])

    # 두 폴더 중 작은 쪽 개수 기준으로 매칭
    for i in range(min(num_queries, num_refs)):
        query_ind = i + 1
        ref_ind = i + 1
        query_name = f"{query_ind:08d}.jpg"  # 8자리
        ref_name = f"offset_0_None/{ref_ind:07d}.jpg"  # 7자리
        writer.writerow([query_ind, query_name, ref_ind, ref_name, 0.0])

print(f"✅ Saved ground-truth file to: {output_path}")