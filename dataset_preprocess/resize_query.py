import os
import cv2
from tqdm import tqdm

# 경로 설정
input_dir = "../data/jeju/query_images"
output_dir = "../data/jeju/query_images_resized"
os.makedirs(output_dir, exist_ok=True)

# 타깃 해상도
target_size = (500, 500)

# 지원 확장자
valid_exts = [".jpg", ".png", ".jpeg"]

# 이미지 파일 목록
images = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in valid_exts]
images.sort()

print(f"총 {len(images)}개 이미지 리사이즈 시작...")

for img_name in tqdm(images):
    input_path = os.path.join(input_dir, img_name)
    output_path = os.path.join(output_dir, img_name)

    # 이미지 읽기
    img = cv2.imread(input_path)
    if img is None:
        print(f"[경고] {img_name} 읽기 실패, 건너뜀")
        continue

    # 다운샘플링
    resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    # 저장
    cv2.imwrite(output_path, resized)

print(f"✅ 모든 이미지가 {output_dir} 폴더에 {target_size} 크기로 저장되었습니다.")