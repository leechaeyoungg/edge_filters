import cv2
import numpy as np
import os
from glob import glob

# ─── 1. 경로 설정 ─────────────────────────────────
input_dir = r"C:\Users\dromii\Downloads\20250710_ori_migum\crack_test_dataset"
output_dir = r"C:\Users\dromii\Downloads\20250710_ori_migum\crack_test_mask_equalizeHist"
os.makedirs(output_dir, exist_ok=True)

# ─── 2. 이미지 파일 목록 불러오기 ────────────────
image_paths = glob(os.path.join(input_dir, "*.jpg"))

# ─── 3. 각 이미지에 대해 마스크 생성 및 저장 ─────
for image_path in image_paths:
    # 1) 이미지 로딩 및 equalize
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    eq = cv2.equalizeHist(img)

    # 2) Threshold → 마스크 생성
    _, mask = cv2.threshold(eq, 70, 255, cv2.THRESH_BINARY)

    # 3) Morphology로 선형 연결
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # 4) Connected Component → 노이즈 제거
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    min_area = 30
    clean_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean_mask[labels == i] = 255

    # 5) 저장 경로 생성 및 저장
    filename = os.path.basename(image_path).replace(".jpg", "_equal_mask.png")
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, clean_mask)

print(f"전체 마스크 저장 완료 ({len(image_paths)}개):\n{output_dir}")
