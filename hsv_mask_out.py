import cv2
import numpy as np

# 1. 원본 이미지 로딩 (컬러)
image_path = r"C:\Users\dromii\Downloads\20250710_ori_migum\20250710_tiled_512\images\DJI_0894_tile_y1979_x2688.jpg"
img_bgr = cv2.imread(image_path)
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

# 2. 채널 분리
h, s, v = cv2.split(img_hsv)

# 3. Value 채널에 히스토그램 평활화 + Denoising
v_eq = cv2.equalizeHist(v)
v_denoised = cv2.fastNlMeansDenoising(v_eq, h=10)

# 4. Adaptive Threshold or HSV 조건으로 크랙 마스크 생성
# ex: V값 낮고 S값 낮은 영역을 크랙으로 가정
crack_mask = cv2.inRange(img_hsv, (0, 0, 0), (180, 60, 140))  # 조건은 조정 가능

# 5. Morphology 후처리
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
crack_mask = cv2.morphologyEx(crack_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

cv2.imwrite("C:/Users/dromii/Downloads/hsv_crack_mask.png", crack_mask)
