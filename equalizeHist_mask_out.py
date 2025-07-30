import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ─── 1. 이미지 로딩 ───────────────────────────────
image_path = r"C:\Users\dromii\Downloads\20250710_ori_migum\20250710_tiled_512\images\DJI_0242_tile_y384_x4103.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# ─── 2. Equalized 이미지 생성 ────────────────────
eq = cv2.equalizeHist(img)                    # Equalized1
eq_bgr = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR) # Equalized2 (시각화용 컬러)

# ─── 3. Threshold 적용 → 마스크 생성 ───────────────
_, mask1 = cv2.threshold(eq, 120, 255, cv2.THRESH_BINARY)  # 강한 영역만
_, mask2 = cv2.threshold(eq, 70, 255, cv2.THRESH_BINARY)   # 약한 경계까지 포함

# ─── 4. Morphology 연결 (선형 연결, 잡음 제거) ─────
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
mask1_closed = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)
mask2_closed = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)

# ─── 5. 시각화 ───────────────────────────────────
plt.figure(figsize=(14, 6))
plt.subplot(2, 3, 1); plt.title("Original"); plt.imshow(img, cmap='gray')
plt.subplot(2, 3, 2); plt.title("Equalized"); plt.imshow(eq, cmap='gray')
plt.subplot(2, 3, 3); plt.title("Threshold >120"); plt.imshow(mask1_closed, cmap='gray')
plt.subplot(2, 3, 5); plt.title("Threshold >70"); plt.imshow(mask2_closed, cmap='gray')
plt.subplot(2, 3, 6); plt.title("Overlay"); plt.imshow(cv2.addWeighted(eq_bgr, 0.7, cv2.cvtColor(mask2_closed, cv2.COLOR_GRAY2BGR), 0.3, 0))
plt.tight_layout(); plt.show()

# ─── 6. 저장 (선택: 세그멘테이션 라벨용 마스크) ─────
output_mask_path = image_path.replace("images", "masks").replace(".jpg", "_eqmask.png")
os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
cv2.imwrite(output_mask_path, mask1_closed)
