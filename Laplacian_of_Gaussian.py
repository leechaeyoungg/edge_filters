import cv2
import numpy as np
import matplotlib.pyplot as plt

# ─── 1. 이미지 경로 및 불러오기 ──────────────
image_path = r"C:\Users\dromii\Downloads\20250710_ori_migum\20250710_tiled_512\images\DJI_0245_tile_y0_x3957.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# ─── 2. Gaussian Blur로 부드럽게 (노이즈 제거) ─
blurred = cv2.GaussianBlur(img, (5, 5), sigmaX=1.0)

# ─── 3. Laplacian 필터로 경계 강조 ─────────────
log = cv2.Laplacian(blurred, cv2.CV_64F)
log_abs = np.uint8(np.absolute(log))  # 음수 제거

# ─── 4. 이진화 (임계값 조절 가능) ───────────────
_, edge_mask = cv2.threshold(log_abs, 10, 255, cv2.THRESH_BINARY)

# ─── 5. 시각화 ──────────────────────────────
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(img, cmap='gray')

plt.subplot(1, 3, 2)
plt.title("Laplacian of Gaussian")
plt.imshow(log_abs, cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Edge Mask")
plt.imshow(edge_mask, cmap='gray')
plt.tight_layout()
plt.show()
