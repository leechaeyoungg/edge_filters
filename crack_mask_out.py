import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import frangi
from skimage.exposure import rescale_intensity

# ─── 1. 이미지 불러오기 ─────────────────────────────────
img_path = r"C:\Users\dromii\Downloads\20250710_ori_migum\20250710_tiled_512\images\DJI_0245_tile_y0_x3957.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# ─── 2. 히스토그램 균일화로 미세 대비 증가 ─────────────
img_eq = cv2.equalizeHist(img)

# ─── 3. Top-hat Transform으로 선형 구조 강조 ────────────
tophat = cv2.morphologyEx(img_eq, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17)))

# ─── 4. Frangi 필터로 얇고 희미한 크랙 강조 ─────────────
frangi_resp = frangi(tophat / 255.0, 
                     scale_range=(0.5, 3.0), 
                     scale_step=0.5, 
                     alpha=0.5, beta=0.5, gamma=15)
frangi_img = rescale_intensity(frangi_resp, out_range=(0, 255)).astype(np.uint8)

# ─── 5. LoG (Laplacian of Gaussian)로 더 부드러운 경계 검출 ─
blurred = cv2.GaussianBlur(frangi_img, (5, 5), 1.0)
log = cv2.Laplacian(blurred, cv2.CV_64F)
log = np.abs(log).astype(np.uint8)

# ─── 6. Adaptive Thresholding (약한 반응도 포착) ─────────
mask = cv2.adaptiveThreshold(log, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv2.THRESH_BINARY, blockSize=11, C=2)

# ─── 7. Morphological 연결 (끊긴 크랙 선 연결) ────────────
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

# ─── 8. 결과 시각화 ──────────────────────────────────────
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(img, cmap='gray')

plt.subplot(1, 3, 2)
plt.title("Frangi + LoG")
plt.imshow(log, cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Final Crack Mask")
plt.imshow(closed, cmap='gray')
plt.tight_layout()
plt.show()

# ─── 9. 마스크 저장 (선택) ────────────────────────────────
save_path = img_path.replace("images", "masks").replace(".jpg", "_crackmask_50m.png")
cv2.imwrite(save_path, closed)
