import cv2
import matplotlib.pyplot as plt

# ─── 1. 이미지 경로 설정 ───────────────────────
image_path = r"C:\Users\dromii\Downloads\20250710_ori_migum\20250710_tiled_512\images\DJI_0245_tile_y0_x3957.jpg"

# ─── 2. 이미지 불러오기 (그레이스케일) ───────────
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# ─── 3. Canny Edge Detection ───────────────────
edges = cv2.Canny(img, threshold1=60, threshold2=150)

# ─── 4. 결과 시각화 ─────────────────────────────
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Canny Edge")
plt.imshow(edges, cmap='gray')
plt.tight_layout()
plt.show()

# ─── 5. 저장 (선택) ─────────────────────────────
save_path = image_path.replace("images", "edges").replace(".jpg", "_canny.png")
cv2.imwrite(save_path, edges)
