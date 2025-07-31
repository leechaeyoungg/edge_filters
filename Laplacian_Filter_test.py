import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 로딩
image_path = r"C:\Users\dromii\Downloads\20250710_ori_migum\20250710_tiled_512\images\DJI_0894_tile_y1979_x2688.jpg"
img = cv2.imread(image_path)
laplacian = cv2.Laplacian(img, ddepth=cv2.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))

# 시각화
plt.figure(figsize=(12,6))
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Laplacian Edge")
plt.imshow(cv2.cvtColor(laplacian, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.tight_layout()
plt.show()
