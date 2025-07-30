import cv2
import matplotlib.pyplot as plt

img = cv2.imread(r"C:\Users\dromii\Downloads\20250710_ori_migum\20250710_tiled_512\images\DJI_0242_tile_y384_x4103.jpg", cv2.IMREAD_GRAYSCALE)
eq = cv2.equalizeHist(img)
eq_bgr = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

plt.figure(figsize=(12,6))
plt.subplot(1,2,1); plt.title("Equalized1"); plt.imshow(img, cmap='gray')
plt.subplot(1,2,2); plt.title("Equalized2"); plt.imshow(eq_bgr)
plt.axis('off'); plt.tight_layout(); plt.show()
