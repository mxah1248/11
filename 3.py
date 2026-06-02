import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

img_path = r"D:\tuxiangkeshe\11\1.jpg"
img_origin = cv2.imdecode(np.fromfile(
    img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
img_gray = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)
img_trans = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
img_equal = cv2.equalizeHist(img_trans)
img_denoise = cv2.medianBlur(img_equal, 5)

_, img_binary = cv2.threshold(img_denoise, 70, 255, cv2.THRESH_BINARY_INV)
kernel = np.ones((2, 2), np.uint8)
img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel)

edge_canny = cv2.Canny(img_binary, 50, 180)

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.imshow(img_binary, cmap="gray")
plt.title("二值分割图")
plt.axis("off")

plt.subplot(122)
plt.imshow(edge_canny, cmap="gray")
plt.title("Canny边缘提取")
plt.axis("off")

plt.tight_layout()
plt.show()
cv2.imwrite("05_Canny边缘图.png", edge_canny)
