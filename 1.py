import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

img_path = r"E:\11\1.jpg"
img_origin = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
img_gray = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)

# 灰度变换拉伸，压制暗区噪声
img_gray_trans = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
img_equal = cv2.equalizeHist(img_gray_trans)

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.imshow(img_gray, cmap="gray")
plt.title("原始灰度图")
plt.axis("off")

plt.subplot(122)
plt.imshow(img_equal, cmap="gray")
plt.title("灰度变换+均衡化（去除暗区噪声）")
plt.axis("off")

plt.tight_layout()
plt.show()
cv2.imwrite("01_灰度均衡增强图.png", img_equal)