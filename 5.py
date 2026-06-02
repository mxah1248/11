import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

img_path = r"F:\11\1.jpg"
img_origin = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
img_gray = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)

# 完整预处理流水线：灰度变换→均衡化→中值降噪
img_trans = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
img_equal = cv2.equalizeHist(img_trans)
img_final = cv2.medianBlur(img_equal, 5)

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.imshow(img_gray, cmap="gray")
plt.title("原始灰度图")
plt.axis("off")

plt.subplot(122)
plt.imshow(img_final, cmap="gray")
plt.title("灰度增强凸显：裂纹、气孔、夹渣")
plt.axis("off")

plt.tight_layout()
plt.show()
cv2.imwrite("灰度增强图.png", img_final)
