import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

img_path = r"F:\11\1.jpg"
img_origin = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
img_gray = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)

# 1 灰度变换+均衡化
img_trans = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
img_equal = cv2.equalizeHist(img_trans)

# 2 三类滤波
blur_mean = cv2.blur(img_equal, (5, 5))
blur_gauss = cv2.GaussianBlur(img_equal, (5, 5), 1.6)
blur_median = cv2.medianBlur(img_equal, 5)

# 3 二值分割
_, img_binary = cv2.threshold(blur_median, 70, 255, cv2.THRESH_BINARY_INV)
kernel = np.ones((2, 2), np.uint8)
img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel)

# 4 Canny边缘
edge_canny = cv2.Canny(img_binary, 50, 180)

# 5 最终凸显成品：纯灰度图，**不绘制任何轮廓线条**
img_show_final = blur_median

# SNR计算
def calc_snr(ori, den):
    ori_f = ori.astype(np.float32)
    den_f = den.astype(np.float32)
    signal = np.sum(den_f ** 2)
    noise = np.sum((ori_f - den_f) ** 2)
    return round(10 * np.log10(signal / noise), 2)

snr_mean = calc_snr(img_gray, blur_mean)
snr_gauss = calc_snr(img_gray, blur_gauss)
snr_median = calc_snr(img_gray, blur_median)

# 九宫格绘图布局
plt.figure(figsize=(16, 12), dpi=100)

# 第一行
plt.subplot(3, 3, 1)
plt.imshow(img_origin[:, :, ::-1])
plt.title("原始焊缝X光图")
plt.axis("off")

plt.subplot(3, 3, 2)
plt.imshow(img_trans, cmap="gray")
plt.title("灰度变换图")
plt.axis("off")

plt.subplot(3, 3, 3)
plt.imshow(img_equal, cmap="gray")
plt.title("直方图均衡化（去除暗区噪声）")
plt.axis("off")

# 第二行
plt.subplot(3, 3, 4)
plt.imshow(blur_mean, cmap="gray")
plt.title(f"均值滤波\nSNR:{snr_mean}dB")
plt.axis("off")

plt.subplot(3, 3, 5)
plt.imshow(blur_gauss, cmap="gray")
plt.title(f"高斯滤波\nSNR:{snr_gauss}dB")
plt.axis("off")

plt.subplot(3, 3, 6)
plt.imshow(blur_median, cmap="gray")
plt.title(f"中值滤波（去除颗粒噪声）\nSNR:{snr_median}dB")
plt.axis("off")

# 第三行
plt.subplot(3, 3, 7)
plt.imshow(edge_canny, cmap="gray")
plt.title("Canny边缘提取")
plt.axis("off")

plt.subplot(3, 3, 8)
plt.imshow(img_binary, cmap="gray")
plt.title("二值分割图")
plt.axis("off")

# 最后一张：纯灰度自然凸显缺陷，无任何标注线条
plt.subplot(3, 3, 9)
plt.imshow(img_show_final, cmap="gray")
plt.title("原图凸显裂纹、气孔、夹渣")
plt.axis("off")

plt.tight_layout()
plt.show()

# 批量保存全部图片
cv2.imwrite("总_灰度变换.png", img_trans)
cv2.imwrite("总_均衡化暗区降噪.png", img_equal)
cv2.imwrite("总_均值滤波.png", blur_mean)
cv2.imwrite("总_高斯滤波.png", blur_gauss)
cv2.imwrite("总_中值滤波.png", blur_median)
cv2.imwrite("总_Canny边缘.png", edge_canny)
cv2.imwrite("总_二值分割图.png", img_binary)
cv2.imwrite("总_最终缺陷凸显图.png", img_show_final)

print(f"SNR：均值{snr_mean}dB 高斯{snr_gauss}dB 中值{snr_median}dB")
print("执行完毕：全程无轮廓画线，最后一张靠灰度明暗自然凸显焊接缺陷")
