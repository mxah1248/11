import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

def calc_snr(ori, den):
    ori_f = ori.astype(np.float32)
    den_f = den.astype(np.float32)
    signal = np.sum(den_f ** 2)
    noise = np.sum((ori_f - den_f) ** 2)
    return round(10 * np.log10(signal / noise), 2)

img_path = r"./data/1.png"
img_origin = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
img_gray = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)
img_trans = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
img_equal = cv2.equalizeHist(img_trans)

blur_mean = cv2.blur(img_equal, (5, 5))
blur_gauss = cv2.GaussianBlur(img_equal, (5, 5), 1.6)
blur_median = cv2.medianBlur(img_equal, 5)

snr_mean = calc_snr(img_gray, blur_mean)
snr_gauss = calc_snr(img_gray, blur_gauss)
snr_median = calc_snr(img_gray, blur_median)

plt.figure(figsize=(14, 7))
plt.subplot(221)
plt.imshow(img_equal, cmap="gray")
plt.title("灰度增强原图")
plt.axis("off")

plt.subplot(222)
plt.imshow(blur_mean, cmap="gray")
plt.title(f"均值滤波\nSNR:{snr_mean}dB")
plt.axis("off")

plt.subplot(223)
plt.imshow(blur_gauss, cmap="gray")
plt.title(f"高斯滤波\nSNR:{snr_gauss}dB")
plt.axis("off")

plt.subplot(224)
plt.imshow(blur_median, cmap="gray")
plt.title(f"中值滤波（去除颗粒噪声）\nSNR:{snr_median}dB")
plt.axis("off")

plt.tight_layout()
plt.show()

cv2.imwrite("02_均值滤波.png", blur_mean)
cv2.imwrite("03_高斯滤波.png", blur_gauss)
cv2.imwrite("04_中值滤波.png", blur_median)
print(f"均值滤波SNR:{snr_mean}dB  高斯:{snr_gauss}dB  中值:{snr_median}dB")
