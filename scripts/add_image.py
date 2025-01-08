import cv2
import os
import numpy as np

image1_path = "/media/cgl/Mamba/experiments/Failure/case_M_20190112093830_0U62372011292129_1_001_001-1_a10_ayy_image0007.jpg"
image2_path = image1_path.replace("Failure", "Failure/prediction").replace("jpg", "png")

image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

rgba = cv2.cvtColor(image2, cv2.COLOR_BGR2RGBA)

alpha_channel = rgba[:, :, 3]
mask = np.zeros_like(alpha_channel)

mask[alpha_channel == 0] == 255

rgba[:, :, 3] = mask

image2 = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

# import ipdb; ipdb.set_trace()

a_1 = 0.8
a_2 = 1.0

gamma = 0

blend = cv2.addWeighted(image1, a_1, image2, a_2, gamma)

save_path = os.path.join(image1_path.replace(image1_path.split('/')[-1], ""), "result")
os.makedirs(save_path, exist_ok=True)

cv2.imwrite(os.path.join(save_path, image1_path.split('/')[-1]), blend)