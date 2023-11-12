from PIL import Image

import pandas as pd
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt

results = pd.read_csv("ISIC2018_Task3_Training_GroundTruth.csv")
df_MEL = results[["image", "MEL"]].loc[results["MEL"] == 1]
df_NV = results[["image", "NV"]].loc[results["NV"] == 1]
df_BCC = results[["image", "BCC"]].loc[results["BCC"] == 1]

MEL_images = df_MEL["image"].to_list()
NV_images = df_NV["image"].to_list()
BCC_images = df_BCC["image"].to_list()

cls_dict = {"MEL_images": MEL_images,
            "NV_images": NV_images,
            "BCC_images": BCC_images
            }

colors = ("red", "green", "blue")

# for cls in cls_dict:
#     plt.figure()
#     plt.xlim([0, 256])
#     hist_R = np.zeros(256)
#     hist_G = np.zeros(256)
#     hist_B = np.zeros(256)
sum_R = np.zeros((450, 600))
sum_G = np.zeros((450, 600))
sum_B = np.zeros((450, 600))
#     for name in cls_dict[cls]:
image1 = np.array(Image.open(f"./ISIC2018_Task3_Training_Input/ISIC_0024306.jpg"))
image2 = np.array(Image.open(f"./ISIC2018_Task3_Training_Input/ISIC_0024307.jpg"))

plt.imshow(image1[:, :, 0], cmap='Reds_r')

sum_R += image1[:, :, 0]
sum_R += image2[:, :, 0]
sum_G += image1[:, :, 1]
sum_G += image2[:, :, 1]
sum_B += image1[:, :, 2]
sum_B += image2[:, :, 2]

color_comp = np.zeros((450, 600, 3))
color_comp[:, :, 0] = (sum_R / 2)
color_comp[:, :, 1] = (sum_G / 2)
color_comp[:, :, 2] = (sum_B / 2)
color_comp = (color_comp/255).astype(float)
print(color_comp)
plt.imshow(color_comp)
plt.show()
