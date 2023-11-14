import os

from PIL import Image

import pandas as pd
import numpy as np
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
os.mkdir("./Results")

for cls in cls_dict:
    plt.figure()
    hist_R = np.zeros(256)
    hist_G = np.zeros(256)
    hist_B = np.zeros(256)
    sum_R = np.zeros((450, 600))
    sum_G = np.zeros((450, 600))
    sum_B = np.zeros((450, 600))
    squares_R = np.zeros((450, 600))
    squares_G = np.zeros((450, 600))
    squares_B = np.zeros((450, 600))
    cnt = len(cls_dict[cls])
    for name in cls_dict[cls]:
        image = np.array(Image.open(f"./ISIC2018_Task3_Training_Input/{name}.jpg"))

        for channel_id, color in enumerate(colors):
            histogram, bin_edges = np.histogram(
                image[:, :, channel_id], bins=256, range=(0, 256)
            )
            if channel_id == 0:
                hist_R += histogram
                sum_R += image[:, :, channel_id]
            elif channel_id == 1:
                hist_G += histogram
                sum_G += image[:, :, channel_id]
            else:
                hist_B += histogram
                sum_B += image[:, :, channel_id]
    for name in cls_dict[cls]:
        image = np.array(Image.open(f"./ISIC2018_Task3_Training_Input/{name}.jpg"))

        for channel_id, color in enumerate(colors):
            if channel_id == 0:
                squares_R += (image[:, :, channel_id] - (sum_R / cnt)) ** 2
            elif channel_id == 1:
                squares_G += (image[:, :, channel_id] - (sum_G / cnt)) ** 2
            else:
                squares_B += (image[:, :, channel_id] - (sum_B / cnt)) ** 2

    bins = np.arange(256)
    for channel_id, color in enumerate(colors):
        if channel_id == 0:
            histogram = hist_R
        elif channel_id == 1:
            histogram = hist_G
        else:
            histogram = hist_B
        plt.plot(bins, histogram, color=color)

        plt.title(f"Color Plot of {cls} class")
        plt.xlabel("Color value")
        plt.ylabel("Pixel count")
    os.mkdir(f"./Results/{cls}")
    plt.savefig(f"./Results/{cls}/{cls}_distribution.png")
    # plt.imsave(f"./Average_images_by_channels/{cls}_R.png", (sum_R / cnt), cmap='Reds_r')
    # plt.imsave(f"./Average_images_by_channels/{cls}_G.png", (sum_G / cnt), cmap='Greens_r')
    # plt.imsave(f"./Average_images_by_channels/{cls}_B.png", (sum_B / cnt), cmap='Blues_r')
    # color_comp = np.zeros((450, 600, 3))
    # color_comp[:, :, 0] = (sum_R / cnt)
    # color_comp[:, :, 1] = (sum_G / cnt)
    # color_comp[:, :, 2] = (sum_B / cnt)
    # color_comp = (color_comp / 255).astype(float)
    # plt.imsave(f"./Average_images_by_classes/{cls}.png", color_comp)
    np.savetxt(f"Results/{cls}/{cls}_average_by_pixels_R.csv", (sum_R / cnt), delimiter=',')
    np.savetxt(f"Results/{cls}/{cls}_average_by_pixels_G.csv", (sum_G / cnt), delimiter=',')
    np.savetxt(f"Results/{cls}/{cls}_average_by_pixels_B.csv", (sum_B / cnt), delimiter=',')
    np.savetxt(f"Results/{cls}/{cls}_variance_by_pixels_R.csv", squares_R / (cnt - 1), delimiter=',')
    np.savetxt(f"Results/{cls}/{cls}_variance_by_pixels_G.csv", squares_G / (cnt - 1), delimiter=',')
    np.savetxt(f"Results/{cls}/{cls}_variance_by_pixels_B.csv", squares_B / (cnt - 1), delimiter=',')
