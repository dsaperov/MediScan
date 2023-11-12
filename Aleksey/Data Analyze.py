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

for cls in cls_dict:
    plt.figure()
    plt.xlim([0, 256])
    hist_R = np.zeros(256)
    hist_G = np.zeros(256)
    hist_B = np.zeros(256)
    for name in cls_dict[cls]:
        image = np.array(Image.open(f"./ISIC2018_Task3_Training_Input/{name}.jpg"))

        for channel_id, color in enumerate(colors):
            histogram, bin_edges = np.histogram(
                image[:, :, channel_id], bins=256, range=(0, 256)
            )
            if channel_id == 0:
                hist_R += histogram
            elif channel_id == 1:
                hist_G += histogram
            else:
                hist_B += histogram

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
    plt.savefig(f"./Distribution_results/{cls}_distribution.png")
