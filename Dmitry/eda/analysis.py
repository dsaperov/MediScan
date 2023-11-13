import os
from typing import Tuple

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

GT_CSV_PATH = os.path.join(os.pardir, "ISIC2018_Task3_Training_GroundTruth.csv")
IMAGE_FOLDER_PATH = os.path.join(os.pardir, "ISIC2018_Task3_Training_Input")

df = pd.read_csv(GT_CSV_PATH)

# -----------------------------
# Identifying class imbalance |
# -----------------------------
df_without_image = df.drop(columns=["image"])
column_sums = df_without_image.sum()

column_sums.plot(kind="bar", figsize=(10, 7))
plt.title("Number of Images by Class")
plt.xlabel("Class Name")
plt.ylabel("Images")
plt.savefig("image_count_by_class.png")


# --------------------------------------
# Identifying image size inconsistency |
# --------------------------------------
def get_size(image_path: str) -> Tuple[int, int]:
    """Returns dimensions for an image"""
    image = Image.open(image_path)
    image_arr = np.array(image)
    height, width, _ = image_arr.shape
    return height, width


image_sizes = []
image_names = os.listdir(IMAGE_FOLDER_PATH)
total_files = len(image_names)

for i, image_name in enumerate(image_names):
    if image_name.endswith(".jpg"):
        image_path = os.path.join(IMAGE_FOLDER_PATH, image_name)
        image_sizes.append(get_size(image_path))
    print(f"Image sizes analysis progress: {i+1}/{total_files} files processed", end='\r')

sizes_df = pd.DataFrame(image_sizes, columns=["height", "width"])

sizes_df.plot.scatter(x="width", y="height")
plt.title("Image Sizes (pixels)")
plt.savefig("image_sizes")


