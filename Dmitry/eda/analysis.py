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


# -----------------------
# RGB channels analysis |
# -----------------------
def plot_rgb_distributions(image_class: str, r: np.array, g: np.array, b: np.array) -> None:
    plt.figure()

    plt.plot(r, color="red")
    plt.plot(g, color="green")
    plt.plot(b, color="blue")

    plt.title(f"Color Plot of {image_class} class")
    plt.xlabel("Color value")
    plt.ylabel("Pixel count")

    plt.savefig(os.path.join("classes", image_class, "pixel_value_distributions" + ".png"))


def get_variance(r_avg: np.array, g_avg: np.array, b_avg: np.array,
                 image_series: pd.Series) -> Tuple[np.array, np.array, np.array]:
    image_count = image_series.size
    r_square_dev, g_square_dev, b_square_dev = np.zeros((3, 450, 600), dtype=np.float64)
    for image_name in image_series:
        image = Image.open(os.path.join(IMAGE_FOLDER_PATH, image_name + ".jpg"))

        r_array, g_array, b_array = [np.array(channel) for channel in image.split()]
        r_square_dev += (r_array - r_avg) ** 2
        g_square_dev += (g_array - g_avg) ** 2
        b_square_dev += (b_array - b_avg) ** 2

    return r_square_dev / image_count + 1, g_square_dev / image_count + 1, b_square_dev / image_count + 1


def write_to_csv(image_class: str, filenames: Tuple[str, str, str],
                 arrays: Tuple[np.array, np.array, np.array]) -> None:
    for filename, arr in zip(filenames, arrays):
        np.savetxt(os.path.join("classes", image_class, filename + ".csv"), arr, delimiter=',')


image_classes = ["AKIEC", "BKL"]
for image_class in image_classes:
    r_values, g_values, b_values = np.zeros((3, 256), dtype=int)
    r_total, g_total, b_total = np.zeros((3, 450, 600), dtype=int)

    # Generate a pandas Series object with the names of images belonging to the specified class
    image_series = df.loc[df[image_class] == 1, "image"]
    for image_name in image_series:
        image = Image.open(os.path.join(IMAGE_FOLDER_PATH, image_name + ".jpg"))

        r_array, g_array, b_array = [np.array(channel) for channel in image.split()]

        # Count the number of occurrences of each value in channel arrays and accumulate them
        r_values += np.bincount(r_array.flatten(), minlength=256)
        g_values += np.bincount(g_array.flatten(), minlength=256)
        b_values += np.bincount(b_array.flatten(), minlength=256)

        r_total, g_total, b_total = r_total + r_array, g_total + g_array, b_total + b_array

    # Plot a histogram of the aggregated pixel intensities for the specified class
    plot_rgb_distributions(image_class, r_values, g_values, b_values)

    image_count = image_series.size

    # Calculate average of pixel values for each of the RGB channels
    r_avg, g_avg, b_avg = r_total / image_count, g_total / image_count, b_total / image_count
    write_to_csv(image_class, ("r_average", "g_average", "b_average"), (r_avg, g_avg, b_avg))

    # Calculate variance of pixel values for each of the RGB channels
    r_var, g_var, b_var = get_variance(r_avg, g_avg, b_avg, image_series)
    write_to_csv(image_class, ("r_variance", "g_variance", "b_variance"), (r_var, g_var, b_var))

    # Combine the color channels
    rgb_avg = np.stack([r_avg, g_avg, b_avg], axis=-1)

    plt.imsave(os.path.join("classes", image_class, "average_image" + ".png"), rgb_avg.astype("uint8"))
