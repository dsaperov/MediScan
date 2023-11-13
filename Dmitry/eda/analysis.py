import os

import pandas as pd
import matplotlib.pyplot as plt

GT_CSV_PATH = os.path.join(os.pardir, "ISIC2018_Task3_Training_GroundTruth.csv")

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
