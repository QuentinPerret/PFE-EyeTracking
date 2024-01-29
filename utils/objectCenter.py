# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
root = "../video/Videos - mask/"

import os

path_mask_videos = []

for path, subdirs, files in os.walk(root):
    for name in files:
        path_mask_videos.append(os.path.join(path, name))

df = pd.DataFrame(path_mask_videos, columns=["maskPath"])
df["videoPath"] = df.apply(
    lambda x: x["maskPath"].replace("video/Videos - mask/", "video/"), axis=1
)


# %%
def gravityCenter(frame, viz=False):
    # Trouver les coordonnées des pixels blancs (valeurs égales à 255)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    coordonnees_blancs = np.column_stack(np.where(frame > 50))

    # Calculer les coordonnées moyennes
    centre_x = np.mean(coordonnees_blancs[:, 1])
    centre_y = np.mean(coordonnees_blancs[:, 0])

    if viz:
        print("Centre de gravité :", centre_x, centre_y)

        # Visualisation
        plt.imshow(frame, cmap="gray")
        plt.scatter(centre_x, centre_y, color="red", marker=".", s=100)
        plt.title("Image avec le centre de gravité")
        plt.show()

    return centre_x / 1920, centre_y / 1080


# %%
def getAllGravityCenter(video_path: str) -> list:
    gravity_center = []
    video = cv2.VideoCapture(video_path)
    while True:
        ret, frame = video.read()
        if ret:
            frame = np.array(frame)
            gravity_center.append(gravityCenter(frame))
        else:
            break
    return gravity_center


df['objectCenter'] = df.apply(lambda x: getAllGravityCenter(x["maskPath"]), axis=1)

# %%
