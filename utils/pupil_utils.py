# %%
import pandas as pd
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt


# %%
def openPsychoPyExperimentMetadata(path):
    df = pd.read_csv(path)
    df = df[~df["videos"].isna()]
    df = df[
        [
            "videos",
            "participant",
            "sessions.thisN",
            "trial.started",
            "trial.stopped",
        ]
    ]
    df.columns.values[2] = "session"
    df["participant"] = df["participant"].astype(int)
    df["session"] = df["session"].astype(int)
    df["videos"] = "../" + df["videos"]
    return df


def buildOneHotEncoding(df):
    filepath = df["videos"]
    filepath = filepath.split("/")[2:-1]
    filepath[0] = filepath[0].split("_")[1:]
    df["init"] = int(filepath[0][0][-1])
    df["perturbation"] = 1 if filepath[0][-1] == "perturbation" else 0
    df["delta"] = 0 if filepath[-1] == "original" else int(filepath[-1][-1])

    return df


# %%
df = openPsychoPyExperimentMetadata("../data/188563_PFE_2024-01-29_18h07.46.466.csv")
df = df.apply(lambda x: buildOneHotEncoding(x), axis=1)
t0 = df["trial.started"][0]
df["trial.started"] = df.apply(lambda x: x["trial.started"] - t0, axis=1)
df["trial.stopped"] = df.apply(lambda x: x["trial.stopped"] - t0, axis=1)

# %%
gaze_positions = pd.read_csv("../pupil_data/gaze_positions.csv")
t0 = gaze_positions.gaze_timestamp[0]
gaze_positions.gaze_timestamp = gaze_positions.apply(
    lambda x: x.gaze_timestamp - t0, axis=1
)


# %%
def retrieve_gaze_positions(df):
    gaze = gaze_positions[
        (gaze_positions["gaze_timestamp"] >= df["trial.started"])
        & (gaze_positions["gaze_timestamp"] < df["trial.stopped"])
    ]
    df["pos_x"] = gaze["norm_pos_x"].tolist()
    df["pos_y"] = gaze["norm_pos_y"].tolist()
    df["timestamps"] = gaze["gaze_timestamp"].tolist()

    return df


df = df.apply(lambda x: retrieve_gaze_positions(x), axis=1)
