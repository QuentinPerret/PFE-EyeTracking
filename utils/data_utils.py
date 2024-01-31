import os
import shutil

from statistics import mean
from math import sqrt

import itertools

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2

from utils.objectCenter import df as objectCenter


def retrieveGazePosition(row):
    # folder = f"{row['participant']}_{row['session']}_"
    folder = f"00{row.name}" if row.name < 10 else f"0{row.name}"
    csv_path = f"../pupil_data/" + folder + "/gaze_positions.csv"
    df = pd.read_csv(csv_path, sep=(","))

    x = df.pos_x.to_list()
    y = df.pos_y.to_list()
    t = df.gaze_timestamp.to_list()

    row["pos_x"] = x
    row["pos_y"] = y
    row["timestamps"] = t
    return row


# Define Euclidean distance between two points
def dist2p(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    return sqrt((dx**2) + (dy**2))


# PeyeMMV parameters:
# file: raw gaze data (x,y,passing time)
# t1,t2: spatial parameters for fixation identification
# min_dur: minimum duration threshold for fixation identification
# report_fix: selecting the value '1', raw gaze data and fixation plot is generated
# Run example (after importing peyemmv module): peyemmv.extract_fixations('demo_data.txt',0.25,0.1,150,'1')


def extract_fixations(df, t1, t2, min_dur, report_fix=False):
    # Initialize x,y,t,p(x,y,t)
    x = df["pos_x"]
    y = df["pos_y"]
    t = df["timestamps"]
    p = []
    for i, j, k in zip(x, y, t):
        p.append([i, j, k])

    # Initialize fixation cluster and fixations list
    fix_clust = []
    fix_clust_t2 = []
    x_t2 = []
    y_t2 = []
    t_t2 = []

    x_gaze = []
    y_gaze = []
    t_gaze = []
    fixations = []

    # Initialize fixation mean point
    fixx = x[0]
    fixy = y[0]

    for point in p:
        dist = dist2p(fixx, fixy, point[0], point[1])

        # check spatial threshold
        if dist < t1:
            x_gaze.append(point[0])
            y_gaze.append(point[1])
            t_gaze.append(point[2])
            fixx = mean(x_gaze)
            fixy = mean(y_gaze)

        else:
            # Put all gaze points in a fixation cluster
            fix_clust.append([x_gaze, y_gaze, t_gaze])
            if len(fix_clust[0][0]) >= 1:
                fixx_clust = mean(fix_clust[0][0])
                fixy_clust = mean(fix_clust[0][1])

                for xg, yg, tg in zip(
                    fix_clust[0][0], fix_clust[0][1], fix_clust[0][2]
                ):
                    if dist2p(fixx_clust, fixy_clust, xg, yg) < t2:
                        x_t2.append(xg)
                        y_t2.append(yg)
                        t_t2.append(tg)

                fixx_clust_t2 = mean(x_t2)
                fixy_clust_t2 = mean(y_t2)
                fixdur_clust_t2 = t_t2[-1] - t_t2[0]

                # Check minimum duration threshold
                if fixdur_clust_t2 >= min_dur:
                    # mean_x,mean_y,dur,start,end
                    fixations.append(
                        [
                            fixx_clust_t2,
                            fixy_clust_t2,
                            fixdur_clust_t2,
                            t_t2[0],
                            t_t2[-1],
                            len(t_t2),
                        ]
                    )

            # Initialize fixation mean point and gaze points
            fixx = point[0]
            fixy = point[1]
            x_gaze = []
            y_gaze = []
            t_gaze = []
            fix_clust = []
            fix_clust_t2 = []
            x_t2 = []
            y_t2 = []
            t_t2 = []

    # Generate fixation report (plot values)
    if report_fix:
        # print final fixations
        x_fix = []
        y_fix = []
        dur_fix = []
        print(
            "Fixation_ID [X_coord, Y_coord, Duration, Start_time, End_time, No_gaze_points]"
        )

        # Define a fixation counter
        fix_count = 0
        for fix in fixations:
            fix_count = fix_count + 1
            print(fix_count, fix)
            x_fix.append(fix[0])
            y_fix.append(fix[1])
            dur_fix.append(fix[2])

        plt.title("Fixation points")
        plt.xlabel("Horizontal coordinates (tracker units)")
        plt.ylabel("Vertical coordinates (tracker units)")

        plt.scatter(x, y, color="blue", marker="+")
        plt.scatter(x_fix, y_fix, color="red", marker=".")
        plt.legend(
            ["raw gaze data", "fixation centers and their durations"], loc="best"
        )
        for i in range(len(x_fix)):
            plt.text(x_fix[i], y_fix[i], "{:.1f}".format(dur_fix[i]), color="red")
        plt.grid()
        plt.show()
    else:
        pass

    return fixations


def show_fixations(row):
    x0 = np.array(row["fixations"])[:, 0]
    y0 = np.array(row["fixations"])[:, 1]
    plt.scatter(x0, y0)
    plt.plot(x0, y0, label="fixations")
    plt.scatter(x0[0], y0[0], color="red")
    plt.show()


def get_fixations(df):
    f = np.array(extract_fixations(df, 0.01, 0.01, 0.001))
    xmin = f[:, 0].min()
    xmax = f[:, 0].max()
    f[:, 0] = (f[:, 0] - xmin + 0.011) / ((xmax - xmin) * 1.22)

    ymax = f[:, 1].min()
    ymin = f[:, 1].max()
    f[:, 1] = (f[:, 1] - ymin) / ((ymax - ymin) * 1.15)

    time_origin = f[0, 3]
    f[:, 3] = f[:, 3] - time_origin
    f[:, 4] = f[:, 4] - time_origin

    return f


def make_video(df,i):
    f = df.fixations
    # Open the video file
    video_path = df.videos
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    T = 1 / 24
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # You can use other codecs like 'XVID'
    output_path = f"../output_video/{i}.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Check if the VideoWriter is opened successfully
    if not out.isOpened():
        print("Error: Could not open output video.")

    i_fix = 0
    i_frame = 0
    # Process each frame
    while cap.isOpened() and i_fix < len(f):
        ret, frame = cap.read()

        if not ret:
            break

        if i_fix == 0:
            fixations = [f[0]]
        elif i_fix == 1:
            fixations = f[0:2]
        else:
            fixations = f[i_fix - 2 : i_fix + 1]

        x0, y0 = None, None
        for fix in fixations:
            x, y = fix[0:2]
            x1, y1 = int(x * width), int(y * height)
            # Draw circles
            cv2.circle(frame, (x1, y1), 2, (0, 0, 255), 2)
            cv2.circle(frame, (x1, y1), 20, (0, 0, 255), 1)

            # Draw lines between the circles
            if (x0, y0) != (None, None):
                cv2.line(frame, (x0, y0), (x1, y1), (255, 0, 0), 1)

            x0, y0 = x1, y1
        # Write the modified frame to the output video
        out.write(frame)
        i_frame += 1
        # print(T * i_frame,fixations[-1][4])
        if T * i_frame > fixations[-1][4]:
            i_fix += 1

    # Release video capture and writer objects
    cap.release()
    out.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

    print("Video processing complete. Output video saved to:", output_path)


def get_relative_distance_from_gravity_center(df):
    video_path = df.videos
    fixations = df.fixations

    center = objectCenter[objectCenter["videoPath"] == video_path][
        "objectCenter"
    ].tolist()[0]

    d = np.zeros(len(center))
    i_frame = 0

    for fix in fixations:
        x, y, _, _, te = fix[0:5]
        t = i_frame * 1 / 24
        while t < te and i_frame < 150:
            t = i_frame * 1 / 24
            x_center, y_center = center[i_frame]
            dist = (x_center - x) ** 2 + (y_center - y) ** 2
            d[i_frame] = dist
            i_frame += 1

    return d
