import cv2
import os
import pandas as pd


times = []
file_names = ["head/", "body/", "tail/"]
cnt = 0
for f_n in file_names:
    video_names = os.listdir(f_n)
    for v_n in video_names:
        #if cnt > 2:
        #   break
        if v_n[-3:] == "avi":
            cap = cv2.VideoCapture(f_n + v_n)
            i = 0
            success, frame = cap.read()  # "/home/ws/er3973/Desktop/research_code/TailTouching.avi"

            while success:
                i += 1
                success, frame = cap.read()
            times.append(i/1000)
            cnt += 1

times_df = pd.DataFrame(times)
times_df.to_csv('time_each_part.csv', index=False, header=False)