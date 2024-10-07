import cv2
from ultralytics import YOLO
import pandas as pd
import time
import numpy as np

cap = cv2.VideoCapture("test2.mp4")
model_1 = YOLO('yolov8s.pt')
model_2 = YOLO("best.pt")

with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

frames_processed = 0
fps_display = 0
start_time_fps = time.time()

while True:
    x1_2, y1_2, x2_2, y2_2=0, 0, 0, 0

    ret, frame = cap.read()
    if not ret:
        break

    results_2 = model_2(frame, verbose=True)
    a_2 = results_2[0].boxes.data.cpu()
    px_2 = pd.DataFrame(a_2).astype("float")

    for index_2, row_2 in px_2.iterrows():
        x1_2, y1_2, x2_2, y2_2, conf_2 = int(row_2[0]), int(row_2[1]), int(row_2[2]), int(row_2[3]), float(row_2[4])
        

    results = model_1(frame, verbose=True)
    a = results[0].boxes.data.cpu()
    px = pd.DataFrame(a).astype("float")

    for index, row in px.iterrows():
        x1, y1, x2, y2, conf, d = int(row[0]), int(row[1]), int(row[2]), int(row[3]), float(row[4]), int(row[5])
        c = class_list[d]
        area= [(x1,y1 ),(x2, y1),(x2, y2),  (x1, y2)]
        
        if d == 2:
            res1 = cv2.pointPolygonTest(np.array(area, np.int32), ((x1_2, y1_2)), False)
            res3 = cv2.pointPolygonTest(np.array(area, np.int32), ((x2_2, y2_2)), False)
            if res1==res3==1:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Accessible vehicle : {(round(conf*100, 1))}% ", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            

    frames_processed += 1
    if time.time() - start_time_fps >= 1:
        fps_display = frames_processed
        frames_processed = 0
        start_time_fps = time.time()
    
    cv2.putText(frame, f"FPS: {fps_display}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 128), 2)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
