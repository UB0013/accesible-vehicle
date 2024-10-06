import cv2
from ultralytics import YOLO
import pandas as pd
import time

cap = cv2.VideoCapture("test1.mp4")
model = YOLO('yolov8s.pt')

with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

frames_processed = 0
fps_display = 0
start_time_fps = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    a = results[0].boxes.data.cpu()
    px = pd.DataFrame(a).astype("float")

    for index, row in px.iterrows():
        x1, y1, x2, y2, conf, d = int(row[0]), int(row[1]), int(row[2]), int(row[3]), float(row[4]), int(row[5])
        c = class_list[d]

        if d == 2:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, c, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    frames_processed += 1
    if time.time() - start_time_fps >= 1:
        fps_display = frames_processed
        frames_processed = 0
        start_time_fps = time.time()

    cv2.putText(frame, f"FPS: {fps_display}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
