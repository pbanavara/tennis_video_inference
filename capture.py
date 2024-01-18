import cv2
import matplotlib.pyplot as plt
import os
from cap_from_youtube import cap_from_youtube
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n-pose.pt")

def get_capture(video_mov):
    cap = cv2.VideoCapture(video_mov)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            #print_frame_bboxes(frame)
            overlay_poses(frame)
        else:
            break
        
    cap.release()


def print_frame_bboxes(frame):
    results = model(frame, device="mps")
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    for bbox,cls in zip(bboxes, classes):
        #if cls == 38 or cls == 32:
            x, y, x2, y2 = bbox
            cv2.imshow('Frame', frame) # OpenCV uses BGR, whereas matplotlib uses RGB
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225), 2)
            cv2.putText(frame, str(cls), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)
            k = cv2.waitKey(1)
            # 113 is ASCII code for q key
            if k == 113:
                break

def overlay_poses(frame):
    results = model(frame, device="mps")
    annotated_frame = results[0].plot()
    cv2.imshow("Annotated frame", annotated_frame)
    k = cv2.waitKey(1)
    if k == 113:
        return
    
        

def get_youtube_video(url):
    cap = cap_from_youtube(url, '')
    while True:
        ret, frame = cap.read()
        if ret:
            print_frame_bboxes(frame)

        else:
            break
    cap.release()
    
        
if __name__ == "__main__":
    file_name = "/Users/pbanavara/dev/tennis_video_inference/video_inference.mov"
    url = "https://www.youtube.com/watch?v=ugQYbOsN5yI"
    print(os.path.exists(file_name))
    frame = get_capture(file_name)
    #frame = get_youtube_video(url)
