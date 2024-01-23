import cv2
import matplotlib.pyplot as plt
import os
from cap_from_youtube import cap_from_youtube
from ultralytics import YOLO
import numpy as np
import torch

mps_device = torch.device("mps")

model = YOLO("/Users/pbanavara/dev/tennis_video_inference/model_train/runs/detect/train14/weights/best.pt")
#model = YOLO("yolov8n-pose.pt")

def get_capture(video_mov):
    cap = cv2.VideoCapture(video_mov)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print_frame_bboxes(frame)
            #overlay_poses(frame)
        else:
            break
        
    cap.release()

def capture_single_image(frame):
    results = model(frame, device=mps_device)
    result = results[0]
    bboxes = np.array(result.boxes.xywh.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    find_racket_ball_contact(bboxes, classes)
    annotated_frame = results[0].plot()
    cv2.imshow("Annotated frame", annotated_frame)

    

def is_racket_and_ball(box, cls):
    if cls == 4 or cls == 0:
        return True
    else:
        return False
    
def find_racket_ball_contact(bboxes, classes):
    boxes_classes = zip(bboxes,classes)
    print(list(filter(is_racket_and_ball, boxes_classes)))
    
    
    
    
def print_frame_bboxes(frame):
    results = model(frame, device=mps_device)
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")

    find_racket_ball_contact(bboxes,classes)
    
    for bbox,cls in zip(bboxes, classes):
        if cls == 0 or cls == 4:
            x, y, x2, y2 = bbox
            cv2.imshow('Frame', frame) # OpenCV uses BGR, whereas matplotlib uses RGB
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225), 4)
            cv2.circle(frame, ((x+x2)//2, (y+y2) //2), 10, (255, 255, 0), 10)
            cv2.putText(frame, str(cls), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 3)
            k = cv2.waitKey(5)
            # 113 is ASCII code for q key
            if k == 113:
                break



def overlay_poses(frame):
    results = model(frame, device=mps_device)
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
            overlay_poses(frame)

        else:
            break
    cap.release()
    
        
if __name__ == "__main__":
    file_name = "/Users/pbanavara/dev/tennis_video_inference/video_inference.mov"
    image = "/Users/pbanavara/dev/tennis_video_inference/high_res_images/stills/my-film-001712.png"
    
    url = "https://www.youtube.com/watch?v=ugQYbOsN5yI"
    url_2 = "https://youtu.be/FZSqzPXs43I?t=18"
    #print(os.path.exists(file_name))
    frame = get_capture(file_name)
    
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    #capture_single_image(img)
    #get_youtube_video(url_2)
    #frame = get_youtube_video(url)
