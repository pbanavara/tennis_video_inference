#!/Users/pbanavara/miniforge3/envs/pytorch/bin/python

import cv2
import matplotlib.pyplot as plt
import os
from cap_from_youtube import cap_from_youtube
from ultralytics import YOLO
import numpy as np
import torch
import math

mps_device = torch.device("mps")

model = YOLO("/Users/pbanavara/dev/tennis_video_inference/model_train/runs/detect/train14/weights/best.pt")
#model = YOLO("yolov8n-pose.pt")

def analyze_video(video_mov):
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
    """
    Placeholder method for single image testing
    """
    results = model(frame, device=mps_device)
    result = results[0]
    bboxes = np.array(result.boxes.xywh.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    find_racket_ball_contact(frame, bboxes, classes)
    annotated_frame = results[0].plot()
    cv2.imshow("Annotated frame", annotated_frame)


def calculate_racket_and_ball_distance(r_centre, b_centre):
    dist = math.sqrt((b_centre[1] - r_centre[1]) ** 2 + (b_centre[0] - r_centre[0]) ** 2)
    return dist

def freeze_frame(frame, b_centre, r_centre, dist):
    new_frame = frame.copy()
    cv2.imshow("Frame", new_frame)
    cv2.circle(new_frame, b_centre, 10, (255, 255, 0), 10)
    cv2.circle(new_frame, r_centre, 10, (255, 0, 255), 10)
    cv2.line(frame, b_centre, r_centre, (245,222,135), 4)
    cv2.imwrite("new_image.png", new_frame)
    

def find_racket_ball_contact(frame, bboxes, classes):
    """
    Need to find the racket center and the ball center and see if they are within a very short delta.
    Now bboxes = [(array([1827,  199, 1926,  421]), 4), (array([1956,   87, 2008,  166]), 0), (array([1986,   87, 2008,  166]), 2)]

    From this I want to extract 4 and 0 and compare the centres.
    Use a map
    
    """

    m = {}
    boxes_classes = zip(bboxes,classes)
    for box, cls in boxes_classes:
        if cls == 0 or cls == 4:
            x, y, x2, y2 = box
            centre = ((x + x2) // 2, (y + y2)//2)
            m[cls] = centre
    # calculate the distance between the centres and set a threshold if only racket  and ball are present
    if len(m) == 2:
        dist = calculate_racket_and_ball_distance(m[0], m[4])
        freeze_frame(frame, m[0], m[4], dist)
        
    
def print_frame_bboxes(frame):
    results = model(frame, device=mps_device)
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")

    find_racket_ball_contact(frame, bboxes,classes)

    """
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
    """


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
    inference_video = "/Users/pbanavara/dev/tennis_video_inference/video_inference.mov"
    image = "/Users/pbanavara/dev/tennis_video_inference/high_res_images/stills/my-film-001712.png"
    
    url = "https://www.youtube.com/watch?v=ugQYbOsN5yI"
    url_2 = "https://youtu.be/FZSqzPXs43I?t=18"
    #print(os.path.exists(file_name))
    analyze_video(inference_video)
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    #capture_single_image(img)
    #get_youtube_video(url_2)
    #frame = get_youtube_video(url)
