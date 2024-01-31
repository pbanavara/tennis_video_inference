#!/Users/pbanavara/miniforge3/envs/pytorch/bin/python

import cv2
import matplotlib.pyplot as plt
import os
from cap_from_youtube import cap_from_youtube
from ultralytics import YOLO
import numpy as np
import torch
import math
import argparse
import ultralytics
from ultralytics.engine.results import Results
from ultralytics.utils.plotting import Annotator

if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
    mps_device = torch.device("mps")

def analyze_video(video_mov, model, pose_flag):
    cap = cv2.VideoCapture(video_mov)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if not pose_flag:
                print_frame_bboxes(frame, model)
            else:
                overlay_poses(frame)
        else:
            break
        
    cap.release()


def capture_single_image(frame, model):
    """
    Placeholder method for single image testing
    """
    results = model(frame, device=mps_device)
    result = results[0]
    bboxes = np.array(result.boxes.xywh.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    find_racket_ball_contact(frame, bboxes, classes)
    annotated_frame= results[0].plot()
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
        if cls == 1 or cls == 5 or cls == 0: #remove this hard coding
            x, y, x2, y2 = box
            centre = ((x + x2) // 2, (y + y2)//2)
            m[cls] = centre
    # calculate the distance between the centres and set a threshold if only racket  and ball are present
    dist = 0
    if len(m) == 2: # This is 2 because we are looking for only the racket and the ball
        dist = calculate_racket_and_ball_distance(m[1], m[5])
        freeze_frame(frame, m[1], m[1], dist)
        
    
def print_frame_bboxes(frame, model):
    results = model(frame, device=mps_device)
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    print(bboxes, classes)
    #find_racket_ball_contact(frame, bboxes,classes)

    
    for bbox,cls in zip(bboxes, classes):
        if cls == 1 or cls == 5:
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
    """
    The pose detection points are in this order
    {
        0:"nose", 1:"left_eye", 2: "right_eye", 3: "left_ear", 4:"right_ear",
        5:"left_shoulder", 6:"right_shoulder", 7:"left_elbow", 8:"right_elbow",
        9: "left_wrist", 10:"right_wrist", 11:"left_hip", 12:"right_hip",
        13:"left_knee", 14:"right_knee", 15: "left_ankle", 16: "right_ankle"
    }

    The main objective of this function is to overlay the angles and the lines for those angles,
    compare the key angles such as the one on the left hand and right hand when the ball makes
    contact with the racket
    """
    overlay_dict = {}
    overlay_dict[0] = 'nose'
    overlay_dict[1] = 'nose'
    results = model(frame, device=mps_device)
    annotated_frame = results[0].plot()

    keypoints = results[0].keypoints.xy.cpu().numpy()[0]
    keypoints = [[int(k[0]), int(k[1])] for k in keypoints]
    ann = Annotator(frame)
    #ann.draw_specific_points(keypoints, indices=[5, 7, 6, 8, 9, 10], shape=(640, 640), radius=2)

    def get_angle(m_i, t_i, b_i):
        """
        Return the angle between 3 points of the indices in the keypoints array
        """
        
        left_elbow = (int(keypoints[m_i][0]), int(keypoints[m_i][1]))
        left_shoulder = (int(keypoints[t_i][0]), int(keypoints[t_i][1]))
    
        left_wrist = (int(keypoints[b_i][0]), int(keypoints[b_i][1]))
        radians = np.arctan2(left_wrist[1] - left_elbow[1], left_wrist[0] - left_elbow[0]) - np.arctan2(left_shoulder[1] - left_elbow[1],
                                                                                                    left_shoulder[0] - left_elbow[0])
        angle = np.abs(radians * 180.0 / np.pi)
    
        if angle > 180.0:
            angle = 360 - angle
        return angle

    left_angle = get_angle(7, 5, 9)
    right_angle = get_angle(8, 6, 10)
    
    """
    for i, kp in enumerate(keypoints):
        x = int(kp[0])
        y = int(kp[1])
        if i in [5,7,9]:
            ann.text((x,y), str(angle))
            cv2.imshow("Frame", ann.result())
    """

    def draw_angle_line(ann, angle, start, mid, end):
        
        ann.plot_angle_and_count_and_stage(angle, 1, "stg", keypoints[7])
        #ann.plot_angle_and_count_and_stage(right_angle, 1, "stg", keypoints[8])
        cv2.line(ann.result(), start, mid, [0, 0, 255], thickness=2, lineType=cv2.LINE_AA)
        cv2.line(ann.result(), mid, end, [255, 255, 0], thickness=2, lineType=cv2.LINE_AA)
        cv2.imshow("Annotated frame", ann.result())

    draw_angle_line(ann, left_angle, keypoints[5], keypoints[7], keypoints[9])
    draw_angle_line(ann, right_angle, keypoints[6], keypoints[8], keypoints[10])
    
    #cv2.line(ann.result(), keypoints[7], keypoints[9], [0, 0, 250], thickness=2, lineType=cv2.LINE_AA)
    #cv2.line(ann.result(), keypoints[6], keypoints[8], [0, 240, 230], thickness=2, lineType=cv2.LINE_AA)
    #cv2.line(ann.result(), keypoints[8], keypoints[10], [0, 240, 230], thickness=2, lineType=cv2.LINE_AA)
    #cv2.imshow("Annotated frame", ann.result())
    #cv2.imwrite('test.png', annotated_frame)

    k = cv2.waitKey(1)
    if k == 113:
        return
    
    
        

def get_youtube_video(url, model):
    cap = cap_from_youtube(url, '')
    count = 0
    while True:
        count += 1
        if count >= 2:
            break
        ret, frame = cap.read()
        if ret:
            overlay_poses(frame, model)
        else:
            break
    cap.release()

def choose_model(pose_flag):
    if not pose_flag:
        model = YOLO("/Users/pbanavara/dev/tennis_video_inference/model_train/runs/detect/train17/weights/best.pt")
    else:
        model = YOLO("yolov8n-pose.pt")
    
    return model
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Inference local video")
    parser.add_argument('video', metavar = 'video', type=str, help = "Local video location for inference")
    parser.add_argument('pose', metavar = 'pose', type=bool, nargs='?', default=False, help = "Pass true for pose detection, defaults to object detection")
    parser.add_argument('yt', metavar='yt', type=str, nargs="?", default=None, help = "Enter an optional youtube url , to supersede the local video")
    parser.add_argument('single_image', metavar='single_image', nargs="?", type=str, help = "Enter the location of single image for single image inference")
    
    args = parser.parse_args()
    
    url = "https://www.youtube.com/watch?v=ugQYbOsN5yI"
    url_2 = "https://youtu.be/FZSqzPXs43I?t=18"

    pose_flag = args.pose
    yt = args.yt
    single_image = args.single_image
    
    model = choose_model(pose_flag)
    if yt is not None:
        get_youtube_video(url_2, model)
    elif single_image:
        capture_single_iamge(single_image, model)
    else:
        analyze_video(args.video, model, pose_flag)
    

    
