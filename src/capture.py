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
from ultralytics.utils import plotting
import time
import logging
import boto3
import botocore
import common
import os
import redis

class AnalyzeVideo(object):
    """Main class for processing and overlaying pose, mask or object detection on the main video

    Args:
        object (_type_): _description_
    """

    def __init__(self):
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not "
                   "built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ " + "and/or you do not have an MPS-enabled device on this machine.")
        else:
            self.mps_device = torch.device("mps")
        self.fourcc = cv2.VideoWriter_fourcc(*'vp09')
        self.fps = 20
        self.r = redis.Redis(host=common.REDIS_HOST, port=common.REDIS_PORT, db=0)
        


    def upload_to_s3(self, output_file_name):
        s3 = boto3.resource('s3')
        try :
             with open(output_file_name, 'rb') as data:
                 s3.Bucket('tennisvideosbucket').put_object(Key = output_file_name, Body = data)
        except botocore.exceptions.ClientError as e:
            logging.error(e)
            raise e 
        except botocore.exceptions.ParamValidationError as error:
            raise ValueError('The parameters you provided are incorrect: {}'.format(error))
        return common.S3_SUCCESS

     
    def analyze_video(self, video_mov, pose_flag, web_flag, output_file_name):
        """Analyzes a given standalone video and overlays poses on the given video

        Args:
            video_mov (_type_): local video file
            model (_type_): Pose or object detection model
            pose_flag (_type_): True for podse detection
        """
        cap = cv2.VideoCapture(video_mov)
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps * 0.30
        out = cv2.VideoWriter(output_file_name, self.fourcc, self.fps, (int(cap.get(3)), int(cap.get(4))))
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if not pose_flag:
                    model = self.choose_model(False)
                    self.print_frame_bboxes(frame, model)
                else:
                    model = self.choose_model(True)
                    result = self.overlay_poses(frame, model)
                    if web_flag:
                        if result.any():
                            out.write(result)
                            count += 1
                        else:
                            out.write(frame)
                    else:
                        logging.debug("Encoding failed")
                        cv2.imshow("Frame", result)
            else:
                 break

        cap.release()
        out.release()
        try:
            if os.path.isfile(output_file_name):
                result = self.upload_to_s3(output_file_name)
                logging.debug("Processs done %s", output_file_name)
            else:
                raise Exception("File creation failed")
            return result
        except Exception as e:
            logging.debug(e)
            raise Exception(e)

    

    def get_youtube_video(self, url, pose_flag, web_flag):
        """Used for processing online youtube videos

        Args:
            url (_type_): _description_
            model (_type_): _description_
        """
        cap = cap_from_youtube(url, '')
        
        output_file = "fed_out.mp4"
        out = cv2.VideoWriter(output_file, self.fourcc, self.fps, (int(cap.get(3)), int(cap.get(4))))
        while True:
            ret, frame = cap.read()
            if ret:
                if pose_flag:
                    model = self.choose_model(True)
                    result = self.overlay_poses(frame, model)
                    if web_flag:
                        if result.any():
                            ret, buffer = cv2.imencode('.jpg', result)
                            out.write(result)
                        else:
                            ret, buffer = cv2.imencode('.jpg', frame)
                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    else:
                        cv2.imshow("Frame", result)
                else:
                    model = self.choose_model(False)
                    self.print_frame_bboxes(frame, model)
            else:
                break
        cap.release()


    def calculate_racket_and_ball_distance(self, r_centre, b_centre):
        """Calculate the racket centre to ball centre distance

        Args:
            r_centre (_type_): _description_
            b_centre (_type_): _description_

        Returns:
            _type_: _description_
        """
        dist = math.sqrt((b_centre[1] - r_centre[1]) ** 2 + (b_centre[0] - r_centre[0]) ** 2)
        return dist

    def freeze_frame(self, frame, b_centre, r_centre):
        """Placeholder method to freeze the action when required

        Args:
            frame (_type_): _description_
            b_centre (_type_): _description_
            r_centre (_type_): _description_
        """
        new_frame = frame.copy()
        cv2.imshow("Frame", new_frame)
        cv2.circle(new_frame, b_centre, 10, (255, 255, 0), 10)
        cv2.circle(new_frame, r_centre, 10, (255, 0, 255), 10)
        cv2.line(frame, b_centre, r_centre, (245,222,135), 4)
        cv2.imwrite("new_image.png", new_frame)


    def find_racket_ball_contact(self, frame, bboxes, classes):
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
        if len(m) == 2: # This is 2 because we are looking for only the racket and the ball
            self.freeze_frame(frame, m[1], m[1])


    def print_frame_bboxes(self, frame, model):
        """Helper method for drawing object detection bounding boxes 

        Args:
            frame (_type_): _description_
            model (_type_): _description_
        """
        results = model(frame, device=self.mps_device)
        result = results[0]
        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        classes = np.array(result.boxes.cls.cpu(), dtype="int")
        print(bboxes, classes)
        self.find_racket_ball_contact(frame, bboxes,classes)


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
    def get_angle(self, t_i, m_i, b_i, keypoints):
        """
        Return the angle between 3 points of the indices in the keypoints array
        """

        top_point = (int(keypoints[t_i][0]), int(keypoints[t_i][1]))
        mid_point = (int(keypoints[m_i][0]), int(keypoints[m_i][1]))

        bottom_point = (int(keypoints[b_i][0]), int(keypoints[b_i][1]))
        radians = np.arctan2(bottom_point[1] - mid_point[1], bottom_point[0] - mid_point[0]) - np.arctan2(top_point[1] - mid_point[1],
                                                                                                        top_point[0] - mid_point[0])
        angle = np.abs(radians * 180 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle
    
    def draw_angle_line(self, ann, angle, start, mid, end, color):
        cv2.circle(ann.result(), start, 5, (255, 255, 255), 2)
        cv2.circle(ann.result(), mid, 5, (255, 255, 255), 2)
        cv2.circle(ann.result(), end, 5, (255, 255, 255), 2)
        cv2.line(ann.result(), start, mid, color, thickness=2, lineType=cv2.LINE_AA)
        cv2.line(ann.result(), mid, end, color, thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(ann.result(), str(int(angle)), (mid[0], mid[1]), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 1)
        #cv2.imshow("Annotated frame", ann.result())
        # Check the angle and tell the difference
        return ann.result()


    def overlay_poses(self, frame, model):
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

        results = model(frame, device=self.mps_device)
        names = results[0].boxes.cls

        keypoints = results[0].keypoints.xy.cpu().numpy()[0]
        keypoints = [[int(k[0]), int(k[1])] for k in keypoints]

        ann = Annotator(frame)
        if names.shape[0] > 0:
            right_hand_angle = self.get_angle(8, 6, 10, keypoints)
            left_leg_angle = self.get_angle(11, 13, 15, keypoints)
            right_leg_angle = self.get_angle(12, 14, 16, keypoints)
            left_hip_angle = self.get_angle(6, 11, 13, keypoints)

            # Comment for specific angles right hand, left hand etc - this needs to improve

            #draw_angle_line(ann, left_angle, keypoints[5], keypoints[7], keypoints[9])
            result = self.draw_angle_line(ann, right_hand_angle, keypoints[6], keypoints[8], keypoints[10], [255, 255, 250])
            
            result = self.draw_angle_line(ann, right_leg_angle, keypoints[12], keypoints[14], keypoints[16], [255, 255, 240] )
            result = self.draw_angle_line(ann, left_leg_angle, keypoints[11], keypoints[13], keypoints[15], [255, 255, 230] )
                #draw_angle_line(ann, left_hip_angle, keypoints[5], keypoints[6], keypoints[11], [255, 255, 220] )
            result = self.draw_angle_line(ann, left_hip_angle, keypoints[6], keypoints[11], keypoints[13], [255, 255, 220] )

            if right_hand_angle > 10:
                cv2.putText(result, "Right hand is bent during forehand", (keypoints[8][0], keypoints[8][1]), cv2.FONT_HERSHEY_PLAIN, 2, (100,100,100), 1)

            k = cv2.waitKey(1)
            if k == ord('p'):
                cv2.waitKey(-1)
            return result
        else:
            # No keypoints found return the unprocessed frame
            return ann.result()


    def choose_model(self, pose_flag):
        """Choose between pose and object detection models

        Args:
            pose_flag (_type_): _description_

        Returns:
            _type_: _description_
        """
        if not pose_flag:
            model = YOLO("/Users/pbanavara/dev/tennis_video_inference/model_train/runs/detect/train17/weights/best.pt")
        else:
            model = YOLO("yolov8n-pose.pt")

        return model
