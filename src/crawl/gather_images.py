"""
Summary:

To enable grabbing of specific images for training the tennis racket and ball classifier. For now I had trained the images with my own images but
that doesn't seem to cut it.

So I need a crawler which can use a seed URL and grab all tennis racket and ball images for labelling. Hence this script
On second thoughts the images on the web are of real bad quality. It's best to download videos and extract images.

"""

import cv2
import numpy as np
import time
import random
import argparse
from cap_from_youtube import cap_from_youtube

class Crawl(object):

    def __init__(self):
        unique_ids = set()
        

    def generate_unique_id(self):
        """
        Use timestamp and 62 bit encoding to generate unique ids that are 7 characters or less
        """
        num = random.randint(1, 1000000000001)
        BASE_62 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        res = []
        while num:
            num, rem = divmod(num, 62)
            res.append(BASE_62[rem])
        return ''.join(x for x in res)

    def generate_file_name(self, base_path):
        start_file_name = "IMG"
        out_file_name = download_path + "/" + start_file_name + "_" + self.generate_unique_id() + ".jpg"
        return out_file_name

    def download_video_into_frames(self, video_url, download_path):
        cap = cv2.VideoCapture(video_url)
        
        while cap.isOpened():
            ret, frame = cap.read()
            out_file_name = self.generate_file_name(download_path)
            cv2.imwrite(out_file_name, frame)
        
        cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Youtube URL")
    parser.add_argument('url', metavar = 'url', type=str, nargs = '+', help = 'youtube URL of the desired player')
    args = parser.parse_args()
    
    crawl = Crawl()
    download_path = "/Users/pbanavara/dev/datasets/tennis_images"
    # test out the id generation
    #print(crawl.generate_unique_id())
    crawl.download_video_into_frames(args.url[0], download_path)
    
    
    
