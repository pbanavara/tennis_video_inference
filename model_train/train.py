from ultralytics import YOLO
import torch
import argparse

mps_device = torch.device("mps")
model = YOLO("yolov8n-pose.pt")
model.to(mps_device)


def train(epoch):
    results = model.train(data = "/Users/pbanavara/dev/tennis_video_inference/datasets/roboflow/data.yaml", epochs=epoch, device=mps_device)
    results = model.val()
    model.export()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "epochs")
    parser.add_argument('epochs', metavar = 'epochs', type=int, help = "No of epochs")
    args = parser.parse_args()
    train(args.epochs)

    
