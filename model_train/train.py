from ultralytics import YOLO
import torch

print(torch.__version__)
mps_device = torch.device("mps")
model = YOLO("../yolov8m.pt")
model.to(mps_device)
results = model.train(data = "../run_3/data.yaml", epochs=50, device=mps_device)

results = model.val()
model.export()

