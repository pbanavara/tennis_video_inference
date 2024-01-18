from ultralytics import YOLO

model = YOLO("../yolov8m.pt")
results = model.train(data = "data.yaml", epochs=10, device='mps')

results = model.val()
model.export()

