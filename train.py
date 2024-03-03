from ultralytics import YOLO
 
# Load the model.
model = YOLO('yolov8n.pt')
 
# Training.
results = model.train(
   data='fabric.yaml',
   imgsz=640,
   epochs=15,
   batch=8,
   name='result'
)