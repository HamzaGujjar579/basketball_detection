from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('best.pt')

# Export to CoreML
model.export(format='coreml')
