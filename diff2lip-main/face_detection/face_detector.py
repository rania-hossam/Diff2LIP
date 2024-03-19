from ultralytics import YOLO


class YoloDetector:
    def __init__(self, weights="yolov8n-face-lindevs.pt"):
        self.model = YOLO(weights)

    def detect_images(self, images):
        detections = []
        for image in images:
            result = self.model(image)[0]
            xyxy = result.boxes.xyxy.detach().numpy().cpu()[0]
            detections.append(xyxy)
        return detections
