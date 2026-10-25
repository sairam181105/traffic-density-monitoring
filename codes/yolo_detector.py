import cv2
import torch
import numpy as np

class YOLOv5Detector:
    """
    A class to handle loading the YOLOv5 model and performing detections.
    """
    def __init__(self, model_name='yolov5s'):
        """
        Initializes the detector by loading the YOLOv5 model from torch.hub.
        Args:
            model_name (str): The name of the YOLOv5 model to use (e.g., 'yolov5s', 'yolov5m').
        """
        try:
            # Load the model from the ultralytics/yolov5 repository
            self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
            print(f"YOLOv5 model ('{model_name}') loaded successfully.")
            
            # Set confidence threshold and other parameters
            self.model.conf = 0.4  # Confidence threshold
            self.model.iou = 0.45   # IoU threshold for NMS
            
        except Exception as e:
            print(f"Error loading YOLOv5 model: {e}")
            print("Please ensure you have an internet connection for the first run to download the model.")
            self.model = None

    def detect(self, frame):
        """
        Performs object detection on a single frame.
        Args:
            frame (np.array): The input image/frame in BGR format (from OpenCV).
        Returns:
            A results object from the YOLOv5 model.
        """
        if self.model is None:
            return None
        # Convert frame from BGR (OpenCV default) to RGB, which the model expects
        rgb_frame = frame[:, :, ::-1]
        results = self.model(rgb_frame)
        return results

    @staticmethod
    def process_and_draw_results(results, frame, target_classes):
        """
        Processes the detection results, draws bounding boxes, and returns the count of target objects.
        Args:
            results: The results object from the model.
            frame (np.array): The frame to draw on.
            target_classes (list): A list of class names to specifically count and draw (e.g., ['car', 'truck']).
        Returns:
            tuple: A tuple containing:
                - annotated_frame (np.array): The frame with bounding boxes and labels.
                - object_count (int): The number of detected objects from the target_classes list.
        """
        if results is None:
            return frame, 0
            
        object_count = 0
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]

        for i in range(n):
            row = cord[i]
            confidence = row[4]
            class_id = int(labels[i])
            class_name = results.names[class_id]

            if class_name in target_classes:
                object_count += 1
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
                
                # Set color for the bounding box
                bgr = (0, 255, 0) # Green
                
                # Draw the rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                
                # Prepare the label text
                label = f"{class_name} {confidence:.2f}"
                
                # Draw the label background and text
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + len(label) * 10, y1), bgr, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
        return frame, object_count
