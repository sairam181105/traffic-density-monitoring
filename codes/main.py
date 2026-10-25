import cv2
import time
from datetime import datetime
from yolo_detector import YOLOv5Detector
import matplotlib.pyplot as plt

VIDEO_PATH = '/home/marias/Downloads/output.mp4'
VEHICLE_CLASSES_TO_COUNT = ['car', 'motorcycle', 'bus', 'truck']
WINDOW_TITLE = "Traffic Density Monitoring from Video"

def main():
    print("Loading YOLOv5 model...")
    detector = YOLOv5Detector(model_name='yolov5s')
    if detector.model is None:
        print("Model failed to load.")
        return

    print(f"Opening video file: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    frame_indices = []
    vehicle_counts_over_time = []

    print("Starting video processing... Press 'q' to quit.")
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached or failed to read frame.")
            break

        results = detector.detect(frame)
        annotated_frame, vehicle_counts = detector.process_and_draw_results(
            results, frame.copy(), VEHICLE_CLASSES_TO_COUNT)

        total_density = vehicle_counts

        # Store vehicle count data
        frame_indices.append(frame_num)
        vehicle_counts_over_time.append(total_density)

        text = f"Traffic Density: {total_density}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        box_x, box_y = 10, 10  # Top-left corner of the box
        box_width = text_width + 10  # Adding some padding
        box_height = text_height + baseline + 10  # Padding

        cv2.rectangle(annotated_frame,
              (box_x, box_y),
              (box_x + box_width, box_y + box_height),
              (0, 0, 0),  # Black background
              cv2.FILLED)

        text_x = box_x + 5
        text_y = box_y + text_height + 5
        cv2.putText(annotated_frame, text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)

        cv2.imshow(WINDOW_TITLE, annotated_frame)

        frame_num += 1

        if cv2.waitKey(30) & 0xFF == ord('q'):
            print("Quitting on user request.")
            break

    cap.release()
    cv2.destroyAllWindows()

    # Plot vehicle counts over time
    plt.figure(figsize=(10, 6))
    plt.plot(frame_indices, vehicle_counts_over_time, label='Number of Vehicles')
    plt.xlabel('Frame Number')
    plt.ylabel('Vehicle Count')
    plt.title('Number of Vehicles on Screen Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Video processing finished and graph displayed.")

if __name__ == "__main__":
    main()
