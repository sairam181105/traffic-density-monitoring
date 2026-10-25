import cv2
import numpy as np
import json
import time
import boto3
import sys
import csv
import os
import pandas as pd # <-- REQUIRED for local graphs
import matplotlib.pyplot as plt # <-- REQUIRED for local graphs
from datetime import datetime

# --- AWS & LANE CONFIGURATION (CRITICAL) -
LANE_NAME = 'NORTH_LANE'
S3_BUCKET_NAME = 'marias181105' # Replace with your actual bucket name
S3_REGION = 'us-east-1' 
S3_FILE_KEY = 'LOCAL_CSV_PATH.csv' # The full path and filename in S3
CSV_COLUMNS = ['Timestamp', 'Vehicle Count', 'Interval_Sec', 'Lane_ID']
LOCAL_CSV_PATH = '/home/marias/venv/LOCAL_CSV_PATH' # Output file for local graphs

# --- GPIO CONFIGURATION ---          
CYCLE_INTERVAL_SECONDS = 30      


s3_client = boto3.client('s3', region_name=S3_REGION)

class TrafficDensityDetector:
    def __init__(self, source=0,output_csv='LOCAL_CSV_PATH'):
        self.source = source
        self.output_csv = output_csv
        self.min_contour_area = 1000
        self.last_phase_start_time = time.time()
        self.data_records = [] # List to hold local data [timestamp, count]
        
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )
    def detect_vehicles(self, frame):
        """Detect vehicles in a frame using background subtraction"""
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove shadows (they appear as gray in the mask)
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours by area and count vehicles
        vehicle_count = 0
        detections = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_contour_area:
                vehicle_count += 1
                x, y, w, h = cv2.boundingRect(contour)
                detections.append((x, y, w, h))
        
        return vehicle_count, detections, fg_mask
    
    def draw_detections(self, frame, detections, count):
        """Draw bounding boxes and vehicle count on frame"""
        for (x, y, w, h) in detections:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Display count
        cv2.putText(
            frame, f'Vehicles: {count}', (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3
        )
        
        return frame
         
    def save_local_csv(self):
        """Saves accumulated data to a local CSV file upon exit for graphing."""
        if not self.data_records:
            return
        
        # Note: We are saving the ACCUMULATED count at the END of the red phase only, 
        # so the header reflects the decision points, not per-second counts.
        with open(LOCAL_CSV_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Vehicle Count']) 
            writer.writerows(self.data_records)
        print(f"\nLocal graph data saved to {LOCAL_CSV_PATH}")

    # --- S3 UPLOAD (Cloud Pipeline) ---
    def upload_decision_log(self, count):
        """Uploads the decision log as a JSON file to the S3 bucket root."""
        timestamp_iso = datetime.utcnow().isoformat()
        
        data_payload = {
            'timestamp': timestamp_iso,
            'lane': LANE_NAME,
            'vehicle_count': count,
            'red_phase_duration_sec': CYCLE_INTERVAL_SECONDS
        }
        
        s3_key = ""

        try:
            self.s3_client.put_object(
                Bucket=marias181105,
                Key=s3_key,
                Body=json.dumps(data_payload),
                ContentType='application/json'
            )
            print(f"--- LOGGED TO S3 --- Count={count}")
        except Exception as e:
            print(f"FATAL S3 ERROR: Failed to upload log ({s3_key}): {e}")

    # --- GRAPHS (Original Matplotlib/Pandas Logic) ---
    def generate_graphs(self):
        """Generate visualization graphs from saved local CSV data."""
        if not os.path.exists(LOCAL_CSV_PATH):
            print(f"Error: {LOCAL_CSV_PATH} not found. Run detection first.")
            return
        
        # Read data
        df = pd.read_csv(LOCAL_CSV_PATH)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # --- Plotting logic identical to your original code ---
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Traffic Density Analysis', fontsize=16, fontweight='bold')
        
        # 1. Time series plot
        axes[0, 0].plot(df['Timestamp'], df['Vehicle Count'], marker='o', linewidth=2, markersize=4)
        axes[0, 0].set_xlabel('Time'); axes[0, 0].set_ylabel('Vehicle Count')
        axes[0, 0].set_title('Vehicle Count Over Time'); axes[0, 0].grid(True, alpha=0.3); axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Distribution histogram
        axes[0, 1].hist(df['Vehicle Count'], bins=20, color='skyblue', edgecolor='black')
        axes[0, 1].set_xlabel('Vehicle Count'); axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Vehicle Count Distribution'); axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Statistics summary
        stats_text = f"Statistics Summary:\n\nTotal Records: {len(df)}\nMean Vehicles: {df['Vehicle Count'].mean():.2f}\nMax Vehicles: {df['Vehicle Count'].max()}"
        axes[1, 0].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center', family='monospace')
        axes[1, 0].axis('off'); axes[1, 0].set_title('Statistics')
        
        # 4. Moving average
        window_size = min(10, len(df))
        df['Moving_Avg'] = df['Vehicle Count'].rolling(window=window_size).mean()
        axes[1, 1].plot(df['Timestamp'], df['Vehicle Count'], alpha=0.5, label='Actual', linewidth=1)
        axes[1, 1].plot(df['Timestamp'], df['Moving_Avg'], color='red', linewidth=2, label=f'Moving Avg (window={window_size})')
        axes[1, 1].set_xlabel('Time'); axes[1, 1].set_ylabel('Vehicle Count')
        axes[1, 1].set_title('Vehicle Count with Moving Average'); axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3); axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        graph_filename = LOCAL_CSV_PATH.replace('.csv', '_analysis.png')
        plt.savefig(graph_filename, dpi=300, bbox_inches='tight')
        print(f"Graphs saved to {graph_filename}")
        
        plt.show()
        
    def save_to_csv(self):
        """Save collected data to CSV file"""
        with open(self.output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Vehicle Count'])
            writer.writerows(self.data_records)
        print(f"\nData saved to {self.output_csv}")

    def run(self, duration=None):
        cap = cv2.VideoCapture(self.source)
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        print("Starting traffic detection...")
        print("Press 'q' to quit")
        
        start_time = time.time()
        last_record_time = start_time
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("End of video or cannot read frame")
                break
            
            # Resize frame for better performance
            frame = cv2.resize(frame, (800, 600))
            
            # Detect vehicles
            count, detections, mask = self.detect_vehicles(frame)
            
            # Draw detections
            display_frame = self.draw_detections(frame.copy(), detections, count)
            
            # Record data every second
            current_time = time.time()
            if current_time - last_record_time >= 1.0:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.data_records.append([timestamp, count])
                last_record_time = current_time
                print(f"{timestamp} - Vehicles: {count}")
            
            # Display frames
            cv2.imshow('Traffic Detection', display_frame)

            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Check duration
            if duration and (time.time() - start_time) >= duration:
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Save data
        if self.data_records:
            self.save_to_csv()

    # --- MAIN RUNNER METHOD ---

if __name__ == "__main__":
    if len(sys.argv) > 1:
        VIDEO_SOURCE = sys.argv[1]
    else:
        VIDEO_SOURCE = '/home/marias/Downloads/output.mp4'

    detector = TrafficDensityDetector(source=VIDEO_SOURCE)
    
    # --- Main Execution Choice ---
    print("\nOptions:")
    print("1. Detect traffic and collect data (and save local CSV)")
    print("2. Generate graphs from existing local CSV")
    
    choice = input("\nEnter your choice (1/2): ").strip()
    
    if choice == "1":
        detector.run()
    elif choice == "2":
        detector.generate_graphs()
    else:
        print("Invalid choice. Running detection by default.")
        detector.run()
