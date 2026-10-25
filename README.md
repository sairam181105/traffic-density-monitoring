# Real-Time Smart Traffic Density Monitoring & Cloud Analytics Pipeline

An end-to-end intelligent transportation monitoring architecture that executes computer vision detection at the edge, calculates real-time lane vehicular densities, pushes telemetry to Amazon Web Services (AWS) S3, and continuously stages structured records into Snowflake Data Warehouse via Snowpipe.

---

## 📌 System Architecture

<p align="center">
  <img src="video/architecture.png" alt="Smart Traffic Density Monitoring Pipeline Architecture" width="850">
</p>

1. **Multi-Camera Edge Ingestion**: Threaded video capture streams frames concurrently across multiple camera indices without stalling detection threads.
2. **Dual Edge Inference Engines**:
   - **YOLOv5 Deep Learning**: Pre-trained object detection filtering specific vehicular classes (`car`, `bus`, `truck`, `motorcycle`).
   - **OpenCV MOG2 Background Subtraction**: Low-latency edge contour tracking with shadow filtering and morphological noise suppression.
3. **Local Telemetry & Time-Series Analytics**: Aggregated vehicle counts, rolling moving averages, and density distribution visualizations.
4. **Cloud Staging (AWS S3)**: Automated scheduled JSON telemetry payloads dispatched to Amazon S3 buckets.
5. **Continuous Data Warehousing (Snowflake & Snowpipe)**: Event-driven ingestion using AWS S3 Storage Integrations and Snowpipe auto-ingest into raw variant staging tables.

---

## 🎥 Demo Video

Watch the complete demonstration of vehicle detection, density tracking, and cloud data dispatch:

> **Click the video below or view it in the `video/` directory:**
> 
> *Direct link: [main_project_recording.mp4](video/main_project_recording.mp4)*

*(Tip: Once pushed to GitHub, you can drag and drop your `.mp4` into the README editor on GitHub to embed an interactive, in-browser media player).*

---

## 📂 Project Structure

```text
├── codes/
│   ├── camera_handler.py       # Threaded multi-camera capture management
│   ├── LOCAL_CSV_PATH.txt      # Sample generated vehicle density telemetry log
│   ├── main_control.py         # Lane detection controller, MOG2 engine & S3 uploader
│   ├── main.py                 # Standalone YOLOv5 video inference & visualizer
│   ├── snowflake_interface.txt # Snowflake DDL, Storage Integration & Snowpipe setup
│   └── yolo_detector.py        # YOLOv5 deep learning detection wrapper
├── papers/                     # Reference papers and research background
├── video/
│   ├── architecture.png        # System architecture diagram
│   ├── AWS final report.docx   # AWS cloud setup & implementation report
│   ├── main_project_recording.mp4 # Video demonstration of live monitoring
│   └── Recording 2025-10-01.mp4   # Raw test footage
├── .gitignore                  # Credential and artifact exclusion rules
├── requirements.txt            # Python dependencies manifest
└── README.md                   # Project documentation