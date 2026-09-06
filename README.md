# IntelliSecure: AI Smart Surveillance & Predictive Crowd Risk Platform

IntelliSecure is an AI-based video surveillance system that combines deep learning and computer vision techniques for violence detection, people detection, and crowd-risk analysis.

## Key Features

- **Violence Detection** — Fine-tuned R3D-18 3D CNN for video-based violence detection using 16-frame video clips.
- **People Detection & Counting** — Uses YOLOv8 to detect and count people across video frames.
- **Crowd Risk Analysis** — Combines people count and motion information for crowd-risk assessment.
- **Video Processing Pipeline** — Integrates multiple computer vision components into a unified analysis pipeline.
- **Interactive Interface** — Streamlit-based interface for uploading videos and viewing analysis results.

## Technologies

- Python
- PyTorch
- R3D-18
- YOLOv8
- OpenCV
- NumPy
- Pandas
- scikit-learn
- Streamlit

## System Architecture

```text
Input Video
     |
     +-------------------+
     |                   |
     v                   v
  R3D-18              YOLOv8
  3D CNN          Person Detection
     |                   |
     v                   v
Violence             People Count
Probability              |
     |                   |
     +---------+---------+
               |
               v
        Crowd Risk Analysis
               |
               v
          Streamlit UI
               |
               v
        Analysis Results
```

## Project Structure

```text
SEM6-Project/
|
├── ai_engine/
│   ├── app.py
│   ├── crowd_count.py
│   ├── crowd_density_monitor.py
│   ├── crowd_motion_analysis.py
│   ├── evaluate_model.py
│   ├── extract_features.py
│   ├── extract_frames.py
│   ├── lstm_model.py
│   ├── lstm_video_test.py
│   ├── object_extraction.py
│   ├── person_tracking.py
│   ├── pipeline.py
│   ├── r3d_model.py
│   ├── r3d_video_test.py
│   ├── stampede_risk_detector.py
│   ├── stampede_video_test.py
│   ├── suspicious_object_detection.py
│   ├── violence_model.py
│   ├── violence_video_test.py
│   └── yolo_detection.py
│
├── frontend/
│   └── app.py
│
├── docs/
│   └── Research_Paper.pdf
│
├── .gitignore
├── yolo8n.pt
└── README.md
```

## My Contributions

- Fine-tuned **R3D-18 3D CNN** for video-based violence detection using 16-frame video clips and generated violence probability scores.
- Implemented **YOLOv8-based person detection and counting** across video frames to support crowd-risk analysis.
- Developed the **Streamlit interface** for video upload and presentation of analysis results.

## Research Paper

The research paper associated with this project is available below:

[View Research Paper](docs/INTELLISECURE_research_paper.pdf)

## Project Status

**Completed Academic Project**

## Team Project

This project was developed as a team project as part of the academic curriculum.
