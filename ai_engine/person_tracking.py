from ultralytics import YOLO
import cv2
import supervision as sv

# Load YOLO model
model = YOLO("yolov8n.pt")

# Tracker
tracker = sv.ByteTrack()

# Open webcam
cap = cv2.VideoCapture(0)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection
    results = model(frame)[0]

    detections = sv.Detections.from_ultralytics(results)

    # Track objects
    detections = tracker.update_with_detections(detections)

    labels = []

    for tracker_id in detections.tracker_id:
        labels.append(f"ID {tracker_id}")

    # Draw boxes
    annotated_frame = box_annotator.annotate(
        scene=frame,
        detections=detections
    )

    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels
    )

    cv2.imshow("Person Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()