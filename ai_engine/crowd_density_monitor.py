from ultralytics import YOLO
import cv2
import supervision as sv

# Load YOLO model
model = YOLO("yolov8n.pt")

# Initialize tracker
tracker = sv.ByteTrack()

# Open webcam
cap = cv2.VideoCapture(0)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    result = model(frame)[0]

    detections = sv.Detections.from_ultralytics(result)

    # Filter only persons (COCO class 0)
    detections = detections[detections.class_id == 0]

    # Track persons
    detections = tracker.update_with_detections(detections)

    person_count = len(detections)

    labels = [f"ID {tracker_id}" for tracker_id in detections.tracker_id]

    # Draw bounding boxes
    annotated_frame = box_annotator.annotate(frame, detections)
    annotated_frame = label_annotator.annotate(
        annotated_frame,
        detections,
        labels
    )

    # Display people count
    cv2.putText(
        annotated_frame,
        f"People: {person_count}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    # Crowd density warning
    if person_count > 5:
        cv2.putText(
            annotated_frame,
            "Crowd Status: HIGH DENSITY",
            (20,80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            3
        )

    cv2.imshow("Crowd Density Monitor", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()