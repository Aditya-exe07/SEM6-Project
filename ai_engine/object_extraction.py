from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Objects we care about
target_objects = ["person", "backpack", "handbag", "suitcase"]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO
    results = model(frame)

    detections = []

    for result in results:
        boxes = result.boxes

        for box in boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0]

            if class_name in target_objects:
                detection = {
                    "object": class_name,
                    "confidence": round(confidence, 2),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)]
                }

                detections.append(detection)

                print(detection)

    # Show annotated frame
    annotated_frame = results[0].plot()
    cv2.imshow("Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()