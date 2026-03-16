from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Objects considered suspicious
suspicious_objects = ["backpack", "handbag", "suitcase"]

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)

    for result in results:
        boxes = result.boxes

        for box in boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0]

            if class_name in suspicious_objects:

                print(f"Suspicious Object Detected: {class_name}")

                # draw red box
                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0,0,255),
                    2
                )

                cv2.putText(
                    frame,
                    class_name,
                    (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,0,255),
                    2
                )

    cv2.imshow("Suspicious Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()