from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLO model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

# Read first frame for motion analysis
ret, frame1 = cap.read()
prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # -------- PERSON DETECTION --------
    results = model(frame)

    person_count = 0

    for result in results:
        boxes = result.boxes

        for box in boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            if class_name == "person":
                person_count += 1

    # -------- MOTION ANALYSIS --------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0
    )

    magnitude, angle = cv2.cartToPolar(flow[...,0], flow[...,1])
    motion_score = np.mean(magnitude)

    prev_gray = gray

    # -------- RISK SCORING --------
    risk_level = "LOW"

    if person_count > 5 and motion_score > 3:
        risk_level = "HIGH"
    elif person_count > 3 and motion_score > 2:
        risk_level = "MEDIUM"

    # -------- DISPLAY INFO --------
    annotated_frame = results[0].plot()

    cv2.putText(
        annotated_frame,
        f"People: {person_count}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Motion Score: {motion_score:.2f}",
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    color = (0,255,0)

    if risk_level == "MEDIUM":
        color = (0,165,255)

    if risk_level == "HIGH":
        color = (0,0,255)

    cv2.putText(
        annotated_frame,
        f"Stampede Risk: {risk_level}",
        (20,120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        3
    )

    cv2.imshow("Stampede Risk Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()