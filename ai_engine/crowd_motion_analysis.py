import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Read first frame
ret, frame1 = cap.read()
prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

while True:

    ret, frame2 = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Optical Flow calculation
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

    # Compute magnitude of motion
    magnitude, angle = cv2.cartToPolar(flow[...,0], flow[...,1])

    motion_score = np.mean(magnitude)

    # Display motion score
    cv2.putText(
        frame2,
        f"Motion Score: {motion_score:.2f}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    # Detect abnormal motion
    if motion_score > 5:
        cv2.putText(
            frame2,
            "ABNORMAL CROWD MOTION",
            (20,80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            3
        )

    cv2.imshow("Crowd Motion Analysis", frame2)

    prev_gray = gray

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()