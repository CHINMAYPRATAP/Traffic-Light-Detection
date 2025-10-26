import cv2
import numpy as np

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Define HSV color ranges (tune if needed)
# Format: lower HSV, upper HSV
colors_hsv = {
    "RED": [([0, 120, 70], [10, 255, 255]), ([170, 120, 70], [180, 255, 255])],  # red has 2 ranges
    "GREEN": [([36, 100, 100], [86, 255, 255])],
    "BLUE": [([94, 80, 2], [126, 255, 255])],
    "YELLOW": [([15, 100, 100], [35, 255, 255])],
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for speed (optional)
    frame = cv2.resize(frame, (640, 480))

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    detected_color = None

    # Check each defined color
    for color_name, ranges in colors_hsv.items():
        mask = None
        for lower, upper in ranges:
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")
            if mask is None:
                mask = cv2.inRange(hsv, lower, upper)
            else:
                mask = mask | cv2.inRange(hsv, lower, upper)

        # Find contours for the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:  # filter small noise
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
                cv2.putText(frame, color_name, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                detected_color = color_name

    # Show the result
    cv2.imshow("Color Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
