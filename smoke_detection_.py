import cv2
import numpy as np
import tkinter
from tkinter import messagebox

# Alert control
alert_shown = False
smoke_detection_count = 0
SMOKE_CONFIRMATION_FRAMES = 5  # Require detection in 5 consecutive frames
smoke_confidence_threshold = 0.5  # Minimum confidence to show detection

# Initialize tkinter for pop-up
root = tkinter.Tk()
root.withdraw()  # Hide the main tkinter window

# Open the webcam
cap = cv2.VideoCapture(0)

# Read the first frame
ret, frame1 = cap.read()
frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
frame1_gray = cv2.GaussianBlur(frame1_gray, (21, 21), 0)

while True:
    # Read the next frame
    ret, frame2 = cap.read()
    if not ret:
        break

    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.GaussianBlur(frame2_gray, (21, 21), 0)

    # Find the difference between frames
    diff = cv2.absdiff(frame1_gray, frame2_gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find moving regions
    contours, _ = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue

        (x, y, w, h) = cv2.boundingRect(contour)
        roi = frame2[y:y+h, x:x+w]
        avg_color = cv2.mean(roi)[:3]

        # Enhanced smoke detection with multiple criteria
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        avg_hsv = cv2.mean(roi_hsv)[:3]

        # Enhanced smoke detection with stricter criteria
        # Calculate confidence score based on multiple factors
        confidence_score = 0.0

        # 1. Color analysis (more restrictive)
        is_grayish = (100 < avg_color[0] < 150 and 100 <
                      avg_color[1] < 150 and 100 < avg_color[2] < 150)
        color_variance = np.std([avg_color[0], avg_color[1], avg_color[2]])
        if is_grayish and color_variance < 15:  # Low color variance = more gray
            confidence_score += 0.3

        # 2. Saturation check (smoke is desaturated)
        if avg_hsv[1] < 30:  # Very low saturation
            confidence_score += 0.25

        # 3. Shape and size analysis
        aspect_ratio = w / h
        area = cv2.contourArea(contour)
        if 0.7 < aspect_ratio < 2.5 and 500 < area < 5000:  # More restrictive
            confidence_score += 0.2

        # 4. Position analysis (smoke rises)
        # More restrictive upper region
        is_upper_region = y < frame2.shape[0] * 0.6
        if is_upper_region:
            confidence_score += 0.15

        # 5. Texture analysis (smoke has low texture variance)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        texture_variance = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
        if texture_variance < 50:  # Smooth texture like smoke
            confidence_score += 0.1

        # Only show detection if confidence is high enough
        if confidence_score >= smoke_confidence_threshold:
            smoke_detection_count += 1

            # Visual feedback with confidence level
            cv2.putText(frame2, f"SMOKE: {confidence_score:.1f} ({smoke_detection_count}/{SMOKE_CONFIRMATION_FRAMES})",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Only alert after consistent high-confidence detection
            if smoke_detection_count >= SMOKE_CONFIRMATION_FRAMES and not alert_shown:
                messagebox.showwarning(
                    "ALERT!", "Smoke Detected! Take Action Immediately!")
                alert_shown = True
        else:
            # Reset counter if confidence is too low
            smoke_detection_count = max(
                0, smoke_detection_count - 2)  # Faster decay

    # Show the camera feed
    cv2.imshow("Smoke Detection", frame2)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Update previous frame
    frame1_gray = frame2_gray.copy()

# Cleanup
cap.release()
cv2.destroyAllWindows()
