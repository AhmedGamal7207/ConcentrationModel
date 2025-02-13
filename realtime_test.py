# Run this file on CV2 in local machine to construct a Concentration Index (CI).
# Video image will show emotion on first line, and engagement on second. Engagement/concentration classification displays either 'Pay attention', 'You are engaged' and 'you are highly engaged' based on CI. Webcam is required.
# Analysis is in 'Util' folder.

from Concentration_Final import AIModel
import cv2
import numpy as np

# Initializing
cap = cv2.VideoCapture(0)
ana = AIModel()
frame_count = 0
# Capture every frame and send to detector
while True:
    _, frame = cap.read()
    if frame_count % 5 == 0:
        bm = ana.detect_face(frame)
        frame_count = 0
    else:
        frame_count = frame_count+1

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
# Exit if 'q' is pressed
    if key == ord('q'):
        break

# Release the memory
cap.release()
cv2.destroyAllWindows()
