import json
from time import time

import cv2
import numpy as np
from mtcnn import MTCNN

from text import put_text

cv2.putText = put_text

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Initialize some variables
current_frame = 0
image_number = 0
face_locations = []
detector = MTCNN()

while True:
    # Only process every 2 frames
    if current_frame % 2 == 0:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        if not ret:
            break
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]

        face_locations_mtcnn = [
            face["box"] for face in detector.detect_faces(rgb_frame)
        ]

        face_locations = [
            (l[0], l[1], l[0] + l[2], l[1] + l[3]) for l in face_locations_mtcnn
        ]

        for left, top, right, bottom in face_locations:
            if (
                min(left, top, right, bottom) < 0
                or min(right - left, bottom - top) < 40
                or not (0.8 < (right - left) / (bottom - top) < 1.2)
            ):
                continue
            image_number += 1
            cv2.imwrite(
                f"faces/{image_number}.jpg",
                frame[
                    max(top - 20, 0) : min(bottom + 20, frame.shape[0]),
                    max(left - 20, 0) : min(right + 20, frame.shape[1]),
                ],
            )

    # Display the results
    for left, top, right, bottom in face_locations:
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    # Display the resulting image
    cv2.imshow("Video", frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
