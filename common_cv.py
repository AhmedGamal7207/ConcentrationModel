from facedet import SCRFD
import cv2

# Identify Face in Video Stream
def get_bbox(video_frame,face_detector):
    bboxes, keypoints = face_detector.detect(video_frame)
    return bboxes, keypoints

