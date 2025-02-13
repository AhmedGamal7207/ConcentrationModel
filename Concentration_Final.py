import cv2 
import dlib
import numpy as np
from tensorflow.keras.models import load_model
from facedet import SCRFD
from math import hypot
from common_cv import get_bbox

class AIModel:
    # Initialise models
    def __init__(self):
        self.emotion_model = load_model('models/emotion_recognition.h5', compile=False)

        # SCRFD Face Detector
        self.face_detector = SCRFD(model_path="models/det_10g.onnx")

        # dlib Face Landmarks Detector
        self.predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
        self.detector = dlib.get_frontal_face_detector()

        self.x = 0
        self.y = 0
        self.emotion = 5
        self.size = 0
        self.frame_count = 0

        self.bboxes = []
        self.keypoints = []

    # Function to detect facial landmarks
    def get_landmarks(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        if faces:
            shape = self.predictor(gray, faces[0])
            landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
            return landmarks
        return None

    # Function for finding midpoint of 2 points
    def midpoint(self, p1, p2):
        return int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)

    # Function for eye size (Blink Detection)
    def get_blinking_ratio(self, frame, eye_points, facial_landmarks):
        if not facial_landmarks:
            return 0
        
        left_point = facial_landmarks[eye_points[0]]
        right_point = facial_landmarks[eye_points[3]]
        center_top = self.midpoint(facial_landmarks[eye_points[1]], facial_landmarks[eye_points[2]])
        center_bottom = self.midpoint(facial_landmarks[eye_points[5]], facial_landmarks[eye_points[4]])

        hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
        ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
        
        return ver_line_length / hor_line_length

    # Gaze Detection Function
    def get_gaze_ratio(self, frame, eye_points, facial_landmarks, gray):
        if not facial_landmarks:
            return 1, 1

        left_eye_region = np.array([facial_landmarks[eye_points[i]] for i in range(6)], np.int32)
        
        height, width, _ = frame.shape
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [left_eye_region], True, 255, 2)
        cv2.fillPoly(mask, [left_eye_region], 255)
        eye = cv2.bitwise_and(gray, gray, mask=mask)

        min_x, max_x = np.min(left_eye_region[:, 0]), np.max(left_eye_region[:, 0])
        min_y, max_y = np.min(left_eye_region[:, 1]), np.max(left_eye_region[:, 1])
        gray_eye = eye[min_y:max_y, min_x:max_x]

        _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)

        height, width = threshold_eye.shape
        left_side_white = cv2.countNonZero(threshold_eye[:, :width // 2])
        right_side_white = cv2.countNonZero(threshold_eye[:, width // 2:])
        up_side_white = cv2.countNonZero(threshold_eye[:height // 2, :])
        down_side_white = cv2.countNonZero(threshold_eye[height // 2:, :])

        lr_gaze_ratio = (left_side_white + 10) / (right_side_white + 10)
        ud_gaze_ratio = (up_side_white + 10) / (down_side_white + 10)

        return lr_gaze_ratio, ud_gaze_ratio

    # Main function for analysis
    def detect_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        font = cv2.FONT_HERSHEY_SIMPLEX

        self.bboxes, self.keypoints = get_bbox(frame,self.face_detector)
        benchmark = []

        for bbox, _ in zip(self.bboxes, self.keypoints):
            x, y, x1, y1 = map(int, bbox[:4])
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
            #frame = frame[x:x1,y:y1]
            landmarks = self.get_landmarks(frame)

            if landmarks:
                left_eye_ratio = self.get_blinking_ratio(frame, [36, 37, 38, 39, 40, 41], landmarks)
                gaze_ratio_lr, gaze_ratio_ud = self.get_gaze_ratio(frame, [36, 37, 38, 39, 40, 41], landmarks, gray)

                benchmark.append([gaze_ratio_lr, gaze_ratio_ud, left_eye_ratio])
                self.detect_emotion(gray)
                ci = self.gen_concentration_index()

                emotions = {0: 'Angry', 1: 'Fear', 2: 'Happy', 3: 'Sad', 4: 'Surprised', 5: 'Neutral'}
                cv2.putText(frame, emotions[self.emotion], (50, 150), font, 2, (0, 0, 255), 3)
                cv2.putText(frame, ci, (50, 250), font, 2, (0, 0, 255), 3)

                self.x = gaze_ratio_lr
                self.y = gaze_ratio_ud
                self.size = left_eye_ratio

        return frame
# Function for detecting emotion 

    def detect_emotion(self, gray):
        emotions = {0: 'Angry', 1: 'Fear', 2: 'Happy', 3: 'Sad', 4: 'Surprised', 5: 'Neutral'}

        # Convert grayscale to RGB to match face detector input expectations
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        # Only predict every 5 frames
        #if self.frame_count % 5 == 0:
        try:
            for bbox, _ in zip(self.bboxes, self.keypoints):
                x, y, x1, y1 = map(int, bbox[:4])
                cropped_face = gray[y:y1, x:x1]

                # Ensure the cropped face is valid before resizing
                if cropped_face.shape[0] > 0 and cropped_face.shape[1] > 0:
                    test_image = cv2.resize(cropped_face, (48, 48))
                    test_image = test_image.reshape([-1, 48, 48, 1])
                    test_image = np.multiply(test_image, 1.0 / 255.0)

                    probab = self.emotion_model.predict(test_image)[0] * 100
                    label = np.argmax(probab)
                    self.emotion = label
        except:
            pass

        #self.frame_count += 1

        # # 	Weights from Sharma et.al. (2019)
        # Neutral	0.9
        # Happy 	0.6
        # Surprised	0.6
        # Sad	    0.3

        # Anger	    0.25
        # Fearful	0.3
        # 0: 'Angry', 1: 'Fear', 2: 'Happy', 3: 'Sad', 4: 'Surprised', 5: 'Neutral'}

    def gen_concentration_index(self):
        weight = 0
        emotionweights = {0: 0.25, 1: 0.3, 2: 0.6,
                          3: 0.3, 4: 0.6, 5: 0.9}


# 	      Open Semi Close
# Centre	5	1.5	0
# Upright	2	1.5	0
# Upleft	2	1.5	0
# Right	    2	1.5	0
# Left	    2	1.5	0
# Downright	2	1.5	0
# Downleft	2	1.5	0
        gaze_weights = 0

        if self.size < 0.2:
            gaze_weights = 0
            print("Closed")
        elif self.size > 0.2 and self.size < 0.3:
            gaze_weights = 1.5
            print("Semi Open")
        else:
            print("Open")
            if self.x < 1.8 and self.x > 0.8: # Center # Changed from 2 to 1 --> 1.8 to 0.8
                print("Center")
                print(f"x: {self.x}")
                print(f"y: {self.y}")
                gaze_weights = 5
            else: 
                gaze_weights = 2

# Concentration index is a percentage : max weights product = 4.5
        concentration_index = (
            emotionweights[self.emotion] * gaze_weights) / 4.5
        if concentration_index > 0.65:
            return "You are highly engaged!"
        elif concentration_index > 0.25 and concentration_index <= 0.65:
            return "You are engaged."
        else:
            return "Pay attention!"
