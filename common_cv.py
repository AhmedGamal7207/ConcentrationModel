from facedet import SCRFD
import cv2
import cv2.data
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from approach.ResEmoteNet import ResEmoteNet
from facedet import SCRFD

# Identify Face in Video Stream
def get_bbox(video_frame,face_detector):
    bboxes, keypoints = face_detector.detect(video_frame)
    return bboxes, keypoints

class EmotionModel:
    def __init__(self,model_path):
        self.device = torch.device("cpu")
        self.emotions = ['Happy', 'Surprised', 'Sad', 'Angry', 'Fear', 'Fear', 'Neutral'] # the first fear was disgust but we don't want it
        self.model = ResEmoteNet().to(self.device)
        self.checkpoint = torch.load(model_path, weights_only=True,map_location=torch.device('cpu') )
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.max_emotion = ''
        self.rounded_scores = []

    def detect_emotion(self,video_frame):
        vid_fr_tensor = self.transform(video_frame).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(vid_fr_tensor)
            probabilities = F.softmax(outputs, dim=1)
        scores = probabilities.cpu().numpy().flatten()
        rounded_scores = [round(score, 2) for score in scores]
        return rounded_scores
    
    def get_max_emotion(self,x, y, w, h, video_frame):
        crop_img = video_frame[y : y + h, x : x + w]
        pil_crop_img = Image.fromarray(crop_img)
        rounded_scores = self.detect_emotion(pil_crop_img)
        max_index = np.argmax(rounded_scores)
        max_emotion = self.emotions[max_index]
        return max_emotion, rounded_scores
    
    def execute_prediction(self,video_frame,bboxes,keypoints):
        for bbox, _ in zip(bboxes, keypoints):
            x_min, y_min, x_max, y_max = map(int, bbox[:4])
            width = x_max - x_min
            height = y_max - y_min
            self.max_emotion,self.rounded_scores = self.get_max_emotion(x_min,y_min,width,height,video_frame)
            return self.max_emotion, self.rounded_scores



