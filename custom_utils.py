from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn as nn
from pytorchvideo.models.hub import slowfast_r50
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AnomalyScorer(nn.Module):
    def __init__(self, input_dim=2304):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()  # Add sigmoid activation
        ).to(device)  # Move model to GPU
    
   
    def forward(self, x):
        return self.net(x.to(device)).squeeze(-1)  # Ensure input is on GPU


# def load_model(model_path):
#     """Load the trained PyTorch model."""
#     model = AnomalyScorer(input_dim=2304)  # Replace with your model class
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
#     return model

def load_model(model_path):
    """Load the trained PyTorch model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AnomalyScorer(input_dim=2304)  # Replace with your model class
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_video(video_path, segment_size=32):
    """Preprocess the video into segments of 32 frames."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    segments = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to PIL image and resize
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        frame = transform(frame)
        frames.append(frame)
        
        # If we have collected enough frames for a segment, process it
        if len(frames) == segment_size:
            segment_tensor = torch.stack(frames)  # Stack frames into a tensor
            segments.append(segment_tensor)
            frames = []  # Reset frames list for the next segment
    
    cap.release()
    
    # Handle leftover frames (if any)
    if frames:
        segment_tensor = torch.stack(frames)
        segments.append(segment_tensor)
    
    return segments