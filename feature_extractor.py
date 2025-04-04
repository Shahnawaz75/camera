import os
import av
import torch
import numpy as np
import cv2
from tqdm import tqdm
from pytorchvideo.models.hub import slowfast_r50
import torch.nn as nn
import torch.cuda.amp as amp
from collections import deque

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Config:
    segment_frames = 32
    stride_frames = 16
    max_segments = 350
    crop_size = 224
    slowfast_mean = [0.45, 0.45, 0.45]
    slowfast_std = [0.225, 0.225, 0.225]

class FeatureExtractor:
    def __init__(self):
        self.model = slowfast_r50(pretrained=True)
        self.model.blocks[-1] = nn.Identity()
        self.model = self.model.to(device).eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.scaler = amp.GradScaler(enabled=False)

    def _process_pathway(self, frames):
        tensor = torch.from_numpy(np.stack(frames).transpose(0, 3, 1, 2)).float().to(device)
        normalized = ((tensor / 255.0 - torch.tensor(Config.slowfast_mean, device=device).view(1, 3, 1, 1)) 
                     / torch.tensor(Config.slowfast_std, device=device).view(1, 3, 1, 1))
        return normalized.permute(1, 0, 2, 3)  

    def extract_features(self, video_path):
        try:
            container = av.open(video_path)
            stream = container.streams.video[0]
            frame_buffer = deque(maxlen=Config.segment_frames + Config.stride_frames)
            features = []
            segment_count = 0
            for frame in container.decode(stream):
                img = frame.to_ndarray(format='rgb24')
                img = cv2.resize(img, (256, 256)).astype(np.uint8)
                frame_buffer.append(img)
                if len(frame_buffer) >= Config.segment_frames:
                    if segment_count % Config.stride_frames == 0:
                        segment = list(frame_buffer)[-Config.segment_frames:]
                        slow_seg = segment[::4]
                        fast_seg = segment
                        
                        slow_tensor = self._process_pathway(slow_seg).unsqueeze(0)
                        fast_tensor = self._process_pathway(fast_seg).unsqueeze(0)
                        with torch.no_grad(), amp.autocast(enabled=False):
                            output = self.model([slow_tensor, fast_tensor])
                            pooled = nn.functional.adaptive_avg_pool2d(output, (1, 1))
                            features.append(pooled.view(1, -1).cpu())
                        del slow_tensor, fast_tensor, output
                        torch.cuda.empty_cache()
                    segment_count += 1
                if len(features) >= Config.max_segments:
                    break
            return torch.cat(features, dim=0) if features else None
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            return None
        
    def extract_features_from_frames(self, frames):
        try:
            frame_buffer = deque(maxlen=Config.segment_frames + Config.stride_frames)
            features = []
            for img in frames:
                img = cv2.resize(img, (256, 256)).astype(np.uint8)
                frame_buffer.append(img)
                if len(frame_buffer) >= Config.segment_frames:
                    segment = list(frame_buffer)[-Config.segment_frames:]
                    slow_seg = segment[::4]
                    fast_seg = segment
                    
                    slow_tensor = self._process_pathway(slow_seg).unsqueeze(0)
                    fast_tensor = self._process_pathway(fast_seg).unsqueeze(0)
                    with torch.no_grad(), amp.autocast(enabled=False):
                        output = self.model([slow_tensor, fast_tensor])
                        pooled = nn.functional.adaptive_avg_pool2d(output, (1, 1))
                        features.append(pooled.view(1, -1).cpu())
                    del slow_tensor, fast_tensor, output
                    torch.cuda.empty_cache()
            return torch.cat(features, dim=0) if features else None
        except Exception as e:
            print(f"Error processing frames: {str(e)}")
            return None