import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import numpy as np


# Model definition using MobileNetV2
class ActionModel(nn.Module):
    def __init__(self, num_frames=5):
        super(ActionModel, self).__init__()
        # Load pretrained MobileNetV2
        mobilenet = models.mobilenet_v2(pretrained=True)
        self.num_frames = num_frames
        # Remove the last classifier layer
        self.features = nn.Sequential(*list(mobilenet.children())[:-1])
        
        # Add custom classifier for 2 classes
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1280 * num_frames, 512),  # 1280 is MobileNetV2's default output channels
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
        
    def forward(self, x):
        # x shape: (batch_size, 3, height * num_frames, width)
        batch_size = x.size(0)
        # Process each frame independently
        chunks = torch.chunk(x, self.num_frames, dim=2)
        frame_features = []
        for i in range(self.num_frames):
            features = self.features(chunks[i])
            frame_features.append(features)
        # Concatenate features from all frames
        x = torch.cat(frame_features, dim=1)
        x = self.classifier(x)
        return x
