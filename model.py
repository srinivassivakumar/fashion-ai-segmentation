import torch
import torch.nn as nn
import torchvision.models as models

class SegmentationModel(nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()
        self.classes = [
            'background',
            'top',
            'bottom',
            'dress',
            'outerwear',
            'shoes',
            'accessories'
        ]
        
        # Use ResNet50 as backbone
        backbone = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, len(self.classes), kernel_size=2, stride=2),
        )
        
    def forward(self, x):
        # Encode
        features = self.encoder(x)
        
        # Decode
        output = self.decoder(features)
        
        return output
