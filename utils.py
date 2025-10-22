import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess image for model input
    """
    # Resize to model input size
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Add batch dimension
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor

def postprocess_output(output: torch.Tensor) -> np.ndarray:
    """
    Convert model output to segmentation map
    """
    # Get predictions
    predictions = torch.argmax(output, dim=1)
    
    # Convert to numpy array
    segmentation_map = predictions.squeeze().cpu().numpy()
    
    return segmentation_map
