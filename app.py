from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import cv2
import numpy as np
from PIL import Image
import io
from model import SegmentationModel
from utils import preprocess_image, postprocess_output

app = FastAPI(title="Fashion AI Segmentation API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
model = SegmentationModel()

@app.get("/")
async def root():
    return {"message": "Fashion AI Segmentation API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Preprocess image
    preprocessed = preprocess_image(image)
    
    # Get prediction
    with torch.no_grad():
        output = model(preprocessed)
    
    # Postprocess output
    segmentation_map = postprocess_output(output)
    
    return {
        "segmentation_map": segmentation_map.tolist(),
        "classes": model.classes
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
