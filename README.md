# Fashion AI Segmentation

An advanced deep learning-based fashion segmentation system using **SegFormer + SAM (Segment Anything Model) + Auto-Center Pipeline** to accurately identify, segment, and extract clothing items from images.

## Features

- **Advanced Segmentation**: Uses SegFormer for initial segmentation and SAM for refinement
- **Auto-Center Cropping**: Automatically centers and crops individual garments
- **Multi-Class Fashion Detection**: Detects hats, upper clothes, pants, dresses, and more
- **FastAPI Web Interface**: Easy-to-use web GUI for image upload and processing
- **REST API**: Programmatic access for integration
- **Docker Support**: Containerized for easy deployment

## Architecture

- **SegFormer**: Initial semantic segmentation (Fashion Segmentation Model)
- **SAM (Segment Anything)**: Mask refinement for precise boundaries
- **Auto-Center Algorithm**: Smart cropping and centering of detected garments
- **FastAPI Backend**: High-performance async API server

## Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA GPU (recommended for faster processing)
- Docker (optional)

## Quick Start

### Using Docker (Recommended)

```bash
# Pull the image from Docker Hub
docker pull srinivassivakumar123/fashion-ai-segmentation

# Run the container
docker run -p 8080:8080 srinivassivakumar123/fashion-ai-segmentation
```

Then open your browser and go to: `http://localhost:8080`

### Manual Setup

1. **Clone the repository:**
```bash
git clone https://github.com/srinivassivakumar/fashion-ai-segmentation.git
cd fashion-ai-segmentation
```

2. **Install dependencies:**
```bash
cd app
pip install -r requirements.txt
```

3. **Download SAM model weights:**
```bash
# Create model directory
mkdir -p data/models

# Download SAM ViT-B model
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O data/models/sam_vit_b_01ec64.pth
```

4. **Run the application:**
```bash
python main.py
```

The server will start at `http://localhost:8080`

## API Endpoints

### Public Endpoints (No Authentication Required)
- `GET /` - Web GUI for image upload and visualization
- `GET /results/{image_name}` - View processing results for a specific image
- `GET /outputs/{path}` - Serve output files (segmentation masks, crops, etc.)

### Protected Endpoints (API Key Required)
- `POST /upload` - Upload and process an image
  - Input: Image file (JPEG/PNG)
  - Headers: `X-API-Key: your_api_key_here`
  - Output: JSON with paths to processed results
  
- `GET /list-outputs/{image_name}` - List all output files for a processed image
  - Headers: `X-API-Key: your_api_key_here`
  - Output: JSON with file paths organized by type

## API Key Authentication

This API uses API key authentication to protect the upload and list endpoints while keeping the web interface publicly accessible.

### Setting Up API Key

#### For Local Development
Set the environment variable before running:

```bash
# Linux/Mac
export FASHION_AI_API_KEY="your-secure-api-key-here"
python app/main.py

# Windows PowerShell
$env:FASHION_AI_API_KEY="your-secure-api-key-here"
python app/main.py
```

If no API key is set, the server will generate and display a temporary one on startup.

#### For Docker
Pass the API key as an environment variable:

```bash
docker run -p 8080:8080 -e FASHION_AI_API_KEY="your-secure-api-key-here" srinivassivakumar123/fashion-ai-segmentation
```

#### For Railway (Production)
1. Go to your Railway project dashboard
2. Click on your service
3. Go to "Variables" tab
4. Add a new variable:
   - Key: `FASHION_AI_API_KEY`
   - Value: Your secure API key (generate using: `python -c "import secrets; print(secrets.token_urlsafe(32))"`)

### Using the API with Authentication

#### cURL Examples

**Upload an image:**
```bash
curl -X POST http://localhost:8000/upload \
  -H "X-API-Key: your_api_key_here" \
  -F "file=@/path/to/image.jpg"
```

**List outputs:**
```bash
curl -H "X-API-Key: your_api_key_here" \
  http://localhost:8000/list-outputs/image_name
```

**Download all crops:**
```bash
curl -s -H "X-API-Key: your_api_key_here" \
  http://localhost:8000/list-outputs/temp_m_2 | \
  jq -r '.files.crops[]' | \
  while read filename; do 
    echo "Downloading: $filename"
    curl -O "http://localhost:8000/outputs/$filename"
  done
```

#### Python Example

```python
import requests

API_KEY = "your_api_key_here"
BASE_URL = "https://fashion-ai-segmentation-production.up.railway.app"

# Upload and process image
headers = {"X-API-Key": API_KEY}
files = {"file": open("image.jpg", "rb")}
response = requests.post(f"{BASE_URL}/upload", headers=headers, files=files)
result = response.json()

print(f"Processed: {result['image_name']}")
print(f"Components: {result['num_components']}")
print(f"Categories: {result['categories']}")

# List all output files
image_name = result["image_name"]
outputs = requests.get(f"{BASE_URL}/list-outputs/{image_name}", headers=headers)
print(outputs.json())

# Download a specific crop
crop_url = f"{BASE_URL}/outputs/crops_centered/{image_name}_upper_clothes_1.png"
crop_response = requests.get(crop_url)
with open("crop.png", "wb") as f:
    f.write(crop_response.content)
```

## Output Structure

After processing an image, the following outputs are generated:

```
outputs/
├── {image_name}_original.jpg          # Original uploaded image
├── {image_name}_visualization.png     # Visualization with segmentation overlay
├── masks_raw/
│   └── {image_name}_segformer_raw.png # Raw SegFormer segmentation
├── masks_refined/
│   └── {image_name}_sam_refined.png   # SAM-refined segmentation
└── crops_centered/
    ├── {image_name}_hat_1.png         # Centered crop of detected hat
    ├── {image_name}_upper_clothes_2.png
    ├── {image_name}_pants_4.png
    └── ...                            # Other detected garments
```

## Supported Garment Classes

1. Hat
2. Upper Clothes (shirts, t-shirts, jackets, etc.)
3. Skirt
4. Pants
5. Dress
6. Belt
7. Left Shoe
8. Right Shoe
9. Face (detected but not cropped)
10. Left Leg
11. Right Leg
12. Left Arm
13. Right Arm

## Docker Build

To build the Docker image locally:

```bash
docker build -t fashion-ai-segmentation .
docker run -p 8080:8080 fashion-ai-segmentation
```

## Project Structure

```
.
├── app/
│   ├── main.py                        # FastAPI application
│   ├── requirements.txt               # Python dependencies
│   ├── src/
│   │   ├── segformer_sam_auto_center.py   # Main pipeline
│   │   ├── segformer_sam_extractor.py     # Segmentation logic
│   │   └── postprocess_center_crops.py    # Crop processing
│   ├── data/
│   │   └── models/                    # Model weights (download separately)
│   └── outputs/                       # Processing results
├── Dockerfile                         # Docker configuration
└── README.md
```

## Performance

- Processing time: ~2-5 seconds per image (GPU)
- Processing time: ~10-20 seconds per image (CPU)
- Supported image formats: JPEG, PNG
- Maximum recommended image size: 4096x4096 pixels

## Troubleshooting

### Model Download Issues
If the SAM model fails to download automatically, manually download it:
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### CUDA/GPU Issues
If you encounter CUDA errors, the application will automatically fall back to CPU mode.

### Memory Issues
For large images, consider resizing them before processing to avoid memory issues.

## License

MIT License

## Citation

If you use this code, please cite the original papers:
- SegFormer: https://arxiv.org/abs/2105.15203
- SAM: https://arxiv.org/abs/2304.02643

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues or questions, please open an issue on GitHub.
