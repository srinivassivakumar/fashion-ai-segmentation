# Fashion AI Segmentation

A deep learning-based fashion segmentation system that can identify and segment different clothing items in images.

## Features

- Clothing item segmentation
- Multi-class classification
- Real-time processing capability
- Support for multiple image formats
- REST API interface
- Docker containerization

## Prerequisites

- Python 3.8+
- PyTorch
- OpenCV
- Docker (optional)

## Quick Start

### Using Docker

```bash
# Pull the image
docker pull srinivassivakumar123/fashion-ai-segmentation

# Run the container
docker run -p 8080:8080 srinivassivakumar123/fashion-ai-segmentation
```

### Manual Setup

1. Clone the repository:
```bash
git clone https://github.com/srinivassivakumar/fashion-ai-segmentation.git
cd fashion-ai-segmentation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

The server will start at `http://localhost:8080`

## API Endpoints

- `POST /segment`: Segment clothing items in an image
  - Input: Image file
  - Output: Segmentation mask and classifications

## Model Architecture

The segmentation model is based on U-Net architecture with a ResNet backbone, trained on a custom fashion dataset.

## Docker Build

To build the Docker image locally:

```bash
docker build -t fashion-ai-segmentation .
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
