# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set the working directory
WORKDIR /workspace

# Copy the app directory contents
COPY app/ /workspace/

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Environment variable for API key authentication
# Set this when running the container: docker run -e FASHION_AI_API_KEY="your-key" ...
# If not set, a temporary key will be generated and displayed on startup
ENV FASHION_AI_API_KEY=""

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


