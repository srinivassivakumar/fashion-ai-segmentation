# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Environment variable for API key authentication
# Set this when running the container: docker run -e FASHION_AI_API_KEY="your-key" ...
# If not set, a temporary key will be generated and displayed on startup
ENV FASHION_AI_API_KEY=""

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run the FastAPI application
# The app is located at app/main.py with the FastAPI instance named 'app'
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]


