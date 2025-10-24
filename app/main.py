#!/usr/bin/env python3
"""
========================================================================================================
FASTAPI WRAPPER FOR SEGFORMER + SAM + AUTO-CENTER PIPELINE
========================================================================================================
This API provides a web interface to upload images and process them using the integrated
SegFormer + SAM + Auto-Center pipeline.

Endpoints:
    GET  /                      - Web GUI for image upload
    POST /upload                - Upload and process image
    GET  /results/{image_name}  - View processing results
    GET  /outputs/{path:path}   - Serve output files (images)

Author: AI-Assisted Fashion Segmentation System
Date: 2025
========================================================================================================
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Optional
import glob
import urllib.request

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Add src directory to path to import the segformer_sam_auto_center module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import functions from existing segformer_sam_auto_center.py
from segformer_sam_auto_center import (
    Config,
    load_segformer_model,
    load_sam_predictor,
    process_single_image
)

# ========================================================================================================
# FASTAPI APPLICATION SETUP
# ========================================================================================================

app = FastAPI(
    title="Fashion Segmentation API",
    description="SegFormer + SAM + Auto-Center Pipeline with Web GUI",
    version="1.0.0"
)

# Global variables for models (loaded once at startup)
segformer_model = None
segformer_processor = None
sam_predictor = None

# ========================================================================================================
# STARTUP EVENT: LOAD MODELS
# ========================================================================================================

@app.on_event("startup")
async def startup_event():
    """Load models once at startup for faster processing"""
    global segformer_model, segformer_processor, sam_predictor
    
    print("\n" + "=" * 80)
    print("STARTING FASTAPI SERVER")
    print("=" * 80)
    
    # Ensure SAM model is available (download if missing)
    sam_checkpoint = os.getenv("SAM_CHECKPOINT", "data/models/sam_vit_b_01ec64.pth")
    if not os.path.exists(sam_checkpoint):
        print(f"\nSAM model not found at {sam_checkpoint}")
        print("Downloading SAM ViT-B model (375MB)... This may take 1-2 minutes...")
        
        # Create models directory
        os.makedirs(os.path.dirname(sam_checkpoint), exist_ok=True)
        
        # Download with progress - Using smaller ViT-B model for better Railway compatibility
        sam_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        try:
            def download_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(100, (downloaded / total_size) * 100)
                print(f"\rDownload progress: {percent:.1f}% ({downloaded / (1024**2):.0f}MB / {total_size / (1024**2):.0f}MB)", end='')
            
            urllib.request.urlretrieve(sam_url, sam_checkpoint, download_progress)
            print("\n[OK] SAM model downloaded successfully!")
        except Exception as e:
            print(f"\n[ERROR] Failed to download SAM model: {e}")
            print("Please download manually from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
            raise
    else:
        print(f"[OK] SAM model found at {sam_checkpoint}")
    
    # Create output directories
    Config.create_output_dirs()
    
    # Load models
    segformer_model, segformer_processor = load_segformer_model()
    sam_predictor = load_sam_predictor()
    
    print("\n" + "=" * 80)
    print("SERVER READY - Models loaded successfully!")
    print("=" * 80)
    print("Access the web interface at: http://localhost:8000")
    print("=" * 80 + "\n")

# ========================================================================================================
# ENDPOINT: HOME PAGE (WEB GUI)
# ========================================================================================================

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the web GUI for image upload"""
    
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Fashion Segmentation - AI Pipeline</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                overflow: hidden;
            }
            
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 40px;
                text-align: center;
            }
            
            .header h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            }
            
            .header p {
                font-size: 1.2em;
                opacity: 0.95;
            }
            
            .content {
                padding: 40px;
            }
            
            .upload-section {
                background: #f8f9fa;
                border: 3px dashed #667eea;
                border-radius: 15px;
                padding: 60px 40px;
                text-align: center;
                margin-bottom: 30px;
                transition: all 0.3s ease;
                cursor: pointer;
            }
            
            .upload-section:hover {
                background: #e8e9ff;
                border-color: #764ba2;
                transform: scale(1.02);
            }
            
            .upload-section.dragover {
                background: #d0d8ff;
                border-color: #764ba2;
                transform: scale(1.05);
            }
            
            .upload-icon {
                font-size: 4em;
                margin-bottom: 20px;
            }
            
            .upload-section h2 {
                color: #333;
                margin-bottom: 15px;
                font-size: 1.8em;
            }
            
            .upload-section p {
                color: #666;
                font-size: 1.1em;
                margin-bottom: 20px;
            }
            
            input[type="file"] {
                display: none;
            }
            
            .btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px 40px;
                border: none;
                border-radius: 30px;
                font-size: 1.1em;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            }
            
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
            }
            
            .btn:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }
            
            .preview-section {
                display: none;
                margin-top: 30px;
                text-align: center;
            }
            
            .preview-section img {
                max-width: 100%;
                max-height: 400px;
                border-radius: 10px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            
            .loading {
                display: none;
                text-align: center;
                margin: 30px 0;
            }
            
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 60px;
                height: 60px;
                animation: spin 1s linear infinite;
                margin: 0 auto 20px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .features {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-top: 40px;
            }
            
            .feature-card {
                background: #f8f9fa;
                padding: 25px;
                border-radius: 10px;
                text-align: center;
                transition: all 0.3s ease;
            }
            
            .feature-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            }
            
            .feature-card h3 {
                color: #667eea;
                margin-bottom: 10px;
                font-size: 1.3em;
            }
            
            .feature-card p {
                color: #666;
                line-height: 1.6;
            }
            
            .status-message {
                padding: 15px;
                border-radius: 10px;
                margin: 20px 0;
                display: none;
            }
            
            .status-success {
                background: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            
            .status-error {
                background: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎨 Fashion AI Segmentation</h1>
                <p>SegFormer + SAM + Auto-Center Pipeline</p>
            </div>
            
            <div class="content">
                <div class="upload-section" id="dropZone" onclick="document.getElementById('fileInput').click()">
                    <div class="upload-icon">📸</div>
                    <h2>Upload Fashion Image</h2>
                    <p>Drag & drop your image here or click to browse</p>
                    <input type="file" id="fileInput" accept="image/*" onchange="handleFileSelect(event)">
                    <button class="btn" type="button">Choose Image</button>
                </div>
                
                <div class="preview-section" id="previewSection">
                    <h3>Preview:</h3>
                    <img id="previewImage" src="" alt="Preview">
                    <br>
                    <button class="btn" onclick="processImage()">🚀 Process Image</button>
                </div>
                
                <div class="loading" id="loadingSection">
                    <div class="spinner"></div>
                    <h3>Processing your image...</h3>
                    <p>This may take a few moments. Please wait.</p>
                </div>
                
                <div class="status-message" id="statusMessage"></div>
                
                <div class="features">
                    <div class="feature-card">
                        <h3>🧠 SegFormer CNN</h3>
                        <p>Fashion-specific semantic segmentation to identify clothing categories</p>
                    </div>
                    <div class="feature-card">
                        <h3>✂️ SAM Refinement</h3>
                        <p>Precise edge detection and boundary refinement using Segment Anything</p>
                    </div>
                    <div class="feature-card">
                        <h3>🎯 Auto-Center</h3>
                        <p>Automatic cropping and centering with minimal white space</p>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            let selectedFile = null;
            
            // Drag and drop handlers
            const dropZone = document.getElementById('dropZone');
            
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                dropZone.addEventListener(eventName, preventDefaults, false);
            });
            
            function preventDefaults(e) {
                e.preventDefault();
                e.stopPropagation();
            }
            
            ['dragenter', 'dragover'].forEach(eventName => {
                dropZone.addEventListener(eventName, () => {
                    dropZone.classList.add('dragover');
                }, false);
            });
            
            ['dragleave', 'drop'].forEach(eventName => {
                dropZone.addEventListener(eventName, () => {
                    dropZone.classList.remove('dragover');
                }, false);
            });
            
            dropZone.addEventListener('drop', handleDrop, false);
            
            function handleDrop(e) {
                const dt = e.dataTransfer;
                const files = dt.files;
                if (files.length > 0) {
                    handleFile(files[0]);
                }
            }
            
            function handleFileSelect(event) {
                const file = event.target.files[0];
                if (file) {
                    handleFile(file);
                }
            }
            
            function handleFile(file) {
                if (!file.type.startsWith('image/')) {
                    showStatus('Please select an image file', 'error');
                    return;
                }
                
                selectedFile = file;
                
                // Show preview
                const reader = new FileReader();
                reader.onload = function(e) {
                    document.getElementById('previewImage').src = e.target.result;
                    document.getElementById('previewSection').style.display = 'block';
                };
                reader.readAsDataURL(file);
                
                showStatus('Image selected: ' + file.name, 'success');
            }
            
            async function processImage() {
                if (!selectedFile) {
                    showStatus('Please select an image first', 'error');
                    return;
                }
                
                // Show loading
                document.getElementById('loadingSection').style.display = 'block';
                document.getElementById('previewSection').style.display = 'none';
                document.getElementById('statusMessage').style.display = 'none';
                
                // Prepare form data
                const formData = new FormData();
                formData.append('file', selectedFile);
                
                try {
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    document.getElementById('loadingSection').style.display = 'none';
                    
                    if (response.ok) {
                        // Redirect to results page
                        window.location.href = '/results/' + result.image_name;
                    } else {
                        showStatus('Error: ' + result.detail, 'error');
                        document.getElementById('previewSection').style.display = 'block';
                    }
                } catch (error) {
                    document.getElementById('loadingSection').style.display = 'none';
                    showStatus('Error processing image: ' + error.message, 'error');
                    document.getElementById('previewSection').style.display = 'block';
                }
            }
            
            function showStatus(message, type) {
                const statusDiv = document.getElementById('statusMessage');
                statusDiv.textContent = message;
                statusDiv.className = 'status-message status-' + type;
                statusDiv.style.display = 'block';
            }
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

# ========================================================================================================
# ENDPOINT: UPLOAD AND PROCESS IMAGE
# ========================================================================================================

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload and process an image through the SegFormer + SAM pipeline
    
    Args:
        file: Uploaded image file
    
    Returns:
        JSON response with processing results and file paths
    """
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Create temporary upload directory
    upload_dir = os.path.join(Config.PROJECT_ROOT, "temp_uploads")
    os.makedirs(upload_dir, exist_ok=True)
    
    # Save uploaded file
    file_path = os.path.join(upload_dir, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"\n[API] Processing uploaded file: {file.filename}")
        
        # Process image using existing pipeline
        result = process_single_image(
            file_path,
            segformer_model,
            segformer_processor,
            sam_predictor
        )
        
        if result is None:
            raise HTTPException(status_code=500, detail="Failed to process image - no clothing found")
        
        # Save original image to outputs for display in results
        original_save_path = os.path.join(Config.OUTPUT_BASE, f"{result['image_name']}_original.jpg")
        shutil.copy2(file_path, original_save_path)
        
        # Clean up uploaded file from temp
        os.remove(file_path)
        
        print(f"[API] Processing complete: {result['num_components']} components extracted")
        
        # Convert numpy types to Python native types for JSON serialization
        return JSONResponse(content={
            "status": "success",
            "message": "Image processed successfully",
            "image_name": result['image_name'],
            "num_components": int(result['num_components']),  # Convert to Python int
            "categories": result['categories'],
            "total_area": int(result['total_area'])  # Convert numpy uint64 to Python int
        })
    
    except Exception as e:
        # Clean up on error
        if os.path.exists(file_path):
            os.remove(file_path)
        
        print(f"[API ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ========================================================================================================
# ENDPOINT: VIEW RESULTS PAGE
# ========================================================================================================

@app.get("/results/{image_name}", response_class=HTMLResponse)
async def view_results(image_name: str):
    """
    Display results page with original image and extracted crops only
    
    Args:
        image_name: Name of the processed image (without extension)
    
    Returns:
        HTML page with results
    """
    
    # Find original uploaded image in temp or input directory
    original_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    original_path = None
    
    # Check if we have the visualization (means processing was done)
    vis_file = f"{image_name}_visualization.png"
    vis_path = os.path.join(Config.OUTPUT_BASE, vis_file)
    
    if not os.path.exists(vis_path):
        raise HTTPException(status_code=404, detail="Results not found")
    
    # Find all crop files
    crop_dir = Config.OUTPUT_CROPS
    crop_pattern = f"{image_name}_*.png"
    crop_files = glob.glob(os.path.join(crop_dir, crop_pattern))
    crop_filenames = [os.path.basename(f) for f in crop_files]
    
    # Generate HTML for crops
    crops_html = ""
    for crop_file in sorted(crop_filenames):
        crop_name = crop_file.replace(f"{image_name}_", "").replace(".png", "").replace("_", " ").title()
        crops_html += f"""
        <div class="crop-item">
            <img src="/outputs/crops_centered/{crop_file}" alt="{crop_name}">
            <div class="crop-label">{crop_name}</div>
            <a href="/outputs/crops_centered/{crop_file}" download class="download-btn">⬇ Download</a>
        </div>
        """
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Results - {image_name}</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }}
            
            .container {{
                max-width: 1600px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                overflow: hidden;
            }}
            
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px 40px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            
            .header h1 {{
                font-size: 2em;
            }}
            
            .btn {{
                background: white;
                color: #667eea;
                padding: 12px 30px;
                border: none;
                border-radius: 25px;
                font-size: 1em;
                cursor: pointer;
                text-decoration: none;
                display: inline-block;
                transition: all 0.3s ease;
                font-weight: 600;
            }}
            
            .btn:hover {{
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }}
            
            .content {{
                padding: 40px;
            }}
            
            .results-layout {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 40px;
                margin-top: 20px;
            }}
            
            @media (max-width: 1024px) {{
                .results-layout {{
                    grid-template-columns: 1fr;
                }}
            }}
            
            .section {{
                background: #f8f9fa;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            }}
            
            .section h2 {{
                color: #333;
                margin-bottom: 25px;
                font-size: 1.6em;
                border-bottom: 3px solid #667eea;
                padding-bottom: 15px;
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            
            .original-image {{
                text-align: center;
            }}
            
            .original-image img {{
                max-width: 100%;
                max-height: 600px;
                border-radius: 12px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.15);
                background: white;
                padding: 15px;
            }}
            
            .crops-section {{
                max-height: 650px;
                overflow-y: auto;
                padding-right: 10px;
            }}
            
            .crops-section::-webkit-scrollbar {{
                width: 8px;
            }}
            
            .crops-section::-webkit-scrollbar-track {{
                background: #e0e0e0;
                border-radius: 10px;
            }}
            
            .crops-section::-webkit-scrollbar-thumb {{
                background: #667eea;
                border-radius: 10px;
            }}
            
            .crops-section::-webkit-scrollbar-thumb:hover {{
                background: #764ba2;
            }}
            
            .crops-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                gap: 20px;
            }}
            
            .crop-item {{
                background: white;
                padding: 15px;
                border-radius: 12px;
                text-align: center;
                transition: all 0.3s ease;
                border: 2px solid #e0e0e0;
            }}
            
            .crop-item:hover {{
                transform: translateY(-5px);
                box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
                border-color: #667eea;
            }}
            
            .crop-item img {{
                max-width: 100%;
                max-height: 250px;
                border-radius: 8px;
                margin-bottom: 12px;
                object-fit: contain;
            }}
            
            .crop-label {{
                color: #333;
                font-weight: 600;
                margin-bottom: 10px;
                font-size: 1em;
                text-transform: capitalize;
            }}
            
            .download-btn {{
                display: inline-block;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 8px 20px;
                border-radius: 20px;
                text-decoration: none;
                transition: all 0.3s ease;
                font-size: 0.9em;
                font-weight: 600;
            }}
            
            .download-btn:hover {{
                transform: scale(1.05);
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            }}
            
            .stats-bar {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px 30px;
                border-radius: 12px;
                margin-bottom: 30px;
                display: flex;
                justify-content: space-around;
                align-items: center;
                flex-wrap: wrap;
                gap: 20px;
            }}
            
            .stat-item {{
                text-align: center;
            }}
            
            .stat-value {{
                font-size: 2em;
                font-weight: bold;
                display: block;
            }}
            
            .stat-label {{
                font-size: 0.9em;
                opacity: 0.9;
                margin-top: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>✅ Extraction Complete</h1>
                <a href="/" class="btn">⬅ Upload Another Image</a>
            </div>
            
            <div class="content">
                <div class="stats-bar">
                    <div class="stat-item">
                        <span class="stat-value">{len(crop_filenames)}</span>
                        <span class="stat-label">Items Extracted</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-value">✓</span>
                        <span class="stat-label">AI Processed</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-value">🎯</span>
                        <span class="stat-label">Auto-Centered</span>
                    </div>
                </div>
                
                <div class="results-layout">
                    <div class="section">
                        <h2>📸 Original Image</h2>
                        <div class="original-image">
                            <img src="/outputs/{image_name}_original.jpg" alt="Original Image">
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>👕 Extracted Clothing Items</h2>
                        <div class="crops-section">
                            <div class="crops-grid">
                                {crops_html}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

# ========================================================================================================
# ENDPOINT: SERVE OUTPUT FILES
# ========================================================================================================

@app.get("/outputs/{file_path:path}")
async def serve_output_file(file_path: str):
    """
    Serve output files (images, masks, crops)
    
    Args:
        file_path: Relative path to file in outputs directory
    
    Returns:
        File response
    """
    full_path = os.path.join(Config.OUTPUT_BASE, file_path)
    
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(full_path)

# ========================================================================================================
# ENDPOINT: LIST OUTPUT FILES FOR AN IMAGE
# ========================================================================================================

@app.get("/list-outputs/{image_name}")
async def list_outputs(image_name: str):
    """List all output files for a processed image"""
    output_dir = Config.OUTPUT_BASE
    
    files = {
        "crops": [],
        "masks_raw": [],
        "masks_refined": [],
        "visualization": None,
        "original": None
    }
    
    # Check for crops
    crops_dir = os.path.join(output_dir, "crops_centered")
    if os.path.exists(crops_dir):
        for file in os.listdir(crops_dir):
            if file.startswith(image_name) and file.endswith('.png'):
                files["crops"].append(f"crops_centered/{file}")
    
    # Check for masks
    masks_raw_dir = os.path.join(output_dir, "masks_raw")
    if os.path.exists(masks_raw_dir):
        for file in os.listdir(masks_raw_dir):
            if file.startswith(image_name) and file.endswith('.png'):
                files["masks_raw"].append(f"masks_raw/{file}")
    
    masks_refined_dir = os.path.join(output_dir, "masks_refined")
    if os.path.exists(masks_refined_dir):
        for file in os.listdir(masks_refined_dir):
            if file.startswith(image_name) and file.endswith('.png'):
                files["masks_refined"].append(f"masks_refined/{file}")
    
    # Check for visualization and original
    for file in os.listdir(output_dir):
        if file.startswith(image_name):
            if file.endswith('_visualization.png'):
                files["visualization"] = file
            elif file.endswith('_original.jpg'):
                files["original"] = file
    
    return JSONResponse({
        "image_name": image_name,
        "files": files,
        "total_files": sum(len(v) if isinstance(v, list) else (1 if v else 0) for v in files.values())
    })

# ========================================================================================================
# MAIN: RUN SERVER
# ========================================================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("STARTING FASHION SEGMENTATION API SERVER")
    print("=" * 80)
    print("Server will be available at: http://localhost:8000")
    print("=" * 80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

