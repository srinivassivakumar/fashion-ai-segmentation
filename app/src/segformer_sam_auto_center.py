#!/usr/bin/env python3
"""
========================================================================================================
INTEGRATED SEGFORMER + SAM + AUTO-CENTER PIPELINE
========================================================================================================

This script combines:
1. SegFormer B2 (Fashion-Specific) semantic segmentation
2. SAM (Segment Anything Model) edge refinement
3. Automatic centering and cropping post-processing

PIPELINE:
    Image → SegFormer → SAM Refinement → Auto-Center Crops → Save

OUTPUT:
    outputs/
    ├── masks_raw/          # SegFormer's raw semantic masks
    ├── masks_refined/      # SAM-refined masks
    ├── crops_centered/     # Centered clothing crops (auto-sized)
    └── visualizations/     # Pipeline visualizations

Author: AI-Assisted Fashion Segmentation System
Date: 2025
========================================================================================================
"""

import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# HuggingFace Transformers for SegFormer
from transformers import SegformerForSemanticSegmentation, AutoImageProcessor

# SAM imports
from segment_anything import sam_model_registry, SamPredictor

# ========================================================================================================
# CONFIGURATION SECTION
# ========================================================================================================

class Config:
    """Configuration for integrated pipeline"""
    
    # Paths - support environment variables for Docker deployment
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "input_images")
    
    # OUTPUT_BASE can be overridden via environment variable
    OUTPUT_BASE = os.environ.get("OUTPUT_BASE", os.path.join(PROJECT_ROOT, "outputs"))
    
    # Output subdirectories
    OUTPUT_MASKS_RAW = os.path.join(OUTPUT_BASE, "masks_raw")
    OUTPUT_MASKS_REFINED = os.path.join(OUTPUT_BASE, "masks_refined")
    OUTPUT_CROPS = os.path.join(OUTPUT_BASE, "crops_centered")  # Changed to centered directly
    
    # Model paths and configurations - SAM_CHECKPOINT can be overridden via environment variable
    SAM_CHECKPOINT = os.environ.get(
        "SAM_CHECKPOINT", 
        os.path.join(PROJECT_ROOT, "data", "models", "sam_vit_b_01ec64.pth")
    )
    SAM_MODEL_TYPE = "vit_b"  # Changed to ViT-B for better Railway compatibility (lower RAM)
    
    # SegFormer model (Fashion-specific)
    SEGFORMER_MODEL = "mattmdjaga/segformer_b2_clothes"
    
    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Segmentation parameters
    MIN_COMPONENT_AREA = 2000  # Minimum pixels for valid clothing item
    CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for SegFormer predictions
    
    # Post-processing parameters (centering)
    CROP_PADDING = 50  # Padding around centered clothing
    WHITE_THRESHOLD = 250  # Pixels above this are considered background
    
    # Visualization settings
    DPI = 100
    FIGSIZE = (15, 10)
    
    # Clothing category mapping (for mattmdjaga/segformer_b2_clothes)
    CLOTHING_CATEGORIES = {
        0: 'Background',
        1: 'Hat',
        2: 'Hair',
        3: 'Sunglasses',
        4: 'Upper-clothes',
        5: 'Skirt',
        6: 'Pants',
        7: 'Dress',
        8: 'Belt',
        9: 'Left-shoe',
        10: 'Right-shoe',
        11: 'Face',
        12: 'Left-leg',
        13: 'Right-leg',
        14: 'Left-arm',
        15: 'Right-arm',
        16: 'Bag',
        17: 'Scarf'
    }
    
    # Focus on actual clothing items (exclude body parts)
    CLOTHING_ONLY_IDS = [1, 4, 5, 6, 7, 8, 16, 17]  # Hat, Upper, Skirt, Pants, Dress, Belt, Bag, Scarf
    
    # Color map for visualization (RGB)
    CATEGORY_COLORS = {
        0: (0, 0, 0),           # Background - Black
        1: (255, 0, 0),         # Hat - Red
        4: (0, 255, 0),         # Upper-clothes - Green
        5: (0, 0, 255),         # Skirt - Blue
        6: (255, 255, 0),       # Pants - Yellow
        7: (255, 0, 255),       # Dress - Magenta
        8: (0, 255, 255),       # Belt - Cyan
        16: (128, 0, 128),      # Bag - Purple
        17: (255, 165, 0),      # Scarf - Orange
    }

    # Behavioral flags
    ONLY_LARGEST_UPPER_AND_BOTTOM = True  # Keep only largest upper and largest bottom wear for SAM

    @classmethod
    def create_output_dirs(cls):
        """Create all output directories"""
        for dir_path in [cls.OUTPUT_MASKS_RAW, cls.OUTPUT_MASKS_REFINED, cls.OUTPUT_CROPS]:
            os.makedirs(dir_path, exist_ok=True)
        print(f"[OK] Output directories created:")
        print(f"    - {cls.OUTPUT_MASKS_RAW}")
        print(f"    - {cls.OUTPUT_MASKS_REFINED}")
        print(f"    - {cls.OUTPUT_CROPS}")

# ========================================================================================================
# POST-PROCESSING: CENTERING FUNCTIONS
# ========================================================================================================

def find_clothing_bbox(image_rgb, white_threshold=250):
    """
    Find the bounding box of the actual clothing item (non-white pixels)
    
    Args:
        image_rgb: RGB image (numpy array)
        white_threshold: Pixel values above this are considered background
    
    Returns:
        (x, y, w, h): Bounding box coordinates, or None if no clothing found
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # Find non-white pixels (the clothing)
    mask = gray < white_threshold
    
    # Find coordinates of non-white pixels
    coords = np.argwhere(mask)
    
    if len(coords) == 0:
        return None
    
    # Get bounding box
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    x = x_min
    y = y_min
    w = x_max - x_min + 1
    h = y_max - y_min + 1
    
    return (x, y, w, h)

def create_centered_crop(image_rgb, padding=50):
    """
    Create a tight centered crop around the clothing
    
    Args:
        image_rgb: Original image with clothing on white background
        padding: Padding around clothing
    
    Returns:
        Tightly cropped and centered image
    """
    bbox = find_clothing_bbox(image_rgb, white_threshold=Config.WHITE_THRESHOLD)
    
    if bbox is None:
        return image_rgb  # Return original if nothing found
    
    x, y, w, h = bbox
    
    # Calculate crop boundaries with padding
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image_rgb.shape[1], x + w + padding)
    y2 = min(image_rgb.shape[0], y + h + padding)
    
    # Crop
    cropped = image_rgb[y1:y2, x1:x2]
    
    return cropped

# ========================================================================================================
# MODEL LOADING
# ========================================================================================================

def load_segformer_model():
    """Load SegFormer B2 model fine-tuned for fashion/clothing segmentation"""
    print("\n" + "=" * 80)
    print("LOADING SEGFORMER B2 (FASHION-SPECIFIC MODEL)")
    print("=" * 80)
    print(f"[INFO] Model: {Config.SEGFORMER_MODEL}")
    print(f"[INFO] Device: {Config.DEVICE}")
    print(f"[INFO] First-time download: ~100MB (cached after initial run)")
    
    try:
        processor = AutoImageProcessor.from_pretrained(Config.SEGFORMER_MODEL)
        model = SegformerForSemanticSegmentation.from_pretrained(Config.SEGFORMER_MODEL)
        model = model.to(Config.DEVICE)
        model.eval()
        
        print(f"[OK] SegFormer loaded successfully!")
        print(f"[INFO] Model supports {len(Config.CLOTHING_CATEGORIES)} clothing categories")
        
        return model, processor
    
    except Exception as e:
        print(f"[ERROR] Failed to load SegFormer: {e}")
        print(f"[HINT] Check internet connection for first-time download")
        sys.exit(1)

def load_sam_predictor():
    """Load SAM (Segment Anything Model) for mask refinement"""
    print("\n" + "=" * 80)
    print("LOADING SAM (SEGMENT ANYTHING MODEL)")
    print("=" * 80)
    
    if not os.path.exists(Config.SAM_CHECKPOINT):
        print(f"[ERROR] SAM checkpoint not found: {Config.SAM_CHECKPOINT}")
        print(f"[HINT] Please download sam_vit_h_4b8939.pth from:")
        print(f"       https://github.com/facebookresearch/segment-anything#model-checkpoints")
        sys.exit(1)
    
    print(f"[INFO] Loading SAM from: {Config.SAM_CHECKPOINT}")
    print(f"[INFO] Device: {Config.DEVICE}")
    
    try:
        sam = sam_model_registry[Config.SAM_MODEL_TYPE](checkpoint=Config.SAM_CHECKPOINT)
        sam = sam.to(Config.DEVICE)
        predictor = SamPredictor(sam)
        
        print(f"[OK] SAM loaded successfully!")
        return predictor
    
    except Exception as e:
        print(f"[ERROR] Failed to load SAM: {e}")
        sys.exit(1)

# ========================================================================================================
# SEGFORMER SEGMENTATION
# ========================================================================================================

def segment_with_segformer(image_rgb, model, processor):
    """Perform semantic segmentation using SegFormer"""
    print("\n[INFO] Running SegFormer semantic segmentation...")
    
    # Preprocess image
    inputs = processor(images=image_rgb, return_tensors="pt")
    inputs = {k: v.to(Config.DEVICE) for k, v in inputs.items()}
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Get predictions
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image_rgb.shape[:2],
        mode="bilinear",
        align_corners=False
    )
    
    probs = torch.nn.functional.softmax(upsampled_logits, dim=1)
    confidence_map, segmentation_mask = torch.max(probs, dim=1)
    
    # Convert to numpy
    segmentation_mask = segmentation_mask.squeeze().cpu().numpy().astype(np.uint8)
    confidence_map = confidence_map.squeeze().cpu().numpy()
    
    # Statistics
    unique_classes = np.unique(segmentation_mask)
    print(f"[INFO] Detected {len(unique_classes)} unique categories")
    
    for class_id in unique_classes:
        if class_id in Config.CLOTHING_CATEGORIES:
            pixel_count = np.sum(segmentation_mask == class_id)
            avg_conf = np.mean(confidence_map[segmentation_mask == class_id])
            print(f"    - {Config.CLOTHING_CATEGORIES[class_id]}: {pixel_count:,} pixels (conf: {avg_conf:.3f})")
    
    return segmentation_mask, confidence_map

# ========================================================================================================
# COMPONENT EXTRACTION
# ========================================================================================================

def extract_clothing_components(segmentation_mask, confidence_map):
    """Extract individual clothing item components from semantic segmentation"""
    print("\n[INFO] Extracting clothing components...")
    
    components = []
    
    for class_id in Config.CLOTHING_ONLY_IDS:
        category_mask = (segmentation_mask == class_id).astype(np.uint8)
        
        if np.sum(category_mask) < Config.MIN_COMPONENT_AREA:
            continue
        
        # Filter by confidence
        high_conf_mask = (confidence_map > Config.CONFIDENCE_THRESHOLD) & (category_mask > 0)
        category_mask = high_conf_mask.astype(np.uint8)
        
        if np.sum(category_mask) < Config.MIN_COMPONENT_AREA:
            continue
        
        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        category_mask = cv2.morphologyEx(category_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        category_mask = cv2.morphologyEx(category_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            category_mask, connectivity=8
        )
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            if area < Config.MIN_COMPONENT_AREA:
                continue
            
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            component_mask = (labels == i).astype(np.uint8)
            
            component = {
                'category_id': class_id,
                'category_name': Config.CLOTHING_CATEGORIES[class_id],
                'mask': component_mask,
                'bbox': (x, y, w, h),
                'area': area,
                'centroid': centroids[i],
                'color': Config.CATEGORY_COLORS.get(class_id, (128, 128, 128))
            }
            
            components.append(component)
            print(f"    - Found: {component['category_name']} ({area:,} pixels)")
    
    print(f"[INFO] Total components extracted: {len(components)}")
    return components

# ========================================================================================================
# COMPONENT SELECTION (KEEP LARGEST UPPER AND BOTTOM)
# ========================================================================================================

def filter_largest_upper_and_bottom(components):
    """Filter components to keep only the largest upper wear and the largest bottom wear.

    - Upper wear: class_id == 4 (Upper-clothes)
    - Bottom wear: class_id in {5 (Skirt), 6 (Pants), 7 (Dress)}
    - All other categories are DROPPED for SAM
    """
    if not components:
        return components

    UPPER_ID = 4
    BOTTOM_IDS = {5, 6, 7}

    largest_upper = None
    largest_bottom = None
    largest_upper_area = -1
    largest_bottom_area = -1

    for comp in components:
        class_id = comp.get('category_id')
        area = comp.get('area', 0)

        if class_id == UPPER_ID:
            if area > largest_upper_area:
                largest_upper = comp
                largest_upper_area = area
        elif class_id in BOTTOM_IDS:
            if area > largest_bottom_area:
                largest_bottom = comp
                largest_bottom_area = area
        else:
            # Drop non upper/bottom categories for SAM
            continue

    filtered = []
    if largest_upper is not None:
        filtered.append(largest_upper)
    if largest_bottom is not None:
        filtered.append(largest_bottom)

    kept_upper = 1 if largest_upper is not None else 0
    kept_bottom = 1 if largest_bottom is not None else 0
    print(f"[INFO] Largest-only filter applied: kept {kept_upper} upper and {kept_bottom} bottom items")
    print(f"[INFO] Components after filter: {len(filtered)} (from {len(components)})")
    return filtered

# ========================================================================================================
# SAM REFINEMENT
# ========================================================================================================

def refine_component_with_sam(image_rgb, component, predictor):
    """Refine a component mask using SAM with box prompt"""
    x, y, w, h = component['bbox']
    
    # Create box prompt (add small padding)
    padding = 10
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image_rgb.shape[1], x + w + padding)
    y2 = min(image_rgb.shape[0], y + h + padding)
    
    box = np.array([x1, y1, x2, y2])
    
    # Run SAM with box prompt
    masks, scores, logits = predictor.predict(
        box=box,
        multimask_output=False
    )
    
    refined_mask = masks[0].astype(np.uint8)
    score = scores[0]
    
    # Merge with original mask
    combined_mask = np.logical_or(refined_mask, component['mask']).astype(np.uint8)
    
    return combined_mask, score

def refine_components_with_sam(image_rgb, components, predictor):
    """Refine all components using SAM"""
    print("\n[INFO] Refining masks with SAM...")
    
    predictor.set_image(image_rgb)
    
    refined_components = []
    
    for i, comp in enumerate(components):
        print(f"    [{i+1}/{len(components)}] Refining {comp['category_name']}...", end=" ")
        
        try:
            refined_mask, score = refine_component_with_sam(image_rgb, comp, predictor)
            
            refined_comp = comp.copy()
            refined_comp['mask_original'] = comp['mask'].copy()
            refined_comp['mask'] = refined_mask
            refined_comp['sam_score'] = score
            refined_comp['area'] = np.sum(refined_mask)
            
            refined_components.append(refined_comp)
            print(f"OK (score: {score:.3f}, area: {refined_comp['area']:,})")
        
        except Exception as e:
            print(f"X (error: {e})")
            refined_components.append(comp)
    
    print(f"[OK] Refinement complete: {len(refined_components)} components")
    return refined_components

# ========================================================================================================
# VISUALIZATION
# ========================================================================================================

def create_colored_mask(segmentation_mask, alpha=0.6):
    """Create color-coded visualization of segmentation mask"""
    h, w = segmentation_mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in Config.CATEGORY_COLORS.items():
        mask = segmentation_mask == class_id
        colored_mask[mask] = color
    
    return colored_mask

def visualize_segmentation_results(image_rgb, segmentation_mask, components, 
                                   refined_components, output_path):
    """Create comprehensive visualization"""
    fig, axes = plt.subplots(2, 3, figsize=Config.FIGSIZE, dpi=Config.DPI)
    fig.suptitle('SegFormer + SAM + Auto-Center Pipeline', fontsize=16, fontweight='bold')
    
    # Row 1: Original and SegFormer results
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original Image', fontweight='bold')
    axes[0, 0].axis('off')
    
    colored_mask = create_colored_mask(segmentation_mask)
    axes[0, 1].imshow(colored_mask)
    axes[0, 1].set_title('SegFormer: Semantic Segmentation', fontweight='bold')
    axes[0, 1].axis('off')
    
    overlay = image_rgb.copy()
    mask_overlay = cv2.addWeighted(overlay, 0.6, colored_mask, 0.4, 0)
    axes[0, 2].imshow(mask_overlay)
    axes[0, 2].set_title('SegFormer: Overlay', fontweight='bold')
    axes[0, 2].axis('off')
    
    # Row 2: SAM refinement results
    comp_vis = image_rgb.copy()
    for comp in components:
        mask = comp['mask']
        comp_vis[mask > 0] = comp_vis[mask > 0] * 0.5 + np.array(comp['color']) * 0.5
    axes[1, 0].imshow(comp_vis.astype(np.uint8))
    axes[1, 0].set_title(f'SegFormer: {len(components)} Components', fontweight='bold')
    axes[1, 0].axis('off')
    
    refined_vis = image_rgb.copy()
    for comp in refined_components:
        mask = comp['mask']
        refined_vis[mask > 0] = refined_vis[mask > 0] * 0.5 + np.array(comp['color']) * 0.5
    axes[1, 1].imshow(refined_vis.astype(np.uint8))
    axes[1, 1].set_title(f'SAM Refined: {len(refined_components)} Components', fontweight='bold')
    axes[1, 1].axis('off')
    
    final_vis = image_rgb.copy()
    for comp in refined_components:
        mask = comp['mask']
        final_vis[mask > 0] = final_vis[mask > 0] * 0.6 + np.array(comp['color']) * 0.4
        
        cx, cy = comp['centroid']
        axes[1, 2].text(cx, cy, comp['category_name'], 
                       color='white', fontsize=10, fontweight='bold',
                       ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    axes[1, 2].imshow(final_vis.astype(np.uint8))
    axes[1, 2].set_title('Final: Auto-Centered Crops', fontweight='bold')
    axes[1, 2].axis('off')
    
    # Legend
    legend_elements = []
    for class_id in sorted(Config.CATEGORY_COLORS.keys()):
        if class_id in Config.CLOTHING_ONLY_IDS:
            color = np.array(Config.CATEGORY_COLORS[class_id]) / 255.0
            label = Config.CLOTHING_CATEGORIES[class_id]
            legend_elements.append(mpatches.Patch(color=color, label=label))
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, 
               frameon=True, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Visualization saved: {output_path}")

# ========================================================================================================
# OUTPUT GENERATION WITH AUTO-CENTERING
# ========================================================================================================

def save_component_crops_centered(image_rgb, components, output_dir, image_name):
    """
    Save individual clothing item crops with automatic centering
    
    Args:
        image_rgb: Original image
        components: List of refined components
        output_dir: Directory to save crops
        image_name: Base name for output files
    """
    print(f"\n[INFO] Saving centered clothing crops to {output_dir}/")
    
    for i, comp in enumerate(components):
        # Create white background with clothing
        result = np.ones_like(image_rgb) * 255
        mask = comp['mask']
        result[mask > 0] = image_rgb[mask > 0]
        
        # Apply automatic centering
        centered_result = create_centered_crop(result, padding=Config.CROP_PADDING)
        
        # Generate filename
        category_name = comp['category_name'].lower().replace('-', '_')
        filename = f"{image_name}_{category_name}_{i+1}.png"
        filepath = os.path.join(output_dir, filename)
        
        # Save
        Image.fromarray(centered_result).save(filepath)
        
        # Calculate size info
        original_size = f"{image_rgb.shape[1]}x{image_rgb.shape[0]}"
        centered_size = f"{centered_result.shape[1]}x{centered_result.shape[0]}"
        reduction = (1 - (centered_result.shape[0] * centered_result.shape[1]) / (image_rgb.shape[0] * image_rgb.shape[1])) * 100
        
        print(f"    - Saved: {filename} ({original_size} -> {centered_size}, {reduction:.1f}% reduction)")
    
    print(f"[OK] Saved {len(components)} centered clothing crops")

def save_masks(segmentation_mask, refined_components, output_raw_dir, 
               output_refined_dir, image_name):
    """Save raw and refined mask visualizations"""
    # Save raw SegFormer mask
    colored_raw = create_colored_mask(segmentation_mask)
    raw_path = os.path.join(output_raw_dir, f"{image_name}_segformer_raw.png")
    Image.fromarray(colored_raw).save(raw_path)
    print(f"[OK] Raw mask saved: {raw_path}")
    
    # Save refined mask
    h, w = segmentation_mask.shape
    refined_mask_vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    for comp in refined_components:
        mask = comp['mask']
        refined_mask_vis[mask > 0] = comp['color']
    
    refined_path = os.path.join(output_refined_dir, f"{image_name}_sam_refined.png")
    Image.fromarray(refined_mask_vis).save(refined_path)
    print(f"[OK] Refined mask saved: {refined_path}")

# ========================================================================================================
# MAIN PROCESSING PIPELINE
# ========================================================================================================

def process_single_image(image_path, segformer_model, segformer_processor, sam_predictor):
    """Process a single image through the complete integrated pipeline"""
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    print("\n" + "=" * 80)
    print(f"PROCESSING: {image_name}")
    print("=" * 80)
    
    # Load image
    print(f"[INFO] Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Failed to load image")
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    print(f"[INFO] Image size: {w}x{h}")
    
    # Stage 1: SegFormer semantic segmentation
    print("\n" + "-" * 80)
    print("STAGE 1: SegFormer Semantic Segmentation")
    print("-" * 80)
    segmentation_mask, confidence_map = segment_with_segformer(
        image_rgb, segformer_model, segformer_processor
    )
    
    # Stage 2: Extract clothing components
    print("\n" + "-" * 80)
    print("STAGE 2: Component Extraction")
    print("-" * 80)
    components = extract_clothing_components(segmentation_mask, confidence_map)

    # Optional: keep only the largest upper and bottom wear for SAM refinement
    if Config.ONLY_LARGEST_UPPER_AND_BOTTOM:
        components = filter_largest_upper_and_bottom(components)
    
    if not components:
        print("[WARN] No valid clothing components found")
        return None
    
    # Stage 3: SAM refinement
    print("\n" + "-" * 80)
    print("STAGE 3: SAM Edge Refinement")
    print("-" * 80)
    refined_components = refine_components_with_sam(image_rgb, components, sam_predictor)
    
    # Stage 4: Visualization
    print("\n" + "-" * 80)
    print("STAGE 4: Visualization & Output")
    print("-" * 80)
    
    vis_path = os.path.join(Config.OUTPUT_BASE, f"{image_name}_visualization.png")
    visualize_segmentation_results(
        image_rgb, segmentation_mask, components, refined_components, vis_path
    )
    
    # Stage 5: Save results with auto-centering
    print("\n" + "-" * 80)
    print("STAGE 5: Save Results (with Auto-Centering)")
    print("-" * 80)
    
    save_masks(segmentation_mask, refined_components, 
               Config.OUTPUT_MASKS_RAW, Config.OUTPUT_MASKS_REFINED, image_name)
    
    save_component_crops_centered(image_rgb, refined_components, Config.OUTPUT_CROPS, image_name)
    
    # Summary
    print("\n" + "=" * 80)
    print(f"[OK] PROCESSING COMPLETE: {image_name}")
    print("=" * 80)
    print(f"Components detected: {len(components)}")
    print(f"Components refined: {len(refined_components)}")
    print(f"Crops saved: Centered with {Config.CROP_PADDING}px padding")
    
    categories_found = set([c['category_name'] for c in refined_components])
    print(f"Categories found: {', '.join(sorted(categories_found))}")
    
    return {
        'image_name': image_name,
        'num_components': len(refined_components),
        'categories': list(categories_found),
        'total_area': sum(c['area'] for c in refined_components)
    }

def batch_process_images(segformer_model, segformer_processor, sam_predictor):
    """Process all images in input directory"""
    import glob
    
    print("\n" + "=" * 80)
    print("BATCH PROCESSING MODE")
    print("=" * 80)
    
    # Find all images
    image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    
    for pattern in image_patterns:
        image_files.extend(glob.glob(os.path.join(Config.INPUT_DIR, pattern)))
    
    if not image_files:
        print(f"[ERROR] No images found in {Config.INPUT_DIR}")
        return
    
    print(f"[INFO] Found {len(image_files)} images to process")
    
    results = []
    successful = 0
    failed = 0
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n{'#' * 80}")
        print(f"# IMAGE {i}/{len(image_files)}")
        print(f"{'#' * 80}")
        
        try:
            result = process_single_image(
                image_path, segformer_model, segformer_processor, sam_predictor
            )
            
            if result:
                results.append(result)
                successful += 1
            else:
                failed += 1
        
        except Exception as e:
            print(f"[ERROR] Failed to process {image_path}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Final summary
    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total images: {len(image_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if results:
        total_components = sum(r['num_components'] for r in results)
        all_categories = set()
        for r in results:
            all_categories.update(r['categories'])
        
        print(f"\nTotal clothing items extracted: {total_components}")
        print(f"Unique categories found: {', '.join(sorted(all_categories))}")
    
    print(f"\nOutputs saved in:")
    print(f"  - Visualizations: {Config.OUTPUT_BASE}")
    print(f"  - Raw masks: {Config.OUTPUT_MASKS_RAW}")
    print(f"  - Refined masks: {Config.OUTPUT_MASKS_REFINED}")
    print(f"  - Centered crops: {Config.OUTPUT_CROPS}")

# ========================================================================================================
# MAIN ENTRY POINT
# ========================================================================================================

def main():
    """Main entry point for integrated pipeline"""
    
    print("=" * 80)
    print("INTEGRATED SEGFORMER + SAM + AUTO-CENTER PIPELINE")
    print("=" * 80)
    print(f"Device: {Config.DEVICE}")
    print(f"SegFormer Model: {Config.SEGFORMER_MODEL}")
    print(f"SAM Checkpoint: {Config.SAM_CHECKPOINT}")
    print(f"Auto-centering: Enabled (padding={Config.CROP_PADDING}px)")
    print("=" * 80)
    
    # Create output directories
    Config.create_output_dirs()
    
    # Load models
    segformer_model, segformer_processor = load_segformer_model()
    sam_predictor = load_sam_predictor()
    
    # Check command line arguments
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--batch':
            # Batch mode
            batch_process_images(segformer_model, segformer_processor, sam_predictor)
        
        elif sys.argv[1] == '--help':
            print("\nUSAGE:")
            print("  python segformer_sam_auto_center.py              # Process single test image")
            print("  python segformer_sam_auto_center.py --batch      # Process all images")
            print("  python segformer_sam_auto_center.py <image.jpg>  # Process specific image")
            print("\nOUTPUT:")
            print("  outputs/masks_raw/      - SegFormer raw semantic masks")
            print("  outputs/masks_refined/  - SAM-refined masks")
            print("  outputs/crops_centered/ - Auto-centered clothing crops")
        
        else:
            # Process specific image
            image_path = sys.argv[1]
            if os.path.exists(image_path):
                process_single_image(image_path, segformer_model, segformer_processor, sam_predictor)
            else:
                print(f"[ERROR] Image not found: {image_path}")
    
    else:
        # Default: process first available image
        import glob
        image_files = glob.glob(os.path.join(Config.INPUT_DIR, "*.jpg"))
        
        if image_files:
            print(f"\n[INFO] Processing first image: {image_files[0]}")
            print(f"[TIP] Use --batch flag to process all {len(image_files)} images")
            process_single_image(image_files[0], segformer_model, segformer_processor, sam_predictor)
        else:
            print(f"[ERROR] No images found in {Config.INPUT_DIR}")

if __name__ == "__main__":
    main()

