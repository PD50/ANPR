#!/usr/bin/env python3
"""
train_anpr_detector.py
Semi-Supervised Training Pipeline for License Plate Detection (SGIE 2)

Workflow:
1. Use Grounding DINO for zero-shot pseudo-labeling
2. Generate YOLO-format annotations
3. Train YOLOv10-N model on pseudo-labeled data
4. Export to TensorRT for DeepStream deployment

Requirements:
pip install groundingdino-py ultralytics opencv-python pillow torch torchvision
"""

import os
import sys
import json
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from PIL import Image

# Grounding DINO imports
try:
    from groundingdino.util.inference import load_model, predict
    from groundingdino.util.utils import clean_state_dict
except ImportError:
    print("[ERROR] Grounding DINO not installed. Install with:")
    print("pip install groundingdino-py")
    sys.exit(1)

# Ultralytics YOLO imports
try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] Ultralytics not installed. Install with:")
    print("pip install ultralytics")
    sys.exit(1)


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Training configuration"""
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "anpr_training_data"
    IMAGES_DIR = DATA_DIR / "images"
    LABELS_DIR = DATA_DIR / "labels"
    RAW_IMAGES_DIR = DATA_DIR / "raw_images"
    
    # Grounding DINO configuration
    GROUNDING_DINO_CONFIG = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT = "weights/groundingdino_swint_ogc.pth"
    TEXT_PROMPT = "license plate . number plate . vehicle registration plate"
    BOX_THRESHOLD = 0.35  # Confidence threshold
    TEXT_THRESHOLD = 0.25  # Text matching threshold
    
    # YOLO training configuration
    YOLO_MODEL = "yolov10n.pt"  # Nano model for speed
    YOLO_EPOCHS = 100
    YOLO_BATCH_SIZE = 32
    YOLO_IMG_SIZE = 640
    YOLO_DEVICE = "0"  # GPU ID or "cpu"
    
    # Dataset split
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.05
    
    # TensorRT export
    EXPORT_TENSORRT = True
    TENSORRT_WORKSPACE = 4  # GB


# ============================================================================
# STEP 1: PSEUDO-LABELING WITH GROUNDING DINO
# ============================================================================

class GroundingDINOLabeler:
    """Generates pseudo-labels using Grounding DINO"""
    
    def __init__(self, config_path: str, checkpoint_path: str):
        """Initialize Grounding DINO model"""
        print("[1/5] Loading Grounding DINO model...")
        
        # Check if files exist
        if not os.path.exists(checkpoint_path):
            print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
            print("Download from: https://github.com/IDEA-Research/GroundingDINO")
            sys.exit(1)
        
        self.model = load_model(config_path, checkpoint_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        print(f"[INFO] Model loaded on {self.device}")
    
    
    def predict_boxes(
        self, 
        image_path: str, 
        text_prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25
    ) -> List[Tuple[float, float, float, float, float]]:
        """
        Predict bounding boxes for license plates
        
        Returns:
            List of (x1, y1, x2, y2, confidence) in pixel coordinates
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        
        # Run inference
        boxes, logits, phrases = predict(
            model=self.model,
            image=image_np,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device
        )
        
        # Convert normalized boxes to pixel coordinates
        h, w = image_np.shape[:2]
        boxes_pixel = []
        
        for box, score in zip(boxes, logits):
            # Grounding DINO returns boxes in (cx, cy, w, h) normalized format
            cx, cy, bw, bh = box.cpu().numpy()
            
            # Convert to (x1, y1, x2, y2) pixel format
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            
            # Clamp to image boundaries
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            boxes_pixel.append((x1, y1, x2, y2, float(score)))
        
        return boxes_pixel
    
    
    def visualize_prediction(
        self, 
        image_path: str, 
        boxes: List[Tuple],
        output_path: str
    ):
        """Save visualization of predictions"""
        image = cv2.imread(image_path)
        
        for box in boxes:
            x1, y1, x2, y2, conf = box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image, 
                f"{conf:.2f}", 
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 0), 
                2
            )
        
        cv2.imwrite(output_path, image)


# ============================================================================
# STEP 2: YOLO FORMAT CONVERSION
# ============================================================================

def convert_to_yolo_format(
    boxes: List[Tuple], 
    image_width: int, 
    image_height: int,
    class_id: int = 0
) -> List[str]:
    """
    Convert bounding boxes to YOLO format
    
    YOLO format: <class_id> <x_center> <y_center> <width> <height> (all normalized)
    
    Args:
        boxes: List of (x1, y1, x2, y2, confidence) boxes
        image_width: Image width in pixels
        image_height: Image height in pixels
        class_id: Class ID (0 for license plate)
        
    Returns:
        List of YOLO format strings
    """
    yolo_lines = []
    
    for box in boxes:
        x1, y1, x2, y2, conf = box
        
        # Calculate center and dimensions
        x_center = (x1 + x2) / 2.0 / image_width
        y_center = (y1 + y2) / 2.0 / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height
        
        # Clamp to [0, 1]
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width = max(0.0, min(1.0, width))
        height = max(0.0, min(1.0, height))
        
        # Format: class_id x_center y_center width height
        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        yolo_lines.append(yolo_line)
    
    return yolo_lines


def save_yolo_annotation(label_path: str, yolo_lines: List[str]):
    """Save YOLO format annotation to file"""
    with open(label_path, 'w') as f:
        f.write('\n'.join(yolo_lines))


# ============================================================================
# STEP 3: DATASET PREPARATION
# ============================================================================

def prepare_dataset_structure(config: Config):
    """Create dataset directory structure"""
    print("[2/5] Preparing dataset structure...")
    
    # Create directories
    config.DATA_DIR.mkdir(exist_ok=True)
    config.RAW_IMAGES_DIR.mkdir(exist_ok=True)
    config.IMAGES_DIR.mkdir(exist_ok=True)
    config.LABELS_DIR.mkdir(exist_ok=True)
    
    # Create train/val/test splits
    for split in ['train', 'val', 'test']:
        (config.IMAGES_DIR / split).mkdir(exist_ok=True)
        (config.LABELS_DIR / split).mkdir(exist_ok=True)
    
    print(f"[INFO] Dataset directory created at: {config.DATA_DIR}")


def create_dataset_yaml(config: Config):
    """Create YOLO dataset configuration file"""
    yaml_content = f"""
# ANPR License Plate Detection Dataset
# Generated by train_anpr_detector.py

path: {config.DATA_DIR.absolute()}
train: images/train
val: images/val
test: images/test

# Classes
nc: 1  # Number of classes
names: ['license_plate']  # Class names
"""
    
    yaml_path = config.DATA_DIR / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content.strip())
    
    print(f"[INFO] Dataset YAML created: {yaml_path}")
    return yaml_path


# ============================================================================
# STEP 4: PSEUDO-LABELING PIPELINE
# ============================================================================

def run_pseudo_labeling(config: Config):
    """Run pseudo-labeling on unlabeled images"""
    print("[3/5] Running pseudo-labeling with Grounding DINO...")
    
    # Initialize labeler
    labeler = GroundingDINOLabeler(
        config.GROUNDING_DINO_CONFIG,
        config.GROUNDING_DINO_CHECKPOINT
    )
    
    # Get all images from raw_images directory
    image_files = list(config.RAW_IMAGES_DIR.glob("*.jpg")) + \
                  list(config.RAW_IMAGES_DIR.glob("*.png"))
    
    if not image_files:
        print(f"[ERROR] No images found in {config.RAW_IMAGES_DIR}")
        print("Please add vehicle images to the raw_images directory")
        sys.exit(1)
    
    print(f"[INFO] Found {len(image_files)} images to label")
    
    # Split into train/val/test
    np.random.shuffle(image_files)
    n_train = int(len(image_files) * config.TRAIN_SPLIT)
    n_val = int(len(image_files) * config.VAL_SPLIT)
    
    splits = {
        'train': image_files[:n_train],
        'val': image_files[n_train:n_train + n_val],
        'test': image_files[n_train + n_val:]
    }
    
    # Process each split
    stats = {'total': 0, 'detected': 0, 'no_detection': 0}
    
    for split_name, split_images in splits.items():
        print(f"\n[INFO] Processing {split_name} set ({len(split_images)} images)...")
        
        for idx, image_path in enumerate(split_images):
            # Predict boxes
            boxes = labeler.predict_boxes(
                str(image_path),
                config.TEXT_PROMPT,
                config.BOX_THRESHOLD,
                config.TEXT_THRESHOLD
            )
            
            stats['total'] += 1
            if boxes:
                stats['detected'] += 1
            else:
                stats['no_detection'] += 1
            
            # Get image dimensions
            image = cv2.imread(str(image_path))
            h, w = image.shape[:2]
            
            # Convert to YOLO format
            yolo_lines = convert_to_yolo_format(boxes, w, h)
            
            # Save image and label
            new_image_name = f"{split_name}_{idx:05d}.jpg"
            new_label_name = f"{split_name}_{idx:05d}.txt"
            
            # Copy image
            import shutil
            shutil.copy2(
                image_path,
                config.IMAGES_DIR / split_name / new_image_name
            )
            
            # Save label
            save_yolo_annotation(
                config.LABELS_DIR / split_name / new_label_name,
                yolo_lines
            )
            
            # Progress
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(split_images)} images...")
    
    print(f"\n[INFO] Pseudo-labeling complete!")
    print(f"  Total images: {stats['total']}")
    print(f"  Plates detected: {stats['detected']}")
    print(f"  No detection: {stats['no_detection']}")
    print(f"  Detection rate: {stats['detected']/stats['total']*100:.1f}%")


# ============================================================================
# STEP 5: YOLO TRAINING
# ============================================================================

def train_yolo_model(config: Config, dataset_yaml: Path):
    """Train YOLOv10 model on pseudo-labeled dataset"""
    print("\n[4/5] Training YOLOv10 model...")
    
    # Initialize model
    model = YOLO(config.YOLO_MODEL)
    
    # Train
    results = model.train(
        data=str(dataset_yaml),
        epochs=config.YOLO_EPOCHS,
        batch=config.YOLO_BATCH_SIZE,
        imgsz=config.YOLO_IMG_SIZE,
        device=config.YOLO_DEVICE,
        project=str(config.DATA_DIR / "runs"),
        name="license_plate_detector",
        patience=10,  # Early stopping
        save=True,
        verbose=True,
        workers=8,
        # Augmentation for robustness
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0
    )
    
    print("[INFO] Training complete!")
    print(f"[INFO] Model saved to: {config.DATA_DIR / 'runs' / 'license_plate_detector'}")
    
    return model


# ============================================================================
# STEP 6: TENSORRT EXPORT
# ============================================================================

def export_to_tensorrt(model: YOLO, config: Config):
    """Export trained model to TensorRT for DeepStream"""
    print("\n[5/5] Exporting to TensorRT...")
    
    try:
        # Export to ONNX first
        onnx_path = model.export(format="onnx", dynamic=False, simplify=True)
        print(f"[INFO] ONNX model exported: {onnx_path}")
        
        # Export to TensorRT
        engine_path = model.export(
            format="engine",
            device=config.YOLO_DEVICE,
            workspace=config.TENSORRT_WORKSPACE,
            half=True  # FP16 for better performance
        )
        print(f"[INFO] TensorRT engine exported: {engine_path}")
        
        # Generate DeepStream config snippet
        generate_deepstream_config(config)
        
    except Exception as e:
        print(f"[WARN] TensorRT export failed: {e}")
        print("[INFO] You can manually convert ONNX to TensorRT using trtexec")


def generate_deepstream_config(config: Config):
    """Generate DeepStream config file for the trained model"""
    config_content = """
[property]
gpu-id=0
net-scale-factor=0.00392156862745098
model-color-format=0
onnx-file=best.onnx
model-engine-file=best.engine
labelfile-path=labels.txt
batch-size=16
network-mode=2
num-detected-classes=1
interval=0
gie-unique-id=3
process-mode=1
network-type=0
cluster-mode=2
maintain-aspect-ratio=1
parse-bbox-func-name=NvDsInferParseCustomYOLOV10
custom-lib-path=/opt/nvidia/deepstream/deepstream/lib/libnvds_infercustomparser.so

[class-attrs-all]
nms-iou-threshold=0.45
pre-cluster-threshold=0.25
topk=300
"""
    
    config_path = config.DATA_DIR / "config_sgie_yolo_lpd.txt"
    with open(config_path, 'w') as f:
        f.write(config_content.strip())
    
    print(f"[INFO] DeepStream config generated: {config_path}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main training pipeline"""
    print("=" * 80)
    print("ANPR License Plate Detector Training Pipeline")
    print("Semi-Supervised Learning with Grounding DINO + YOLOv10")
    print("=" * 80)
    
    config = Config()
    
    # Check for GPU
    if not torch.cuda.is_available():
        print("[WARN] CUDA not available. Training will be slow on CPU.")
        config.YOLO_DEVICE = "cpu"
    
    # Step 1: Prepare dataset structure
    prepare_dataset_structure(config)
    
    # Step 2: Create dataset YAML
    dataset_yaml = create_dataset_yaml(config)
    
    # Step 3: Run pseudo-labeling (if images exist)
    if list(config.RAW_IMAGES_DIR.glob("*.jpg")) or list(config.RAW_IMAGES_DIR.glob("*.png")):
        run_pseudo_labeling(config)
    else:
        print("\n[INFO] Skipping pseudo-labeling (no images in raw_images directory)")
        print(f"[INFO] Add images to: {config.RAW_IMAGES_DIR}")
        print("[INFO] Or manually create annotations in YOLO format")
    
    # Check if we have training data
    train_images = list((config.IMAGES_DIR / 'train').glob("*.jpg"))
    if not train_images:
        print("\n[ERROR] No training data found. Exiting.")
        sys.exit(1)
    
    # Step 4: Train YOLO model
    model = train_yolo_model(config, dataset_yaml)
    
    # Step 5: Export to TensorRT
    if config.EXPORT_TENSORRT:
        export_to_tensorrt(model, config)
    
    print("\n" + "=" * 80)
    print("Training pipeline complete!")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"1. Copy the TensorRT engine to your DeepStream project")
    print(f"2. Update config_sgie_yolo_lpd.txt with the model path")
    print(f"3. Run the DeepStream pipeline with main_pipeline.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
