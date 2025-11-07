# Real-Time Vehicle Analytics Pipeline - Deployment Guide

## System Architecture Overview

This is a production-grade, 20fps vehicle analytics system built on NVIDIA DeepStream 7.0+. The architecture follows a decoupled, multi-model ensemble design for maximum throughput and fault tolerance.

```
┌─────────────┐
│  RTSP/File  │
│   Stream    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│              NVIDIA DeepStream Pipeline                     │
│  ┌──────────┐  ┌────────┐  ┌──────────────────────────┐   │
│  │ YOLOv10  ├─→│ByteTrack├─→│  Custom Direction Probe  │   │
│  │  (PGIE)  │  │         │  │  (Line Crossing Logic)   │   │
│  └──────────┘  └────┬───┘  └──────────────────────────┘   │
│                     │                                        │
│           ┌─────────┴─────────┐                            │
│           ▼                   ▼                             │
│    ┌──────────┐        ┌──────────┐                        │
│    │ ResNet50 │        │ YOLO-LPD │                        │
│    │  Color   │        │  (SGIE1) │                        │
│    │ (SGIE0)  │        └────┬─────┘                        │
│    └──────────┘             │                              │
│                              ▼                              │
│                       ┌──────────┐                         │
│                       │  LPRNet  │                         │
│                       │   OCR    │                         │
│                       │ (SGIE2)  │                         │
│                       └────┬─────┘                         │
│                            │                                │
│                    ┌───────┴────────┐                      │
│                    │  nvmsgbroker   │                      │
│                    │   (RabbitMQ)   │                      │
│                    └────────────────┘                      │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │   RabbitMQ     │
                    │  Message Queue │
                    └────────┬───────┘
                             │
                             ▼
                    ┌────────────────┐
                    │  db_consumer   │
                    │   (Python)     │
                    └────────┬───────┘
                             │
                             ▼
                    ┌────────────────┐
                    │   PostgreSQL   │
                    │    Database    │
                    └────────────────┘
```

## Key Design Principles

1. **Decoupled Architecture**: Pipeline never blocks on database writes
2. **Zero Frame Drops**: 20fps guaranteed through GPU-accelerated inference
3. **Fault Tolerance**: RabbitMQ queue buffers messages if database is slow
4. **Scalability**: Add more consumers to handle write spikes

## File Descriptions

### Core Pipeline Files

- **`deepstream_config.txt`**: Main DeepStream configuration defining the complete GStreamer pipeline
- **`main_pipeline.py`**: Python orchestrator with custom probe for direction detection
- **`line_crossing_logic.py`**: Virtual line-crossing module using trajectory analysis
- **`db_consumer.py`**: Async RabbitMQ→PostgreSQL consumer with batch inserts
- **`schema.sql`**: Complete database schema with indexes and materialized views

### Training Files

- **`train_anpr_detector.py`**: Semi-supervised training pipeline for license plate detector using Grounding DINO

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with compute capability ≥ 7.0 (Volta+)
  - Recommended: RTX 3060 or higher
  - Minimum VRAM: 6GB
- **CPU**: 8+ cores for optimal performance
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 50GB for models, data, and logs

### Software Requirements
```bash
# NVIDIA Drivers
nvidia-driver-535 or higher

# CUDA Toolkit
CUDA 12.1 or higher

# TensorRT
TensorRT 8.6 or higher

# DeepStream SDK
DeepStream 7.0 or higher

# Python
Python 3.8+
```

## Installation Steps

### 1. Install DeepStream SDK

```bash
# Download from NVIDIA Developer
wget https://developer.nvidia.com/downloads/deepstream-7.0

# Install dependencies
sudo apt-get install \
    libssl-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstrtspserver-1.0-dev \
    libx11-dev

# Install DeepStream
sudo tar -xvf deepstream_sdk_v7.0_x86_64.tbz2 -C /
cd /opt/nvidia/deepstream/deepstream
sudo ./install.sh
```

### 2. Install Python Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install core dependencies
pip install --upgrade pip
pip install numpy opencv-python pika psycopg2-binary

# Install DeepStream Python bindings
cd /opt/nvidia/deepstream/deepstream/lib
pip install pyds-1.1.10-py3-none-linux_x86_64.whl
```

### 3. Install RabbitMQ

```bash
# Install RabbitMQ server
sudo apt-get update
sudo apt-get install rabbitmq-server

# Start service
sudo systemctl start rabbitmq-server
sudo systemctl enable rabbitmq-server

# Enable management plugin (optional)
sudo rabbitmq-plugins enable rabbitmq_management

# Access web UI at http://localhost:15672
# Default credentials: guest/guest
```

### 4. Install PostgreSQL

```bash
# Install PostgreSQL 14+
sudo apt-get install postgresql postgresql-contrib

# Start service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database
sudo -u postgres psql
CREATE DATABASE vehicle_analytics;
CREATE USER analytics_app WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE vehicle_analytics TO analytics_app;
\q

# Run schema
psql -U analytics_app -d vehicle_analytics -f schema.sql
```

### 5. Download Pre-trained Models

```bash
# Create model directory
mkdir -p models/

# YOLOv10 for vehicle detection
wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt -O models/yolov10n.pt

# ResNet50 for color classification (train or download pre-trained)
# LPRNet for OCR (train or download pre-trained)
# YOLO-LPD for plate detection (use train_anpr_detector.py)

# Convert to TensorRT
# See model conversion section below
```

## Model Preparation

### Converting Models to TensorRT

```bash
# YOLOv10 Vehicle Detector
trtexec --onnx=models/yolov10n.onnx \
        --saveEngine=models/yolov10n.engine \
        --fp16 \
        --workspace=4096

# ResNet50 Color Classifier
trtexec --onnx=models/resnet50_color.onnx \
        --saveEngine=models/resnet50_color.engine \
        --fp16

# License Plate Detector (after training)
python train_anpr_detector.py
# This will generate the TensorRT engine automatically

# LPRNet OCR
trtexec --onnx=models/lprnet.onnx \
        --saveEngine=models/lprnet.engine \
        --fp16
```

### Training Custom Models

For license plate detection:
```bash
# 1. Collect vehicle images and place in raw_images/
mkdir -p anpr_training_data/raw_images/
# Add 500-1000 vehicle images

# 2. Run training pipeline
python train_anpr_detector.py

# 3. Model will be auto-exported to TensorRT
```

## Configuration

### 1. Update `deepstream_config.txt`

```ini
# Update video source
[source0]
uri=file:///path/to/your/video.mp4
# OR for RTSP
uri=rtsp://username:password@camera-ip:554/stream

# Update model paths
[primary-gie]
config-file=/path/to/config_pgie_yolov10.txt

[secondary-gie0]
config-file=/path/to/config_sgie_color_resnet50.txt

# ... (update all SGIE paths)

# Update RabbitMQ connection
[msgbroker]
conn-str=localhost;5672;guest;guest
```

### 2. Update `main_pipeline.py`

```python
# Adjust virtual line coordinates for your camera view
VIRTUAL_LINE_A = [(960, 400), (960, 680)]

# For horizontal lines:
# VIRTUAL_LINE_A = [(200, 540), (1720, 540)]
```

### 3. Update `db_consumer.py`

```python
# PostgreSQL credentials
POSTGRES_CONFIG = {
    'host': 'localhost',
    'database': 'vehicle_analytics',
    'user': 'analytics_app',
    'password': 'your_secure_password',
}

# RabbitMQ credentials
RABBITMQ_CONFIG = {
    'host': 'localhost',
    'username': 'guest',
    'password': 'guest',
}
```

## Running the System

### Terminal 1: Start Database Consumer
```bash
source venv/bin/activate
python db_consumer.py
```

### Terminal 2: Start DeepStream Pipeline
```bash
source venv/bin/activate
python main_pipeline.py
```

### Expected Output

**DeepStream Pipeline:**
```
[INFO] Starting pipeline...
[INFO] Pipeline running. Press Ctrl+C to stop.

[VEHICLE SIGHTING] ID:42 | Plate:MH12AB1234 | Color:Red | Direction:towards_camera | Confidence:0.94
[VEHICLE SIGHTING] ID:43 | Plate:DL05CD5678 | Color:White | Direction:away_from_camera | Confidence:0.91
```

**Database Consumer:**
```
[DB] Connection pool created: 2-10 connections
[MQ] Connected to RabbitMQ: localhost:5672
[MQ] Listening on queue: vehicle_sightings

[MSG] Processed: MH12AB1234 | Red | towards_camera
[MSG] Processed: DL05CD5678 | White | away_from_camera
[DB] ✓ Flushed 50 records to database
```

## Performance Tuning

### Achieving 20fps Target

1. **GPU Memory Optimization**
   ```ini
   # In deepstream_config.txt
   [streammux]
   nvbuf-memory-type=0  # Device memory
   
   [primary-gie]
   batch-size=1  # Start with 1, increase if multi-stream
   ```

2. **Inference Interval Tuning**
   ```ini
   # Run SGIEs less frequently if needed
   [secondary-gie0]
   interval=2  # Run every 2nd frame
   ```

3. **Tracker Optimization**
   ```yaml
   # config_tracker_bytetrack.yml
   track_algo: BYTE
   fast_mode: 1
   ```

### Database Performance

```sql
-- Monitor insert performance
SELECT COUNT(*) as records_per_min
FROM vehicle_sightings
WHERE timestamp > NOW() - INTERVAL '1 minute';

-- Check index usage
SELECT schemaname, tablename, indexname, idx_scan
FROM pg_stat_user_indexes
WHERE schemaname = 'public';
```

## Monitoring & Debugging

### Pipeline Performance
```bash
# Check GPU utilization
nvidia-smi -l 1

# DeepStream performance metrics
# Automatically printed every 5 seconds in console
```

### RabbitMQ Queue Depth
```bash
# Check queue status
sudo rabbitmqctl list_queues name messages consumers

# Web UI
http://localhost:15672
```

### Database Queries
```sql
-- Recent sightings
SELECT * FROM vehicle_sightings
ORDER BY timestamp DESC
LIMIT 10;

-- Vehicle history
SELECT * FROM get_vehicle_history('MH12AB1234', 7);

-- Traffic flow in last hour
SELECT * FROM get_traffic_flow(1, 60);

-- Hourly statistics
SELECT * FROM analytics_summary
WHERE hour > NOW() - INTERVAL '24 hours'
ORDER BY hour DESC;
```

## Troubleshooting

### Issue: Pipeline drops frames

**Solution:**
- Reduce SGIE batch sizes
- Increase `interval` parameter for SGIEs
- Use lower resolution models (e.g., YOLOv10-N instead of YOLOv10-S)

### Issue: RabbitMQ queue growing

**Solution:**
- Start multiple db_consumer instances
- Increase `BATCH_INSERT_SIZE` in db_consumer.py
- Add database indexes on frequently queried columns

### Issue: Low detection accuracy

**Solution:**
- Collect more training data for YOLO-LPD
- Adjust `BOX_THRESHOLD` in Grounding DINO labeling
- Fine-tune confidence thresholds in config files

### Issue: Direction detection not working

**Solution:**
- Verify virtual line placement using visualizations
- Adjust line coordinates in `main_pipeline.py`
- Check tracker is generating trajectories

## Production Deployment Checklist

- [ ] GPU drivers and CUDA installed
- [ ] DeepStream SDK 7.0+ installed
- [ ] All models converted to TensorRT
- [ ] RabbitMQ running with persistent queues
- [ ] PostgreSQL with proper indexes
- [ ] Database backup strategy in place
- [ ] Monitoring tools configured (Grafana, Prometheus)
- [ ] Log rotation configured
- [ ] Systemd services for auto-restart
- [ ] Virtual line calibrated for camera view
- [ ] Accuracy validated on test footage
- [ ] Performance benchmarks achieved (20fps)

## Scaling to Multiple Cameras

```python
# Update deepstream_config.txt
[source0]
uri=rtsp://camera1-ip/stream

[source1]
uri=rtsp://camera2-ip/stream

[source2]
uri=rtsp://camera3-ip/stream

# Update streammux
[streammux]
batch-size=3  # Number of cameras
```

## API Integration (Future Enhancement)

Consider adding a REST API layer:
```python
# api_server.py (pseudocode)
from fastapi import FastAPI
import psycopg2

app = FastAPI()

@app.get("/vehicles/{plate}")
def get_vehicle_history(plate: str):
    # Query database
    # Return JSON
    pass

@app.get("/traffic/live")
def get_live_traffic():
    # Query recent sightings
    # Return JSON
    pass
```

## License & Credits

- NVIDIA DeepStream SDK: [https://developer.nvidia.com/deepstream-sdk](https://developer.nvidia.com/deepstream-sdk)
- YOLOv10: [https://github.com/THU-MIG/yolov10](https://github.com/THU-MIG/yolov10)
- ByteTrack: [https://github.com/ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack)
- Grounding DINO: [https://github.com/IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)

## Support

For issues, questions, or improvements:
1. Check DeepStream documentation: [https://docs.nvidia.com/metropolis/deepstream/](https://docs.nvidia.com/metropolis/deepstream/)
2. Review code comments in individual files
3. Monitor system logs for error messages

---
**System Version:** 1.0  
**Last Updated:** November 2025  
**Target Performance:** 20fps @ 1080p
