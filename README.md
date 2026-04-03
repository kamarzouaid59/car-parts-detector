
This repository is part of an InsureVision damage assessment pipeline:
1. **car-parts-detector** (this repo) — Detects car parts from image, orchestrates pipeline
2. **damage_classification_models** (separate service) — Classifies damage in each detected part
3. **InsureVision webapp** — Presents results to user

All services run locally on your machine.

## Quick Start

```bash
uv pip install -r requirements.txt
uv run ex2.py --image-path car.jpg        # Local test with visuals
# OR
uv run ex2.py --server                     # Start API server
# OR  
uv run ex2.py --api --endpoint pipeline --image-path car.jpg  # Test API
```

## What is ex2.py?

[ex2.py](ex2.py) is a unified script with three independent modes:

### 1️⃣ Local Detection (Image Testing Mode)

Run YOLO detection locally with visual outputs:

```bash
uv run ex2.py --image-path car.jpg
```

Output:
- Prints detected parts and confidence
- Shows annotated image (`output_full.jpg`)
- Saves cropped parts to `cropped_parts/` folder
- Optional: `uv run ex2.py --image-path car.jpg --output-dir my_crops`

### 2️⃣ API Server Mode

Start the FastAPI service:

```bash
uv run ex2.py --server
```

This starts a server on **http://127.0.0.1:8000** with three endpoints:

- `GET /health` — Returns service status
- `POST /predict` — Parts detection (returns bounding boxes + confidence)
- `POST /pipeline` — Full pipeline (parts detection + damage classification via separate service)

### 3️⃣ API Testing Mode

Test an already-running API server:

```bash
# Test /predict endpoint
uv run ex2.py --api --endpoint predict --image-path car.jpg

# Test /pipeline endpoint (requires damage_classifier on port 8001)
uv run ex2.py --api --endpoint pipeline --image-path car.jpg

# Custom API URL
uv run ex2.py --api --api-url http://other-host:8000 --endpoint pipeline --image-path car.jpg
```

Prints results in JSON format for debugging/integration.

---

## Full Pipeline Flow

When `/pipeline` endpoint is called:

```
Client uploads car image
    ↓
POST /pipeline (parts-detector, port 8000)
    │
    ├─→ Load YOLO model (epoch3.pt)
    ├─→ Detect car parts
    └─→ For each detected part crop:
            │
            └─→ POST crop to damage classifier (port 8001)
                    ↓
                 EfficientNetB0 model classifies damage type
                    ↓
                 Returns damage class + confidence
    │
    └─→ Combine part detection + damage classification
        ↓
        Return unified JSON response to client
```

## Configuration

Control behavior with environment variables:

```powershell
$env:MODEL_PATH = "epoch3.pt"           # YOLO weights
$env:PARTS_CONF_THRESHOLD = "0.3"      # Detection confidence
$env:PARTS_IOU_THRESHOLD = "0.5"       # NMS threshold
$env:DAMAGE_SERVICE_URL = "http://127.0.0.1:8001"  # Damage classifier URL
$env:SAVE_ANNOTATED = "true"           # Optional: save annotated images
$env:SAVE_CROPS = "true"               # Optional: save part crops
$env:OUTPUT_DIR = "./outputs"          # Where to save artifacts

uv run ex2.py --server
```

Defaults all work for local-only testing.

## Usage Examples

### Local Testing (Simplest)

```bash
uv run ex2.py --image-path my_car.jpg
```

You'll see:
- Console output with detected parts
- `output_full.jpg` with bounding boxes drawn
- Cropped images in `cropped_parts/` folder

### Full Integrated Testing

**Terminal 1** — Start damage classifier:
```bash
cd D:\damage_classification_models
uv run uvicorn final_model:app --host 127.0.0.1 --port 8001
```

**Terminal 2** — Start parts detector API:
```bash
cd D:\car-parts-detector
uv run ex2.py --server
```

**Terminal 3** — Test the full pipeline:
```bash
uv run ex2.py --api --endpoint pipeline --image-path car.jpg
```

Sample output:
```json
{
  "image_id": "car",
  "image_width": 1920,
  "image_height": 1080,
  "detections": [
    {
      "part_label": "front_bumper",
      "bbox": [120, 310, 640, 760],
      "detection_confidence": 0.943,
      "damage": {
        "prediction": "dent",
        "confidence": 87.65,
        "probabilities": {
          "crack": 2.1,
          "dent": 87.65,
          "glass shatter": 0.5
        }
      }
    }
  ]
}
```

### cURL Commands (For Webapp Integration)

Health check:
```bash
curl http://127.0.0.1:8000/health
```

Parts detection only:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@car.jpg"
```

Full pipeline:
```bash
curl -X POST "http://127.0.0.1:8000/pipeline" \
  -F "file=@car.jpg"
```

## File Structure

```
d:\car-parts-detector\
├── ex2.py                      ← Unified script (local, server, or API test modes)
├── epoch3.pt                   ← YOLO model weights
├── requirements.txt            ← Python dependencies  
└── README.md                   ← This file
```

There are no separate api.py or detector_core.py files—all functionality is in [ex2.py](ex2.py).
# Car Parts Detector

This repository is part of an InsureVision damage assessment pipeline:
1. **car-parts-detector** (this repo) — Detects car parts from image, orchestrates pipeline
2. **damage_classification_models** (separate service) — Classifies damage in each detected part
3. **InsureVision webapp** — Presents results to user

All services run locally on your machine.

## Architecture

Two separate FastAPI services communicate locally:

```
Webapp uploads car image to /pipeline
    ↓
car-parts-detector (port 8000)
    ├─ Detects parts using YOLO (epoch3.pt)
    └─ For each crop, calls damage classifier service
         ↓
    damage_classification_models (port 8001)
         └─ Classifies damage type & confidence
    ↓
Returns combined results to webapp
```

## API Endpoints

- `GET /health` — Health check
- `POST /predict` — Parts detection only
- `POST /pipeline` — Full flow (parts detection + damage classification)

## Pipeline: Full Assessment (`POST /pipeline`)

**Primary endpoint for InsureVision integration.**

User uploads car image → detector finds parts → damage classifier analyzes each part → results to webapp.

### Input
Multipart form with `file` field (image upload).

### Output
```json
{
  "image_id": "test_image",
  "image_width": 1920,
  "image_height": 1080,
  "detections": [
    {
      "part_label": "front_bumper",
      "bbox": [120, 310, 640, 760],
      "detection_confidence": 0.9432,
      "damage": {
        "damage_class": "dent",
        "confidence": 87.65,
        "probabilities": {
          "crack": 2.1,
          "dent": 87.65,
          "glass shatter": 0.5,
          "lamp broken": 0.8,
          "scratch": 5.2,
          "tire flat": 3.9
        }
      }
    }
  ]
}
```

Each detection contains:
- `part_label`: detected car part (hood, bumper, wheel, etc.)
- `bbox`: absolute bounding box coordinates `[x_min, y_min, x_max, y_max]`
- `detection_confidence`: part detection confidence
- `damage`: damage classification for that part

## Parts Detection Only (`POST /predict`)

If you only need part detection without damage classification:

### Input
Multipart form with `file` field.

### Output
```json
{
  "image_id": "test_image",
  "image_width": 1920,
  "image_height": 1080,
  "detections": [
    {
      "part_label": "front_bumper",
      "bbox": [120, 310, 640, 760],
      "confidence": 0.9432
    }
  ]
}
```

## Configuration (Environment Variables)

- `MODEL_PATH` (default: `./epoch3.pt`)
- `PARTS_CONF_THRESHOLD` (default: `0.3`)
- `PARTS_IOU_THRESHOLD` (default: `0.5`)
- `SAVE_ANNOTATED` (`true`/`false`, default: `false`)
- `SAVE_CROPS` (`true`/`false`, default: `false`)
- `OUTPUT_DIR` (default: `./outputs`)

## Setup & Run

### Prerequisites

Both services must be running for the full pipeline to work.

### 1. Start Damage Classifier Service

In the `D:\damage_classification_models` folder:

```bash
uv run uvicorn final_model:app --host 127.0.0.1 --port 8001 --reload
```

Check health: `curl http://127.0.0.1:8001/`

### 2. Install & Start Parts Detector

In this folder (`D:\car-parts-detector`):

```bash
uv pip install -r requirements.txt
uv run uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```

Check health: `curl http://127.0.0.1:8000/health`

### 3. Configure Service URLs (Optional)

By default, car-parts-detector looks for damage classifier at `http://127.0.0.1:8001`.

To override:

```bash
set DAMAGE_SERVICE_URL=http://your-damage-service:port
uv run uvicorn api:app --host 127.0.0.1 --port 8000
```

## Usage Examples

### Full Pipeline (Parts + Damage Classification)

Requires both services running.

```bash
curl -X POST "http://127.0.0.1:8000/pipeline" -F "file=@car_image.jpg"
```

Response:
```json
{
  "image_id": "car_image",
  "image_width": 1920,
  "image_height": 1080,
  "detections": [
    {
      "part_label": "front_bumper",
      "bbox": [120, 310, 640, 760],
      "detection_confidence": 0.9432,
      "damage": {
        "prediction": "dent",
        "confidence": 87.65,
        "probabilities": {
          "crack": 2.1,
          "dent": 87.65,
          "glass shatter": 0.5,
          "lamp broken": 0.8,
          "scratch": 5.2,
          "tire flat": 3.9
        }
      }
    }
  ]
}
```

### Parts Detection Only

(Damage classifier service not required)

```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@car_image.jpg"
```

Response includes only detected parts and bounding boxes.

### Health Checks

Parts detector:
```bash
curl http://127.0.0.1:8000/health
```

Damage classifier:
```bash
curl http://127.0.0.1:8001/
```

## Testing Locally

Use the offline script for standalone validation (no API needed):

```bash
uv run ex2.py --image-path your_image.jpg
```

This runs the parts detector directly with visual outputs (useful for debugging and model validation).


