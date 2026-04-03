import argparse
import asyncio
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO


# ═══════════════════════════════════════════════════════════════════════════
# DETECTOR CORE LOGIC
# ═══════════════════════════════════════════════════════════════════════════

def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    model_path: str
    conf_threshold: float
    iou_threshold: float
    save_annotated: bool
    save_crops: bool
    output_dir: Path


def load_settings() -> Settings:
    return Settings(
        model_path=os.getenv("MODEL_PATH", "./epoch3.pt"),
        conf_threshold=_env_float("PARTS_CONF_THRESHOLD", 0.3),
        iou_threshold=_env_float("PARTS_IOU_THRESHOLD", 0.5),
        save_annotated=_env_bool("SAVE_ANNOTATED", False),
        save_crops=_env_bool("SAVE_CROPS", False),
        output_dir=Path(os.getenv("OUTPUT_DIR", "./outputs")),
    )


class DetectorService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = YOLO(settings.model_path)

    def predict_from_image(self, image_bgr: Any, image_id: str | None = None) -> dict[str, Any]:
        if image_bgr is None:
            raise ValueError("Input image is empty or invalid")

        image_id = image_id or str(uuid.uuid4())
        results = self.model.predict(
            source=image_bgr,
            conf=self.settings.conf_threshold,
            iou=self.settings.iou_threshold,
            save=False,
            verbose=False,
        )
        result = results[0]
        orig_img = result.orig_img
        image_height, image_width = int(orig_img.shape[0]), int(orig_img.shape[1])

        detections: list[dict[str, Any]] = []
        for box in result.boxes:
            part_label = result.names[int(box.cls)]
            confidence = float(box.conf)

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1 = max(0, min(image_width, int(round(x1))))
            y1 = max(0, min(image_height, int(round(y1))))
            x2 = max(0, min(image_width, int(round(x2))))
            y2 = max(0, min(image_height, int(round(y2))))

            detections.append(
                {
                    "part_label": str(part_label),
                    "bbox": [x1, y1, x2, y2],
                    "confidence": confidence,
                }
            )

        payload = {
            "image_id": str(image_id),
            "image_width": image_width,
            "image_height": image_height,
            "detections": detections,
        }

        if self.settings.save_annotated or self.settings.save_crops:
            self.settings.output_dir.mkdir(parents=True, exist_ok=True)

        if self.settings.save_annotated:
            annotated = result.plot()
            out_annotated = self.settings.output_dir / f"{image_id}_annotated.jpg"
            cv2.imwrite(str(out_annotated), annotated)

        if self.settings.save_crops:
            for idx, det in enumerate(detections):
                x1, y1, x2, y2 = det["bbox"]
                crop = orig_img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                safe_label = det["part_label"].replace(" ", "_")
                out_crop = self.settings.output_dir / f"{image_id}_{safe_label}_{idx}.jpg"
                cv2.imwrite(str(out_crop), crop)

        return payload

    def predict_from_path(self, image_path: str, image_id: str | None = None) -> dict[str, Any]:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to load image from path: {image_path}")
        image_id = image_id or Path(image_path).stem
        return self.predict_from_image(image, image_id=image_id)


def load_image_from_url(image_url: str) -> tuple[np.ndarray, str]:
    """Download and decode an image from URL."""
    try:
        response = requests.get(image_url, timeout=20)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ValueError(f"Unable to fetch image_url: {exc}") from exc

    image_array = np.frombuffer(response.content, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Fetched URL content is not a valid image")

    parsed = urlparse(image_url)
    image_id = Path(parsed.path).stem or str(uuid.uuid4())
    return image, image_id


# ═══════════════════════════════════════════════════════════════════════════
# API SERVER LOGIC
# ═══════════════════════════════════════════════════════════════════════════

def create_app(detector: DetectorService) -> FastAPI:
    """Create FastAPI app with detection endpoints."""
    app = FastAPI(title="Car Parts Detector API", version="1.0.0")
    damage_service_url = os.getenv("DAMAGE_SERVICE_URL", "http://127.0.0.1:8001")
    damage_input_mode = os.getenv("DAMAGE_INPUT_MODE", "file").strip().lower()
    damage_url_field = os.getenv("DAMAGE_URL_FIELD", "image_url")
    predict_with_damage = _env_bool("PREDICT_WITH_DAMAGE", True)

    print("[startup] Stage 1 runtime wiring")
    print(f"[startup] DAMAGE_SERVICE_URL={damage_service_url}")
    print(f"[startup] DAMAGE_INPUT_MODE={damage_input_mode}")
    print(f"[startup] DAMAGE_URL_FIELD={damage_url_field}")
    print(f"[startup] PREDICT_WITH_DAMAGE={predict_with_damage}")

    # Serve generated crops so a second model can consume them by URL.
    pipeline_crop_dir = detector.settings.output_dir / "pipeline_crops"
    pipeline_crop_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/pipeline-crops", StaticFiles(directory=str(pipeline_crop_dir)), name="pipeline-crops")

    async def classify_detections(
        request: Request,
        image_bgr: np.ndarray,
        detection_result: dict[str, Any],
    ) -> list[dict[str, Any]]:
        annotated_detections = []
        for detection in detection_result["detections"]:
            x1, y1, x2, y2 = detection["bbox"]
            crop = image_bgr[y1:y2, x1:x2]

            if crop.size == 0:
                damage_info = {
                    "damage_class": "unknown",
                    "confidence": 0.0,
                    "probabilities": {},
                }
            else:
                try:
                    crop_name = f"{detection_result['image_id']}_{len(annotated_detections)}.jpg"
                    if damage_input_mode == "url":
                        crop_path = pipeline_crop_dir / crop_name
                        cv2.imwrite(str(crop_path), crop)

                        base_url = str(request.base_url).rstrip("/")
                        crop_url = f"{base_url}/pipeline-crops/{crop_name}"
                        print(f"[pipeline] calling stage2 /predict for crop={crop_name} mode=url")
                        damage_response = await asyncio.to_thread(
                            requests.post,
                            f"{damage_service_url}/predict",
                            json={damage_url_field: crop_url},
                            timeout=30,
                        )
                    else:
                        _, crop_bytes = cv2.imencode(".jpg", crop)
                        print(f"[pipeline] calling stage2 /predict for crop={crop_name} mode=file")
                        damage_response = await asyncio.to_thread(
                            requests.post,
                            f"{damage_service_url}/predict",
                            files={"file": ("crop.jpg", crop_bytes.tobytes(), "image/jpeg")},
                            timeout=30,
                        )

                    if damage_response.status_code == 200:
                        damage_info = damage_response.json()
                    else:
                        damage_info = {
                            "damage_class": "error",
                            "confidence": 0.0,
                            "error": f"Service returned {damage_response.status_code}",
                        }
                except requests.exceptions.RequestException as exc:
                    damage_info = {
                        "damage_class": "error",
                        "confidence": 0.0,
                        "error": f"Failed to reach damage service: {str(exc)}",
                    }

            annotated_detections.append(
                {
                    "part_label": detection["part_label"],
                    "bbox": detection["bbox"],
                    "detection_confidence": detection["confidence"],
                    "confidence": detection["confidence"],
                    "damage": damage_info,
                }
            )

        return annotated_detections

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "model_path": detector.settings.model_path,
        }

    @app.post("/predict")
    async def predict(request: Request) -> dict:
        content_type = request.headers.get("content-type", "")
        image_bytes: bytes | None = None
        image_id: str | None = None
        image_path: str | None = None
        image_url: str | None = None

        if "multipart/form-data" in content_type:
            form = await request.form()
            upload = form.get("file")
            image_path = form.get("image_path")
            image_url = form.get("image_url")

            if upload is not None and hasattr(upload, "read"):
                image_bytes = await upload.read()
                filename = getattr(upload, "filename", None)
                if filename:
                    image_id = Path(filename).stem
        else:
            try:
                body = await request.json()
            except Exception:
                body = {}
            image_path = body.get("image_path") if isinstance(body, dict) else None
            image_url = body.get("image_url") if isinstance(body, dict) else None

        if image_bytes:
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None:
                raise HTTPException(status_code=400, detail="Invalid uploaded image")
            detection_result = detector.predict_from_image(image, image_id=image_id)
            if predict_with_damage:
                detection_result["detections"] = await classify_detections(request, image, detection_result)
            return detection_result

        if image_path:
            try:
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Unable to load image from path: {image_path}")
                detection_result = detector.predict_from_image(image, image_id=Path(image_path).stem)
                if predict_with_damage:
                    detection_result["detections"] = await classify_detections(request, image, detection_result)
                return detection_result
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        if image_url:
            try:
                image, image_id_from_url = load_image_from_url(image_url)
                detection_result = detector.predict_from_image(image, image_id=image_id or image_id_from_url)
                if predict_with_damage:
                    detection_result["detections"] = await classify_detections(request, image, detection_result)
                return detection_result
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        raise HTTPException(
            status_code=400,
            detail="Provide multipart 'file' upload, or JSON/form 'image_path' or 'image_url'",
        )

    @app.post("/pipeline")
    async def pipeline(request: Request) -> dict:
        """Full pipeline: detect parts → classify damage per part via external service."""
        content_type = request.headers.get("content-type", "")
        image_bytes: bytes | None = None
        image_id: str | None = None
        image_url: str | None = None

        if "multipart/form-data" in content_type:
            form = await request.form()
            upload = form.get("file")
            image_url = form.get("image_url")
            if upload is not None and hasattr(upload, "read"):
                image_bytes = await upload.read()
                filename = getattr(upload, "filename", None)
                if filename:
                    image_id = Path(filename).stem
        else:
            try:
                body = await request.json()
            except Exception:
                body = {}
            image_url = body.get("image_url") if isinstance(body, dict) else None

        # Step 1: Resolve image from upload or URL
        if image_bytes:
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise HTTPException(status_code=400, detail="Invalid uploaded image")
        elif image_url:
            try:
                image_bgr, image_id_from_url = load_image_from_url(image_url)
                image_id = image_id or image_id_from_url
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        else:
            raise HTTPException(status_code=400, detail="Provide multipart 'file' upload or 'image_url'")

        detection_result = detector.predict_from_image(image_bgr, image_id=image_id)

        # Step 2: Classify damage for each detection via external service
        annotated_detections = await classify_detections(request, image_bgr, detection_result)

        return {
            "image_id": detection_result["image_id"],
            "image_width": detection_result["image_width"],
            "image_height": detection_result["image_height"],
            "detections": annotated_detections,
        }

    return app


# ═══════════════════════════════════════════════════════════════════════════
# SCRIPT MODES
# ═══════════════════════════════════════════════════════════════════════════

def run_local(settings: Settings, args) -> None:
    """Run detection locally without API."""
    output_dir = Path(args.output_dir)
    matplotlib.use("TkAgg")

    print(f"[1] Loading model: {settings.model_path}")
    detector = DetectorService(settings)
    print("    Model loaded")

    print(f"[2] Running inference on: {args.image_path}")
    result_payload = detector.predict_from_path(args.image_path)
    detections = result_payload["detections"]
    print(f"    Total detections: {len(detections)}")

    if not detections:
        print("No detections. Try lowering PARTS_CONF_THRESHOLD.")
    else:
        print("\nDetected parts:")
        for i, det in enumerate(detections):
            print(f"  [{i}] {det['part_label']} conf={det['confidence']:.3f}")

    # Load image for visualization
    image_bgr = cv2.imread(args.image_path)
    if image_bgr is None:
        print(f"Error: Could not load image {args.image_path}")
        return

    print("\n[3] Displaying full image with detections...")
    try:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(14, 10))
        plt.imshow(image_rgb)
        plt.axis("off")
        plt.title(f"Detected Car Parts - {len(detections)} detections")
        plt.tight_layout()
        plt.savefig("output_full.jpg", dpi=150, bbox_inches="tight")
        print("    Saved: output_full.jpg")
        plt.show()
    except Exception as exc:
        print(f"    [warn] display error: {exc}")

    print("\n[4] Cropping parts...")
    output_dir.mkdir(exist_ok=True)
    orig_img = image_bgr
    saved = 0

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        crop = orig_img[y1:y2, x1:x2]

        if crop.size == 0:
            print(f"    [{i}] skipped empty crop")
            continue

        crop_path = output_dir / f"{det['part_label']}_{i}_{det['confidence']:.2f}.jpg"
        cv2.imwrite(str(crop_path), crop)
        print(f"    Saved: {crop_path}")
        saved += 1

        try:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(4, 4))
            plt.title(f"{det['part_label']} ({det['confidence']:.2f})")
            plt.imshow(crop_rgb)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(str(output_dir / f"{det['part_label']}_{i}.png"))
            plt.show()
            plt.close()
        except Exception as exc:
            print(f"    [warn] plot error: {exc}")

    print(f"\nDone - {saved}/{len(detections)} parts saved to '{output_dir}/'")
    print("Check output_full.jpg for the full annotated image")


def run_api(args) -> None:
    """Call API endpoint and display results."""
    api_url = args.api_url.rstrip("/")
    endpoint = args.endpoint

    use_image_url = bool(args.image_url)
    if use_image_url:
        print(f"[1] Calling {api_url}/{endpoint}")
        print(f"    Image URL: {args.image_url}")
    else:
        if not Path(args.image_path).exists():
            print(f"Error: Image not found: {args.image_path}")
            return
        print(f"[1] Calling {api_url}/{endpoint}")
        print(f"    Image: {args.image_path}")

    try:
        if use_image_url:
            response = requests.post(
                f"{api_url}/{endpoint}",
                json={"image_url": args.image_url},
                timeout=60,
            )
        else:
            with open(args.image_path, "rb") as f:
                files = {"file": (Path(args.image_path).name, f, "application/octet-stream")}
                response = requests.post(f"{api_url}/{endpoint}", files=files, timeout=60)

        if response.status_code != 200:
            print(f"Error: API returned {response.status_code}")
            print(response.text)
            return

        result = response.json()
        print("    ✓ Success")

    except requests.exceptions.RequestException as exc:
        print(f"Error: Failed to reach API: {exc}")
        return

    # Display results
    print(f"\n[2] Results:")
    print(f"    Image ID: {result.get('image_id')}")
    print(f"    Dimensions: {result.get('image_width')} x {result.get('image_height')}")

    detections = result.get("detections", [])
    print(f"    Total detections: {len(detections)}")

    if not detections:
        print("    No detections found.")
        return

    print("\n[3] Detected parts:")
    for i, det in enumerate(detections):
        part = det.get("part_label", "?")
        conf = det.get("detection_confidence", det.get("confidence", 0))
        print(f"  [{i}] {part} (conf={conf:.3f})")

        if endpoint == "pipeline":
            damage = det.get("damage", {})
            damage_class = damage.get("damage_class", damage.get("prediction", "?"))
            damage_conf = damage.get("confidence", 0)
            print(f"       → Damage: {damage_class} (conf={damage_conf}%)")

    print(f"\n[4] Full JSON response:")
    print(json.dumps(result, indent=2))


def run_server(args) -> None:
    """Start the FastAPI server."""
    import uvicorn

    settings = load_settings()
    detector = DetectorService(settings)
    app = create_app(detector)

    host = args.host
    port = args.port

    print(f"Starting API server on {host}:{port}")
    print(f"Model: {settings.model_path}")
    print(f"Damage classifier: {os.getenv('DAMAGE_SERVICE_URL', 'http://127.0.0.1:8001')}")
    print("\nEndpoints:")
    print(f"  GET  http://{host}:{port}/health")
    print(f"  POST http://{host}:{port}/predict")
    print(f"  POST http://{host}:{port}/pipeline")

    uvicorn.run(app, host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Car Parts Detector - local testing, API testing, or server mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Local detection with visuals:
    uv run ex2.py --image-path car.jpg
  
  Test API /predict endpoint:
    uv run ex2.py --api --endpoint predict --image-path car.jpg
  
  Test full /pipeline (parts + damage):
    uv run ex2.py --api --endpoint pipeline --image-path car.jpg
  
  Start API server:
    uv run ex2.py --server
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--server", action="store_true",
                           help="Start FastAPI server (don't use --image-path)")
    mode_group.add_argument("--api", action="store_true",
                           help="Call API endpoint instead of running locally")

    # Common options
    parser.add_argument("--image-path", default="i8.jpg", help="Path to input image")
    parser.add_argument("--image-url", default="", help="Image URL (for --api mode)")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000",
                       help="API server URL (for --api mode)")
    parser.add_argument("--endpoint", choices=["predict", "pipeline"], default="pipeline",
                       help="API endpoint to call (requires --api)")

    # Local inference options
    parser.add_argument("--output-dir", default="cropped_parts",
                       help="Crop output folder (local mode only)")

    # Server options
    parser.add_argument("--host", default="127.0.0.1",
                       help="Server host (for --server mode)")
    parser.add_argument("--port", type=int, default=8000,
                       help="Server port (for --server mode)")

    args = parser.parse_args()

    if args.server:
        run_server(args)
    elif args.api:
        run_api(args)
    else:
        settings = load_settings()
        run_local(settings, args)


if __name__ == "__main__":
    main()
