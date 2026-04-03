import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2
from pathlib import Path
import numpy as np

# ── CONFIG ───────────────────────────────────────────────────
MODEL_PATH  = "epoch3.pt"          # or "parts_seg_m_90ep_best.pt"
IMAGE_PATH  = "i8.jpg"
CONF        = 0.3
IOU         = 0.5
OUTPUT_DIR  = Path("cropped_parts")

# ── LOAD MODEL ───────────────────────────────────────────────
print(f"[1] Loading model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print("    Model loaded ✅")

# ── PREDICT ──────────────────────────────────────────────────
print(f"[2] Running inference on: {IMAGE_PATH}")
results = model.predict(
    source=IMAGE_PATH,
    conf=CONF,
    iou=IOU,
    save=False,
    verbose=True
)
result = results[0]
print(f"    Total detections: {len(result.boxes)}")

if len(result.boxes) == 0:
    print("⚠️  No detections. Try lowering CONF further (e.g. 0.05)")
else:
    # Print all detections
    print("\n    Detected parts:")
    for i, box in enumerate(result.boxes):
        name = result.names[int(box.cls)]
        conf = float(box.conf)
        print(f"      [{i}] {name} — conf={conf:.3f}")

# ── SHOW FULL IMAGE ──────────────────────────────────────────
print("\n[3] Displaying full image with detections...")
try:
    img = result.plot()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(14, 10))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.title(f"Detected Car Parts — {len(result.boxes)} detections")
    plt.tight_layout()
    plt.savefig("output_full.jpg", dpi=150, bbox_inches="tight")
    print("    Saved: output_full.jpg ✅")
    plt.show()
except Exception as e:
    print(f"    [warn] display error: {e}")

# ── CROP & SAVE EACH PART ────────────────────────────────────
print("\n[4] Cropping parts...")
OUTPUT_DIR.mkdir(exist_ok=True)
orig_img = np.array(result.orig_img)
saved = 0

for i, box in enumerate(result.boxes):
    part_name  = result.names[int(box.cls)]
    confidence = float(box.conf)

    x1, y1, x2, y2 = map(int, box.xyxy[0])
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(orig_img.shape[1], x2)
    y2 = min(orig_img.shape[0], y2)
    crop = orig_img[y1:y2, x1:x2]

    if crop.size == 0:
        print(f"    [{i}] skipped — empty crop")
        continue

    crop_path = OUTPUT_DIR / f"{part_name}_{i}_{confidence:.2f}.jpg"
    cv2.imwrite(str(crop_path), crop)
    print(f"    Saved: {crop_path}")
    saved += 1

    try:
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(4, 4))
        plt.title(f"{part_name} ({confidence:.2f})")
        plt.imshow(crop_rgb)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(str(OUTPUT_DIR / f"{part_name}_{i}.png"))
        plt.show()
        plt.close()
    except Exception as e:
        print(f"    [warn] plot error: {e}")

print(f"\n✅ Done — {saved}/{len(result.boxes)} parts saved to '{OUTPUT_DIR}/'")
print(f"📁 Check 'output_full.jpg' for the full annotated image")