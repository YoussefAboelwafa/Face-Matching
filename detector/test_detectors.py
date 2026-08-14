import os
import csv
import sys
from pathlib import Path
from tqdm import tqdm
import torch
import tensorflow as tf
os.environ["TF_USE_LEGACY_KERAS"] = "1"
sys.path.insert(0, str(Path.cwd().parent))

from deepface import DeepFace
from config import TRAIN_DATASET_PATH, VAL_DATASET_PATH

TRAIN_PATH = Path(TRAIN_DATASET_PATH)
VAL_PATH = Path(VAL_DATASET_PATH)

detectors = [
    "retinaface",
    "mtcnn",
    "yolov8n",
    "yolov11n",
    "opencv",
    "dlib",
    "yunet",
    "centerface",
]

results = {}

for detector in detectors:
    print(f"\n{'='*60}")
    print(f"Testing detector: {detector}")

    if torch.cuda.is_available():
        print(f"Device: GPU ({torch.cuda.get_device_name(0)})")
    elif len(tf.config.list_physical_devices("GPU")) > 0:
        print(f"Device: GPU (TensorFlow)")
    else:
        print(f"Device: CPU")

    print(f"{'='*60}")

    detected_count = 0
    failed_count = 0
    total_count = 0
    error_samples = []

    all_images = []
    for split in ["train", "val"]:
        split_path = TRAIN_PATH if split == "train" else VAL_PATH

        for age in range(91):
            age_folder = split_path / str(age)
            if not age_folder.exists():
                continue

            all_images.extend(list(age_folder.iterdir()))

    for img_path in tqdm(all_images, desc=detector):
        total_count += 1
        try:
            faces = DeepFace.extract_faces(
                img_path=str(img_path),
                detector_backend=detector,
                enforce_detection=False,
                align=True,
            )
            if len(faces) > 0:
                detected_count += 1
            else:
                failed_count += 1
        except Exception as e:
            failed_count += 1
            if len(error_samples) < 3:
                error_samples.append(f"{img_path}: {e}")

    accuracy = detected_count / total_count if total_count > 0 else 0
    results[detector] = {
        "detected": detected_count,
        "failed": failed_count,
        "total": total_count,
        "accuracy": accuracy,
    }

    print(f"\nDetector: {detector}")
    print(f"  Detected: {detected_count}/{total_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    if error_samples:
        print(f"\n  Error samples (first {len(error_samples)}):")
        for sample in error_samples:
            print(f"    {sample}")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(
    f"{'Detector':<15} {'Detected':<10} {'Failed':<10} {'Total':<10} {'Accuracy':<10}"
)
print(f"{'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
for detector, stats in results.items():
    print(
        f"{detector:<15} {stats['detected']:<10} {stats['failed']:<10} {stats['total']:<10} {stats['accuracy']:.4f}"
    )

sorted_results = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)

with open("detectors_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Detector", "Accuracy"])
    for detector, stats in sorted_results:
        writer.writerow([detector, f"{stats['accuracy']:.4f}"])

print(f"\nResults saved to detectors_results.csv")
