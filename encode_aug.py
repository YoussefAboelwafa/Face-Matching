import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import cv2
from deepface import DeepFace
from tqdm import tqdm
from utils import save_embeddings
from config import TRAIN_DATASET_PATH, TRAIN_EMBEDDINGS_AUG_PATH


SEED = 42
AUG_PROB = 0.6

AUGMENTATIONS = {
    "horizontal_flip": lambda img: cv2.flip(img, 1),
    "brightness_up": lambda img: cv2.convertScaleAbs(img, alpha=1.3, beta=0),
    "brightness_down": lambda img: cv2.convertScaleAbs(img, alpha=0.7, beta=0),
    "blur": lambda img: cv2.GaussianBlur(img, (5, 5), 0),
    "noise": lambda img: add_gaussian_noise(img),
}


def add_gaussian_noise(image, mean=0, std=25):
    noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
    noisy = cv2.add(image, noise)
    return noisy


def embed_dataset_with_augmentations(dataset_path, embeddings_path, seed=SEED, aug_prob=AUG_PROB):
    np.random.seed(seed)
    data = []
    
    all_images = []
    class_counts = {}
    for age in range(0, 91):
        class_path = os.path.join(dataset_path, str(age))
        if os.path.exists(class_path):
            img_names = os.listdir(class_path)
            class_counts[str(age)] = len(img_names)
            for img_name in img_names:
                img_path = os.path.join(class_path, img_name)
                all_images.append((img_path, str(age)))

    median_count = np.median(list(class_counts.values()))
    minority_classes = {age for age, count in class_counts.items() if count < median_count}
    
    print(f"Median class count: {median_count}")
    print(f"Minority classes (below median): {len(minority_classes)}")
    print(f"Majority classes (at/above median): {len(class_counts) - len(minority_classes)}")

    pbar = tqdm(all_images, desc="Processing images")
    for img_path, age in pbar:
        pbar.set_description(f"Age {age}")
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue

            try:
                emb = DeepFace.represent(
                    img,
                    model_name="Facenet512",
                    detector_backend="retinaface",
                    enforce_detection=False,
                    align=True,
                )
                data.append((emb[0]["embedding"], age))
            except Exception as e:
                print(f"Could not embed original {img_path}: {e}")

            if age in minority_classes:
                for aug_name, aug_fn in AUGMENTATIONS.items():
                    if np.random.random() < aug_prob:
                        try:
                            augmented_img = aug_fn(img)
                            emb = DeepFace.represent(
                                augmented_img,
                                model_name="Facenet512",
                                detector_backend="retinaface",
                                enforce_detection=False,
                                align=True,
                            )
                            data.append((emb[0]["embedding"], age))
                        except Exception as e:
                            print(f"Could not embed {img_path} with {aug_name}: {e}")

        except Exception as e:
            print(f"Could not process {img_path}: {e}")

    save_embeddings(embeddings_path, data)
    print(f"Total embeddings saved: {len(data)}")


if __name__ == "__main__":
    print("Embedding the training dataset with augmentations...")
    embed_dataset_with_augmentations(TRAIN_DATASET_PATH, TRAIN_EMBEDDINGS_AUG_PATH)
    print("Done embedding the training dataset with augmentations.")
