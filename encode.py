import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from deepface import DeepFace
from tqdm import tqdm
from utils import save_embeddings
from config import *


def embed_dataset(dataset_path, embeddings_path):

    data = []
    folders = [str(i) for i in range(0, 91)]

    for folder in tqdm(folders, desc="Processing folders"):
        class_path = os.path.join(dataset_path, folder)

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            try:
                emb = DeepFace.represent(
                    img_path,
                    model_name="Facenet512",
                    detector_backend="retinaface",
                    enforce_detection=False,
                    align=True,
                )
                data.append((emb[0]["embedding"], str(folder)))
                print(f"Processed {img_path}")
            except Exception as e:
                print(f"Could not process {img_path}: {e}")

    save_embeddings(embeddings_path, data)


if __name__ == "__main__":
    
    print("Embedding the training dataset...")
    embed_dataset(TRAIN_DATASET_PATH, TRAIN_EMBEDDINGS_PATH)
    print("Done embedding the training dataset.")
    
    # print("Embedding the val dataset...")
    # embed_dataset(VAL_DATASET_PATH, VAL_EMBEDDINGS_PATH)
    # print("Done embedding the val dataset.")
