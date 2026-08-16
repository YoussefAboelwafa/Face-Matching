import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import torch
import numpy as np
from deepface import DeepFace
from age_model.model import AgeModel
from config import CHECKPOINTS_PATH


def cosine_distance(emb1, emb2):
    emb1 = np.asarray(emb1)
    emb2 = np.asarray(emb2)
    dot_product = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    similarity = dot_product / (norm1 * norm2)
    return 1 - similarity


def predict_age(embedding, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)
    class_output, _ = model(emb_tensor)
    predicted_age = class_output.argmax(dim=1).item()
    return predicted_age

def load_model(checkpoint_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AgeModel(embedding_dim=512, num_classes=91).to(device)
    checkpoint_path = os.path.join(CHECKPOINTS_PATH, checkpoint_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    return model

def extract_embedding(image_path):
    result = DeepFace.represent(
        image_path,
        model_name="Facenet512",
        detector_backend="retinaface",
        enforce_detection=False,
        align=True,
    )
    embedding = result[0]["embedding"]
    return embedding

def verify(img1_path, img2_path, model, threshold=0.5):
    emb1 = extract_embedding(img1_path)
    emb2 = extract_embedding(img2_path)

    distance = cosine_distance(emb1, emb2)
    verified = distance <= threshold

    age1 = predict_age(emb1, model)
    age2 = predict_age(emb2, model)

    return {
        "verified": verified,
        "distance": distance,
        "threshold": threshold,
        "age1": age1,
        "age2": age2
    }

def main():
    parser = argparse.ArgumentParser(description="Compare two faces for similarity and estimate ages")
    parser.add_argument("--img1", type=str, required=True, help="Path to first image")
    parser.add_argument("--img2", type=str, required=True, help="Path to second image")
    parser.add_argument("--checkpoint", type=str, default="age_model_best.pth", help="Relative path to model checkpoint from CHECKPOINTS_PATH")
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    result = verify(args.img1, args.img2, model)
    
    print("Verification Result:")
    print(f"Verified: {result['verified']}")
    print(f"Distance: {result['distance']:.4f}")
    print(f"Threshold: {result['threshold']}")
    print(f"Estimated Age of Image 1: {result['age1']}")
    print(f"Estimated Age of Image 2: {result['age2']}")


if __name__ == "__main__":
    main()
