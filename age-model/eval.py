import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import argparse
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from utils import load_embeddings
from config import VAL_EMBEDDINGS_PATH, CHECKPOINTS_PATH
from model import AgeModel
from tqdm import tqdm


def load_model(device, checkpoint_name):
    model = AgeModel(embedding_dim=512, num_classes=91).to(device)
    checkpoint_path = os.path.join(CHECKPOINTS_PATH, checkpoint_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    return model


def evaluate_model(model, val_data, device):
    predictions = []
    true_ages = []

    with torch.no_grad():
        for embedding, age in val_data:
            embedding = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)
            class_output, reg_output = model(embedding)
            predicted_age = class_output.argmax(dim=1).item()
            predictions.append(predicted_age)
            true_ages.append(int(age))

    return np.array(predictions), np.array(true_ages)


def calculate_metrics(predictions, true_ages):
    errors = predictions - true_ages
    abs_errors = np.abs(errors)

    metrics = {}

    metrics["MAE"] = np.mean(abs_errors)
    metrics["Error Std"] = np.std(errors)

    metrics["Acc_Exact"] = np.mean(abs_errors == 0) * 100
    metrics["Acc_±3"] = np.mean(abs_errors <= 3) * 100
    metrics["Acc_±5"] = np.mean(abs_errors <= 5) * 100
    metrics["Acc_±10"] = np.mean(abs_errors <= 10) * 100

    child_mask = true_ages < 18
    adult_mask = (true_ages >= 18) & (true_ages < 60)
    elderly_mask = true_ages >= 60

    if child_mask.sum() > 0:
        metrics["MAE_Child"] = np.mean(abs_errors[child_mask])
        metrics["Acc_±5_Child"] = np.mean(abs_errors[child_mask] <= 5) * 100
    else:
        metrics["MAE_Child"] = 0
        metrics["Acc_±5_Child"] = 0

    if adult_mask.sum() > 0:
        metrics["MAE_Adult"] = np.mean(abs_errors[adult_mask])
        metrics["Acc_±5_Adult"] = np.mean(abs_errors[adult_mask] <= 5) * 100
    else:
        metrics["MAE_Adult"] = 0
        metrics["Acc_±5_Adult"] = 0

    if elderly_mask.sum() > 0:
        metrics["MAE_Elderly"] = np.mean(abs_errors[elderly_mask])
        metrics["Acc_±5_Elderly"] = np.mean(abs_errors[elderly_mask] <= 5) * 100
    else:
        metrics["MAE_Elderly"] = 0
        metrics["Acc_±5_Elderly"] = 0

    return metrics


def print_metrics(metrics):
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    
    print("\nOverall Metrics:")
    print(f"  MAE: {metrics['MAE']:.2f} years")
    print(f"  Error Std: {metrics['Error Std']:.2f} years")
    
    print("\nAccuracy Metrics:")
    print(f"  Exact Match: {metrics['Acc_Exact']:.2f}%")
    print(f"  Within ±3 years: {metrics['Acc_±3']:.2f}%")
    print(f"  Within ±5 years: {metrics['Acc_±5']:.2f}%")
    print(f"  Within ±10 years: {metrics['Acc_±10']:.2f}%")
    
    print("\nAge Group Performance:")
    print(f"  Child (<18):  MAE={metrics['MAE_Child']:.2f}, Acc_±5={metrics['Acc_±5_Child']:.2f}%")
    print(f"  Adult (18-60): MAE={metrics['MAE_Adult']:.2f}, Acc_±5={metrics['Acc_±5_Adult']:.2f}%")
    print(f"  Elderly (≥60): MAE={metrics['MAE_Elderly']:.2f}, Acc_±5={metrics['Acc_±5_Elderly']:.2f}%")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate age prediction model")
    parser.add_argument("--checkpoint", type=str, default="age_model_best.pth", help="Relative path to model checkpoint from CHECKPOINTS_PATH")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading validation embeddings...")
    val_data = load_embeddings(VAL_EMBEDDINGS_PATH)
    print(f"Validation samples: {len(val_data)}")

    print("Loading model...")
    model = load_model(device, args.checkpoint)

    print("Evaluating model...")
    predictions, true_ages = evaluate_model(model, val_data, device)

    metrics = calculate_metrics(predictions, true_ages)
    print_metrics(metrics)


if __name__ == "__main__":
    main()
