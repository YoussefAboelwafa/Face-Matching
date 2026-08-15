import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import load_embeddings
from config import TRAIN_EMBEDDINGS_PATH, VAL_EMBEDDINGS_PATH, CHECKPOINTS_PATH
from model import AgeModel, OrdinalLoss
from data import EmbeddingDataset
from comet_ml import Experiment
import optuna


GPUS = [0, 1, 2, 3]


def objective(trial):
    gpu_id = GPUS[trial.number % len(GPUS)]
    device = torch.device(f"cuda:{gpu_id}")
    
    trial_name = f"trial_{trial.number}"
    trial_checkpoint_dir = os.path.join(CHECKPOINTS_PATH, "TUNE", trial_name)
    os.makedirs(trial_checkpoint_dir, exist_ok=True)
    
    experiment = Experiment(
        api_key="rwyMmTQC0QDIH0oF5XaSzgmh4",
        project_name="face-matching",
        workspace="youssefaboelwafa",
    )
    experiment.set_name(trial_name)
    
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    alpha = trial.suggest_float("alpha", 0.1, 0.9)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    epochs = trial.suggest_int("epochs", 50, 150)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    
    experiment.log_parameter("device", str(device))
    experiment.log_parameter("gpu_id", gpu_id)
    experiment.log_parameter("epochs", epochs)
    experiment.log_parameter("batch_size", batch_size)
    experiment.log_parameter("learning_rate", lr)
    experiment.log_parameter("weight_decay", weight_decay)
    experiment.log_parameter("alpha", alpha)
    experiment.log_parameter("dropout", dropout)
    
    print(f"[{trial_name}] Using device: {device}")
    print(f"[{trial_name}] Loading training embeddings...")
    train_data = load_embeddings(TRAIN_EMBEDDINGS_PATH)
    train_dataset = EmbeddingDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    print(f"[{trial_name}] Loading validation embeddings...")
    val_data = load_embeddings(VAL_EMBEDDINGS_PATH)
    val_dataset = EmbeddingDataset(val_data)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"[{trial_name}] Training samples: {len(train_data)}")
    print(f"[{trial_name}] Validation samples: {len(val_data)}")
    
    experiment.log_parameter("train_samples", len(train_data))
    experiment.log_parameter("val_samples", len(val_data))
    
    model = AgeModel(embedding_dim=512, num_classes=91, dropout=dropout).to(device)
    criterion = OrdinalLoss(num_classes=91, alpha=alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc_3 = 0.0
    patience = 20
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_correct_3 = 0
        train_correct_5 = 0
        train_correct_10 = 0
        train_total = 0
        
        for embeddings, ages in train_loader:
            embeddings, ages = embeddings.to(device), ages.to(device)
            
            optimizer.zero_grad()
            class_outputs, reg_outputs = model(embeddings)
            loss = criterion(class_outputs, reg_outputs, ages)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = class_outputs.max(1)
            train_total += ages.size(0)
            train_correct += predicted.eq(ages).sum().item()
            errors = torch.abs(predicted - ages)
            train_correct_3 += (errors <= 3).sum().item()
            train_correct_5 += (errors <= 5).sum().item()
            train_correct_10 += (errors <= 10).sum().item()
        
        train_acc = 100.0 * train_correct / train_total
        train_acc_3 = 100.0 * train_correct_3 / train_total
        train_acc_5 = 100.0 * train_correct_5 / train_total
        train_acc_10 = 100.0 * train_correct_10 / train_total
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_correct_3 = 0
        val_correct_5 = 0
        val_correct_10 = 0
        val_total = 0
        
        with torch.no_grad():
            for embeddings, ages in val_loader:
                embeddings, ages = embeddings.to(device), ages.to(device)
                class_outputs, reg_outputs = model(embeddings)
                loss = criterion(class_outputs, reg_outputs, ages)
                
                val_loss += loss.item()
                _, predicted = class_outputs.max(1)
                val_total += ages.size(0)
                val_correct += predicted.eq(ages).sum().item()
                errors = torch.abs(predicted - ages)
                val_correct_3 += (errors <= 3).sum().item()
                val_correct_5 += (errors <= 5).sum().item()
                val_correct_10 += (errors <= 10).sum().item()
        
        val_acc = 100.0 * val_correct / val_total
        val_acc_3 = 100.0 * val_correct_3 / val_total
        val_acc_5 = 100.0 * val_correct_5 / val_total
        val_acc_10 = 100.0 * val_correct_10 / val_total
        val_loss_avg = val_loss / len(val_loader)
        
        scheduler.step()
        
        metrics = {
            "train_loss": train_loss / len(train_loader),
            "train_acc": train_acc,
            "train_acc_3": train_acc_3,
            "train_acc_5": train_acc_5,
            "train_acc_10": train_acc_10,
            "val_loss": val_loss_avg,
            "val_acc": val_acc,
            "val_acc_3": val_acc_3,
            "val_acc_5": val_acc_5,
            "val_acc_10": val_acc_10,
        }
        experiment.log_metrics(metrics, step=epoch)
        
        print(f"[{trial_name}] Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.2f}% (±3:{train_acc_3:.1f}%) | "
              f"Val Loss: {val_loss_avg:.4f} | "
              f"Val Acc: {val_acc:.2f}% (±3:{val_acc_3:.1f}%)")
        
        if val_acc_3 > best_val_acc_3:
            best_val_acc_3 = val_acc_3
            patience_counter = 0
            checkpoint_path = os.path.join(trial_checkpoint_dir, "best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_acc_3': val_acc_3,
                'val_acc_5': val_acc_5,
                'val_acc_10': val_acc_10,
            }, checkpoint_path)
            print(f"[{trial_name}] Saved best model with val_acc_3: {val_acc_3:.2f}%")
            experiment.log_asset(checkpoint_path, file_name="best.pth")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"[{trial_name}] Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break
    
    print(f"[{trial_name}] Training complete. Best validation accuracy (±3): {best_val_acc_3:.2f}%")
    experiment.log_parameter("best_val_acc_3", best_val_acc_3)
    experiment.end()
    
    return best_val_acc_3


if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        study_name="age_model_hyperparameter_tuning",
        storage="sqlite:///tuning.db",
        load_if_exists=True,
    )
    
    n_trials = 100
    print(f"Starting hyperparameter tuning with {n_trials} trials on GPUs {GPUS}")
    
    study.optimize(objective, n_trials=n_trials, n_jobs=len(GPUS))
    
    print("\n" + "="*100)
    print("BEST TRIAL")
    print("="*100)
    print(f"Trial number: {study.best_trial.number}")
    print(f"Best val_acc_3: {study.best_value:.2f}%")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("="*100)
