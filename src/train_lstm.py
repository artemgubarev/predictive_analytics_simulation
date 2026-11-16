from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.lstm_utils import create_sequences

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss для бинарной классификации с дисбалансом.
    gamma > 0 уменьшает вес легких примеров, фокусируясь на сложных.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss) 
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss
        
        if self.pos_weight is not None:
            focal_loss = focal_loss * (1 + (self.pos_weight - 1) * targets)
            
        return focal_loss.mean()


class SimpleGRU(nn.Module):
    """Двунаправленная GRU (BiGRU) с механизмом Attention."""
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        self.attention_weights = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        attn_scores = self.attention_weights(out)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context_vector = torch.sum(out * attn_weights, dim=1)
        context_vector = self.dropout(context_vector)
        logits = self.fc(context_vector)
        return logits.squeeze(1)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    epochs: int,
) -> float:
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
    
    for X_batch, y_batch in pbar:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    return total_loss / len(dataloader)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    model.eval()
    all_preds, all_targets = [], []
    total_loss = 0.0
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            
            if criterion is not None:
                loss = criterion(logits, y_batch)
                total_loss += loss.item()
                
            probs = torch.sigmoid(logits)
            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
            
    mean_loss = total_loss / len(dataloader) if criterion is not None else None
    return np.array(all_preds).flatten(), np.array(all_targets), mean_loss


def train_lstm(df_features: pd.DataFrame, config: dict) -> tuple[Any, float, Any, dict]:
    logger.info("=== Train (BiGRU with Focal Loss) started ===")
    cfg = config["train"]
    lstm_cfg = cfg.get("lstm", {})
    
    drop_cols = ["target", "machineID", "datetime"]
    feature_cols = [c for c in df_features.columns if c not in drop_cols]
    
    df = df_features.sort_values(["machineID", "datetime"]).reset_index(drop=True)
    
    seq_length = lstm_cfg.get("seq_length", 48)
    batch_size = lstm_cfg.get("batch_size", 256)
    epochs = lstm_cfg.get("epochs", 20)
    lr = lstm_cfg.get("learning_rate", 1e-3)
    patience = lstm_cfg.get("patience", 3)
    max_pos_weight = lstm_cfg.get("max_pos_weight", 3.0)
    
    logger.info("Создание последовательностей (длина=%d)...", seq_length)
    X_seq, y_seq, groups_seq = create_sequences(df, feature_cols, "target", seq_length)
    logger.info("Форма тензора: %s | Positive rate: %.4f", X_seq.shape, y_seq.mean())
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Устройство: %s", device)
    
    gkf = GroupKFold(n_splits=cfg["n_splits"])
    oof_preds = np.zeros(len(y_seq))
    pr_auc_scores = []
    
    with mlflow.start_run(run_name="bigru_study", nested=True):
        mlflow.log_params({
            "model": "BiGRU_Attention_FocalLoss",
            "seq_length": seq_length, 
            "batch_size": batch_size, 
            "epochs": epochs,
            "patience": patience,
            "hidden_size": lstm_cfg.get("hidden_size", 64),
            "num_layers": lstm_cfg.get("num_layers", 1),
            "max_pos_weight": max_pos_weight
        })
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_seq, y_seq, groups_seq)):
            logger.info("Fold %d/%d", fold+1, cfg["n_splits"])
            
            X_train, X_val = X_seq[train_idx], X_seq[val_idx]
            y_train, y_val = y_seq[train_idx], y_seq[val_idx]
            
            scaler = StandardScaler()
            X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
            X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
            X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
            X_val_scaled = scaler.transform(X_val_reshaped).reshape(X_val.shape)
            
            train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train))
            val_dataset = TensorDataset(torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val))
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            model = SimpleGRU(
                input_size=X_seq.shape[2], 
                hidden_size=lstm_cfg.get("hidden_size", 64),
                num_layers=lstm_cfg.get("num_layers", 1),
                dropout=lstm_cfg.get("dropout", 0.3)
            ).to(device)
            
            # --- FOCAL LOSS ---
            pos_weight = torch.tensor([min(len(y_train) / (y_train.sum() + 1e-6), max_pos_weight)]).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            best_val_loss = float('inf')
            patience_counter = 0
            best_model_state = None
            
            for epoch in range(epochs):
                train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, epochs)
                _, _, val_loss = evaluate_model(model, val_loader, device, criterion)
                
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    logger.info("  Epoch %d/%d | Train Loss: %.4f | Val Loss: %.4f", 
                                epoch+1, epochs, train_loss, val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info("  Early stopping на эпохе %d (лучший Val Loss: %.4f)", epoch+1, best_val_loss)
                        break
            
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                model.to(device)
            
            preds, targets, _ = evaluate_model(model, val_loader, device)
            oof_preds[val_idx] = preds
            pr_auc = average_precision_score(y_val, preds)
            pr_auc_scores.append(pr_auc)
            logger.info("Fold %d PR-AUC: %.4f", fold+1, pr_auc)
            
        mean_pr_auc = float(np.mean(pr_auc_scores))
        oof_pr_auc = float(average_precision_score(y_seq, oof_preds))
        
        mlflow.log_metric("cv_pr_auc_mean", mean_pr_auc)
        mlflow.log_metric("oof_pr_auc", oof_pr_auc)
        logger.info("BiGRU OOF PR-AUC: %.4f", oof_pr_auc)
        
        best_thr, best_f1 = 0.5, 0.0
        for thr in np.arange(0.1, 0.9, 0.01):
            f1 = f1_score(y_seq, (oof_preds > thr).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr
                
        mlflow.log_metric("final_threshold", best_thr)
        mlflow.log_metric("final_oof_f1", best_f1)
        logger.info("Лучший порог: %.2f (F1=%.4f)", best_thr, best_f1)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "bigru_weights.pth"
            torch.save(model.state_dict(), model_path)
            mlflow.log_artifact(str(model_path), artifact_path="model")
            mlflow.log_params({"input_size": X_seq.shape[2]})
            
    return model, best_thr, scaler, {"model_type": "bigru_focal", "oof_pr_auc": oof_pr_auc, "threshold": best_thr}