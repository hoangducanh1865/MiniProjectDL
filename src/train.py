import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
import joblib
import logging

from src.dataset import ICUTabularDataset
from src.model import ICUMortalityMLP
from src.utils import set_seed

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    import platform
    is_apple_silicon = (platform.system() == "Darwin" and
                        platform.machine() == "arm64")
    if is_apple_silicon:
        logger.info("Apple Silicon detected → using CPU (MPS disabled for stability)")
        return torch.device("cpu")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        logger.info(f"CUDA detected → using GPU: {name}")
        return torch.device("cuda")
    logger.warning("No GPU detected → using CPU. "
                   "On Colab: Runtime → Change runtime type → GPU")
    return torch.device("cpu")


# ────────────────────────────────────────────────
# Preprocessing
# ────────────────────────────────────────────────

def build_preprocessor(X_train: np.ndarray):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imp = imputer.fit_transform(X_train)
    X_scaled = scaler.fit_transform(X_imp)
    return imputer, scaler, X_scaled


def apply_preprocessor(X, imputer, scaler):
    return scaler.transform(imputer.transform(X))


# ────────────────────────────────────────────────
# Single-fold MLP training
# ────────────────────────────────────────────────

def train_one_fold(
    X_tr, y_tr, X_val, y_val,
    input_dim, device,
    # architecture
    hidden_dims=(512, 256, 128),
    dropout=0.3,
    # optimisation
    epochs=80,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-4,
    # scheduler
    scheduler_type="cosine",   # "cosine" | "step" | "none"
    step_size=20,
    gamma=0.5,
    # label smoothing
    label_smoothing=0.05,
    seed=42,
):
    set_seed(seed)

    class_counts = np.bincount(y_tr.astype(int))
    weights = 1.0 / class_counts[y_tr.astype(int)]
    sampler = WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.float64),
        num_samples=len(weights),
        replacement=True,
    )

    train_ds = ICUTabularDataset(X_tr, y_tr)
    val_ds   = ICUTabularDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size * 4,
                              num_workers=0, pin_memory=False)

    model = ICUMortalityMLP(input_dim, hidden_dims=hidden_dims, dropout=dropout).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        scheduler = None

    pos_weight = torch.tensor(
        [float(class_counts[0]) / max(float(class_counts[1]), 1)],
        dtype=torch.float32,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc   = 0.0
    best_state = None
    history    = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            # Label smoothing for positive class
            yb_smooth = yb * (1 - label_smoothing) + 0.5 * label_smoothing
            optimizer.zero_grad()
            loss = criterion(model(xb), yb_smooth)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        if scheduler is not None:
            scheduler.step()

        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            preds = []
            with torch.no_grad():
                for xb, _ in val_loader:
                    logits = model(xb.to(device))
                    preds.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            auc = roc_auc_score(y_val, preds)
            ap  = average_precision_score(y_val, preds)
            logger.info(f"  Epoch {epoch:3d}  Val AUC={auc:.4f}  AP={ap:.4f}")
            history.append({"epoch": epoch, "auc": auc, "ap": ap})
            if auc > best_auc:
                best_auc   = auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_auc, history


# ────────────────────────────────────────────────
# Full k-fold training pipeline
# ────────────────────────────────────────────────

def run_training(
    X_df, y, output_dir,
    # CV
    k_fold=5,
    seed=42,
    # architecture
    hidden_dims=(512, 256, 128),
    dropout=0.3,
    # optimisation
    epochs=80,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-4,
    # scheduler
    scheduler_type="cosine",
    step_size=20,
    gamma=0.5,
    # label smoothing
    label_smoothing=0.05,
):
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    device  = get_device()
    skf     = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=seed)
    X_np    = X_df.values.astype(np.float32)
    col_names = list(X_df.columns)

    fold_results = []
    oof_probs    = np.zeros(len(y))
    best_fold_auc = 0.0
    best_fold_idx = 1

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_np, y), 1):
        logger.info(f"\n{'='*50}\nFold {fold}/{k_fold}\n{'='*50}")

        X_tr_raw,  X_val_raw  = X_np[tr_idx], X_np[val_idx]
        y_tr,      y_val      = y[tr_idx],    y[val_idx]

        imputer, scaler, X_tr_scaled = build_preprocessor(X_tr_raw)
        X_val_scaled = apply_preprocessor(X_val_raw, imputer, scaler)

        input_dim = X_tr_scaled.shape[1]

        model, fold_auc, history = train_one_fold(
            X_tr_scaled, y_tr, X_val_scaled, y_val,
            input_dim, device,
            hidden_dims=hidden_dims,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            scheduler_type=scheduler_type,
            step_size=step_size,
            gamma=gamma,
            label_smoothing=label_smoothing,
            seed=seed + fold,
        )

        # OOF predictions
        with torch.no_grad():
            t = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
            oof_probs[val_idx] = torch.sigmoid(model(t)).cpu().numpy()

        # Save fold artifacts
        torch.save(
            {"model_state": model.state_dict(),
             "input_dim": input_dim,
             "hidden_dims": list(hidden_dims),
             "dropout": dropout,
             "fold_auc": fold_auc},
            os.path.join(output_dir, f"model_fold{fold}.pt"),
        )
        joblib.dump(
            {"imputer": imputer, "scaler": scaler},
            os.path.join(output_dir, f"preprocessor_fold{fold}.pkl"),
        )
        logger.info(f"Fold {fold} saved → model_fold{fold}.pt  (AUC={fold_auc:.4f})")

        fold_results.append({
            "fold": fold,
            "auc": fold_auc,
            "val_idx": val_idx,
            "history": history,
        })

        if fold_auc > best_fold_auc:
            best_fold_auc = fold_auc
            best_fold_idx = fold

    # Save shared metadata
    joblib.dump(col_names, os.path.join(output_dir, "col_names.pkl"))
    joblib.dump({"best_fold": best_fold_idx, "best_auc": best_fold_auc},
                os.path.join(output_dir, "best_fold.pkl"))

    # OOF evaluation
    oof_auc = roc_auc_score(y, oof_probs)
    oof_ap  = average_precision_score(y, oof_probs)

    logger.info(f"\nBest fold: Fold {best_fold_idx}  AUC={best_fold_auc:.4f}")
    logger.info(f"OOF AUC={oof_auc:.4f}  AUC-PR={oof_ap:.4f}")
    for r in fold_results:
        logger.info(f"  Fold {r['fold']}: AUC={r['auc']:.4f}")

    # Persist OOF data for evaluation plots
    joblib.dump(
        {"oof_probs": oof_probs,
         "y_true": y,
         "fold_results": fold_results,
         "best_fold": best_fold_idx},
        os.path.join(output_dir, "oof_results.pkl"),
    )

    return fold_results, oof_auc, oof_ap
