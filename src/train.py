import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
from catboost import CatBoostClassifier
import joblib
import logging

from src.dataset import ICUTabularDataset
from src.model import ICUMortalityMLP
from src.utils import set_seed

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """
    Safe device selection:
      - Apple Silicon (M1/M2): force CPU — MPS segfaults with certain ops
      - CUDA available: use GPU
      - Otherwise: CPU
    """
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
# Preprocessing pipeline
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
# MLP training
# ────────────────────────────────────────────────

def train_mlp(X_train, y_train, X_val, y_val, input_dim,
              device, epochs=80, batch_size=256, lr=1e-3, seed=42):
    set_seed(seed)

    class_counts = np.bincount(y_train.astype(int))
    weights = 1.0 / class_counts[y_train.astype(int)]
    # num_workers=0 avoids multiprocessing segfaults on macOS
    sampler = WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.float64),
        num_samples=len(weights),
        replacement=True,
    )

    train_ds = ICUTabularDataset(X_train, y_train)
    val_ds = ICUTabularDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 4,
                            num_workers=0, pin_memory=False)

    model = ICUMortalityMLP(input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    pos_weight = torch.tensor(
        [float(class_counts[0]) / max(float(class_counts[1]), 1)],
        dtype=torch.float32,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            preds = []
            with torch.no_grad():
                for xb, _ in val_loader:
                    logits = model(xb.to(device))
                    preds.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            auc = roc_auc_score(y_val, preds)
            ap = average_precision_score(y_val, preds)
            logger.info(f"  MLP Epoch {epoch:3d}  Val AUC={auc:.4f}  AP={ap:.4f}")
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_auc


# ────────────────────────────────────────────────
# CatBoost training
# ────────────────────────────────────────────────

def train_catboost(X_train_raw, y_train, X_val_raw, y_val, seed=42):
    scale_pos = float(np.sum(y_train == 0)) / max(float(np.sum(y_train == 1)), 1)
    clf = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=3,
        loss_function="Logloss",
        eval_metric="AUC",
        scale_pos_weight=scale_pos,
        early_stopping_rounds=100,
        random_seed=seed,
        verbose=200,
        # Disable GPU on Apple Silicon — use CPU only
        task_type="CPU",
        thread_count=-1,
    )
    clf.fit(
        X_train_raw, y_train,
        eval_set=(X_val_raw, y_val),
        use_best_model=True,
    )
    val_prob = clf.predict_proba(X_val_raw)[:, 1]
    auc = roc_auc_score(y_val, val_prob)
    ap = average_precision_score(y_val, val_prob)
    logger.info(f"  CatBoost  Val AUC={auc:.4f}  AP={ap:.4f}")
    return clf, auc


# ────────────────────────────────────────────────
# Full training pipeline
# ────────────────────────────────────────────────

def run_training(X_df, y, output_dir, seed=42, n_folds=5, epochs=80):
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    device = get_device()
    logger.info(f"Device: {device}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    X_np = X_df.values.astype(np.float32)
    col_names = list(X_df.columns)

    fold_results = []
    oof_probs_mlp = np.zeros(len(y))
    oof_probs_cb = np.zeros(len(y))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_np, y), 1):
        logger.info(f"\n{'='*50}\nFold {fold}/{n_folds}\n{'='*50}")

        X_tr_raw, X_val_raw = X_np[tr_idx], X_np[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        imputer, scaler, X_tr_scaled = build_preprocessor(X_tr_raw)
        X_val_scaled = apply_preprocessor(X_val_raw, imputer, scaler)
        X_tr_imp = imputer.transform(X_tr_raw)
        X_val_imp = imputer.transform(X_val_raw)

        input_dim = X_tr_scaled.shape[1]

        mlp, mlp_auc = train_mlp(
            X_tr_scaled, y_tr, X_val_scaled, y_val,
            input_dim, device, epochs=epochs, seed=seed + fold)

        cb, cb_auc = train_catboost(
            X_tr_imp, y_tr, X_val_imp, y_val, seed=seed + fold)

        # OOF predictions
        mlp.eval()
        with torch.no_grad():
            t = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
            oof_probs_mlp[val_idx] = torch.sigmoid(mlp(t)).cpu().numpy()
        oof_probs_cb[val_idx] = cb.predict_proba(X_val_imp)[:, 1]

        # Save artifacts
        torch.save({"model_state": mlp.state_dict(), "input_dim": input_dim},
                   os.path.join(output_dir, f"mlp_fold{fold}.pt"))
        joblib.dump(cb, os.path.join(output_dir, f"catboost_fold{fold}.pkl"))
        joblib.dump({"imputer": imputer, "scaler": scaler},
                    os.path.join(output_dir, f"preprocessor_fold{fold}.pkl"))

        fold_results.append({
            "fold": fold,
            "mlp_auc": mlp_auc,
            "cb_auc": cb_auc,
            "val_idx": val_idx,
        })

    joblib.dump(col_names, os.path.join(output_dir, "col_names.pkl"))

    oof_ensemble = 0.5 * oof_probs_mlp + 0.5 * oof_probs_cb
    oof_auc = roc_auc_score(y, oof_ensemble)
    oof_ap = average_precision_score(y, oof_ensemble)

    logger.info(f"\nOOF Ensemble  AUC={oof_auc:.4f}  AUC-PR={oof_ap:.4f}")
    for res in fold_results:
        logger.info(f"  Fold {res['fold']}: MLP={res['mlp_auc']:.4f}  CB={res['cb_auc']:.4f}")

    # Persist OOF probabilities for evaluation plots
    joblib.dump({
        "oof_probs_mlp": oof_probs_mlp,
        "oof_probs_cb": oof_probs_cb,
        "oof_ensemble": oof_ensemble,
        "y_true": y,
        "fold_results": fold_results,
    }, os.path.join(output_dir, "oof_results.pkl"))

    return fold_results, oof_auc, oof_ap
