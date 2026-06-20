import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve, roc_curve,
)
import joblib
import logging

from src.dataset import ICUTabularDataset
from src.model import build_model
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

class FoldPreprocessor:
    def __init__(self, clip_quantile=0.005, add_missing_indicators=True):
        self.clip_quantile = clip_quantile
        self.add_missing_indicators = add_missing_indicators
        self.lower_ = None
        self.upper_ = None
        self.imputer = SimpleImputer(
            strategy="median",
            add_indicator=add_missing_indicators,
        )
        self.scaler = StandardScaler()

    def fit_transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.lower_ = np.nanquantile(X, self.clip_quantile, axis=0)
        self.upper_ = np.nanquantile(X, 1.0 - self.clip_quantile, axis=0)
        self.lower_ = np.where(np.isfinite(self.lower_), self.lower_, np.nan)
        self.upper_ = np.where(np.isfinite(self.upper_), self.upper_, np.nan)
        X_clip = self._clip(X)
        X_imp = self.imputer.fit_transform(X_clip)
        return self.scaler.fit_transform(X_imp)

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        X_clip = self._clip(X)
        X_imp = self.imputer.transform(X_clip)
        return self.scaler.transform(X_imp)

    def _clip(self, X):
        lower = np.where(np.isfinite(self.lower_), self.lower_, -np.inf)
        upper = np.where(np.isfinite(self.upper_), self.upper_, np.inf)
        return np.clip(X, lower, upper)


def build_preprocessor(X_train: np.ndarray):
    preprocessor = FoldPreprocessor(clip_quantile=0.005, add_missing_indicators=True)
    X_scaled = preprocessor.fit_transform(X_train)
    return preprocessor, None, X_scaled


def apply_preprocessor(X, imputer, scaler):
    if hasattr(imputer, "transform") and scaler is None:
        return imputer.transform(X)
    return scaler.transform(imputer.transform(X))


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - pt).pow(self.gamma) * bce).mean()


def pairwise_auc_loss(logits, targets, margin=1.0):
    hard_targets = (targets >= 0.5)
    pos = logits[hard_targets]
    neg = logits[~hard_targets]
    if pos.numel() == 0 or neg.numel() == 0:
        return logits.new_tensor(0.0)
    diffs = pos[:, None] - neg[None, :]
    return nn.functional.softplus(margin - diffs).mean()


def augment_feature_space(
    X, y, factor=0, noise_std=0.03, mixup_alpha=0.4,
    include_negatives=False, seed=42,
):
    """Fold-safe augmentation after split/preprocessing."""
    if factor <= 0:
        return X, y
    rng = np.random.default_rng(seed)
    X_parts = [X]
    y_parts = [y]

    classes = [1]
    if include_negatives:
        classes.append(0)

    for cls in classes:
        idx = np.where(y.astype(int) == cls)[0]
        if len(idx) < 2:
            continue
        for _ in range(factor):
            base_idx = rng.choice(idx, size=len(idx), replace=True)
            other_idx = rng.choice(idx, size=len(idx), replace=True)
            lam = rng.beta(mixup_alpha, mixup_alpha, size=(len(idx), 1))
            mixed = lam * X[base_idx] + (1.0 - lam) * X[other_idx]
            noise = rng.normal(0.0, noise_std, size=mixed.shape)
            X_aug = (mixed + noise).astype(np.float32)
            y_aug = np.full(len(idx), cls, dtype=y.dtype)
            X_parts.append(X_aug)
            y_parts.append(y_aug)

    return np.vstack(X_parts).astype(np.float32), np.concatenate(y_parts)


def _make_ema_state(model):
    return {
        name: param.detach().cpu().clone()
        for name, param in model.state_dict().items()
        if torch.is_floating_point(param)
    }


def _update_ema_state(model, ema_state, decay):
    for name, param in model.state_dict().items():
        if name in ema_state:
            ema_state[name].mul_(decay).add_(param.detach().cpu(), alpha=1.0 - decay)


def _copy_ema_to_model(model, ema_state):
    state = model.state_dict()
    backup = {}
    for name, value in ema_state.items():
        backup[name] = state[name].detach().cpu().clone()
        state[name].copy_(value.to(state[name].device))
    return backup


def _restore_state(model, backup):
    state = model.state_dict()
    for name, value in backup.items():
        state[name].copy_(value.to(state[name].device))


def _predict_loader(model, loader, device, ema_state=None):
    backup = None
    if ema_state is not None:
        backup = _copy_ema_to_model(model, ema_state)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            logits = model(xb.to(device))
            preds.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    if backup is not None:
        _restore_state(model, backup)
    return preds


def _select_threshold(y_true, probs, strategy="youden"):
    if strategy == "fixed":
        return 0.5
    if strategy == "f1":
        precision, recall, thresholds = precision_recall_curve(y_true, probs)
        if len(thresholds) == 0:
            return 0.5
        f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
            precision[:-1] + recall[:-1], 1e-12
        )
        return float(thresholds[int(np.nanargmax(f1))])
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    scores = tpr - fpr
    return float(thresholds[int(np.nanargmax(scores))])


# ────────────────────────────────────────────────
# Single-fold MLP training
# ────────────────────────────────────────────────

def train_one_fold(
    X_tr, y_tr, X_val, y_val,
    input_dim, device,
    # architecture
    model_type="mlp",
    hidden_dims=(256, 128, 64),
    dropout=0.35,
    input_dropout=0.05,
    num_res_blocks=2,
    transformer_dim=96,
    transformer_layers=4,
    transformer_heads=8,
    # optimisation
    epochs=100,
    batch_size=256,
    lr=5e-4,
    weight_decay=1e-4,
    # scheduler
    scheduler_type="cosine",   # "cosine" | "step" | "none"
    step_size=20,
    gamma=0.5,
    # label smoothing
    label_smoothing=0.05,
    # regularisation / validation
    eval_every=1,
    patience=12,
    min_delta=1e-4,
    sampler_type="none",
    use_pos_weight=True,
    loss_type="bce",
    focal_alpha=0.25,
    focal_gamma=2.0,
    auc_loss_weight=0.25,
    auc_margin=1.0,
    batch_mixup_alpha=0.0,
    batch_mixup_prob=0.0,
    ema_decay=0.0,
    seed=42,
):
    set_seed(seed)

    class_counts = np.bincount(y_tr.astype(int))
    sampler = None
    if sampler_type == "balanced":
        weights = 1.0 / class_counts[y_tr.astype(int)]
        sampler = WeightedRandomSampler(
            torch.tensor(weights, dtype=torch.float64),
            num_samples=len(weights),
            replacement=True,
        )

    train_ds = ICUTabularDataset(X_tr, y_tr)
    val_ds   = ICUTabularDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, shuffle=sampler is None,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size * 4,
                              num_workers=0, pin_memory=False)

    model = build_model(
        model_type,
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        input_dropout=input_dropout,
        num_res_blocks=num_res_blocks,
        transformer_dim=transformer_dim,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=max(len(train_loader), 1),
            pct_start=0.2,
            div_factor=10.0,
            final_div_factor=100.0,
        )
    else:
        scheduler = None
    step_scheduler_per_batch = scheduler_type == "onecycle"

    pos_weight = None
    if use_pos_weight:
        pos_weight = torch.tensor(
            [float(class_counts[0]) / max(float(class_counts[1]), 1)],
            dtype=torch.float32,
        ).to(device)
    if loss_type == "focal":
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma,
                              pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    ema_state = _make_ema_state(model) if ema_decay > 0 else None
    best_auc   = 0.0
    best_ap    = 0.0
    best_epoch = 0
    best_state = None
    history    = []
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            if batch_mixup_alpha > 0 and np.random.rand() < batch_mixup_prob:
                perm = torch.randperm(xb.size(0), device=device)
                lam = np.random.beta(batch_mixup_alpha, batch_mixup_alpha)
                xb = lam * xb + (1.0 - lam) * xb[perm]
                yb = lam * yb + (1.0 - lam) * yb[perm]

            yb_smooth = yb * (1 - label_smoothing) + 0.5 * label_smoothing
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb_smooth)
            if loss_type == "bce_auc":
                loss = loss + auc_loss_weight * pairwise_auc_loss(
                    logits, yb, margin=auc_margin
                )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if ema_state is not None:
                _update_ema_state(model, ema_state, ema_decay)
            if scheduler is not None and step_scheduler_per_batch:
                scheduler.step()
            epoch_loss += loss.item()
            batch_count += 1

        if scheduler is not None and not step_scheduler_per_batch:
            scheduler.step()

        if epoch % eval_every == 0 or epoch == epochs:
            preds = _predict_loader(model, val_loader, device, ema_state=ema_state)
            auc = roc_auc_score(y_val, preds)
            ap  = average_precision_score(y_val, preds)
            train_loss = epoch_loss / max(batch_count, 1)
            logger.info(
                f"  Epoch {epoch:3d}  loss={train_loss:.4f}  "
                f"Val AUC={auc:.4f}  AP={ap:.4f}"
            )
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "auc": auc,
                "ap": ap,
            })
            if auc > best_auc + min_delta:
                best_auc   = auc
                best_ap    = ap
                best_epoch = epoch
                if ema_state is not None:
                    best_state = {
                        k: v.clone() if k in ema_state else v.cpu().clone()
                        for k, v in model.state_dict().items()
                    }
                    for name, value in ema_state.items():
                        best_state[name] = value.clone()
                else:
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += eval_every
                if epochs_without_improvement >= patience:
                    logger.info(
                        f"  Early stopping at epoch {epoch} "
                        f"(best epoch {best_epoch}, AUC={best_auc:.4f})"
                    )
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_auc, best_ap, best_epoch, history


def train_fixed_epochs(
    X_tr, y_tr, input_dim, device,
    model_type="mlp",
    hidden_dims=(256, 128, 64),
    dropout=0.34,
    input_dropout=0.04,
    num_res_blocks=2,
    transformer_dim=96,
    transformer_layers=4,
    transformer_heads=8,
    epochs=20,
    batch_size=256,
    lr=5e-4,
    weight_decay=1e-4,
    scheduler_type="cosine",
    step_size=20,
    gamma=0.5,
    label_smoothing=0.05,
    sampler_type="none",
    use_pos_weight=True,
    loss_type="bce",
    focal_alpha=0.25,
    focal_gamma=2.0,
    auc_loss_weight=0.25,
    auc_margin=1.0,
    batch_mixup_alpha=0.0,
    batch_mixup_prob=0.0,
    ema_decay=0.0,
    seed=42,
):
    set_seed(seed)

    class_counts = np.bincount(y_tr.astype(int))
    sampler = None
    if sampler_type == "balanced":
        weights = 1.0 / class_counts[y_tr.astype(int)]
        sampler = WeightedRandomSampler(
            torch.tensor(weights, dtype=torch.float64),
            num_samples=len(weights),
            replacement=True,
        )

    train_ds = ICUTabularDataset(X_tr, y_tr)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=0,
        pin_memory=False,
    )

    model = build_model(
        model_type,
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        input_dropout=input_dropout,
        num_res_blocks=num_res_blocks,
        transformer_dim=transformer_dim,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=max(len(train_loader), 1),
            pct_start=0.2,
            div_factor=10.0,
            final_div_factor=100.0,
        )
    else:
        scheduler = None
    step_scheduler_per_batch = scheduler_type == "onecycle"

    pos_weight = None
    if use_pos_weight:
        pos_weight = torch.tensor(
            [float(class_counts[0]) / max(float(class_counts[1]), 1)],
            dtype=torch.float32,
        ).to(device)
    if loss_type == "focal":
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma,
                              pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    ema_state = _make_ema_state(model) if ema_decay > 0 else None
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            if batch_mixup_alpha > 0 and np.random.rand() < batch_mixup_prob:
                perm = torch.randperm(xb.size(0), device=device)
                lam = np.random.beta(batch_mixup_alpha, batch_mixup_alpha)
                xb = lam * xb + (1.0 - lam) * xb[perm]
                yb = lam * yb + (1.0 - lam) * yb[perm]

            yb_smooth = yb * (1 - label_smoothing) + 0.5 * label_smoothing
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb_smooth)
            if loss_type == "bce_auc":
                loss = loss + auc_loss_weight * pairwise_auc_loss(
                    logits, yb, margin=auc_margin
                )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if ema_state is not None:
                _update_ema_state(model, ema_state, ema_decay)
            if scheduler is not None and step_scheduler_per_batch:
                scheduler.step()
            epoch_loss += loss.item()
            batch_count += 1

        if scheduler is not None and not step_scheduler_per_batch:
            scheduler.step()
        logger.info(
            f"  Final refit epoch {epoch:3d}/{epochs}  "
            f"loss={epoch_loss / max(batch_count, 1):.4f}"
        )

    if ema_state is not None:
        _copy_ema_to_model(model, ema_state)
    model.eval()
    return model


# ────────────────────────────────────────────────
# Full k-fold training pipeline
# ────────────────────────────────────────────────

def run_training(
    X_df, y, output_dir,
    # CV
    k_fold=5,
    seed=42,
    # architecture
    model_type="mlp",
    hidden_dims=(256, 128, 64),
    dropout=0.34,
    input_dropout=0.04,
    num_res_blocks=2,
    transformer_dim=96,
    transformer_layers=4,
    transformer_heads=8,
    # optimisation
    epochs=100,
    batch_size=256,
    lr=5e-4,
    weight_decay=1e-4,
    # scheduler
    scheduler_type="cosine",
    step_size=20,
    gamma=0.5,
    # label smoothing
    label_smoothing=0.05,
    eval_every=1,
    patience=12,
    min_delta=1e-4,
    sampler_type="none",
    use_pos_weight=True,
    loss_type="bce",
    focal_alpha=0.25,
    focal_gamma=2.0,
    auc_loss_weight=0.25,
    auc_margin=1.0,
    batch_mixup_alpha=0.0,
    batch_mixup_prob=0.0,
    ema_decay=0.0,
    feature_augment_factor=0,
    feature_noise_std=0.03,
    feature_mixup_alpha=0.4,
    feature_augment_negatives=False,
    final_refit=True,
    final_epochs=None,
    final_epoch_multiplier=1.15,
    threshold_strategy="youden",
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
        X_tr_scaled, y_tr_train = augment_feature_space(
            X_tr_scaled,
            y_tr,
            factor=feature_augment_factor,
            noise_std=feature_noise_std,
            mixup_alpha=feature_mixup_alpha,
            include_negatives=feature_augment_negatives,
            seed=seed + 1000 + fold,
        )
        if len(y_tr_train) != len(y_tr):
            logger.info(
                f"Fold-safe feature augmentation: "
                f"{len(y_tr)} → {len(y_tr_train)} training rows"
            )
        else:
            y_tr_train = y_tr

        input_dim = X_tr_scaled.shape[1]

        model, fold_auc, fold_ap, best_epoch, history = train_one_fold(
            X_tr_scaled, y_tr_train, X_val_scaled, y_val,
            input_dim, device,
            model_type=model_type,
            hidden_dims=hidden_dims,
            dropout=dropout,
            input_dropout=input_dropout,
            num_res_blocks=num_res_blocks,
            transformer_dim=transformer_dim,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            scheduler_type=scheduler_type,
            step_size=step_size,
            gamma=gamma,
            label_smoothing=label_smoothing,
            eval_every=eval_every,
            patience=patience,
            min_delta=min_delta,
            sampler_type=sampler_type,
            use_pos_weight=use_pos_weight,
            loss_type=loss_type,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            auc_loss_weight=auc_loss_weight,
            auc_margin=auc_margin,
            batch_mixup_alpha=batch_mixup_alpha,
            batch_mixup_prob=batch_mixup_prob,
            ema_decay=ema_decay,
            seed=seed + fold,
        )

        # OOF predictions
        with torch.no_grad():
            t = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
            oof_probs[val_idx] = torch.sigmoid(model(t)).cpu().numpy()

        # Save fold artifacts
        torch.save(
            {"model_state": model.state_dict(),
             "model_type": model_type,
             "input_dim": input_dim,
             "hidden_dims": list(hidden_dims),
             "dropout": dropout,
             "input_dropout": input_dropout,
             "num_res_blocks": num_res_blocks,
             "transformer_dim": transformer_dim,
             "transformer_layers": transformer_layers,
             "transformer_heads": transformer_heads,
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
            "ap": fold_ap,
            "best_epoch": best_epoch,
            "val_idx": val_idx,
            "history": history,
        })

        if fold_auc > best_fold_auc:
            best_fold_auc = fold_auc
            best_fold_idx = fold

    # Save shared metadata
    joblib.dump(col_names, os.path.join(output_dir, "col_names.pkl"))
    threshold = _select_threshold(y, oof_probs, strategy=threshold_strategy)
    joblib.dump({"best_fold": best_fold_idx, "best_auc": best_fold_auc,
                 "threshold": threshold,
                 "threshold_strategy": threshold_strategy},
                os.path.join(output_dir, "best_fold.pkl"))

    # OOF evaluation
    oof_auc = roc_auc_score(y, oof_probs)
    oof_ap  = average_precision_score(y, oof_probs)

    logger.info(f"\nBest fold: Fold {best_fold_idx}  AUC={best_fold_auc:.4f}")
    logger.info(f"OOF AUC={oof_auc:.4f}  AUC-PR={oof_ap:.4f}")
    logger.info(f"OOF threshold ({threshold_strategy}) = {threshold:.4f}")
    for r in fold_results:
        logger.info(
            f"  Fold {r['fold']}: AUC={r['auc']:.4f}  "
            f"AP={r['ap']:.4f}  best_epoch={r['best_epoch']}"
        )

    # Persist OOF data for evaluation plots
    joblib.dump(
        {"oof_probs": oof_probs,
         "y_true": y,
         "fold_results": fold_results,
         "best_fold": best_fold_idx,
         "threshold": threshold,
         "threshold_strategy": threshold_strategy},
        os.path.join(output_dir, "oof_results.pkl"),
    )

    if final_refit:
        best_epochs = [int(r["best_epoch"]) for r in fold_results
                       if int(r.get("best_epoch", 0)) > 0]
        if final_epochs is None:
            refit_epochs = int(np.ceil(np.median(best_epochs) * final_epoch_multiplier))
            refit_epochs = max(refit_epochs, 1)
        else:
            refit_epochs = int(final_epochs)

        logger.info(
            f"\nFinal refit on 100% train data for {refit_epochs} epochs "
            f"(median best fold epoch={np.median(best_epochs):.1f})"
        )
        imputer, scaler, X_full_scaled = build_preprocessor(X_np)
        y_full_train = y
        X_full_scaled, y_full_train = augment_feature_space(
            X_full_scaled,
            y_full_train,
            factor=feature_augment_factor,
            noise_std=feature_noise_std,
            mixup_alpha=feature_mixup_alpha,
            include_negatives=feature_augment_negatives,
            seed=seed + 9999,
        )
        if len(y_full_train) != len(y):
            logger.info(
                f"Final fold-safe feature augmentation: "
                f"{len(y)} → {len(y_full_train)} training rows"
            )

        final_model = train_fixed_epochs(
            X_full_scaled,
            y_full_train,
            input_dim=X_full_scaled.shape[1],
            device=device,
            model_type=model_type,
            hidden_dims=hidden_dims,
            dropout=dropout,
            input_dropout=input_dropout,
            num_res_blocks=num_res_blocks,
            transformer_dim=transformer_dim,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            epochs=refit_epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            scheduler_type=scheduler_type,
            step_size=step_size,
            gamma=gamma,
            label_smoothing=label_smoothing,
            sampler_type=sampler_type,
            use_pos_weight=use_pos_weight,
            loss_type=loss_type,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            auc_loss_weight=auc_loss_weight,
            auc_margin=auc_margin,
            batch_mixup_alpha=batch_mixup_alpha,
            batch_mixup_prob=batch_mixup_prob,
            ema_decay=ema_decay,
            seed=seed + 999,
        )
        torch.save(
            {"model_state": final_model.state_dict(),
             "model_type": model_type,
             "input_dim": X_full_scaled.shape[1],
             "hidden_dims": list(hidden_dims),
             "dropout": dropout,
             "input_dropout": input_dropout,
             "num_res_blocks": num_res_blocks,
             "transformer_dim": transformer_dim,
             "transformer_layers": transformer_layers,
             "transformer_heads": transformer_heads,
             "refit_epochs": refit_epochs},
            os.path.join(output_dir, "final_model.pt"),
        )
        joblib.dump(
            {"imputer": imputer, "scaler": scaler},
            os.path.join(output_dir, "preprocessor_final.pkl"),
        )
        info = joblib.load(os.path.join(output_dir, "best_fold.pkl"))
        info["use_final_model"] = True
        info["final_epochs"] = refit_epochs
        joblib.dump(info, os.path.join(output_dir, "best_fold.pkl"))
        logger.info("Final refit model saved → final_model.pt")

    return fold_results, oof_auc, oof_ap
