import os
import numpy as np
import pandas as pd
import torch
import joblib
import logging

from src.model import ICUMortalityMLP
from src.train import apply_preprocessor, get_device
from src.utils import build_feature_matrix

logger = logging.getLogger(__name__)


def predict(test_data, model_dir, output_csv="group1.csv", threshold=None):
    """
    Load the best fold model (by val AUC), run inference, save CSV.
    """
    device = get_device()

    # Determine best fold
    best_fold_path = os.path.join(model_dir, "best_fold.pkl")
    if os.path.exists(best_fold_path):
        info = joblib.load(best_fold_path)
        best_fold = info["best_fold"]
        logger.info(f"Using best fold: Fold {best_fold}  (Val AUC={info['best_auc']:.4f})")
        if threshold is None:
            threshold = float(info.get("threshold", 0.5))
    else:
        # Fallback: scan available folds and pick highest AUC
        best_fold, best_auc = 1, 0.0
        fold = 1
        while os.path.exists(os.path.join(model_dir, f"model_fold{fold}.pt")):
            ckpt = torch.load(os.path.join(model_dir, f"model_fold{fold}.pt"),
                              map_location="cpu", weights_only=False)
            if ckpt.get("fold_auc", 0) > best_auc:
                best_auc  = ckpt["fold_auc"]
                best_fold = fold
            fold += 1
        logger.info(f"best_fold.pkl not found — scanning folds. "
                    f"Best: Fold {best_fold} (AUC={best_auc:.4f})")
    if threshold is None:
        threshold = 0.5
    logger.info(f"Using classification threshold: {threshold:.4f}")

    # Build test feature matrix
    X_test_df = build_feature_matrix(test_data, has_target=False)
    test_ids  = list(X_test_df.index)

    col_names = joblib.load(os.path.join(model_dir, "col_names.pkl"))
    for col in col_names:
        if col not in X_test_df.columns:
            X_test_df[col] = np.nan
    X_test_df = X_test_df[col_names]
    X_test_np = X_test_df.values.astype(np.float32)

    # Load best fold artifacts
    prep  = joblib.load(os.path.join(model_dir, f"preprocessor_fold{best_fold}.pkl"))
    ckpt  = torch.load(os.path.join(model_dir, f"model_fold{best_fold}.pt"),
                       map_location=device, weights_only=False)

    X_scaled = apply_preprocessor(X_test_np, prep["imputer"], prep["scaler"])

    model = ICUMortalityMLP(
        input_dim   = ckpt["input_dim"],
        hidden_dims = tuple(ckpt["hidden_dims"]),
        dropout     = ckpt["dropout"],
        input_dropout = ckpt.get("input_dropout", 0.0),
        num_res_blocks = ckpt.get("num_res_blocks", 2),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        probs = torch.sigmoid(
            model(torch.tensor(X_scaled, dtype=torch.float32).to(device))
        ).cpu().numpy()

    predictions = (probs >= threshold).astype(int)

    df_out = pd.DataFrame({
        "id":          test_ids,
        "probability": np.round(probs, 6),
        "prediction":  predictions,
    })
    df_out.to_csv(output_csv, index=False)
    logger.info(f"Saved → {output_csv}  "
                f"({predictions.sum()} positive / {len(predictions)} total)")
    return df_out
