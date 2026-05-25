import os
import numpy as np
import pandas as pd
import torch
import joblib
import logging

from src.model import ICUMortalityMLP
from src.train import apply_preprocessor
from src.utils import build_feature_matrix

logger = logging.getLogger(__name__)


def predict(test_data, model_dir, output_csv="group1.csv",
            threshold=0.5, mlp_weight=0.5, cb_weight=0.5):
    """
    Load all fold models, run inference, average predictions,
    and write the output CSV.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build test feature matrix
    X_test_df = build_feature_matrix(test_data, has_target=False)
    test_ids = list(X_test_df.index)

    # Align columns to training columns
    col_names = joblib.load(os.path.join(model_dir, "col_names.pkl"))
    for col in col_names:
        if col not in X_test_df.columns:
            X_test_df[col] = np.nan
    X_test_df = X_test_df[col_names]
    X_test_np = X_test_df.values.astype(np.float32)

    # Collect predictions across folds
    fold_probs_mlp = []
    fold_probs_cb = []

    fold = 1
    while True:
        mlp_path = os.path.join(model_dir, f"mlp_fold{fold}.pt")
        if not os.path.exists(mlp_path):
            break

        prep = joblib.load(os.path.join(model_dir, f"preprocessor_fold{fold}.pkl"))
        imputer, scaler = prep["imputer"], prep["scaler"]

        X_scaled = apply_preprocessor(X_test_np, imputer, scaler)
        X_imp = imputer.transform(X_test_np)

        # MLP
        ckpt = torch.load(mlp_path, map_location=device)
        mlp = ICUMortalityMLP(ckpt["input_dim"]).to(device)
        mlp.load_state_dict(ckpt["model_state"])
        mlp.eval()
        with torch.no_grad():
            probs_mlp = torch.sigmoid(
                mlp(torch.tensor(X_scaled).to(device))
            ).cpu().numpy()
        fold_probs_mlp.append(probs_mlp)

        # CatBoost
        cb = joblib.load(os.path.join(model_dir, f"catboost_fold{fold}.pkl"))
        probs_cb = cb.predict_proba(X_imp)[:, 1]
        fold_probs_cb.append(probs_cb)

        fold += 1

    avg_mlp = np.mean(fold_probs_mlp, axis=0)
    avg_cb = np.mean(fold_probs_cb, axis=0)
    final_probs = mlp_weight * avg_mlp + cb_weight * avg_cb
    predictions = (final_probs >= threshold).astype(int)

    df_out = pd.DataFrame({
        "id": test_ids,
        "probability": np.round(final_probs, 6),
        "prediction": predictions,
    })
    df_out.to_csv(output_csv, index=False)
    logger.info(f"Saved predictions to {output_csv}  "
                f"({predictions.sum()} positives / {len(predictions)} total)")
    return df_out
