"""
Deep-learning ablation runner for ICU mortality prediction.

Each ablation trains one standalone neural model. No fold ensembling and no
classical ML models are used. The best single fold of each ablation is used for
that ablation's prediction file, matching main.py.
"""

import argparse
import json
import logging
import os
import pickle
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.evaluate import run_evaluation
from src.predict import predict
from src.train import run_training
from src.utils import build_feature_matrix


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


PAPER_CORE_PREFIXES = (
    "age_at_admission",
    "duration",
    "urineoutput__",
    "pao2__",
    "pao2fio2ratio__",
    "pneumonia",
    "stroke",
    "aki",
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Run detailed deep-learning-only ablations."
    )
    p.add_argument("--train", default="data/train.pkl")
    p.add_argument("--test", default="data/test.pkl")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--k-fold", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--variants", nargs="*", default=None,
                   help="Optional subset of variant names to run.")
    p.add_argument("--fast", action="store_true",
                   help="Short smoke-test run: 3 folds, 35 epochs, 6 patience.")
    return p.parse_args()


def make_run_dir(results_dir, run_dir=None):
    if run_dir:
        path = Path(run_dir)
    else:
        stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        path = Path(results_dir) / stamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def filter_features(X: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "all":
        return X
    if mode == "no_severity_scores":
        drop_prefixes = ("sofa__", "sapsii__")
        drop_exact = {"lods"}
        cols = [
            c for c in X.columns
            if c not in drop_exact and not c.startswith(drop_prefixes)
        ]
        return X[cols]
    if mode == "no_count_features":
        cols = [c for c in X.columns if not c.endswith("__count")]
        return X[cols]
    if mode == "paper_core_features":
        cols = [
            c for c in X.columns
            if c in PAPER_CORE_PREFIXES
            or any(c.startswith(prefix) for prefix in PAPER_CORE_PREFIXES)
        ]
        return X[cols]
    raise ValueError(f"Unknown feature mode: {mode}")


def get_variants():
    return [
        {
            "name": "a100_strong_mlp_auc_aug",
            "description": "Large MLP with fold-safe augmentation, mixup, EMA, BCE+AUC loss.",
            "feature_mode": "all",
            "kwargs": {
                "model_type": "mlp",
                "hidden_dims": (1024, 512, 256, 128),
                "dropout": 0.25,
                "input_dropout": 0.03,
                "num_res_blocks": 4,
                "lr": 8e-4,
                "weight_decay": 2e-4,
                "scheduler_type": "onecycle",
                "sampler_type": "none",
                "use_pos_weight": True,
                "loss_type": "bce_auc",
                "auc_loss_weight": 0.20,
                "batch_mixup_alpha": 0.20,
                "batch_mixup_prob": 0.35,
                "ema_decay": 0.995,
                "feature_augment_factor": 1,
                "feature_noise_std": 0.025,
                "feature_mixup_alpha": 0.40,
                "label_smoothing": 0.03,
                "threshold_strategy": "youden",
            },
        },
        {
            "name": "a100_transformer_auc_aug",
            "description": "FT-Transformer-style tabular DL with fold-safe augmentation and EMA.",
            "feature_mode": "all",
            "kwargs": {
                "model_type": "transformer",
                "transformer_dim": 128,
                "transformer_layers": 4,
                "transformer_heads": 8,
                "dropout": 0.12,
                "lr": 5e-4,
                "weight_decay": 1e-4,
                "scheduler_type": "onecycle",
                "sampler_type": "none",
                "use_pos_weight": True,
                "loss_type": "bce_auc",
                "auc_loss_weight": 0.15,
                "batch_mixup_alpha": 0.15,
                "batch_mixup_prob": 0.25,
                "ema_decay": 0.995,
                "feature_augment_factor": 1,
                "feature_noise_std": 0.02,
                "feature_mixup_alpha": 0.35,
                "label_smoothing": 0.03,
                "threshold_strategy": "youden",
            },
        },
        {
            "name": "a100_strong_mlp_no_aug",
            "description": "Large MLP with AUC loss and EMA, no synthetic rows.",
            "feature_mode": "all",
            "kwargs": {
                "model_type": "mlp",
                "hidden_dims": (1024, 512, 256, 128),
                "dropout": 0.25,
                "input_dropout": 0.03,
                "num_res_blocks": 4,
                "lr": 8e-4,
                "weight_decay": 2e-4,
                "scheduler_type": "onecycle",
                "sampler_type": "none",
                "use_pos_weight": True,
                "loss_type": "bce_auc",
                "auc_loss_weight": 0.20,
                "batch_mixup_alpha": 0.20,
                "batch_mixup_prob": 0.35,
                "ema_decay": 0.995,
                "feature_augment_factor": 0,
                "label_smoothing": 0.03,
                "threshold_strategy": "youden",
            },
        },
        {
            "name": "improved_regularized",
            "description": "Smaller residual MLP, no sampler, pos_weight BCE.",
            "feature_mode": "all",
            "kwargs": {
                "hidden_dims": (256, 128, 64),
                "dropout": 0.35,
                "input_dropout": 0.05,
                "num_res_blocks": 2,
                "lr": 5e-4,
                "weight_decay": 1e-4,
                "sampler_type": "none",
                "use_pos_weight": True,
                "loss_type": "bce",
                "label_smoothing": 0.05,
                "threshold_strategy": "youden",
            },
        },
        {
            "name": "compact_high_dropout",
            "description": "Lower-capacity MLP to test overfit sensitivity.",
            "feature_mode": "all",
            "kwargs": {
                "hidden_dims": (128, 64),
                "dropout": 0.45,
                "input_dropout": 0.10,
                "num_res_blocks": 1,
                "lr": 7e-4,
                "weight_decay": 2e-4,
                "sampler_type": "none",
                "use_pos_weight": True,
                "loss_type": "bce",
                "label_smoothing": 0.05,
                "threshold_strategy": "youden",
            },
        },
        {
            "name": "wide_current_like",
            "description": "Original capacity and double class compensation.",
            "feature_mode": "all",
            "kwargs": {
                "hidden_dims": (512, 256, 128),
                "dropout": 0.30,
                "input_dropout": 0.0,
                "num_res_blocks": 2,
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "sampler_type": "balanced",
                "use_pos_weight": True,
                "loss_type": "bce",
                "label_smoothing": 0.05,
                "threshold_strategy": "youden",
            },
        },
        {
            "name": "balanced_sampler_only",
            "description": "Balanced sampler without pos_weight.",
            "feature_mode": "all",
            "kwargs": {
                "hidden_dims": (256, 128, 64),
                "dropout": 0.35,
                "input_dropout": 0.05,
                "num_res_blocks": 2,
                "lr": 5e-4,
                "weight_decay": 1e-4,
                "sampler_type": "balanced",
                "use_pos_weight": False,
                "loss_type": "bce",
                "label_smoothing": 0.05,
                "threshold_strategy": "youden",
            },
        },
        {
            "name": "focal_loss",
            "description": "Focal loss for class imbalance.",
            "feature_mode": "all",
            "kwargs": {
                "hidden_dims": (256, 128, 64),
                "dropout": 0.35,
                "input_dropout": 0.05,
                "num_res_blocks": 2,
                "lr": 5e-4,
                "weight_decay": 1e-4,
                "sampler_type": "none",
                "use_pos_weight": True,
                "loss_type": "focal",
                "focal_alpha": 0.35,
                "focal_gamma": 2.0,
                "label_smoothing": 0.0,
                "threshold_strategy": "youden",
            },
        },
        {
            "name": "no_severity_scores",
            "description": "Drop SOFA/SAPS-II/LODS-like severity score features.",
            "feature_mode": "no_severity_scores",
            "kwargs": {
                "hidden_dims": (256, 128, 64),
                "dropout": 0.35,
                "input_dropout": 0.05,
                "num_res_blocks": 2,
                "lr": 5e-4,
                "weight_decay": 1e-4,
                "sampler_type": "none",
                "use_pos_weight": True,
                "loss_type": "bce",
                "label_smoothing": 0.05,
                "threshold_strategy": "youden",
            },
        },
        {
            "name": "no_count_features",
            "description": "Drop observation-count features to test care-intensity leakage.",
            "feature_mode": "no_count_features",
            "kwargs": {
                "hidden_dims": (256, 128, 64),
                "dropout": 0.35,
                "input_dropout": 0.05,
                "num_res_blocks": 2,
                "lr": 5e-4,
                "weight_decay": 1e-4,
                "sampler_type": "none",
                "use_pos_weight": True,
                "loss_type": "bce",
                "label_smoothing": 0.05,
                "threshold_strategy": "youden",
            },
        },
        {
            "name": "paper_core_features",
            "description": "DL model restricted to paper-inspired core features.",
            "feature_mode": "paper_core_features",
            "kwargs": {
                "hidden_dims": (64, 32),
                "dropout": 0.35,
                "input_dropout": 0.05,
                "num_res_blocks": 1,
                "lr": 7e-4,
                "weight_decay": 2e-4,
                "sampler_type": "none",
                "use_pos_weight": True,
                "loss_type": "bce",
                "label_smoothing": 0.05,
                "threshold_strategy": "youden",
            },
        },
    ]


def save_table_image(df: pd.DataFrame, path: Path, title: str):
    height = max(3.0, 0.48 * len(df) + 1.3)
    width = max(9.0, 1.25 * len(df.columns))
    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis("off")
    ax.set_title(title, fontsize=13, pad=12)
    table = ax.table(
        cellText=df.astype(str).values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.35)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_bars(df: pd.DataFrame, path: Path):
    plot_df = df.sort_values("oof_auc", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.55 * len(plot_df))))
    ax.barh(plot_df["variant"], plot_df["oof_auc"], color="#4C72B0", alpha=0.85)
    ax.set_xlabel("OOF AUC-ROC")
    ax.set_title("Deep Learning Ablation OOF AUC")
    ax.grid(axis="x", alpha=0.25)
    for i, value in enumerate(plot_df["oof_auc"]):
        ax.text(value + 0.002, i, f"{value:.4f}", va="center", fontsize=9)
    ax.set_xlim(max(0.0, plot_df["oof_auc"].min() - 0.03),
                min(1.0, plot_df["oof_auc"].max() + 0.04))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    if args.fast:
        args.k_fold = 3
        args.epochs = min(args.epochs, 35)
        args.patience = min(args.patience, 6)

    run_dir = make_run_dir(args.results_dir, args.run_dir)
    logger.info(f"Ablation run directory: {run_dir}")

    with open(args.train, "rb") as f:
        train_data = pickle.load(f)
    with open(args.test, "rb") as f:
        test_data = pickle.load(f)

    logger.info("Building feature matrix once from original train.pkl")
    X_all, y = build_feature_matrix(train_data, has_target=True)
    logger.info(f"Feature matrix: {X_all.shape}")

    variants = get_variants()
    if args.variants:
        wanted = set(args.variants)
        variants = [v for v in variants if v["name"] in wanted]
        missing = wanted - {v["name"] for v in variants}
        if missing:
            raise ValueError(f"Unknown variants: {sorted(missing)}")

    rows = []
    for variant in variants:
        name = variant["name"]
        variant_dir = run_dir / name
        model_dir = variant_dir / "models"
        variant_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        logger.info("\n" + "=" * 70)
        logger.info(f"Running ablation: {name}")
        logger.info(variant["description"])

        X = filter_features(X_all, variant["feature_mode"])
        config = {
            "name": name,
            "description": variant["description"],
            "feature_mode": variant["feature_mode"],
            "n_features": int(X.shape[1]),
            "k_fold": args.k_fold,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "seed": args.seed,
            "kwargs": {
                k: list(v) if isinstance(v, tuple) else v
                for k, v in variant["kwargs"].items()
            },
        }
        with open(variant_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        fold_results, oof_auc, oof_ap = run_training(
            X, y,
            output_dir=str(model_dir),
            k_fold=args.k_fold,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            eval_every=args.eval_every,
            patience=args.patience,
            **variant["kwargs"],
        )
        run_evaluation(str(model_dir), str(variant_dir))
        predict(
            test_data,
            model_dir=str(model_dir),
            output_csv=str(variant_dir / "group1.csv"),
        )

        best_fold = max(fold_results, key=lambda r: r["auc"])
        row = {
            "variant": name,
            "oof_auc": round(float(oof_auc), 5),
            "oof_ap": round(float(oof_ap), 5),
            "best_fold": int(best_fold["fold"]),
            "best_fold_auc": round(float(best_fold["auc"]), 5),
            "best_epoch": int(best_fold.get("best_epoch", 0)),
            "n_features": int(X.shape[1]),
            "feature_mode": variant["feature_mode"],
            "description": variant["description"],
        }
        rows.append(row)
        pd.DataFrame(rows).sort_values("oof_auc", ascending=False).to_csv(
            run_dir / "ablation_summary_partial.csv", index=False
        )

    summary = pd.DataFrame(rows).sort_values("oof_auc", ascending=False)
    summary.to_csv(run_dir / "ablation_summary.csv", index=False)
    save_table_image(summary, run_dir / "ablation_summary.png",
                     "Deep Learning Ablation Summary")
    plot_ablation_bars(summary, run_dir / "ablation_auc_bar.png")

    best_variant = str(summary.iloc[0]["variant"])
    shutil.copyfile(run_dir / best_variant / "group1.csv", run_dir / "group1.csv")
    with open(run_dir / "best_variant.txt", "w") as f:
        f.write(best_variant + "\n")

    logger.info("\nAblation complete.")
    logger.info(f"Best variant: {best_variant}")
    logger.info(f"Summary: {run_dir / 'ablation_summary.csv'}")
    logger.info(f"Best prediction copied to: {run_dir / 'group1.csv'}")


if __name__ == "__main__":
    main()
