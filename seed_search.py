"""
Seed search for the single deep-learning ICU model.

This trains several independent single-model runs and selects the one with the
best OOF AUC. It does not ensemble predictions.
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


def parse_args():
    p = argparse.ArgumentParser(
        description="Search random seeds for the single DL ICU model."
    )
    p.add_argument("--train", default="data/train.pkl")
    p.add_argument("--test", default="data/test.pkl")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[7, 21, 42, 77, 123])
    p.add_argument("--k-fold", type=int, default=5)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--no-evaluate", action="store_true")
    p.add_argument("--metric", default="auc", choices=["auc", "ap"],
                   help="Metric used to choose the winning single model.")
    return p.parse_args()


def make_run_dir(results_dir, run_dir=None):
    if run_dir:
        path = Path(run_dir)
    else:
        stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        path = Path(results_dir) / f"{stamp}_seed_search"
    path.mkdir(parents=True, exist_ok=True)
    return path


def main():
    args = parse_args()
    run_dir = make_run_dir(args.results_dir, args.run_dir)
    logger.info(f"Seed-search run directory: {run_dir}")
    logger.info("This selects one single model; predictions are not ensembled.")

    with open(args.train, "rb") as f:
        train_data = pickle.load(f)
    with open(args.test, "rb") as f:
        test_data = pickle.load(f)

    logger.info("Extracting features once from train.pkl")
    X_df, y = build_feature_matrix(train_data, has_target=True)
    logger.info(f"Feature matrix: {X_df.shape}  positives={int(y.sum())}/{len(y)}")

    rows = []
    for seed in args.seeds:
        seed_dir = run_dir / f"seed_{seed}"
        model_dir = seed_dir / "models"
        seed_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        logger.info("\n" + "=" * 70)
        logger.info(f"Training seed {seed}")

        fold_results, oof_auc, oof_ap = run_training(
            X_df, y,
            output_dir=str(model_dir),
            k_fold=args.k_fold,
            seed=seed,
            model_type="mlp",
            hidden_dims=(256, 128, 64),
            dropout=0.34,
            input_dropout=0.04,
            num_res_blocks=2,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=5e-4,
            weight_decay=1e-4,
            scheduler_type="cosine",
            label_smoothing=0.05,
            eval_every=args.eval_every,
            patience=args.patience,
            min_delta=1e-4,
            sampler_type="none",
            use_pos_weight=True,
            loss_type="bce",
            final_refit=True,
            final_epoch_multiplier=1.15,
            threshold_strategy="youden",
        )

        if not args.no_evaluate:
            run_evaluation(str(model_dir), str(seed_dir))
        predict(
            test_data,
            model_dir=str(model_dir),
            output_csv=str(seed_dir / "group1.csv"),
        )

        best_fold = max(fold_results, key=lambda r: r["auc"])
        row = {
            "seed": seed,
            "oof_auc": round(float(oof_auc), 5),
            "oof_ap": round(float(oof_ap), 5),
            "best_fold": int(best_fold["fold"]),
            "best_fold_auc": round(float(best_fold["auc"]), 5),
            "best_epoch": int(best_fold.get("best_epoch", 0)),
            "path": str(seed_dir),
        }
        rows.append(row)
        pd.DataFrame(rows).sort_values(
            f"oof_{args.metric}", ascending=False
        ).to_csv(run_dir / "seed_search_partial.csv", index=False)

    summary = pd.DataFrame(rows).sort_values(f"oof_{args.metric}", ascending=False)
    summary.to_csv(run_dir / "seed_search_summary.csv", index=False)

    best = summary.iloc[0]
    best_dir = Path(best["path"])
    shutil.copyfile(best_dir / "group1.csv", run_dir / "group1.csv")
    with open(run_dir / "best_seed.json", "w") as f:
        json.dump(best.to_dict(), f, indent=2)

    logger.info("\nSeed search complete.")
    logger.info(f"Best seed: {int(best['seed'])}")
    logger.info(
        f"Best OOF AUC={best['oof_auc']:.5f}  AUC-PR={best['oof_ap']:.5f}"
    )
    logger.info(f"Summary: {run_dir / 'seed_search_summary.csv'}")
    logger.info(f"Best single-model prediction copied to: {run_dir / 'group1.csv'}")


if __name__ == "__main__":
    main()
