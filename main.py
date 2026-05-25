"""
main.py — Entry point for ICU Mortality Prediction pipeline.

Usage:
    python main.py                        # train + predict + evaluate
    python main.py --augment              # with data augmentation
    python main.py --skip-train           # inference only (models already saved)
    python main.py --evaluate-only        # regenerate plots from saved OOF results
"""

# ── Must be FIRST before any torch/multiprocessing import ──────────────────
# Fixes segfault on macOS (Apple Silicon + Intel) caused by the default
# "fork" multiprocessing start method conflicting with Accelerate/vecLib.
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
# ───────────────────────────────────────────────────────────────────────────

import argparse
import logging
import os
import pickle
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train",            default="data/train.pkl")
    p.add_argument("--test",             default="data/test.pkl")
    p.add_argument("--output",           default="group1.csv")
    p.add_argument("--model-dir",        default="models/")
    p.add_argument("--results-dir",      default="results/")
    p.add_argument("--augment",          action="store_true")
    p.add_argument("--augment-factor",   type=int,   default=2)
    p.add_argument("--folds",            type=int,   default=5)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--epochs",           type=int,   default=80)
    p.add_argument("--skip-train",       action="store_true",
                   help="Skip training; use saved models for inference")
    p.add_argument("--evaluate-only",    action="store_true",
                   help="Only regenerate evaluation plots from saved OOF results")
    p.add_argument("--no-evaluate",      action="store_true",
                   help="Skip evaluation plots")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Evaluate-only mode ─────────────────────────────────────────────────
    if args.evaluate_only:
        from src.evaluate import run_evaluation
        run_evaluation(args.model_dir, args.results_dir)
        return

    # ── Load data ──────────────────────────────────────────────────────────
    logger.info(f"Loading train data from {args.train}")
    with open(args.train, "rb") as f:
        train_data = pickle.load(f)

    logger.info(f"Loading test data from {args.test}")
    with open(args.test, "rb") as f:
        test_data = pickle.load(f)

    logger.info(f"Train: {len(train_data)} patients | Test: {len(test_data)} patients")

    # ── Augmentation ───────────────────────────────────────────────────────
    if args.augment and not args.skip_train:
        from src.augment import augment_data
        aug_path = args.train.replace(".pkl", "_augmented.pkl")
        logger.info(f"Augmenting training data (factor={args.augment_factor})...")
        train_data = augment_data(train_data,
                                  augment_factor=args.augment_factor,
                                  seed=args.seed)
        with open(aug_path, "wb") as f:
            pickle.dump(train_data, f)
        logger.info(f"Augmented data saved to {aug_path}")

    # ── Feature extraction ─────────────────────────────────────────────────
    if not args.skip_train:
        from src.utils import build_feature_matrix
        logger.info("Extracting features...")
        X_df, y = build_feature_matrix(train_data, has_target=True)
        logger.info(f"Feature matrix: {X_df.shape}  "
                    f"Positives: {int(y.sum())}/{len(y)} "
                    f"({100*y.mean():.1f}%)")

    # ── Training ───────────────────────────────────────────────────────────
    if not args.skip_train:
        from src.train import run_training
        logger.info("Starting training...")
        fold_results, oof_auc, oof_ap = run_training(
            X_df, y,
            output_dir=args.model_dir,
            seed=args.seed,
            n_folds=args.folds,
            epochs=args.epochs,
        )
        logger.info(f"\n✓ Training complete  OOF AUC={oof_auc:.4f}  AUC-PR={oof_ap:.4f}")

    # ── Inference ──────────────────────────────────────────────────────────
    from src.predict import predict
    logger.info("Running inference on test set...")
    df_out = predict(
        test_data,
        model_dir=args.model_dir,
        output_csv=args.output,
    )
    logger.info(f"✓ Predictions saved to {args.output}")
    pos = int(df_out["prediction"].sum())
    logger.info(f"  Predicted positive (mortality): {pos}/{len(df_out)} "
                f"({100*pos/len(df_out):.1f}%)")

    # ── Evaluation plots ───────────────────────────────────────────────────
    if not args.no_evaluate and not args.skip_train:
        from src.evaluate import run_evaluation
        logger.info("Generating evaluation plots...")
        run_evaluation(args.model_dir, args.results_dir)
        logger.info(f"✓ Plots saved to {args.results_dir}")


if __name__ == "__main__":
    main()
