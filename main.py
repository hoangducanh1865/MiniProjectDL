"""
main.py — ICU Mortality Prediction (single Deep Learning model, k-fold CV)

Usage:
    python main.py                     # full pipeline
    python main.py --augment           # with data augmentation
    python main.py --skip-train        # inference only (best saved model)
    python main.py --evaluate-only     # regenerate plots from saved OOF
"""

# Must be FIRST — fixes segfault on macOS (fork + Accelerate conflict)
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import argparse
import logging
import os
import pickle
import sys
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="ICU Mortality Prediction — Deep Learning pipeline"
    )

    # ── Data ──────────────────────────────────────────────────────────────
    p.add_argument("--train",       default="data/train.pkl")
    p.add_argument("--test",        default="data/test.pkl")
    p.add_argument("--output",      default=None)
    p.add_argument("--model-dir",   default=None)
    p.add_argument("--results-dir", default="results/")
    p.add_argument("--run-dir",     default=None,
                   help="Timestamped run directory. Defaults to results/YYYY_MM_DD_HH_MM_SS")

    # ── Augmentation ──────────────────────────────────────────────────────
    p.add_argument("--augment",         action="store_true")
    p.add_argument("--augment-factor",  type=int,   default=2)

    # ── Cross-validation ──────────────────────────────────────────────────
    p.add_argument("--k-fold",  type=int,   default=5,
                   help="Number of CV folds (default: 5)")
    p.add_argument("--seed",    type=int,   default=42)

    # ── Architecture ──────────────────────────────────────────────────────
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128, 64],
                   help="Hidden layer sizes, e.g. --hidden-dims 512 256 128")
    p.add_argument("--dropout",     type=float, default=0.35)
    p.add_argument("--input-dropout", type=float, default=0.05)
    p.add_argument("--num-res-blocks", type=int, default=2)

    # ── Optimisation ──────────────────────────────────────────────────────
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch-size",    type=int,   default=256)
    p.add_argument("--lr",            type=float, default=5e-4,
                   help="Learning rate (default: 5e-4)")
    p.add_argument("--weight-decay",  type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--eval-every",     type=int,   default=1)
    p.add_argument("--patience",       type=int,   default=12)
    p.add_argument("--min-delta",      type=float, default=1e-4)
    p.add_argument("--sampler",        default="none",
                   choices=["none", "balanced"])
    p.add_argument("--no-pos-weight",  action="store_true")
    p.add_argument("--loss",           default="bce",
                   choices=["bce", "focal"])
    p.add_argument("--focal-alpha",    type=float, default=0.25)
    p.add_argument("--focal-gamma",    type=float, default=2.0)
    p.add_argument("--threshold-strategy", default="youden",
                   choices=["youden", "f1", "fixed"])
    p.add_argument("--threshold",      type=float, default=None,
                   help="Override inference threshold. Defaults to OOF-tuned threshold.")

    # ── LR Scheduler ──────────────────────────────────────────────────────
    p.add_argument("--scheduler",  default="cosine",
                   choices=["cosine", "step", "none"],
                   help="LR scheduler (default: cosine)")
    p.add_argument("--step-size",  type=int,   default=20,
                   help="StepLR step size (used when --scheduler=step)")
    p.add_argument("--gamma",      type=float, default=0.5,
                   help="StepLR gamma (used when --scheduler=step)")

    # ── Modes ─────────────────────────────────────────────────────────────
    p.add_argument("--skip-train",    action="store_true",
                   help="Skip training; run inference with best saved model")
    p.add_argument("--evaluate-only", action="store_true",
                   help="Only regenerate evaluation plots")
    p.add_argument("--no-evaluate",   action="store_true",
                   help="Skip evaluation plots after training")

    return p.parse_args()


def main():
    args = parse_args()
    provided_model_dir = args.model_dir
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_dir = args.run_dir or os.path.join(args.results_dir, timestamp)
    args.run_dir = run_dir
    args.model_dir = args.model_dir or os.path.join(run_dir, "models")
    args.output = args.output or os.path.join(run_dir, "group1.csv")
    eval_dir = run_dir

    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Model directory: {args.model_dir}")
    logger.info(f"Prediction CSV: {args.output}")

    # ── Evaluate-only ─────────────────────────────────────────────────────
    if args.evaluate_only:
        from src.evaluate import run_evaluation
        run_evaluation(provided_model_dir or "models/", eval_dir)
        return

    # ── Load data ─────────────────────────────────────────────────────────
    logger.info(f"Loading train data from {args.train}")
    with open(args.train, "rb") as f:
        train_data = pickle.load(f)

    logger.info(f"Loading test data from {args.test}")
    with open(args.test, "rb") as f:
        test_data = pickle.load(f)

    logger.info(f"Train: {len(train_data)} | Test: {len(test_data)}")

    # ── Augmentation ──────────────────────────────────────────────────────
    if args.augment and not args.skip_train:
        from src.augment import augment_data
        aug_path = args.train.replace(".pkl", "_augmented.pkl")
        logger.info(f"Augmenting (factor={args.augment_factor})...")
        train_data = augment_data(train_data,
                                  augment_factor=args.augment_factor,
                                  seed=args.seed)
        with open(aug_path, "wb") as f:
            pickle.dump(train_data, f)
        logger.info(f"Augmented data saved to {aug_path}")

    # ── Feature extraction ────────────────────────────────────────────────
    if not args.skip_train:
        from src.utils import build_feature_matrix
        logger.info("Extracting features...")
        X_df, y = build_feature_matrix(train_data, has_target=True)
        logger.info(f"Feature matrix: {X_df.shape}  "
                    f"Positives: {int(y.sum())}/{len(y)} ({100*y.mean():.1f}%)")

    # ── Training ──────────────────────────────────────────────────────────
    if not args.skip_train:
        from src.train import run_training
        logger.info(
            f"Starting {args.k_fold}-fold training  "
            f"epochs={args.epochs}  lr={args.lr}  "
            f"batch={args.batch_size}  hidden={args.hidden_dims}  "
            f"dropout={args.dropout}  scheduler={args.scheduler}"
        )
        fold_results, oof_auc, oof_ap = run_training(
            X_df, y,
            output_dir      = args.model_dir,
            k_fold          = args.k_fold,
            seed            = args.seed,
            hidden_dims     = tuple(args.hidden_dims),
            dropout         = args.dropout,
            input_dropout   = args.input_dropout,
            num_res_blocks  = args.num_res_blocks,
            epochs          = args.epochs,
            batch_size      = args.batch_size,
            lr              = args.lr,
            weight_decay    = args.weight_decay,
            scheduler_type  = args.scheduler,
            step_size       = args.step_size,
            gamma           = args.gamma,
            label_smoothing = args.label_smoothing,
            eval_every      = args.eval_every,
            patience        = args.patience,
            min_delta       = args.min_delta,
            sampler_type    = args.sampler,
            use_pos_weight  = not args.no_pos_weight,
            loss_type       = args.loss,
            focal_alpha     = args.focal_alpha,
            focal_gamma     = args.focal_gamma,
            threshold_strategy = args.threshold_strategy,
        )
        logger.info(f"✓ Training done  OOF AUC={oof_auc:.4f}  AUC-PR={oof_ap:.4f}")

    # ── Inference ─────────────────────────────────────────────────────────
    from src.predict import predict
    logger.info("Running inference (best fold model)...")
    df_out = predict(test_data, model_dir=args.model_dir, output_csv=args.output,
                     threshold=args.threshold)
    logger.info(f"✓ Predictions saved to {args.output}")

    # ── Evaluation plots ──────────────────────────────────────────────────
    if not args.no_evaluate and not args.skip_train:
        from src.evaluate import run_evaluation
        logger.info("Generating evaluation plots...")
        run_evaluation(args.model_dir, eval_dir)
        logger.info(f"✓ Plots saved to {eval_dir}")


if __name__ == "__main__":
    main()
