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
    p.add_argument("--preset",  default="regularized",
                   choices=["regularized", "strong_mlp", "transformer"],
                   help="Convenience preset for stronger DL configurations")

    # ── Architecture ──────────────────────────────────────────────────────
    p.add_argument("--model-type", default="mlp",
                   choices=["mlp", "transformer"])
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128, 64],
                   help="Hidden layer sizes, e.g. --hidden-dims 512 256 128")
    p.add_argument("--dropout",     type=float, default=0.34)
    p.add_argument("--input-dropout", type=float, default=0.04)
    p.add_argument("--num-res-blocks", type=int, default=2)
    p.add_argument("--transformer-dim", type=int, default=96)
    p.add_argument("--transformer-layers", type=int, default=4)
    p.add_argument("--transformer-heads", type=int, default=8)

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
                   choices=["bce", "focal", "bce_auc"])
    p.add_argument("--focal-alpha",    type=float, default=0.25)
    p.add_argument("--focal-gamma",    type=float, default=2.0)
    p.add_argument("--auc-loss-weight", type=float, default=0.25)
    p.add_argument("--auc-margin",      type=float, default=1.0)
    p.add_argument("--batch-mixup-alpha", type=float, default=0.0)
    p.add_argument("--batch-mixup-prob",  type=float, default=0.0)
    p.add_argument("--ema-decay",        type=float, default=0.0)
    p.add_argument("--feature-augment-factor", type=int, default=0)
    p.add_argument("--feature-noise-std", type=float, default=0.03)
    p.add_argument("--feature-mixup-alpha", type=float, default=0.4)
    p.add_argument("--feature-augment-negatives", action="store_true")
    p.add_argument("--threshold-strategy", default="youden",
                   choices=["youden", "f1", "fixed"])
    p.add_argument("--threshold",      type=float, default=None,
                   help="Override inference threshold. Defaults to OOF-tuned threshold.")
    p.add_argument("--no-final-refit", action="store_true",
                   help="Disable final refit on 100% train data after CV.")
    p.add_argument("--final-epochs", type=int, default=None,
                   help="Fixed epoch count for final refit. Defaults to median best fold epoch.")
    p.add_argument("--final-epoch-multiplier", type=float, default=1.15)

    # ── LR Scheduler ──────────────────────────────────────────────────────
    p.add_argument("--scheduler",  default="cosine",
                   choices=["cosine", "step", "onecycle", "none"],
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
    if args.preset == "strong_mlp":
        args.model_type = "mlp"
        args.hidden_dims = [1024, 512, 256, 128]
        args.dropout = 0.25
        args.input_dropout = 0.03
        args.num_res_blocks = 4
        args.epochs = max(args.epochs, 160)
        args.lr = 8e-4
        args.weight_decay = 2e-4
        args.scheduler = "onecycle"
        args.loss = "bce_auc"
        args.auc_loss_weight = 0.20
        args.batch_mixup_alpha = 0.20
        args.batch_mixup_prob = 0.35
        args.ema_decay = 0.995
        args.feature_augment_factor = max(args.feature_augment_factor, 1)
        args.patience = max(args.patience, 25)
    elif args.preset == "transformer":
        args.model_type = "transformer"
        args.transformer_dim = max(args.transformer_dim, 128)
        args.transformer_layers = max(args.transformer_layers, 4)
        args.transformer_heads = max(args.transformer_heads, 8)
        args.dropout = 0.12
        args.epochs = max(args.epochs, 140)
        args.lr = 5e-4
        args.weight_decay = 1e-4
        args.scheduler = "onecycle"
        args.loss = "bce_auc"
        args.auc_loss_weight = 0.15
        args.batch_mixup_alpha = 0.15
        args.batch_mixup_prob = 0.25
        args.ema_decay = 0.995
        args.feature_augment_factor = max(args.feature_augment_factor, 1)
        args.patience = max(args.patience, 22)
    else:
        args.model_type = "mlp"
        args.hidden_dims = [256, 128, 64]
        args.dropout = 0.34
        args.input_dropout = 0.04
        args.num_res_blocks = 2

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
            model_type      = args.model_type,
            hidden_dims     = tuple(args.hidden_dims),
            dropout         = args.dropout,
            input_dropout   = args.input_dropout,
            num_res_blocks  = args.num_res_blocks,
            transformer_dim = args.transformer_dim,
            transformer_layers = args.transformer_layers,
            transformer_heads = args.transformer_heads,
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
            auc_loss_weight = args.auc_loss_weight,
            auc_margin      = args.auc_margin,
            batch_mixup_alpha = args.batch_mixup_alpha,
            batch_mixup_prob  = args.batch_mixup_prob,
            ema_decay       = args.ema_decay,
            feature_augment_factor = args.feature_augment_factor,
            feature_noise_std = args.feature_noise_std,
            feature_mixup_alpha = args.feature_mixup_alpha,
            feature_augment_negatives = args.feature_augment_negatives,
            final_refit    = not args.no_final_refit,
            final_epochs   = args.final_epochs,
            final_epoch_multiplier = args.final_epoch_multiplier,
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
