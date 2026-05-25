"""
evaluate.py — Evaluation plots from OOF results (single DL model, k-fold CV).

Plots saved to results/:
  1. roc_curves.png          — ROC per fold + OOF overall
  2. pr_curves.png           — Precision-Recall per fold + OOF overall
  3. calibration.png         — Reliability diagram
  4. score_distribution.png  — Score histogram by class
  5. fold_comparison.png     — AUC bar chart per fold
  6. confusion_matrix.png    — Confusion matrix at threshold=0.5
  7. training_history.png    — AUC/AP per epoch for each fold
"""

import os
import argparse
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, brier_score_loss,
)
from sklearn.calibration import calibration_curve
import logging

logger = logging.getLogger(__name__)

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
           "#937860", "#DA8BC3", "#8C8C8C"]
BEST_COLOR   = "#e63946"
OVERALL_COLOR = "#2d2d2d"


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {path}")


def plot_roc_curves(fold_results, oof_probs, y_true, best_fold, output_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    for res in fold_results:
        vi    = res["val_idx"]
        fpr, tpr, _ = roc_curve(y_true[vi], oof_probs[vi])
        fold_auc = auc(fpr, tpr)
        is_best  = res["fold"] == best_fold
        color    = BEST_COLOR if is_best else PALETTE[res["fold"] % len(PALETTE)]
        lw       = 2.2 if is_best else 1.0
        label    = (f"Fold {res['fold']} (AUC={fold_auc:.3f})"
                    + (" ★ best" if is_best else ""))
        ax.plot(fpr, tpr, color=color, linewidth=lw, alpha=0.8, label=label)

    fpr_o, tpr_o, _ = roc_curve(y_true, oof_probs)
    oof_auc = auc(fpr_o, tpr_o)
    ax.plot(fpr_o, tpr_o, "--", color=OVERALL_COLOR, linewidth=2.5,
            label=f"OOF Overall (AUC={oof_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k:", linewidth=0.8, alpha=0.4)

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Per Fold + OOF Overall", fontsize=13)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.3)
    _save(fig, os.path.join(output_dir, "roc_curves.png"))


def plot_pr_curves(fold_results, oof_probs, y_true, best_fold, output_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    baseline = y_true.mean()
    for res in fold_results:
        vi   = res["val_idx"]
        prec, rec, _ = precision_recall_curve(y_true[vi], oof_probs[vi])
        ap   = average_precision_score(y_true[vi], oof_probs[vi])
        is_best = res["fold"] == best_fold
        color   = BEST_COLOR if is_best else PALETTE[res["fold"] % len(PALETTE)]
        lw      = 2.2 if is_best else 1.0
        label   = (f"Fold {res['fold']} (AP={ap:.3f})"
                   + (" ★ best" if is_best else ""))
        ax.plot(rec, prec, color=color, linewidth=lw, alpha=0.8, label=label)

    prec_o, rec_o, _ = precision_recall_curve(y_true, oof_probs)
    oof_ap = average_precision_score(y_true, oof_probs)
    ax.plot(rec_o, prec_o, "--", color=OVERALL_COLOR, linewidth=2.5,
            label=f"OOF Overall (AP={oof_ap:.3f})")
    ax.axhline(baseline, color="gray", linestyle=":", linewidth=0.9,
               label=f"Baseline prevalence={baseline:.2f}")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — Per Fold + OOF Overall", fontsize=13)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)
    _save(fig, os.path.join(output_dir, "pr_curves.png"))


def plot_calibration(oof_probs, y_true, output_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    frac_pos, mean_pred = calibration_curve(y_true, oof_probs, n_bins=10)
    bs = brier_score_loss(y_true, oof_probs)
    ax.plot(mean_pred, frac_pos, "o-", color=OVERALL_COLOR, linewidth=2,
            label=f"Model (Brier={bs:.3f})")
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title("Calibration Plot (OOF)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    _save(fig, os.path.join(output_dir, "calibration.png"))


def plot_score_distribution(oof_probs, y_true, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 41)
    ax.hist(oof_probs[y_true == 0], bins=bins, alpha=0.65, color=PALETTE[0],
            label="Survived (y=0)", density=True)
    ax.hist(oof_probs[y_true == 1], bins=bins, alpha=0.65, color=PALETTE[3],
            label="Died (y=1)", density=True)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.2, label="Threshold=0.5")
    ax.set_xlabel("Predicted Mortality Probability", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Score Distribution by Outcome (OOF)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    _save(fig, os.path.join(output_dir, "score_distribution.png"))


def plot_fold_comparison(fold_results, oof_probs, y_true, best_fold, output_dir):
    folds    = [r["fold"] for r in fold_results]
    aucs     = [roc_auc_score(y_true[r["val_idx"]], oof_probs[r["val_idx"]])
                for r in fold_results]
    colors   = [BEST_COLOR if f == best_fold else PALETTE[i % len(PALETTE)]
                for i, f in enumerate(folds)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(folds)), aucs, color=colors, alpha=0.85, width=0.55)
    for bar, v in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    oof_auc = roc_auc_score(y_true, oof_probs)
    ax.axhline(oof_auc, color=OVERALL_COLOR, linestyle="--", linewidth=1.5,
               label=f"OOF Overall AUC={oof_auc:.4f}")

    ax.set_xticks(range(len(folds)))
    ax.set_xticklabels([f"Fold {f}" + (" ★" if f == best_fold else "")
                        for f in folds])
    ax.set_ylabel("Val AUC-ROC", fontsize=12)
    ax.set_title("Per-Fold Validation AUC (★ = best, used for inference)", fontsize=12)
    ax.set_ylim(max(0, min(aucs) - 0.05), 1.0)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, os.path.join(output_dir, "fold_comparison.png"))


def plot_confusion_matrix(oof_probs, y_true, output_dir, threshold=0.5):
    y_pred = (oof_probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Survived", "Died"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    tn, fp, fn, tp = cm.ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ax.set_title(f"Confusion Matrix (threshold={threshold})\n"
                 f"Sensitivity={sens:.3f}  Specificity={spec:.3f}", fontsize=11)
    _save(fig, os.path.join(output_dir, "confusion_matrix.png"))


def plot_training_history(fold_results, best_fold, output_dir):
    """Plot val AUC per epoch for each fold."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for res in fold_results:
        history = res.get("history", [])
        if not history:
            continue
        epochs = [h["epoch"] for h in history]
        aucs   = [h["auc"]   for h in history]
        is_best = res["fold"] == best_fold
        color   = BEST_COLOR if is_best else PALETTE[res["fold"] % len(PALETTE)]
        lw      = 2.5 if is_best else 1.2
        label   = (f"Fold {res['fold']}"
                   + (" ★ best" if is_best else ""))
        ax.plot(epochs, aucs, "o-", color=color, linewidth=lw,
                markersize=4, alpha=0.85, label=label)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Val AUC-ROC", fontsize=12)
    ax.set_title("Training History — Val AUC per Epoch", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    _save(fig, os.path.join(output_dir, "training_history.png"))


def run_evaluation(model_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    oof_path = os.path.join(model_dir, "oof_results.pkl")
    if not os.path.exists(oof_path):
        raise FileNotFoundError(
            f"{oof_path} not found. Run training first.")

    data         = joblib.load(oof_path)
    oof_probs    = data["oof_probs"]
    y_true       = data["y_true"]
    fold_results = data["fold_results"]
    best_fold    = data.get("best_fold", 1)

    # Attach per-fold OOF slice
    for res in fold_results:
        res["oof_slice"] = oof_probs[res["val_idx"]]

    logger.info(f"Generating plots → {output_dir}/  (best fold: {best_fold})")

    plot_roc_curves(fold_results, oof_probs, y_true, best_fold, output_dir)
    plot_pr_curves(fold_results, oof_probs, y_true, best_fold, output_dir)
    plot_calibration(oof_probs, y_true, output_dir)
    plot_score_distribution(oof_probs, y_true, output_dir)
    plot_fold_comparison(fold_results, oof_probs, y_true, best_fold, output_dir)
    plot_confusion_matrix(oof_probs, y_true, output_dir)
    plot_training_history(fold_results, best_fold, output_dir)

    oof_auc = roc_auc_score(y_true, oof_probs)
    oof_ap  = average_precision_score(y_true, oof_probs)
    logger.info(
        f"\n{'='*45}\n"
        f"  OOF AUC-ROC = {oof_auc:.4f}\n"
        f"  OOF AUC-PR  = {oof_ap:.4f}\n"
        f"  Best fold   = Fold {best_fold}\n"
        f"  Plots saved → {output_dir}/\n"
        f"{'='*45}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir",  default="models/")
    p.add_argument("--output-dir", default="results/")
    args = p.parse_args()
    run_evaluation(args.model_dir, args.output_dir)
