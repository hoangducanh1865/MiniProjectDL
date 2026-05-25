"""
evaluate.py — Generate evaluation plots from OOF results.

Plots saved to results/ (or --output-dir):
  1. roc_curves.png        — ROC curve per fold + ensemble
  2. pr_curves.png         — Precision-Recall curve per fold + ensemble
  3. calibration.png       — Calibration plot (reliability diagram)
  4. score_distribution.png — Predicted probability distribution by class
  5. fold_comparison.png   — AUC bar chart: MLP vs CatBoost vs Ensemble per fold
  6. confusion_matrix.png  — Confusion matrix at threshold=0.5

Usage (standalone):
    python evaluate.py [--model-dir models/] [--output-dir results/]
"""

import os
import argparse
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
import logging

logger = logging.getLogger(__name__)

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
ENSEMBLE_COLOR = "#2d2d2d"


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {path}")


# ── 1. ROC curves ────────────────────────────────────────────────────────────

def plot_roc_curves(fold_results, oof_ensemble, y_true, output_dir):
    fig, ax = plt.subplots(figsize=(7, 6))

    for i, res in enumerate(fold_results):
        val_idx = res["val_idx"]
        y_v = y_true[val_idx]

        for label, probs, ls in [
            ("MLP", res.get("oof_mlp", oof_ensemble[val_idx]), "--"),
            ("CB",  res.get("oof_cb",  oof_ensemble[val_idx]), ":"),
        ]:
            fpr, tpr, _ = roc_curve(y_v, probs)
            fold_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, ls, color=PALETTE[i % len(PALETTE)],
                    alpha=0.5, linewidth=1,
                    label=f"Fold {res['fold']} {label} (AUC={fold_auc:.3f})")

    fpr_e, tpr_e, _ = roc_curve(y_true, oof_ensemble)
    ens_auc = auc(fpr_e, tpr_e)
    ax.plot(fpr_e, tpr_e, "-", color=ENSEMBLE_COLOR, linewidth=2.5,
            label=f"OOF Ensemble (AUC={ens_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4)

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — OOF per Fold + Ensemble", fontsize=13)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.3)
    _save(fig, os.path.join(output_dir, "roc_curves.png"))


# ── 2. Precision-Recall curves ───────────────────────────────────────────────

def plot_pr_curves(fold_results, oof_ensemble, y_true, output_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    baseline = y_true.mean()

    for i, res in enumerate(fold_results):
        val_idx = res["val_idx"]
        y_v = y_true[val_idx]
        probs = res.get("oof_mlp", oof_ensemble[val_idx])
        prec, rec, _ = precision_recall_curve(y_v, probs)
        ap = average_precision_score(y_v, probs)
        ax.plot(rec, prec, "-", color=PALETTE[i % len(PALETTE)],
                alpha=0.55, linewidth=1,
                label=f"Fold {res['fold']} MLP (AP={ap:.3f})")

    prec_e, rec_e, _ = precision_recall_curve(y_true, oof_ensemble)
    ens_ap = average_precision_score(y_true, oof_ensemble)
    ax.plot(rec_e, prec_e, "-", color=ENSEMBLE_COLOR, linewidth=2.5,
            label=f"OOF Ensemble (AP={ens_ap:.3f})")
    ax.axhline(baseline, color="gray", linestyle="--", linewidth=0.9,
               label=f"Baseline (prevalence={baseline:.2f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — OOF per Fold + Ensemble", fontsize=13)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    _save(fig, os.path.join(output_dir, "pr_curves.png"))


# ── 3. Calibration plot ──────────────────────────────────────────────────────

def plot_calibration(oof_probs_mlp, oof_probs_cb, oof_ensemble, y_true, output_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")

    for label, probs, color in [
        ("MLP",      oof_probs_mlp, PALETTE[0]),
        ("CatBoost", oof_probs_cb,  PALETTE[1]),
        ("Ensemble", oof_ensemble,  ENSEMBLE_COLOR),
    ]:
        frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10)
        bs = brier_score_loss(y_true, probs)
        lw = 2.5 if label == "Ensemble" else 1.5
        ax.plot(mean_pred, frac_pos, "o-", color=color, linewidth=lw,
                label=f"{label} (Brier={bs:.3f})")

    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title("Calibration Plot (Reliability Diagram)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    _save(fig, os.path.join(output_dir, "calibration.png"))


# ── 4. Score distribution ────────────────────────────────────────────────────

def plot_score_distribution(oof_ensemble, y_true, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 41)

    ax.hist(oof_ensemble[y_true == 0], bins=bins, alpha=0.65, color=PALETTE[0],
            label="Survived (y=0)", density=True)
    ax.hist(oof_ensemble[y_true == 1], bins=bins, alpha=0.65, color=PALETTE[3],
            label="Died (y=1)", density=True)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.2, label="Threshold=0.5")

    ax.set_xlabel("Predicted Mortality Probability", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Ensemble Score Distribution by Outcome", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    _save(fig, os.path.join(output_dir, "score_distribution.png"))


# ── 5. Fold comparison bar chart ─────────────────────────────────────────────

def plot_fold_comparison(fold_results, oof_probs_mlp, oof_probs_cb,
                         oof_ensemble, y_true, output_dir):
    folds = [r["fold"] for r in fold_results]
    mlp_aucs = [r["mlp_auc"] for r in fold_results]
    cb_aucs  = [r["cb_auc"]  for r in fold_results]

    # Per-fold ensemble AUC using OOF slice
    ens_aucs = []
    for r in fold_results:
        vi = r["val_idx"]
        ens_fold = 0.5 * oof_probs_mlp[vi] + 0.5 * oof_probs_cb[vi]
        ens_aucs.append(roc_auc_score(y_true[vi], ens_fold))

    x = np.arange(len(folds))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width, mlp_aucs, width, label="MLP",      color=PALETTE[0], alpha=0.85)
    b2 = ax.bar(x,          cb_aucs,  width, label="CatBoost", color=PALETTE[1], alpha=0.85)
    b3 = ax.bar(x + width,  ens_aucs, width, label="Ensemble", color=ENSEMBLE_COLOR, alpha=0.85)

    # Overall OOF lines
    overall_mlp = roc_auc_score(y_true, oof_probs_mlp)
    overall_cb  = roc_auc_score(y_true, oof_probs_cb)
    overall_ens = roc_auc_score(y_true, oof_ensemble)
    ax.axhline(overall_mlp, color=PALETTE[0],    linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(overall_cb,  color=PALETTE[1],    linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(overall_ens, color=ENSEMBLE_COLOR, linestyle="--", linewidth=1.5)

    for bars in [b1, b2, b3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.set_ylabel("AUC-ROC", fontsize=12)
    ax.set_title("Per-Fold AUC: MLP vs CatBoost vs Ensemble\n"
                 f"(Overall OOF — MLP: {overall_mlp:.3f}  CB: {overall_cb:.3f}  "
                 f"Ens: {overall_ens:.3f})", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(max(0, min(mlp_aucs + cb_aucs + ens_aucs) - 0.05), 1.0)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, os.path.join(output_dir, "fold_comparison.png"))


# ── 6. Confusion matrix ──────────────────────────────────────────────────────

def plot_confusion_matrix(oof_ensemble, y_true, output_dir, threshold=0.5):
    y_pred = (oof_ensemble >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Survived", "Died"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")

    tn, fp, fn, tp = cm.ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ax.set_title(
        f"Confusion Matrix (threshold={threshold})\n"
        f"Sensitivity={sens:.3f}  Specificity={spec:.3f}",
        fontsize=11,
    )
    _save(fig, os.path.join(output_dir, "confusion_matrix.png"))


# ── Master function ──────────────────────────────────────────────────────────

def run_evaluation(model_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    oof_path = os.path.join(model_dir, "oof_results.pkl")
    if not os.path.exists(oof_path):
        raise FileNotFoundError(
            f"{oof_path} not found. Run training first (python main.py).")

    data = joblib.load(oof_path)
    oof_probs_mlp = data["oof_probs_mlp"]
    oof_probs_cb  = data["oof_probs_cb"]
    oof_ensemble  = data["oof_ensemble"]
    y_true        = data["y_true"]
    fold_results  = data["fold_results"]

    # Attach per-fold OOF slices for per-fold plots
    for res in fold_results:
        vi = res["val_idx"]
        res["oof_mlp"] = oof_probs_mlp[vi]
        res["oof_cb"]  = oof_probs_cb[vi]

    logger.info(f"Generating evaluation plots → {output_dir}/")

    plot_roc_curves(fold_results, oof_ensemble, y_true, output_dir)
    plot_pr_curves(fold_results, oof_ensemble, y_true, output_dir)
    plot_calibration(oof_probs_mlp, oof_probs_cb, oof_ensemble, y_true, output_dir)
    plot_score_distribution(oof_ensemble, y_true, output_dir)
    plot_fold_comparison(fold_results, oof_probs_mlp, oof_probs_cb,
                         oof_ensemble, y_true, output_dir)
    plot_confusion_matrix(oof_ensemble, y_true, output_dir)

    # Print summary
    ens_auc = roc_auc_score(y_true, oof_ensemble)
    ens_ap  = average_precision_score(y_true, oof_ensemble)
    logger.info(
        f"\n{'='*45}\n"
        f"  OOF Ensemble  AUC-ROC = {ens_auc:.4f}\n"
        f"  OOF Ensemble  AUC-PR  = {ens_ap:.4f}\n"
        f"  Plots saved to: {output_dir}/\n"
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
