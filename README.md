# ICU Mortality Prediction — Mini Project DL

Dự đoán tử vong bệnh nhân ICU từ dữ liệu lâm sàng MIMIC-IV.

**Kiến trúc:** Ensemble Deep Residual MLP + CatBoost, 5-fold cross-validation.

---

## Setup

```bash
conda env create -f environment.yml
conda activate mini_project_dl
```

## Đặt dữ liệu

```
MiniProjectDL/
└── data/
    ├── train.pkl
    └── test.pkl
```

## Chạy

```bash
# Train + predict + evaluation plots (mặc định)
python main.py

# Với data augmentation
python main.py --augment

# Chỉ tạo lại plots (đã có model)
python main.py --evaluate-only

# Chỉ inference, bỏ qua train
python main.py --skip-train
```

## Output

| Path | Nội dung |
|---|---|
| `group1.csv` | Predictions: id, probability, prediction |
| `models/` | Saved model checkpoints |
| `results/roc_curves.png` | ROC curve per fold + ensemble |
| `results/pr_curves.png` | Precision-Recall curves |
| `results/calibration.png` | Reliability diagram |
| `results/score_distribution.png` | Score histogram by class |
| `results/fold_comparison.png` | AUC bar chart per fold |
| `results/confusion_matrix.png` | Confusion matrix at threshold=0.5 |

## Tùy chọn

| Flag | Default | Mô tả |
|---|---|---|
| `--folds` | 5 | Số fold CV |
| `--epochs` | 80 | Epoch MLP |
| `--seed` | 42 | Random seed |
| `--augment-factor` | 2 | Hệ số augment minority class |
| `--results-dir` | `results/` | Thư mục lưu PNG |
