# ICU Mortality Prediction — Mini Project DL

Dự đoán tử vong bệnh nhân ICU từ dữ liệu lâm sàng MIMIC-IV.

**Kiến trúc:** Deep Residual MLP, huấn luyện qua k-fold cross-validation. Model tốt nhất (theo val AUC) được dùng để inference.

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

# Chỉ tạo lại plots (đã train xong)
python main.py --evaluate-only

# Chỉ inference với best saved model, bỏ qua train
python main.py --skip-train
```

Mỗi lần chạy sẽ tạo một thư mục kết quả riêng:

```
results/YYYY_MM_DD_HH_MM_SS/
├── group1.csv
├── models/
├── *.png
└── *_metrics.csv
```

Chạy ablation deep-learning-only:

```bash
python ablation.py
```

Smoke test ablation nhanh:

```bash
python ablation.py --fast --variants improved_regularized focal_loss
```

## Output

| Path | Nội dung |
|---|---|
| `group1.csv` | Predictions: id, probability, prediction |
| `models/model_fold{k}.pt` | Weights MLP fold k |
| `models/preprocessor_fold{k}.pkl` | Imputer + scaler fold k |
| `models/best_fold.pkl` | Thông tin fold tốt nhất |
| `models/oof_results.pkl` | OOF predictions (dùng để vẽ plots) |
| `results/roc_curves.png` | ROC curve per fold + OOF overall |
| `results/pr_curves.png` | Precision-Recall curves |
| `results/calibration.png` | Reliability diagram |
| `results/score_distribution.png` | Score histogram by class |
| `results/fold_comparison.png` | AUC bar chart per fold |
| `results/confusion_matrix.png` | Confusion matrix at threshold=0.5 |
| `results/training_history.png` | Val AUC per epoch per fold |

## Tất cả hyperparameter

| Argument | Default | Mô tả |
|---|---|---|
| `--train` | `data/train.pkl` | Đường dẫn train data |
| `--test` | `data/test.pkl` | Đường dẫn test data |
| `--output` | `group1.csv` | File CSV output |
| `--model-dir` | `models/` | Thư mục lưu model |
| `--results-dir` | `results/` | Thư mục lưu plots |
| `--k-fold` | `5` | Số fold cross-validation |
| `--seed` | `42` | Random seed |
| `--hidden-dims` | `512 256 128` | Kích thước các hidden layer |
| `--dropout` | `0.3` | Dropout rate |
| `--epochs` | `80` | Số epoch train |
| `--batch-size` | `256` | Batch size |
| `--lr` | `1e-3` | Learning rate |
| `--weight-decay` | `1e-4` | L2 regularization |
| `--label-smoothing` | `0.05` | Label smoothing |
| `--scheduler` | `cosine` | LR scheduler: `cosine` / `step` / `none` |
| `--step-size` | `20` | StepLR step size (dùng khi `--scheduler=step`) |
| `--gamma` | `0.5` | StepLR gamma (dùng khi `--scheduler=step`) |
| `--augment` | off | Bật data augmentation |
| `--augment-factor` | `2` | Hệ số augment minority class |

## Ví dụ

```bash
# Tùy chỉnh kiến trúc và training
python main.py --k-fold 3 --epochs 100 --lr 5e-4 \
               --hidden-dims 1024 512 256 128 --dropout 0.2

# Train với augmentation, scheduler step
python main.py --augment --augment-factor 3 \
               --scheduler step --step-size 20 --gamma 0.5
```
