import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import random
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ────────────────────────────────────────────────
# Feature definitions
# ────────────────────────────────────────────────

TIME_SERIES_FEATURES = [
    "heart_rate", "resp_rate", "temperature", "spo2",
    "pao2", "paco2", "pao2fio2ratio", "totalco2", "baseexcess", "ph", "lactatebg",
    "hematocrit", "hemoglobin", "platelet", "wbc", "rbc",
    "lymphocytes_abs", "monocytes_abs", "neutrophils_abs", "eosinophils_abs",
    "basophils_abs", "bands",
    "albumin", "aniongap", "bicarbonate", "creatinine", "bun", "glucose",
    "sodium", "potassium", "calcium", "magnesium", "chloride",
    "alt", "alp", "ast", "bilirubin_total", "ld_ldh",
    "inr", "pt", "ptt",
    "sofa", "sapsii",
    "urineoutput",
    "weight", "weight_admit", "weight_min", "weight_max",
    "gcs_min", "gcs_motor", "gcs_verbal", "gcs_eyes",
]

STATIC_BINARY_FEATURES = [
    "sepsis", "hepatitis", "liver_cirrhosis", "ventricular_arrhythmia",
    "atrial_fibrillation", "pneumonia", "ami", "omi", "pleural_effusion",
    "valvular_disease", "hypertension", "anemia", "ckd", "stroke", "copd",
    "diabetes", "aki", "pe", "angina_pectoris",
]

SCALAR_FEATURES = ["bmi", "duration", "age_at_admission", "lods"]


def extract_ts_stats(records: Optional[List[Dict]]) -> Dict[str, float]:
    if not records:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan,
                "first": np.nan, "last": np.nan, "count": 0.0, "trend": np.nan}
    values = [r["value"] for r in records if r.get("value") is not None]
    if not values:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan,
                "first": np.nan, "last": np.nan, "count": 0.0, "trend": np.nan}
    arr = np.array(values, dtype=float)
    trend = (arr[-1] - arr[0]) / max(len(arr) - 1, 1)
    return {
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)) if len(arr) > 1 else 0.0,
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "first": float(arr[0]),
        "last": float(arr[-1]),
        "count": float(len(arr)),
        "trend": float(trend),
    }


def extract_features(patient: Dict[str, Any]) -> Dict[str, float]:
    row: Dict[str, float] = {}

    # Time-series → statistics
    for feat in TIME_SERIES_FEATURES:
        stats = extract_ts_stats(patient.get(feat))
        for stat_name, val in stats.items():
            row[f"{feat}__{stat_name}"] = val

    # Static binary
    for feat in STATIC_BINARY_FEATURES:
        row[feat] = float(patient.get(feat, 0))

    # Scalar
    for feat in SCALAR_FEATURES:
        val = patient.get(feat, np.nan)
        if isinstance(val, list):
            val = val[0]["value"] if val else np.nan
        row[feat] = float(val) if val is not None else np.nan

    # Gender encoding
    gender = patient.get("gender", "")
    row["gender_male"] = 1.0 if str(gender).upper() == "M" else 0.0

    # Race encoding (top categories)
    race = str(patient.get("race", "UNKNOWN")).upper()
    for cat in ["WHITE", "BLACK", "HISPANIC", "ASIAN", "UNKNOWN"]:
        row[f"race_{cat}"] = 1.0 if cat in race else 0.0

    return row


def build_feature_matrix(data: Dict[str, Any], has_target: bool = True):
    rows = []
    ids = []
    targets = []

    for pid, patient in data.items():
        row = extract_features(patient)
        rows.append(row)
        ids.append(pid)
        if has_target:
            targets.append(int(patient.get("target", 0)))

    df = pd.DataFrame(rows, index=ids)

    # Drop columns with >50% missing
    thresh = int(0.5 * len(df))
    df = df.dropna(axis=1, thresh=thresh)

    if has_target:
        return df, np.array(targets)
    return df
