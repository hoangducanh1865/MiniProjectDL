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

LOG_FEATURE_PREFIXES = (
    "lactatebg__", "creatinine__", "bun__", "glucose__", "wbc__",
    "platelet__", "aniongap__", "bilirubin_total__", "alt__", "ast__",
    "alp__", "ld_ldh__", "inr__", "pt__", "ptt__", "urineoutput__",
    "sapsii__", "sofa__", "duration", "lods", "bmi",
)


def _safe_ratio(num, den):
    if pd.isna(num) or pd.isna(den) or abs(den) < 1e-8:
        return np.nan
    return float(num) / float(den)


def _flag_gt(series: pd.Series, threshold: float) -> pd.Series:
    return np.where(series.isna(), np.nan, (series > threshold).astype(float))


def _flag_lt(series: pd.Series, threshold: float) -> pd.Series:
    return np.where(series.isna(), np.nan, (series < threshold).astype(float))


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
    df = add_engineered_features(df)

    # Drop columns with >50% missing
    thresh = int(0.5 * len(df))
    df = df.dropna(axis=1, thresh=thresh)

    if has_target:
        return df, np.array(targets)
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Clinically motivated interactions. Missing values are kept as NaN so the
    # fold-local imputer and missingness indicators can handle them safely.
    specs = {
        "bun_creatinine_ratio": ("bun__last", "creatinine__last"),
        "aniongap_bicarbonate_ratio": ("aniongap__last", "bicarbonate__last"),
        "neutrophil_lymphocyte_ratio": ("neutrophils_abs__last", "lymphocytes_abs__last"),
        "pao2_spo2_ratio": ("pao2__last", "spo2__last"),
        "pao2fio2_lactate_ratio": ("pao2fio2ratio__last", "lactatebg__last"),
        "urineoutput_duration_ratio": ("urineoutput__mean", "duration"),
        "sofa_sapsii_sum": ("sofa__mean", "sapsii__mean"),
        "kidney_stress": ("bun__last", "urineoutput__mean"),
        "oxygenation_stress": ("lactatebg__last", "pao2fio2ratio__last"),
        "coagulation_stress": ("inr__last", "platelet__last"),
    }

    for name, (a, b) in specs.items():
        if a not in df.columns or b not in df.columns:
            continue
        if name in {
            "sofa_sapsii_sum",
            "kidney_stress",
            "oxygenation_stress",
            "coagulation_stress",
        }:
            if name == "sofa_sapsii_sum":
                df[name] = df[a] + df[b]
            else:
                df[name] = df[a] * df[b]
        else:
            df[name] = [
                _safe_ratio(num, den) for num, den in zip(df[a].values, df[b].values)
            ]

    # Encode measurement intensity across organ systems. Count features often
    # capture acuity/care-intensity signals that neural nets use well.
    count_cols = [c for c in df.columns if c.endswith("__count")]
    if count_cols:
        count_df = df[count_cols]
        df["measurement_count_total"] = count_df.sum(axis=1, skipna=True)
        df["measurement_count_nonzero"] = (count_df.fillna(0) > 0).sum(axis=1)
        df["measurement_count_max"] = count_df.max(axis=1, skipna=True)

    # Threshold features give the neural net crisp clinical cut-points while
    # keeping the original continuous variables available.
    flags = {}
    high_specs = {
        "age_gt_65": ("age_at_admission", 65),
        "age_gt_80": ("age_at_admission", 80),
        "duration_gt_48h": ("duration", 48),
        "bmi_gt_30": ("bmi", 30),
        "lactate_gt_2": ("lactatebg__last", 2.0),
        "lactate_gt_4": ("lactatebg__last", 4.0),
        "creatinine_gt_2": ("creatinine__last", 2.0),
        "bun_gt_40": ("bun__last", 40.0),
        "aniongap_gt_16": ("aniongap__last", 16.0),
        "wbc_gt_12": ("wbc__last", 12.0),
        "inr_gt_15": ("inr__last", 1.5),
        "sofa_gt_6": ("sofa__mean", 6.0),
        "sapsii_gt_50": ("sapsii__mean", 50.0),
        "lods_gt_8": ("lods", 8.0),
        "heart_rate_gt_110": ("heart_rate__last", 110.0),
        "resp_rate_gt_24": ("resp_rate__last", 24.0),
    }
    low_specs = {
        "spo2_lt_92": ("spo2__last", 92.0),
        "pao2fio2_lt_200": ("pao2fio2ratio__last", 200.0),
        "pao2fio2_lt_100": ("pao2fio2ratio__last", 100.0),
        "ph_lt_735": ("ph__last", 7.35),
        "bicarbonate_lt_22": ("bicarbonate__last", 22.0),
        "platelet_lt_150": ("platelet__last", 150.0),
        "hemoglobin_lt_10": ("hemoglobin__last", 10.0),
        "albumin_lt_35": ("albumin__last", 3.5),
        "urineoutput_lt_500": ("urineoutput__mean", 500.0),
        "gcs_min_lt_13": ("gcs_min__mean", 13.0),
    }
    for name, (col, threshold) in high_specs.items():
        if col in df.columns:
            flags[name] = _flag_gt(df[col], threshold)
    for name, (col, threshold) in low_specs.items():
        if col in df.columns:
            flags[name] = _flag_lt(df[col], threshold)

    if flags:
        flag_df = pd.DataFrame(flags, index=df.index)
        df = pd.concat([df, flag_df], axis=1)
        respiratory_flags = [
            c for c in ["spo2_lt_92", "pao2fio2_lt_200", "pao2fio2_lt_100", "resp_rate_gt_24"]
            if c in flag_df.columns
        ]
        renal_flags = [
            c for c in ["creatinine_gt_2", "bun_gt_40", "urineoutput_lt_500"]
            if c in flag_df.columns
        ]
        global_flags = [
            c for c in ["age_gt_80", "lactate_gt_4", "sofa_gt_6", "sapsii_gt_50", "lods_gt_8"]
            if c in flag_df.columns
        ]
        if respiratory_flags:
            df["respiratory_risk_flags"] = flag_df[respiratory_flags].sum(axis=1, skipna=True)
        if renal_flags:
            df["renal_risk_flags"] = flag_df[renal_flags].sum(axis=1, skipna=True)
        if global_flags:
            df["global_risk_flags"] = flag_df[global_flags].sum(axis=1, skipna=True)

    # Add selected log1p transforms for skewed non-negative clinical variables.
    log_features = {}
    for col in list(df.columns):
        if not col.startswith(LOG_FEATURE_PREFIXES):
            continue
        series = df[col]
        finite = series[np.isfinite(series)]
        if finite.empty or finite.min() < 0:
            continue
        log_features[f"{col}__log1p"] = np.log1p(series)

    if log_features:
        df = pd.concat([df, pd.DataFrame(log_features, index=df.index)], axis=1)

    return df
