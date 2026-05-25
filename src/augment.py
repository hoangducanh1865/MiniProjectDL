"""
Data augmentation for ICU time-series data.

Strategies implemented:
  1. Gaussian jitter on time-series values
  2. Random temporal dropout (mask some observations)
  3. Interpolation-based synthetic minority oversampling

Only augments the minority class (target=1).
"""

import copy
import pickle
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

TIME_SERIES_FIELDS = [
    "heart_rate", "resp_rate", "temperature", "spo2",
    "pao2", "paco2", "pao2fio2ratio", "totalco2", "baseexcess", "ph", "lactatebg",
    "hematocrit", "hemoglobin", "platelet", "wbc", "rbc",
    "lymphocytes_abs", "monocytes_abs", "neutrophils_abs",
    "eosinophils_abs", "basophils_abs", "bands",
    "albumin", "aniongap", "bicarbonate", "creatinine", "bun", "glucose",
    "sodium", "potassium", "calcium", "magnesium", "chloride",
    "alt", "alp", "ast", "bilirubin_total", "ld_ldh",
    "inr", "pt", "ptt", "sofa", "sapsii",
    "urineoutput", "weight", "gcs_min", "gcs_motor", "gcs_verbal", "gcs_eyes",
]


def _jitter_patient(patient: Dict, noise_std: float = 0.02) -> Dict:
    """Add small Gaussian noise proportional to the signal range."""
    p = copy.deepcopy(patient)
    for field in TIME_SERIES_FIELDS:
        records = p.get(field)
        if not records:
            continue
        vals = [r["value"] for r in records if r.get("value") is not None]
        if not vals:
            continue
        sig_range = max(np.std(vals), 1e-6)
        for r in records:
            if r.get("value") is not None:
                r["value"] = r["value"] + np.random.normal(0, noise_std * sig_range)
    return p


def _temporal_dropout(patient: Dict, drop_prob: float = 0.15) -> Dict:
    """Randomly drop observations from each time-series."""
    p = copy.deepcopy(patient)
    for field in TIME_SERIES_FIELDS:
        records = p.get(field)
        if not records or len(records) <= 1:
            continue
        p[field] = [r for r in records if np.random.rand() > drop_prob]
        if not p[field]:
            p[field] = [records[0]]  # keep at least one
    return p


def _interpolate_patients(p1: Dict, p2: Dict, alpha: float = 0.5) -> Dict:
    """
    Create synthetic patient by interpolating static features between two patients.
    Time-series is taken from p1 (structural interpolation is complex).
    """
    p = copy.deepcopy(p1)
    scalar_fields = ["bmi", "duration", "age_at_admission", "lods"]
    for field in scalar_fields:
        v1 = p1.get(field)
        v2 = p2.get(field)
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            p[field] = alpha * v1 + (1 - alpha) * v2
    return p


def augment_data(train_data: Dict[str, Any], augment_factor: int = 2,
                 seed: int = 42) -> Dict[str, Any]:
    """
    Augment minority class (target=1) samples.

    augment_factor: how many synthetic copies per original positive sample.
    """
    np.random.seed(seed)
    augmented = dict(train_data)

    positives = {k: v for k, v in train_data.items() if v.get("target") == 1}
    pos_keys = list(positives.keys())
    logger.info(f"Augmenting {len(pos_keys)} positive samples x{augment_factor}")

    new_id_counter = 0
    for _ in range(augment_factor):
        for pid in pos_keys:
            patient = positives[pid]
            method = np.random.choice(["jitter", "dropout", "interpolate"])

            if method == "jitter":
                new_patient = _jitter_patient(patient)
            elif method == "dropout":
                new_patient = _temporal_dropout(patient)
            else:  # interpolate
                other_pid = np.random.choice(pos_keys)
                new_patient = _interpolate_patients(patient, positives[other_pid],
                                                     alpha=np.random.uniform(0.3, 0.7))
            new_patient["target"] = 1
            new_id = f"aug_{new_id_counter:06d}"
            augmented[new_id] = new_patient
            new_id_counter += 1

    logger.info(f"Dataset size: {len(train_data)} → {len(augmented)}")
    return augmented


def run_augmentation(train_pkl: str, output_pkl: str,
                     augment_factor: int = 2, seed: int = 42):
    with open(train_pkl, "rb") as f:
        train_data = pickle.load(f)
    augmented = augment_data(train_data, augment_factor=augment_factor, seed=seed)
    with open(output_pkl, "wb") as f:
        pickle.dump(augmented, f)
    logger.info(f"Saved augmented data to {output_pkl}")
