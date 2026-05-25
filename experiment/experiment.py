#!/usr/bin/env python3
from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path
import sys
from datetime import datetime

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import STATIC_BINARY_FEATURES, TIME_SERIES_FEATURES, SCALAR_FEATURES, build_feature_matrix


PLACEHOLDER_TIME_SERIES = {
    "sofa": 0.0,
    "sapsii": 0.0,
    "weight": 0.0,
    "weight_admit": 0.0,
    "weight_min": 0.0,
    "weight_max": 0.0,
    "gcs_min": 0.0,
    "gcs_motor": 0.0,
    "gcs_verbal": 0.0,
    "gcs_eyes": 0.0,
}

DIAG_KEYWORDS = {
    "sepsis": ["sepsis"],
    "hepatitis": ["hepatitis"],
    "liver_cirrhosis": ["cirrhosis", "liver cirrhosis"],
    "ventricular_arrhythmia": ["ventricular arrhythmia"],
    "atrial_fibrillation": ["atrial fibrillation"],
    "pneumonia": ["pneumonia"],
    "ami": ["myocardial infarction", "heart attack", "ami"],
    "omi": ["old myocardial infarction", "old mi", "omi"],
    "pleural_effusion": ["pleural effusion"],
    "valvular_disease": ["valvular", "valve disease"],
    "hypertension": ["hypertension"],
    "anemia": ["anemia"],
    "ckd": ["chronic kidney disease", "ckd"],
    "stroke": ["stroke", "cva"],
    "copd": ["copd", "chronic obstructive pulmonary"],
    "diabetes": ["diabetes"],
    "aki": ["acute kidney injury", "aki"],
    "pe": ["pulmonary embol"],
    "angina_pectoris": ["angina"],
}

FEATURE_SPECS = {
    "heart_rate": ("chartevents", ["heart rate", "hr"]),
    "resp_rate": ("chartevents", ["respiratory rate", "resp rate"]),
    "temperature": ("chartevents", ["temperature", "temp"]),
    "spo2": ("chartevents", ["spo2", "o2 saturation", "oxygen saturation"]),
    "pao2": ("labevents", ["pao2", "po2"]),
    "paco2": ("labevents", ["paco2", "pco2"]),
    "pao2fio2ratio": ("labevents", ["pao2fio2ratio", "pao2/fio2", "p/f ratio"]),
    "totalco2": ("labevents", ["total co2", "tco2", "co2"]),
    "baseexcess": ("labevents", ["base excess", "baseexcess"]),
    "ph": ("labevents", ["ph"]),
    "lactatebg": ("labevents", ["lactate"]),
    "hematocrit": ("labevents", ["hematocrit"]),
    "hemoglobin": ("labevents", ["hemoglobin"]),
    "platelet": ("labevents", ["platelet"]),
    "wbc": ("labevents", ["white blood cell", "wbc"]),
    "rbc": ("labevents", ["red blood cell", "rbc"]),
    "lymphocytes_abs": ("labevents", ["lymphocyte"]),
    "monocytes_abs": ("labevents", ["monocyte"]),
    "neutrophils_abs": ("labevents", ["neutrophil"]),
    "eosinophils_abs": ("labevents", ["eosinophil"]),
    "basophils_abs": ("labevents", ["basophil"]),
    "bands": ("labevents", ["band"]),
    "albumin": ("labevents", ["albumin"]),
    "aniongap": ("labevents", ["anion gap"]),
    "bicarbonate": ("labevents", ["bicarbonate", "hco3"]),
    "creatinine": ("labevents", ["creatinine"]),
    "bun": ("labevents", ["bun", "urea nitrogen"]),
    "glucose": ("labevents", ["glucose"]),
    "sodium": ("labevents", ["sodium"]),
    "potassium": ("labevents", ["potassium"]),
    "calcium": ("labevents", ["calcium"]),
    "magnesium": ("labevents", ["magnesium"]),
    "chloride": ("labevents", ["chloride"]),
    "alt": ("labevents", ["alanine aminotransferase", "alt"]),
    "alp": ("labevents", ["alkaline phosphatase", "alp"]),
    "ast": ("labevents", ["asparate aminotransferase", "aspartate aminotransferase", "ast"]),
    "bilirubin_total": ("labevents", ["bilirubin"]),
    "ld_ldh": ("labevents", ["lactate dehydrogenase", "ldh"]),
    "inr": ("labevents", ["inr"]),
    "pt": ("labevents", ["prothrombin time", "pt(", "pt "]),
    "ptt": ("labevents", ["partial thromboplastin time", "ptt"]),
    "urineoutput": ("outputevents", ["urine"]),
    "specimen": ("chartevents", ["specimen"]),
    "gcs_min": ("chartevents", ["gcs"]),
    "gcs_motor": ("chartevents", ["gcs motor"]),
    "gcs_verbal": ("chartevents", ["gcs verbal"]),
    "gcs_eyes": ("chartevents", ["gcs eyes"]),
}


def parse_time(value):
    if pd.isna(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime().strftime("%Y-%m-%d %H:%M:%S")


def build_series(rows: pd.DataFrame, time_col: str, value_col: str, as_string: bool = False, limit: int | None = None):
    if rows.empty:
        return []
    rows = rows.sort_values(time_col)
    if limit is not None:
        rows = rows.head(limit)
    series = []
    for _, row in rows.iterrows():
        ts = parse_time(row.get(time_col))
        if ts is None:
            continue
        value = row.get(value_col)
        if as_string:
            if pd.isna(value):
                continue
            series.append({"charttime": ts, "value": str(value)})
            continue
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            continue
        series.append({"charttime": ts, "value": float(numeric)})
    return series


def add_placeholder_series(patient: dict, key: str, base_time: str):
    if key == "specimen":
        patient[key] = [{"charttime": base_time, "value": "UNK"}]
    else:
        patient[key] = [{"charttime": base_time, "value": float(PLACEHOLDER_TIME_SERIES.get(key, 0.0))}]


def build_patient_subset() -> dict:
    subject_ids = pd.read_csv(DEMO / "demo_subject_id.csv")["subject_id"].astype(str).tolist()

    patients = pd.read_csv(DEMO / "hosp" / "patients.csv")
    admissions = pd.read_csv(DEMO / "hosp" / "admissions.csv")
    icustays = pd.read_csv(DEMO / "icu" / "icustays.csv")
    diagnoses = pd.read_csv(DEMO / "hosp" / "diagnoses_icd.csv")
    d_icd = pd.read_csv(DEMO / "hosp" / "d_icd_diagnoses.csv")
    chartevents = pd.read_csv(DEMO / "icu" / "chartevents.csv")
    d_items = pd.read_csv(DEMO / "icu" / "d_items.csv")
    labevents = pd.read_csv(DEMO / "hosp" / "labevents.csv")
    d_labitems = pd.read_csv(DEMO / "hosp" / "d_labitems.csv")
    outputevents = pd.read_csv(DEMO / "icu" / "outputevents.csv")
    omr = pd.read_csv(DEMO / "hosp" / "omr.csv")

    patients["subject_id"] = patients["subject_id"].astype(str)
    admissions["subject_id"] = admissions["subject_id"].astype(str)
    icustays["subject_id"] = icustays["subject_id"].astype(str)
    diagnoses["subject_id"] = diagnoses["subject_id"].astype(str)
    chartevents["subject_id"] = chartevents["subject_id"].astype(str)
    labevents["subject_id"] = labevents["subject_id"].astype(str)
    outputevents["subject_id"] = outputevents["subject_id"].astype(str)
    omr["subject_id"] = omr["subject_id"].astype(str)

    admissions = admissions.sort_values(["subject_id", "admittime"])
    icustays = icustays.sort_values(["subject_id", "intime"])

    diagnoses = diagnoses.merge(d_icd[["icd_code", "long_title"]], on="icd_code", how="left")
    chartevents = chartevents.merge(d_items[["itemid", "label"]], on="itemid", how="left")
    labevents = labevents.merge(d_labitems[["itemid", "label"]], on="itemid", how="left")
    outputevents = outputevents.merge(d_items[["itemid", "label"]], on="itemid", how="left")

    result = {}
    for subject_id in subject_ids:
        patient_row = patients[patients["subject_id"] == subject_id].head(1)
        admission_row = admissions[admissions["subject_id"] == subject_id].head(1)
        icu_row = icustays[icustays["subject_id"] == subject_id].head(1)

        if patient_row.empty or admission_row.empty or icu_row.empty:
            continue

        patient_row = patient_row.iloc[0]
        admission_row = admission_row.iloc[0]
        icu_row = icu_row.iloc[0]

        admittime = pd.to_datetime(admission_row["admittime"], errors="coerce")
        intime = pd.to_datetime(icu_row["intime"], errors="coerce")
        base_time = parse_time(admittime) or "1970-01-01 00:00:00"

        patient = {}
        patient["gender"] = str(patient_row.get("gender", ""))
        patient["race"] = str(admission_row.get("race", "UNKNOWN"))
        patient["target"] = int(admission_row.get("hospital_expire_flag", 0))

        age_at_admission = int(patient_row.get("anchor_age", 0))
        anchor_year = patient_row.get("anchor_year")
        if not pd.isna(anchor_year) and not pd.isna(admittime):
            age_at_admission = int(patient_row.get("anchor_age", 0)) + int(admittime.year - int(anchor_year))
        patient["age_at_admission"] = age_at_admission
        patient["duration"] = float(icu_row.get("los", 0.0))

        bmi_row = omr[omr["subject_id"] == subject_id]
        bmi = np.nan
        if not bmi_row.empty:
            height = pd.to_numeric(bmi_row[bmi_row["result_name"].astype(str).str.contains("height", case=False, na=False)]["result_value"], errors="coerce")
            weight = pd.to_numeric(bmi_row[bmi_row["result_name"].astype(str).str.contains("weight", case=False, na=False)]["result_value"], errors="coerce")
            if not height.empty and not weight.empty:
                h = float(height.dropna().iloc[0])
                w = float(weight.dropna().iloc[0])
                if h > 0 and w > 0:
                    if h > 3:  # likely inches
                        h = h * 0.0254
                    bmi = w / (h * h)
        if pd.isna(bmi):
            bmi = 0.0
        patient["bmi"] = float(bmi)
        patient["lods"] = int(0)

        diag_text = " ".join(diagnoses.loc[diagnoses["subject_id"] == subject_id, "long_title"].dropna().astype(str).tolist()).lower()
        for feat, kws in DIAG_KEYWORDS.items():
            patient[feat] = 1 if any(kw in diag_text for kw in kws) else 0

        time_tables = {
            "chartevents": chartevents[chartevents["subject_id"] == subject_id].copy(),
            "labevents": labevents[labevents["subject_id"] == subject_id].copy(),
            "outputevents": outputevents[outputevents["subject_id"] == subject_id].copy(),
        }

        for feat in TIME_SERIES_FEATURES:
            source, keywords = FEATURE_SPECS.get(feat, (None, []))
            if source is None:
                add_placeholder_series(patient, feat, base_time)
                continue
            df = time_tables[source]
            if df.empty:
                add_placeholder_series(patient, feat, base_time)
                continue

            labels = df["label"].astype(str).str.lower()
            mask = pd.Series(False, index=df.index)
            for kw in keywords:
                mask = mask | labels.str.contains(kw.lower(), na=False, regex=False)

            matched = df[mask].copy()
            if feat == "specimen":
                series = build_series(matched, "charttime" if "charttime" in matched.columns else matched.columns[0], "value", as_string=True, limit=10)
            else:
                time_col = "charttime" if "charttime" in matched.columns else ("storetime" if "storetime" in matched.columns else matched.columns[0])
                value_col = "valuenum" if "valuenum" in matched.columns else "value"
                series = build_series(matched, time_col, value_col)

            if feat in PLACEHOLDER_TIME_SERIES:
                if not series:
                    add_placeholder_series(patient, feat, base_time)
                else:
                    patient[feat] = series
            else:
                if feat in ["gcs_min", "gcs_motor", "gcs_verbal", "gcs_eyes"]:
                    gcs_rows = matched[matched["label"].astype(str).str.contains("gcs", case=False, na=False)].copy()
                    series = build_series(gcs_rows, "charttime", "valuenum")
                if not series:
                    add_placeholder_series(patient, feat, base_time)
                else:
                    patient[feat] = series

        # Ensure all expected keys exist
        for feat in TIME_SERIES_FEATURES:
            patient.setdefault(feat, [])
        for feat in STATIC_BINARY_FEATURES:
            patient.setdefault(feat, 0)
        for feat in SCALAR_FEATURES:
            patient.setdefault(feat, 0.0)
        patient.setdefault("gender", "")
        patient.setdefault("race", "UNKNOWN")
        patient.setdefault("target", 0)

        result[subject_id] = patient

    return result


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
DEMO = DATA / "mimic-iv-clinical-database-demo-2.2"


def describe_object(name: str, obj) -> None:
    print(f"\n== {name} ==")
    print(f"type: {type(obj)}")

    if isinstance(obj, pd.DataFrame):
        print(f"shape: {obj.shape}")
        print(obj.head(3).to_string())
        return

    if isinstance(obj, dict):
        keys = list(obj.keys())
        print(f"dict keys ({len(keys)}): {keys[:20]}")
        for key, value in list(obj.items())[:10]:
            shape = getattr(value, "shape", None)
            print(f"  - {key!r}: {type(value)} shape={shape}")
            if isinstance(value, pd.DataFrame):
                print(value.head(2).to_string())
            elif hasattr(value, "head"):
                try:
                    print(value.head(2))
                except Exception:
                    pass
        return

    if hasattr(obj, "shape"):
        print(f"shape: {obj.shape}")

    try:
        print(pd.DataFrame(obj).head(3).to_string())
    except Exception:
        print(repr(obj)[:500])


def summarize_patient_record(name: str, patient: dict) -> None:
    print(f"\n== {name} field summary ==")
    type_counts = Counter(type(v).__name__ for v in patient.values())
    print(f"field type counts: {dict(type_counts)}")

    for key in list(patient.keys())[:25]:
        value = patient[key]
        if isinstance(value, list):
            preview = value[:2]
            print(f"  - {key}: list len={len(value)} preview={preview}")
        else:
            print(f"  - {key}: {type(value).__name__} value={value!r}")

    for key in ["bmi", "duration", "age_at_admission", "lods", "gender", "race", "target"]:
        if key in patient:
            value = patient[key]
            print(f"  * scalar {key}: {type(value).__name__} value={value!r}")


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def inspect_demo_files() -> None:
    print("\n== data folder ==")
    for path in sorted(DATA.iterdir()):
        kind = "dir" if path.is_dir() else "file"
        print(f"- {path.name} [{kind}]")

    print("\n== demo folder top-level ==")
    for path in sorted(DEMO.iterdir()):
        kind = "dir" if path.is_dir() else "file"
        print(f"- {path.name} [{kind}]")

    for rel in [
        "hosp/patients.csv",
        "hosp/admissions.csv",
        "icu/icustays.csv",
        "icu/chartevents.csv",
    ]:
        path = DEMO / rel
        if not path.exists():
            continue
        df = pd.read_csv(path, nrows=3)
        print(f"\n== sample {rel} ==")
        print(f"columns: {df.columns.tolist()}")
        print(df.to_string(index=False))


def main() -> None:
    inspect_demo_files()

    demo_subject_ids = pd.read_csv(DEMO / "demo_subject_id.csv")
    print(f"\n== demo_subject_id.csv ==")
    print(f"rows: {len(demo_subject_ids)}")
    print(demo_subject_ids.head(5).to_string(index=False))

    d_items = pd.read_csv(DEMO / "icu" / "d_items.csv")
    d_labitems = pd.read_csv(DEMO / "hosp" / "d_labitems.csv")

    def show_matches(feature_names, table, label_col, table_name, limit=5):
        print(f"\n== keyword matches in {table_name} ==")
        for feat in feature_names:
            needle = feat.replace("_", " ")
            matches = table[table[label_col].astype(str).str.contains(needle, case=False, na=False)]
            if matches.empty:
                alt = table[table[label_col].astype(str).str.contains(feat, case=False, na=False)]
                matches = alt
            if not matches.empty:
                sample = matches[[c for c in ["itemid", "label"] if c in matches.columns]].head(limit)
                print(f"- {feat}: {len(matches)} matches")
                print(sample.to_string(index=False))

    show_matches(TIME_SERIES_FEATURES[:20], d_items, "label", "d_items")
    show_matches(TIME_SERIES_FEATURES, d_labitems, "label", "d_labitems")
    show_matches(STATIC_BINARY_FEATURES + SCALAR_FEATURES, d_labitems, "label", "d_labitems")

    train = load_pickle(DATA / "train.pkl")
    test = load_pickle(DATA / "test.pkl")

    describe_object("train.pkl", train)
    describe_object("test.pkl", test)

    if isinstance(train, dict) and train:
        first_key = next(iter(train))
        first_value = train[first_key]
        print(f"\n== sample train record: {first_key} ==")
        describe_object(f"train[{first_key}]", first_value)
        summarize_patient_record(f"train[{first_key}]", first_value)

    if isinstance(test, dict) and test:
        first_key = next(iter(test))
        first_value = test[first_key]
        print(f"\n== sample test record: {first_key} ==")
        describe_object(f"test[{first_key}]", first_value)
        summarize_patient_record(f"test[{first_key}]", first_value)

    print("\n== similarity hints ==")
    if isinstance(train, dict) and isinstance(test, dict):
        train_keys = set(train.keys())
        test_keys = set(test.keys())
        print(f"train/test same keys: {train_keys == test_keys}")
        print(f"shared keys: {sorted(train_keys & test_keys)[:20]}")
        print(f"train only: {sorted(train_keys - test_keys)[:20]}")
        print(f"test only: {sorted(test_keys - train_keys)[:20]}")
    elif isinstance(train, pd.DataFrame) and isinstance(test, pd.DataFrame):
        print(f"same columns: {list(train.columns) == list(test.columns)}")
        print(f"train columns: {train.columns.tolist()[:20]}")
        print(f"test columns: {test.columns.tolist()[:20]}")

    print("\nConclusion: if train/test are patient-level dicts/feature tables, they are processed artifacts, not raw MIMIC-IV CSVs.")

    subset = build_patient_subset()
    output_path = DATA / "test_subset100.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(subset, f)
    print(f"\nSaved subset pickle: {output_path} ({len(subset)} patients)")

    ground_truth_path = DATA / "test_subset100.csv"
    ground_truth = pd.DataFrame([
        {"id": pid, "target": int(patient.get("target", 0))}
        for pid, patient in subset.items()
    ])
    ground_truth.to_csv(ground_truth_path, index=False)
    print(f"Saved ground truth CSV: {ground_truth_path} ({len(ground_truth)} rows)")

    X_subset = pd.DataFrame()
    y_subset = None
    try:
        X_subset, y_subset = build_feature_matrix(subset, has_target=False)
        print(f"Subset feature matrix shape: {X_subset.shape}")
        print(f"Subset columns sample: {X_subset.columns.tolist()[:15]}")
    except Exception as exc:
        print(f"Feature-matrix validation failed: {exc}")

    if len(subset) != 100:
        print(f"Warning: expected 100 subjects, got {len(subset)}")


if __name__ == "__main__":
    main()