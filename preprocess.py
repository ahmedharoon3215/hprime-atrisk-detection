"""
hPrime Assessment Data Preprocessing Pipeline
CSCI323 Modern Artificial Intelligence - Spring 2026
University of Wollongong in Dubai

Parses raw hPrime assessment export (Mini-CEX + CBD) into a clean,
ML-ready dataset for at-risk trainee detection.
"""

import pandas as pd
import numpy as np
import json
import re
import os
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_FILE  = "/mnt/user-data/uploads/3_years_Assessment_data.xlsx"
OUTPUT_DIR  = "/mnt/user-data/outputs"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "hprime_clean.csv")
REPORT_FILE = os.path.join(OUTPUT_DIR, "preprocessing_report.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# CBD score column names (4 dimensions, rubric order)
CBD_SCORE_COLS = [
    "score_record_keeping",
    "score_history_examination",
    "score_management_plan",
    "score_clinical_judgement",
]

# MiniCEX score column names (7 dimensions)
MINICEX_SCORE_COLS = [
    "score_history",
    "score_physical_exam",
    "score_communication",
    "score_clinical_judgement",
    "score_professionalism",
    "score_organisation",
    "score_overall_care",
]

# Seniority tiers derived from assessor title keywords
SENIOR_KEYWORDS   = ["facem", "director", "consultant", "senior specialist",
                      "staff specialist", "ed ss", "supervisor", "ed clinical"]
REGISTRAR_KEYWORDS = ["registrar", "advanced trainee", "vmo"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_html(text: str) -> str:
    """Strip HTML tags and normalise whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_time_mins(token: str) -> float:
    """Convert '15 minutes', '35 or more minutes', etc. → numeric minutes."""
    token = str(token).lower().strip()
    if "35 or more" in token:
        return 35.0
    if "less than 5" in token:
        return 4.0
    m = re.search(r"(\d+)", token)
    return float(m.group(1)) if m else np.nan


def parse_age(token: str) -> float:
    """'70 years', '13 days', '9 months' → approximate age in years."""
    token = str(token).strip().lower()
    m = re.search(r"(\d+)", token)
    if not m:
        return np.nan
    val = float(m.group(1))
    if "day" in token:
        return round(val / 365, 3)
    if "month" in token:
        return round(val / 12, 2)
    return val


def parse_scores(scores_json: str) -> dict:
    """
    Parse embedded JSON scores object.
    Returns dict keyed by integer index → ColumnNum value.
    """
    try:
        obj = json.loads(scores_json)
        return {int(k): v["ColumnNum"] for k, v in obj.items()}
    except Exception:
        return {}


def derive_seniority(title: str) -> str:
    t = str(title).lower()
    if any(k in t for k in SENIOR_KEYWORDS):
        return "senior"
    if any(k in t for k in REGISTRAR_KEYWORDS):
        return "registrar"
    return "other"


def parse_cbd_row(parts: list) -> dict:
    """
    CBD pipe-split structure (after stripping HTML rows):
    [0]  date
    [1]  competency areas
    [2]  patient age
    [3]  patient gender
    [4]  setting
    [5]  meets_standard_flag (pre-score, sometimes 'Yes')
    [6]  case summary
    [7]  scores JSON
    [8]  meets_standard boolean
    [9]  feedback
    [10] discussion time
    [11] prep/feedback time
    [12] assessor name
    [13] assessor title
    """
    rec = {}
    try:
        rec["assessment_date_raw"] = parts[0].strip() if len(parts) > 0 else ""
        rec["competency_areas"]    = parts[1].strip() if len(parts) > 1 else ""
        rec["patient_age_raw"]     = parts[2].strip() if len(parts) > 2 else ""
        rec["patient_gender"]      = parts[3].strip().lower() if len(parts) > 3 else ""
        rec["setting"]             = parts[4].strip().lower() if len(parts) > 4 else ""
        rec["case_summary"]        = clean_html(parts[6].strip()) if len(parts) > 6 else ""
        scores_raw                 = parts[7].strip() if len(parts) > 7 else "{}"
        rec["meets_standard_raw"]  = parts[8].strip().lower() if len(parts) > 8 else ""
        rec["feedback"]            = clean_html(parts[9].strip()) if len(parts) > 9 else ""
        rec["discussion_time_raw"] = parts[10].strip() if len(parts) > 10 else ""
        rec["prep_time_raw"]       = parts[11].strip() if len(parts) > 11 else ""
        rec["assessor_name"]       = parts[12].strip() if len(parts) > 12 else ""
        rec["assessor_title"]      = parts[13].strip() if len(parts) > 13 else ""

        scores = parse_scores(scores_raw)
        for i, col in enumerate(CBD_SCORE_COLS):
            rec[col] = scores.get(i, np.nan)

        rec["num_competencies"] = len([
            c.strip() for c in rec["competency_areas"].split(",") if c.strip()
        ])

    except Exception as e:
        rec["_parse_error"] = str(e)

    return rec


def parse_minicex_row(parts: list) -> dict:
    """
    MiniCEX pipe-split structure:
    [0]  focus area
    [1]  date
    [2]  patient age
    [3]  patient gender
    [4]  case summary
    [5]  setting
    [6]  scores JSON
    [7]  meets_standard boolean
    [8]  feedback
    [9]  observation time
    [10] feedback time
    [11] assessor name
    [12] assessor title
    """
    rec = {}
    try:
        rec["focus_area"]          = parts[0].strip() if len(parts) > 0 else ""
        rec["assessment_date_raw"] = parts[1].strip() if len(parts) > 1 else ""
        rec["patient_age_raw"]     = parts[2].strip() if len(parts) > 2 else ""
        rec["patient_gender"]      = parts[3].strip().lower() if len(parts) > 3 else ""
        rec["case_summary"]        = clean_html(parts[4].strip()) if len(parts) > 4 else ""
        rec["setting"]             = parts[5].strip().lower() if len(parts) > 5 else ""
        scores_raw                 = parts[6].strip() if len(parts) > 6 else "{}"
        rec["meets_standard_raw"]  = parts[7].strip().lower() if len(parts) > 7 else ""
        rec["feedback"]            = clean_html(parts[8].strip()) if len(parts) > 8 else ""
        rec["discussion_time_raw"] = parts[9].strip() if len(parts) > 9 else ""
        rec["prep_time_raw"]       = parts[10].strip() if len(parts) > 10 else ""
        rec["assessor_name"]       = parts[11].strip() if len(parts) > 11 else ""
        rec["assessor_title"]      = parts[12].strip() if len(parts) > 12 else ""

        scores = parse_scores(scores_raw)
        for i, col in enumerate(MINICEX_SCORE_COLS):
            rec[col] = scores.get(i, np.nan)

    except Exception as e:
        rec["_parse_error"] = str(e)

    return rec


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    report_lines = []
    log = lambda msg: (print(msg), report_lines.append(msg))

    log("=" * 60)
    log("hPrime Assessment Data Preprocessing Report")
    log(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 60)

    # Load raw data
    df_raw = pd.read_excel(INPUT_FILE)
    log(f"\nRaw dataset: {len(df_raw)} rows, {df_raw['Candidate ID'].nunique()} unique candidates")
    log(f"Form type counts: {df_raw['Form Type'].value_counts().to_dict()}")

    # Drop rows with no submission values
    df_raw = df_raw[df_raw["Submission Values"].notna()].copy()
    log(f"After dropping null Submission Values: {len(df_raw)} rows")

    # Drop HTML bloat rows (form description rows, not real assessments)
    html_mask = df_raw["Submission Values"].str.contains("<p style", na=False)
    log(f"Dropping {html_mask.sum()} HTML form-description rows")
    df_raw = df_raw[~html_mask].copy()
    log(f"After dropping HTML rows: {len(df_raw)} rows")

    # Parse each row
    records = []
    parse_errors = 0

    for _, row in df_raw.iterrows():
        form_type  = str(row["Form Type"]).strip()
        subm_val   = str(row["Submission Values"])
        parts      = [p for p in subm_val.split("|")]

        if form_type == "Case Based Discussion":
            parsed = parse_cbd_row(parts)
        elif form_type == "Mini-CEX":
            parsed = parse_minicex_row(parts)
        else:
            continue

        parsed["candidate_id"] = row["Candidate ID"]
        parsed["form_type"]    = form_type

        if "_parse_error" in parsed:
            parse_errors += 1

        records.append(parsed)

    log(f"\nParsed {len(records)} records ({parse_errors} with partial errors)")

    df = pd.DataFrame(records)

    # ── Feature engineering ──────────────────────────────────────────────────

    # 1. Assessment date → proper datetime
    df["assessment_date"] = pd.to_datetime(
        df["assessment_date_raw"], dayfirst=True, errors="coerce"
    )

    # 2. Patient age in years
    df["patient_age_years"] = df["patient_age_raw"].apply(parse_age)

    # 3. Meets standard → binary
    df["meets_standard"] = df["meets_standard_raw"].map(
        {"true": 1, "yes": 1, "false": 0, "no": 0}
    ).astype("Int64")

    # 4. Gender → binary
    df["patient_is_female"] = df["patient_gender"].map(
        lambda x: 1 if "female" in str(x) else (0 if "male" in str(x) else np.nan)
    )

    # 5. Setting → binary
    df["is_emergency"] = df["setting"].map(
        lambda x: 1 if "emergency" in str(x) else (0 if "inpatient" in str(x) else np.nan)
    )

    # 6. Time fields → numeric
    df["discussion_time_mins"] = df["discussion_time_raw"].apply(parse_time_mins)
    df["prep_time_mins"]       = df["prep_time_raw"].apply(parse_time_mins)

    # 7. Assessor seniority
    df["assessor_seniority"] = df["assessor_title"].apply(derive_seniority)

    # 8. Average score (across whichever score columns exist for each form type)
    cbd_mask    = df["form_type"] == "Case Based Discussion"
    minicex_mask = df["form_type"] == "Mini-CEX"

    df["avg_score"] = np.nan
    df.loc[cbd_mask, "avg_score"] = df.loc[cbd_mask, CBD_SCORE_COLS].mean(axis=1)
    df.loc[minicex_mask, "avg_score"] = df.loc[minicex_mask, MINICEX_SCORE_COLS].mean(axis=1)

    # 9. Score std (measure of consistency across dimensions)
    df["score_std"] = np.nan
    df.loc[cbd_mask, "score_std"]     = df.loc[cbd_mask, CBD_SCORE_COLS].std(axis=1)
    df.loc[minicex_mask, "score_std"] = df.loc[minicex_mask, MINICEX_SCORE_COLS].std(axis=1)

    # 10. Score below standard on any dimension
    df["any_score_1"] = np.nan
    df.loc[cbd_mask, "any_score_1"] = (
        df.loc[cbd_mask, CBD_SCORE_COLS].eq(1).any(axis=1).astype(int)
    )
    df.loc[minicex_mask, "any_score_1"] = (
        df.loc[minicex_mask, MINICEX_SCORE_COLS].eq(1).any(axis=1).astype(int)
    )

    # 11. Feedback length (proxy for detail of concern)
    df["feedback_length"] = df["feedback"].apply(lambda x: len(str(x).split()))

    # 12. Sort by candidate and date for temporal features
    df = df.sort_values(["candidate_id", "assessment_date"]).reset_index(drop=True)

    # 13. Per-candidate temporal features
    df["assessment_number"]    = df.groupby("candidate_id").cumcount() + 1
    df["days_since_last"]      = df.groupby("candidate_id")["assessment_date"].diff().dt.days
    df["rolling_avg_3"]        = (
        df.groupby("candidate_id")["avg_score"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )
    df["rolling_fail_rate_3"]  = (
        df.groupby("candidate_id")["meets_standard"]
        .transform(lambda x: x.rolling(3, min_periods=1)
                   .apply(lambda w: (w == 0).sum() / len(w)))
    )
    df["cumulative_fail_count"] = (
        df.groupby("candidate_id")["meets_standard"]
        .transform(lambda x: (x == 0).cumsum())
    )

    # 14. Drop raw intermediate columns
    drop_cols = [
        "assessment_date_raw", "patient_age_raw", "patient_gender",
        "meets_standard_raw", "discussion_time_raw", "prep_time_raw", "_parse_error"
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # ── Final column order ───────────────────────────────────────────────────
    id_cols      = ["candidate_id", "form_type", "assessment_date", "assessment_number"]
    patient_cols = ["patient_age_years", "patient_is_female", "is_emergency"]
    score_cols   = CBD_SCORE_COLS + MINICEX_SCORE_COLS + [
        "avg_score", "score_std", "any_score_1"
    ]
    temporal_cols = [
        "days_since_last", "rolling_avg_3",
        "rolling_fail_rate_3", "cumulative_fail_count"
    ]
    outcome_col  = ["meets_standard"]
    time_cols    = ["discussion_time_mins", "prep_time_mins"]
    assessor_cols = ["assessor_name", "assessor_title", "assessor_seniority"]
    text_cols    = ["competency_areas", "focus_area", "case_summary",
                    "feedback", "feedback_length"]
    misc_cols    = ["num_competencies", "score_std"]

    ordered = (id_cols + patient_cols + score_cols + temporal_cols +
               outcome_col + time_cols + assessor_cols + text_cols)
    remaining = [c for c in df.columns if c not in ordered]
    df = df[[c for c in ordered if c in df.columns] + remaining]

    # ── Save ─────────────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_FILE, index=False)

    # ── Report ───────────────────────────────────────────────────────────────
    log(f"\n{'─'*40}")
    log("CLEANED DATASET SUMMARY")
    log(f"{'─'*40}")
    log(f"Total records:        {len(df)}")
    log(f"Unique candidates:    {df['candidate_id'].nunique()}")
    log(f"Mini-CEX records:     {(df['form_type'] == 'Mini-CEX').sum()}")
    log(f"CBD records:          {(df['form_type'] == 'Case Based Discussion').sum()}")
    log(f"Date range:           {df['assessment_date'].min().date()} → {df['assessment_date'].max().date()}")
    log(f"Meets standard:       {df['meets_standard'].eq(1).sum()} pass / {df['meets_standard'].eq(0).sum()} fail")
    log(f"Fail rate:            {df['meets_standard'].eq(0).sum() / df['meets_standard'].notna().sum():.1%}")
    log(f"\nAvg score stats:")
    log(f"  Mean:   {df['avg_score'].mean():.2f}")
    log(f"  Std:    {df['avg_score'].std():.2f}")
    log(f"  Min:    {df['avg_score'].min():.2f}")
    log(f"  Max:    {df['avg_score'].max():.2f}")
    log(f"\nAssessments per candidate:")
    counts = df.groupby("candidate_id").size()
    log(f"  Mean:   {counts.mean():.1f}")
    log(f"  Median: {counts.median():.0f}")
    log(f"  Min:    {counts.min()}")
    log(f"  Max:    {counts.max()}")
    log(f"\nMissing values (key columns):")
    key_cols = ["assessment_date", "avg_score", "meets_standard",
                "patient_age_years", "is_emergency"]
    for c in key_cols:
        if c in df.columns:
            missing = df[c].isna().sum()
            log(f"  {c:<25} {missing} ({missing/len(df):.1%})")
    log(f"\nOutput saved to: {OUTPUT_FILE}")
    log("=" * 60)

    # Save report
    with open(REPORT_FILE, "w") as f:
        f.write("\n".join([str(l[1]) if isinstance(l, tuple) else str(l)
                           for l in report_lines]))

    return df


if __name__ == "__main__":
    df = main()
    print(f"\nDone. Columns in clean dataset ({len(df.columns)}):")
    print(df.columns.tolist())
