"""
PD vs PD-plus classifier — XGBoost, 8 variants.

Variants = 2 targets × 2 tiers × 2 class-balance settings
  Targets  : binary (PD=0 vs PD-plus=1)
             multiclass (PD=0, PSP=1, MSA=2, DLB=3, CBS=4)
  Tiers    : A   — self-report only (Level 1+2, ~33 features)
             AB  — full clinical  (Level 1+2+3, ~50 features)
  Balance  : balanced (scale_pos_weight / sample_weight)
             unbalanced (no correction)

Outputs saved to output/model/ :
  feature_matrix.csv          — joined feature table (IDs × features)
  results_summary.csv         — one row per variant (AUC, F1, …)
  {variant}/confusion_matrix.png
  {variant}/shap_summary.png
  {variant}/metrics.json
"""

import json
import math
import os
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import chi2_contingency
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths — uses clean pipeline (enrolled + complete rows only)
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).parent.parent
CLEAN_DIR = ROOT / "output" / "clean_pipeline" / "full_enrolled"
OUTPUT    = ROOT / "output" / "model"

OUTPUT.mkdir(parents=True, exist_ok=True)

# Clinical "Determined diagnosis" numeric codes
DIAG_COL  = next(c for c in pd.read_csv(CLEAN_DIR / "clinical.csv").columns
                 if "Determined diagnosis" in c)
DIAG_MAP  = {0.0: "PD", 1.0: "PSP", 2.0: "MSA", 3.0: "CBS", 4.0: "DLB"}
PDPLUS    = {1.0, 2.0, 3.0, 4.0}  # PSP, MSA, CBS, DLB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(name: str) -> pd.DataFrame:
    df = pd.read_csv(CLEAN_DIR / name)
    return df.drop(columns=[c for c in df.columns if c.startswith("Unnamed:")],
                   errors="ignore")


def find_col(df: pd.DataFrame, prefix: str) -> str | None:
    """Return the first column whose stripped text starts with prefix."""
    prefix = prefix.strip()
    for c in df.columns:
        if c.strip().startswith(prefix):
            return c
    return None


def extract_int(val) -> float:
    """Pull leading integer from UPDRS text like '2: Mild: …' or '<strong>1:…'."""
    if pd.isna(val):
        return np.nan
    s = re.sub(r"<[^>]+>", "", str(val)).strip()
    m = re.match(r"^(\d+)", s)
    return float(m.group(1)) if m else np.nan


def to_numeric_col(series: pd.Series) -> pd.Series:
    """Try direct numeric conversion; fall back to extract_int on text."""
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().mean() >= 0.5:
        return num
    return series.apply(extract_int)


def encode_yesno(series: pd.Series,
                 yes_vals=("Yes/Oui", "Yes"),
                 no_vals=("No/Non", "No")) -> pd.Series:
    result = pd.Series(np.nan, index=series.index)
    result[series.isin(yes_vals)] = 1.0
    result[series.isin(no_vals)]  = 0.0
    return result


# ---------------------------------------------------------------------------
# Load data files (once, reused)
# ---------------------------------------------------------------------------
print("Loading data files…")
clin_df  = load_csv("clinical.csv")
demo_df  = load_csv("demographic.csv")
epi_df   = load_csv("epidemiological.csv")
upd_df   = load_csv("mds_updrs.csv")       # ALL UPDRS data (Part 1 + 2 + 3)
pdq_df   = load_csv("pdq_39.csv")
moca_df  = load_csv("moca.csv")
neuro_df = load_csv("neuropsychological.csv")


# ---------------------------------------------------------------------------
# 1 + 2. Diagnosis — clean pipeline is already enrolled+complete filtered
# ---------------------------------------------------------------------------
clin_sub = clin_df.copy()
clin_sub["_diag_code"] = pd.to_numeric(clin_sub[DIAG_COL], errors="coerce")
clin_sub = clin_sub[clin_sub["_diag_code"].isin(DIAG_MAP.keys())].copy()
clin_sub["diagnosis"] = clin_sub["_diag_code"].map(DIAG_MAP)

diag_ids = set(clin_sub["Project key"])
print(f"  PD/PD-plus participants: {len(diag_ids)}")
print("  Counts:", clin_sub["diagnosis"].value_counts().to_dict())


# ---------------------------------------------------------------------------
# 3. Feature extraction helpers per variable
# ---------------------------------------------------------------------------

def get_series(df: pd.DataFrame, col_prefix: str,
               pk_col="Project key") -> pd.Series:
    """Find column by prefix, return Series indexed by Project key, numeric."""
    col = find_col(df, col_prefix)
    if col is None:
        return pd.Series(dtype=float, name=col_prefix)
    s = df.set_index(pk_col)[col]
    return to_numeric_col(s)


# ---------------------------------------------------------------------------
# 4. Build raw feature columns
# ---------------------------------------------------------------------------
print("\nExtracting features…")

def build_features(ids: set) -> pd.DataFrame:
    fts: dict[str, pd.Series] = {}

    # ----- Demographics -----
    d = demo_df[demo_df["Project key"].isin(ids)].set_index("Project key")

    age_col = find_col(demo_df, "Age at study visit (automatic calculated field)")
    fts["study_visit_age"] = to_numeric_col(d[age_col]) if age_col else pd.Series(dtype=float)

    gen_col = find_col(demo_df, "1. Gender")
    if gen_col:
        gen_s = d[gen_col].astype(str)
        gen_enc = pd.Series(np.nan, index=gen_s.index)
        gen_enc[gen_s.str.lower().str.startswith("male") | gen_s.str.lower().str.startswith("homme")] = 1.0
        gen_enc[gen_s.str.lower().str.startswith("female") | gen_s.str.lower().str.startswith("femme")] = 0.0
        fts["gender"] = gen_enc

    edu_col = find_col(demo_df, "5. Years of education")
    fts["yrs_education"] = to_numeric_col(d[edu_col]) if edu_col else pd.Series(dtype=float)

    # ----- Epidemiological -----
    e = epi_df[epi_df["Project key"].isin(ids)].set_index("Project key")

    smell_col = find_col(epi_df, "8. Do you feel or have people told you")
    if smell_col:
        s = e[smell_col].astype(str)
        enc = pd.Series(np.nan, index=s.index)
        enc[s.str.startswith("Yes") | s.str.contains("Already lost")] = 1.0
        enc[s.str.startswith("No/")] = 0.0
        fts["smell"] = enc

    dreams_col = find_col(epi_df, "14. Have you ever been told")
    if dreams_col:
        fts["dreams"] = encode_yesno(e[dreams_col].astype(str))

    constip_col = find_col(epi_df, "5. Do you suffer from constipation")
    if constip_col:
        fts["constipation"] = encode_yesno(e[constip_col].astype(str))

    comorb_col = find_col(epi_df, "1. Have you ever been diagnosed with")
    if comorb_col:
        s = e[comorb_col].astype(str)
        enc = pd.Series(np.nan, index=s.index)
        enc[s.str.contains("None of these|Aucun")] = 0.0
        enc[~s.str.contains("None of these|Aucun") & ~s.str.contains("nan")] = 1.0
        fts["comorbidities"] = enc

    blow_col = find_col(epi_df, "3. Have you ever experienced a hard blow")
    if blow_col:
        fts["head_blow"] = encode_yesno(e[blow_col].astype(str))

    pest_col = find_col(epi_df, "18. Have you ever used or been exposed to pesticides")
    if pest_col:
        fts["pesticides"] = encode_yesno(e[pest_col].astype(str))

    gp_col = find_col(epi_df, "29. Do you have grandparents with Parkinson")
    if gp_col:
        fts["grandparents"] = encode_yesno(e[gp_col].astype(str),
                                            yes_vals=("Yes/Oui", "Yes"))

    au_col = find_col(epi_df, "30. Do you have any uncles or aunts with Parkinson")
    if au_col:
        s = e[au_col].astype(str)
        enc = pd.Series(np.nan, index=s.index)
        enc[s.str.contains("None/Aucun")] = 0.0
        enc[s.str.contains("Maternal|Paternal|Maternel|Paternel")] = 1.0
        fts["aunts_uncles_pd"] = enc

    # ----- Clinical -----
    cl = clin_df[clin_df["Project key"].isin(ids)].set_index("Project key")

    # symptom_asymmetry Q8: Right/Left = asymmetric (1), Both = symmetric (0)
    sym8_col = find_col(clin_df, "8. When the symptoms began")
    if sym8_col:
        s = cl[sym8_col].astype(str)
        enc = pd.Series(np.nan, index=s.index)
        enc[s.str.contains("Right|Left|Droit|Gauche")] = 1.0
        enc[s.str.contains("Both sides|Deux côtés")] = 0.0
        fts["symptom_asymmetry"] = enc

    # current_asymmetry Q10: still one side (1) vs both (0/0.5)
    sym10_col = find_col(clin_df, "10. Do the symptoms affect both sides")
    if sym10_col:
        s = cl[sym10_col].astype(str)
        enc = pd.Series(np.nan, index=s.index)
        enc[s.str.contains("No .still affects only one")] = 1.0
        enc[s.str.contains("one side affected more")] = 0.5
        enc[s.str.contains("equally")] = 0.0
        fts["current_asymmetry"] = enc

    # first_symptoms Q4: multi-select → three binary flags
    fs_col = find_col(clin_df, "4. What were the first symptoms")
    if fs_col:
        s = cl[fs_col].astype(str)
        fts["first_sx_tremor"]    = (s.str.contains("Tremor|Tremblements", na=False)).astype(float)
        fts["first_sx_brady"]     = (s.str.contains("Bradykinesia|Bradykinésie", na=False)).astype(float)
        fts["first_sx_rigidity"]  = (s.str.contains("Rigidity|Rigidité", na=False)).astype(float)

    # ----- MDS-UPDRS: Part 1A value cols, Part 1B text cols, Part 2, Part 3 -----
    u = upd_df[upd_df["Project key"].isin(ids)].set_index("Project key")

    # Part 1A — clinician-rated, have computed value columns
    for var, vcol in [
        ("updrs_1_1", "Updrs_1_1 value"),
        ("updrs_1_2", "Updrs_1_2 value"),
        ("updrs_1_4", "Updrs_1_4 value"),
        ("updrs_1_5", "Updrs_1_5 value"),
    ]:
        if vcol in upd_df.columns:
            fts[var] = pd.to_numeric(u[vcol], errors="coerce")

    # 1.3 DEPRESSED MOOD — has Updrs_1_3 computed col in MDS-UPDRS
    if "Updrs_1_3" in upd_df.columns:
        fts["updrs_1_3"] = pd.to_numeric(u["Updrs_1_3"], errors="coerce")

    # Part 1B — self-rated text responses
    for var, prefix in [
        ("updrs_1_10", "1.10 URINARY"),
        ("updrs_1_11", "1.11 CONSTIPATION"),
        ("updrs_1_12", "1.12 LIGHT HEADEDNESS"),
        ("updrs_1_13", "1.13 FATIGUE"),
    ]:
        col = find_col(upd_df, prefix)
        if col:
            fts[var] = u[col].apply(extract_int)

    # Part 2 items — self-rated text responses
    for var, prefix in [
        ("updrs_2_1",  "2.1 SPEECH"),
        ("updrs_2_5",  "2.5 DRESSING"),
        ("updrs_2_7",  "2.7 HANDWRITING"),
        ("updrs_2_8",  "2.8 DOING HOBBIES"),
        ("updrs_2_10", "2.10 TREMOR"),
        ("updrs_2_12", "2.12 WALKING AND BALANCE"),
        ("updrs_2_13", "2.13 FREEZING"),
    ]:
        col = find_col(upd_df, prefix)
        if col:
            fts[var] = u[col].apply(extract_int)

    # ----- MDS-UPDRS Part 3 (value columns) -----
    for var, vcol in [
        ("updrs_3_1",     "Updrs_3_1 value"),
        ("updrs_3_2",     "Updrs_3_2 value"),
        ("updrs_3_3_neck","Updrs_3_3_neck value"),
        ("updrs_3_10",    "Updrs_3_10 value"),
        ("updrs_3_11",    "Updrs_3_11 value"),
        ("updrs_3_12",    "Updrs_3_12 value"),
        ("updrs_3_13",    "Updrs_3_13 value"),
        ("updrs_3_14",    "Updrs_3_14"),
        ("updrs_3_18",    "Updrs_3_18 value"),
    ]:
        if vcol in upd_df.columns:
            fts[var] = pd.to_numeric(u[vcol], errors="coerce")

    # ----- Composite scores -----
    def mean_value_cols(df_sub, cols):
        available = [c for c in cols if c in upd_df.columns]
        if not available:
            return pd.Series(dtype=float)
        return df_sub[available].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    brady_cols = [f"Updrs_3_{n} value" for n in
                  ["4_r","4_l","5_r","5_l","6_r","6_l","7_r","7_l","8_r","8_l"]]
    brady_r_cols = [f"Updrs_3_{n} value" for n in ["4_r","5_r","6_r","7_r","8_r"]]
    brady_l_cols = [f"Updrs_3_{n} value" for n in ["4_l","5_l","6_l","7_l","8_l"]]
    rest_cols    = ["Updrs_3_17_rue value","Updrs_3_17_lue value",
                    "Updrs_3_17_rle value","Updrs_3_17_lle value"]
    action_cols  = ["Updrs_3_15_r value","Updrs_3_15_l value",
                    "Updrs_3_16_r value","Updrs_3_16_l value"]
    rigid_cols   = ["Updrs_3_3_rue value","Updrs_3_3_lue value",
                    "Updrs_3_3_rle value","Updrs_3_3_lle value"]

    fts["bradykinesia_mean"]  = mean_value_cols(u, brady_cols)
    fts["rest_tremor_mean"]   = mean_value_cols(u, rest_cols)
    fts["action_tremor_mean"] = mean_value_cols(u, action_cols)
    fts["rigidity_limb_mean"] = mean_value_cols(u, rigid_cols)

    # asymmetry_index = |R_brady - L_brady| / (R_brady + L_brady + 1)
    r_avail = [c for c in brady_r_cols if c in upd_df.columns]
    l_avail = [c for c in brady_l_cols if c in upd_df.columns]
    if r_avail and l_avail:
        R = u[r_avail].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        L = u[l_avail].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        fts["asymmetry_index"] = (R - L).abs() / (R + L + 1)

    # ----- PDQ-39 -----
    p = pdq_df[pdq_df["Project key"].isin(ids)].set_index("Project key")
    for var, prefix in [
        ("pdq39_mobility",      "PDQ-39 Mobility (Raw Score)"),
        ("pdq39_communication", "PDQ-39 Communication (Raw Score)"),
        ("pdq39_summary",       "PDQ-39 Summary Index"),
    ]:
        col = find_col(pdq_df, prefix)
        if col:
            fts[var] = pd.to_numeric(p[col], errors="coerce")

    # ----- MoCA -----
    m = moca_df[moca_df["Project key"].isin(ids)].set_index("Project key")
    for var, prefix in [
        ("moca_total",     "TOTAL SCORE"),
        ("moca_visuosp",   "Visuospatial/Executive Score"),
        ("moca_attention", "Attention Score"),
    ]:
        col = find_col(moca_df, prefix)
        if col:
            fts[var] = pd.to_numeric(m[col], errors="coerce")

    # ----- Neuropsychological (Trail B) -----
    n = neuro_df[neuro_df["Project key"].isin(ids)].set_index("Project key")
    trail_col = find_col(neuro_df, "Trail B raw score")
    if trail_col:
        fts["trail_b"] = pd.to_numeric(n[trail_col], errors="coerce")

    # ----- Assemble dataframe -----
    feat_df = pd.DataFrame(fts)
    feat_df.index.name = "Project key"
    return feat_df.reset_index()


feat_df = build_features(diag_ids)
print(f"  Feature matrix: {feat_df.shape[0]} rows × {feat_df.shape[1]-1} feature cols")
print(f"  Missing rate per feature (top 10):")
miss = (feat_df.set_index("Project key").isna().mean() * 100).sort_values(ascending=False)
print(miss.head(10).to_string())


# ---------------------------------------------------------------------------
# 5. Merge features + diagnosis
# ---------------------------------------------------------------------------
diag_sub = clin_sub[["Project key","diagnosis","_diag_code"]].copy()
df_full  = feat_df.merge(diag_sub, on="Project key", how="inner")

print(f"\nFull joined table: {df_full.shape[0]} rows")
print("Diagnosis distribution:", df_full["diagnosis"].value_counts().to_dict())

# Save feature matrix
df_full.to_csv(OUTPUT / "feature_matrix.csv", index=False)
print(f"  → feature_matrix.csv saved")


# ---------------------------------------------------------------------------
# 6. Feature sets per tier
# ---------------------------------------------------------------------------
TIER_A_VARS = [
    "study_visit_age","gender","yrs_education",
    "updrs_2_10","updrs_2_7","updrs_2_8","updrs_2_5","updrs_2_1",
    "updrs_2_12","updrs_2_13",
    "symptom_asymmetry","current_asymmetry",
    "first_sx_tremor","first_sx_brady","first_sx_rigidity",
    "smell","dreams","constipation",
    "updrs_1_4","updrs_1_3","updrs_1_5","updrs_1_13",
    "updrs_1_11","updrs_1_10","updrs_1_12","updrs_1_1","updrs_1_2",
    "comorbidities","head_blow","pesticides","grandparents","aunts_uncles_pd",
    "pdq39_mobility","pdq39_communication","pdq39_summary",
]
TIER_B_VARS = [
    "bradykinesia_mean","updrs_3_14","rest_tremor_mean","updrs_3_18",
    "action_tremor_mean","rigidity_limb_mean","updrs_3_3_neck",
    "updrs_3_12","updrs_3_11","updrs_3_10","updrs_3_13","asymmetry_index",
    "updrs_3_1","updrs_3_2",
    "moca_total","moca_visuosp","moca_attention",
]
TIER_C_VARS = ["trail_b"]  # excluded from main models (too few cases)

TIER_MAP = {
    "TierA":  TIER_A_VARS,
    "TierAB": TIER_A_VARS + TIER_B_VARS,
}


# ---------------------------------------------------------------------------
# 7. XGBoost model runner
# ---------------------------------------------------------------------------

def run_variant(X: pd.DataFrame, y: pd.Series, task: str, balanced: bool,
                label_names: list[str], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    n_classes = len(label_names)

    skf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_arr = y.values

    all_preds, all_proba, all_true = [], [], []

    for fold_i, (tr_idx, te_idx) in enumerate(skf.split(X, y_arr)):
        X_tr, X_te = X.iloc[tr_idx].copy(), X.iloc[te_idx].copy()
        y_tr, y_te = y_arr[tr_idx], y_arr[te_idx]

        if task == "binary":
            model = XGBClassifier(
                n_estimators=400, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                use_label_encoder=False, eval_metric="logloss",
                random_state=42, n_jobs=-1,
                scale_pos_weight=((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
                if balanced else 1.0,
            )
        else:
            model = XGBClassifier(
                n_estimators=400, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                use_label_encoder=False, eval_metric="mlogloss",
                objective="multi:softprob", num_class=n_classes,
                random_state=42, n_jobs=-1,
            )

        fit_kwargs = {}
        if task == "multi" and balanced:
            # Per-sample weights: inverse class frequency, robust to absent classes in fold
            cls_counts = np.bincount(y_tr, minlength=n_classes).astype(float)
            cls_counts[cls_counts == 0] = 1.0  # avoid div-by-zero for absent classes
            w_per_class = len(y_tr) / (n_classes * cls_counts)
            fit_kwargs["sample_weight"] = w_per_class[y_tr]

        model.fit(X_tr, y_tr, **fit_kwargs)

        preds = model.predict(X_te)
        proba = model.predict_proba(X_te)

        all_preds.append(preds)
        all_proba.append(proba)
        all_true.append(y_te)

    y_true  = np.concatenate(all_true)
    y_pred  = np.concatenate(all_preds)
    y_proba = np.concatenate(all_proba, axis=0)

    # --- Metrics ---
    if task == "binary":
        auc  = roc_auc_score(y_true, y_proba[:, 1])
        f1   = f1_score(y_true, y_pred, average="binary")
        f1_w = f1_score(y_true, y_pred, average="weighted")
    else:
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        valid = [i for i in range(n_classes) if (y_true == i).sum() > 0]
        auc   = roc_auc_score(y_bin[:, valid], y_proba[:, valid],
                              multi_class="ovr", average="macro")
        f1    = f1_score(y_true, y_pred, average="macro")
        f1_w  = f1_score(y_true, y_pred, average="weighted")

    present_labels = sorted(np.unique(y_true))
    present_names  = [label_names[i] for i in present_labels]
    report = classification_report(y_true, y_pred,
                                   labels=present_labels,
                                   target_names=present_names,
                                   output_dict=True, zero_division=0)

    macro_avg  = report.get("macro avg", {})
    weight_avg = report.get("weighted avg", {})
    metrics = {
        "roc_auc":            round(float(auc), 4),
        "f1_macro":           round(float(f1), 4),
        "f1_weighted":        round(float(f1_w), 4),
        "precision_macro":    round(float(macro_avg.get("precision", 0)), 4),
        "recall_macro":       round(float(macro_avg.get("recall", 0)), 4),
        "precision_weighted": round(float(weight_avg.get("precision", 0)), 4),
        "recall_weighted":    round(float(weight_avg.get("recall", 0)), 4),
        "n_samples":          int(len(y_true)),
        "per_class":          {name: report.get(name, {}) for name in present_names},
    }
    def _to_json(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _to_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_json(v) for v in obj]
        return obj

    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(_to_json(metrics), fh, indent=2)

    # --- Confusion matrix ---
    cm   = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    disp = ConfusionMatrixDisplay(cm, display_labels=label_names)
    fig, ax = plt.subplots(figsize=(max(5, n_classes), max(4, n_classes)))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{out_dir.name}\nAUC={auc:.3f}  F1={f1:.3f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    # --- SHAP (train final model on all data — already complete cases) ---
    X_full = X.copy()
    y_full = y.values

    if task == "binary":
        final_model = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1,
            scale_pos_weight=((y_full == 0).sum() / max((y_full == 1).sum(), 1))
            if balanced else 1.0,
        )
    else:
        cls_counts = np.bincount(y_full, minlength=n_classes).astype(float)
        cls_counts[cls_counts == 0] = 1.0
        weights    = (len(y_full) / (n_classes * cls_counts))[y_full] if balanced else None
        final_model = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="mlogloss",
            objective="multi:softprob", num_class=n_classes,
            random_state=42, n_jobs=-1,
        )

    fit_kw = {"sample_weight": weights} if (task == "multi" and balanced) else {}
    final_model.fit(X_full, y_full, **fit_kw)

    explainer   = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_full)

    # --- Compute mean |SHAP| per feature ---
    if task == "binary":
        sv = shap_values if not isinstance(shap_values, list) else shap_values[1]
        if isinstance(sv, np.ndarray) and sv.ndim == 3:
            sv = sv[:, :, 1]
        mean_shap_vals = np.abs(sv).mean(axis=0)
    else:
        if isinstance(shap_values, list):
            sv_arr = np.abs(np.stack(shap_values, axis=-1))
            mean_shap_vals = sv_arr.mean(axis=(0, 2))
        elif shap_values.ndim == 3:
            mean_shap_vals = np.abs(shap_values).mean(axis=(0, 2))
        else:
            mean_shap_vals = np.abs(shap_values).mean(axis=0)

    mean_shap = (pd.Series(mean_shap_vals, index=X.columns)
                   .sort_values(ascending=False)
                   .reset_index())
    mean_shap.columns = ["feature", "mean_abs_shap"]
    mean_shap.insert(0, "rank", range(1, len(mean_shap) + 1))
    mean_shap["mean_abs_shap"] = mean_shap["mean_abs_shap"].round(5)
    mean_shap.to_csv(out_dir / "shap_values.csv", index=False)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, max(4, len(X.columns) * 0.3)))
    if task == "binary":
        shap.summary_plot(sv, X_full, show=False, max_display=30, plot_size=None)
        plt.tight_layout()
    else:
        mean_shap.set_index("feature")["mean_abs_shap"].sort_values().tail(30).plot(
            kind="barh", ax=ax)
        ax.set_xlabel("Mean |SHAP|")
        ax.set_title(f"Feature importance — {out_dir.name}", fontsize=9)
        plt.tight_layout()

    plt.savefig(out_dir / "shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close()

    return metrics


# ---------------------------------------------------------------------------
# 8. Run all 8 variants
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("Running model variants…")

all_cols = [c for c in df_full.columns
            if c not in ("Project key","diagnosis","_diag_code")]

results = []

for tier_name, tier_vars in TIER_MAP.items():
    feat_cols = [v for v in tier_vars if v in df_full.columns]
    X_base    = df_full[feat_cols].copy()
    n_feat    = X_base.shape[1]
    n_present = X_base.notna().all(axis=1).sum()
    print(f"\n{tier_name}: {n_feat} features present")

    for task in ["binary", "multi"]:
        if task == "binary":
            y    = (df_full["_diag_code"] != 0.0).astype(int)
            lbl  = ["PD", "PD-plus"]
            mask = pd.Series(True, index=df_full.index)
        else:
            # Multiclass: PD=0, PSP=1, MSA=2, DLB=3, CBS=4
            code_to_mc = {0.0: 0, 1.0: 1, 2.0: 2, 4.0: 3, 3.0: 4}
            mc = df_full["_diag_code"].map(code_to_mc)
            mask = mc.notna()
            y    = mc[mask].astype(int)
            lbl  = ["PD","PSP","MSA","DLB","CBS"]

        X_task = X_base[mask].copy()

        for balanced in [True, False]:
            bal_str  = "balanced" if balanced else "unbalanced"
            var_name = f"{tier_name}_{task}_{bal_str}"
            out_dir  = OUTPUT / var_name
            print(f"  → {var_name}  ({X_task.shape[0]} samples, {n_feat} features)", flush=True)

            try:
                y_task  = y.reset_index(drop=True)
                X_reset = X_task.reset_index(drop=True)

                # Drop features with >80% missing (too sparse to be informative)
                keep = [c for c in X_reset.columns if X_reset[c].isna().mean() <= 0.80]
                X_reset = X_reset[keep]
                # Remaining NaN values passed directly to XGBoost (native split handling)

                m = run_variant(X_reset, y_task, task, balanced, lbl, out_dir)
                results.append({
                    "variant":            var_name,
                    "tier":               tier_name,
                    "task":               task,
                    "balanced":           balanced,
                    "n_samples":          m["n_samples"],
                    "n_features":         len(keep),
                    "roc_auc":            m["roc_auc"],
                    "f1_macro":           m["f1_macro"],
                    "f1_weighted":        m["f1_weighted"],
                    "precision_macro":    m["precision_macro"],
                    "recall_macro":       m["recall_macro"],
                    "precision_weighted": m["precision_weighted"],
                    "recall_weighted":    m["recall_weighted"],
                })
                print(f"     AUC={m['roc_auc']:.3f}  F1={m['f1_macro']:.3f}  "
                      f"Prec={m['precision_macro']:.3f}  Rec={m['recall_macro']:.3f}")
            except Exception as exc:
                print(f"     ERROR: {exc}")
                results.append({"variant": var_name, "error": str(exc)})

# ---------------------------------------------------------------------------
# 9. Summary table (full feature set)
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT / "results_summary.csv", index=False)
print("\n" + "="*60)
print("Results summary (full features):")
print(results_df[["variant","n_samples","n_features","roc_auc",
                   "f1_macro","precision_macro","recall_macro"]].to_string(index=False))


# ---------------------------------------------------------------------------
# 10. Top-20 SHAP feature selection pass
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("Top-20 SHAP feature selection pass…")

# Use TierAB_binary_balanced SHAP as the selector (best model)
shap_ref_path = OUTPUT / "TierAB_binary_balanced" / "shap_values.csv"
shap_ref = pd.read_csv(shap_ref_path)
top20_features = shap_ref.head(20)["feature"].tolist()
print(f"  Top 20 features (from TierAB_binary_balanced SHAP):")
for i, f in enumerate(top20_features, 1):
    score = shap_ref.loc[shap_ref["feature"] == f, "mean_abs_shap"].values[0]
    print(f"    {i:2}. {f:<30}  {score:.5f}")

results_top20 = []

for tier_name, tier_vars in TIER_MAP.items():
    # Keep only features that are both in this tier AND in top 20
    feat_cols = [v for v in tier_vars if v in df_full.columns and v in top20_features]
    if not feat_cols:
        print(f"\n{tier_name}: no top-20 features — skipping")
        continue
    n_eligible = len(feat_cols)  # tier-eligible subset of the top-20 pool
    X_base = df_full[feat_cols].copy()
    print(f"\n{tier_name} (top-20 subset): {n_eligible}/20 eligible — {feat_cols}")

    for task in ["binary", "multi"]:
        if task == "binary":
            y    = (df_full["_diag_code"] != 0.0).astype(int)
            lbl  = ["PD", "PD-plus"]
            mask = pd.Series(True, index=df_full.index)
        else:
            code_to_mc = {0.0: 0, 1.0: 1, 2.0: 2, 4.0: 3, 3.0: 4}
            mc   = df_full["_diag_code"].map(code_to_mc)
            mask = mc.notna()
            y    = mc[mask].astype(int)
            lbl  = ["PD","PSP","MSA","DLB","CBS"]

        X_task = X_base[mask].copy()

        for balanced in [True, False]:
            bal_str  = "balanced" if balanced else "unbalanced"
            var_name = f"{tier_name}_{task}_{bal_str}_top20"
            out_dir  = OUTPUT / var_name
            print(f"  → {var_name}", flush=True)

            try:
                y_task  = y.reset_index(drop=True)
                X_reset = X_task.reset_index(drop=True)
                keep    = [c for c in X_reset.columns if X_reset[c].isna().mean() <= 0.80]
                X_reset = X_reset[keep]

                m = run_variant(X_reset, y_task, task, balanced, lbl, out_dir)
                results_top20.append({
                    "variant":            var_name,
                    "tier":               tier_name,
                    "task":               task,
                    "balanced":           balanced,
                    "n_top_pool":         20,
                    "n_eligible":         n_eligible,
                    "n_features":         len(keep),
                    "roc_auc":            m["roc_auc"],
                    "f1_macro":           m["f1_macro"],
                    "f1_weighted":        m["f1_weighted"],
                    "precision_macro":    m["precision_macro"],
                    "recall_macro":       m["recall_macro"],
                    "precision_weighted": m["precision_weighted"],
                    "recall_weighted":    m["recall_weighted"],
                })
                print(f"     AUC={m['roc_auc']:.3f}  F1={m['f1_macro']:.3f}  "
                      f"Prec={m['precision_macro']:.3f}  Rec={m['recall_macro']:.3f}")
            except Exception as exc:
                print(f"     ERROR: {exc}")
                results_top20.append({"variant": var_name, "error": str(exc)})

# ---------------------------------------------------------------------------
# 11. Comparison table: full vs top-20
# ---------------------------------------------------------------------------
top20_df = pd.DataFrame(results_top20)
top20_df.to_csv(OUTPUT / "results_summary_top20.csv", index=False)

print("\n" + "="*60)
print("Comparison: full features vs top-20 SHAP selection")
print(f"\n{'Variant':<35} {'AUC(F)':>7} {'AUC(20)':>8} {'ΔAUC':>6}  "
      f"{'Pre(F)':>7} {'Pre(20)':>8}  {'Rec(F)':>7} {'Rec(20)':>8}  {'F1(F)':>6} {'F1(20)':>7}")
print("-" * 110)

full_lookup = results_df.set_index("variant")
for row in results_top20:
    if "error" in row:
        continue
    v20   = row["variant"]
    vfull = v20.replace("_top20", "")
    if vfull not in full_lookup.index:
        continue
    fl = full_lookup.loc[vfull]
    d_auc = row["roc_auc"] - fl["roc_auc"]
    print(f"{vfull:<35} {fl['roc_auc']:>7.4f} {row['roc_auc']:>8.4f} {d_auc:>+6.4f}  "
          f"{fl['precision_macro']:>7.4f} {row['precision_macro']:>8.4f}  "
          f"{fl['recall_macro']:>7.4f} {row['recall_macro']:>8.4f}  "
          f"{fl['f1_macro']:>6.4f} {row['f1_macro']:>7.4f}")

print(f"\n→ All outputs saved to {OUTPUT}/")

# ---------------------------------------------------------------------------
# 12. Top-15 SHAP — binary only
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("Top-15 SHAP feature selection — binary only…")

top15_features = shap_ref.head(15)["feature"].tolist()
print(f"  Top 15 features:")
for i, f in enumerate(top15_features, 1):
    score = shap_ref.loc[shap_ref["feature"] == f, "mean_abs_shap"].values[0]
    print(f"    {i:2}. {f:<30}  {score:.5f}")

results_top15 = []

for tier_name, tier_vars in TIER_MAP.items():
    feat_cols = [v for v in tier_vars if v in df_full.columns and v in top15_features]
    if not feat_cols:
        continue
    n_eligible = len(feat_cols)  # tier-eligible subset of the top-15 pool
    X_base = df_full[feat_cols].copy()
    print(f"\n{tier_name} (top-15 subset): {n_eligible}/15 eligible — {feat_cols}")

    y    = (df_full["_diag_code"] != 0.0).astype(int)
    lbl  = ["PD", "PD-plus"]

    for balanced in [True, False]:
        bal_str  = "balanced" if balanced else "unbalanced"
        var_name = f"{tier_name}_binary_{bal_str}_top15"
        out_dir  = OUTPUT / var_name
        print(f"  → {var_name}", flush=True)

        try:
            y_task  = y.reset_index(drop=True)
            X_reset = X_base.reset_index(drop=True)
            keep    = [c for c in X_reset.columns if X_reset[c].isna().mean() <= 0.80]
            X_reset = X_reset[keep]

            m = run_variant(X_reset, y_task, "binary", balanced, lbl, out_dir)
            results_top15.append({
                "variant":            var_name,
                "tier":               tier_name,
                "task":               "binary",
                "balanced":           balanced,
                "n_top_pool":         15,
                "n_eligible":         n_eligible,
                "n_features":         len(keep),
                "roc_auc":            m["roc_auc"],
                "f1_macro":           m["f1_macro"],
                "f1_weighted":        m["f1_weighted"],
                "precision_macro":    m["precision_macro"],
                "recall_macro":       m["recall_macro"],
                "precision_weighted": m["precision_weighted"],
                "recall_weighted":    m["recall_weighted"],
            })
            print(f"     AUC={m['roc_auc']:.3f}  F1={m['f1_macro']:.3f}  "
                  f"Prec={m['precision_macro']:.3f}  Rec={m['recall_macro']:.3f}")
        except Exception as exc:
            print(f"     ERROR: {exc}")
            results_top15.append({"variant": var_name, "error": str(exc)})

top15_df = pd.DataFrame(results_top15)
top15_df.to_csv(OUTPUT / "results_summary_top15.csv", index=False)

# ---------------------------------------------------------------------------
# 13. Final comparison: full vs top-20 vs top-15 (binary only)
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("Binary comparison: full vs top-20 vs top-15")

top20_lookup = pd.DataFrame(results_top20).set_index("variant") \
    if results_top20 else pd.DataFrame()
top15_lookup = top15_df.set_index("variant") if not top15_df.empty else pd.DataFrame()

hdr = (f"\n{'Variant':<30} "
       f"{'AUC':>6} {'P':>6} {'R':>6} {'F1':>6}  "
       f"{'AUC20':>6} {'P20':>6} {'R20':>6} {'F120':>6}  "
       f"{'AUC15':>6} {'P15':>6} {'R15':>6} {'F115':>6}")
print(hdr)
print("-" * len(hdr))

for tier_name in TIER_MAP:
    for bal_str in ["balanced", "unbalanced"]:
        vfull = f"{tier_name}_binary_{bal_str}"
        v20   = vfull + "_top20"
        v15   = vfull + "_top15"
        if vfull not in full_lookup.index:
            continue
        fl  = full_lookup.loc[vfull]
        r20 = top20_lookup.loc[v20] if v20 in top20_lookup.index else None
        r15 = top15_lookup.loc[v15] if v15 in top15_lookup.index else None

        def _v(row, key): return f"{row[key]:>6.4f}" if row is not None else f"{'—':>6}"

        print(f"{vfull:<30} "
              f"{fl['roc_auc']:>6.4f} {fl['precision_macro']:>6.4f} "
              f"{fl['recall_macro']:>6.4f} {fl['f1_macro']:>6.4f}  "
              f"{_v(r20,'roc_auc')} {_v(r20,'precision_macro')} "
              f"{_v(r20,'recall_macro')} {_v(r20,'f1_macro')}  "
              f"{_v(r15,'roc_auc')} {_v(r15,'precision_macro')} "
              f"{_v(r15,'recall_macro')} {_v(r15,'f1_macro')}")

print(f"\n→ All outputs saved to {OUTPUT}/")
