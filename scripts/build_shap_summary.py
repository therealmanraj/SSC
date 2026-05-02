"""
Extracts and saves SHAP feature importance from XGBoost across all 5 tasks.

Reuses the same feature engineering as build_ml_v3.py but:
  - Trains XGBoost only (fastest, SHAP-native)
  - Saves mean |SHAP| per feature per task to Excel
  - Generates ranked bar chart per task + cross-task heatmap

Output:
  output/shap_summary/
    shap_values.xlsx   one sheet per task — features ranked by mean |SHAP|
    plots/
      task_a.png  task_b.png  ...  heatmap.png
"""

import os, re, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings("ignore")

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR  = os.path.join(ROOT, "output", "shap_summary")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers (mirrors build_ml_v3.py)
# ---------------------------------------------------------------------------

def load(name):
    df = pd.read_csv(os.path.join(DATA_DIR, name))
    if df.columns[0].startswith("Unnamed"):
        df = df.iloc[:, 1:]
    return df

def first_col(df, *keywords):
    kws = [k.lower() for k in keywords]
    for c in df.columns:
        if all(k in c.lower() for k in kws):
            return c
    return None

def yn(v):
    s = str(v).lower()
    return 1 if ("yes" in s or "oui" in s) else (0 if ("no" in s or "non" in s) else np.nan)

def make_xgb(scale_pos=None):
    params = dict(n_estimators=400, learning_rate=0.05, max_depth=5,
                  random_state=42, verbosity=0, eval_metric="logloss", tree_method="hist")
    if scale_pos:
        params["scale_pos_weight"] = scale_pos
    return XGBClassifier(**params)


# ---------------------------------------------------------------------------
# Feature engineering (same as build_ml_v3.py)
# ---------------------------------------------------------------------------

print("Loading data…")
enroll = load("Enrollement"); clin = load("Clinical"); moca = load("MoCA")
updrs  = load("MDS-UPDRS");   updrs3 = load("UPDRS 1.2 part 3")
med    = load("Medication");   scopa  = load("SCOPA")
demo   = load("Demographic");  neuro  = load("Neuropsychological")

DIAG_LABEL = {0:"PD",1:"PSP",2:"MSA",3:"CBS",4:"DLB",6:"ET",7:"RBD",10:"HC"}
AP_CODES       = {1,2,3,4,6,7}
AP_CODES_NO_ET = {1,2,3,4,7}

enrol_col  = first_col(enroll, "nrolment", "roup")
status_col = first_col(enroll, "Study Status")
diag_col   = first_col(clin,   "Determined diagnosis")

enrolled_complete = enroll[
    enroll[status_col].str.contains("Enrolled", na=False) &
    (enroll["Complete?"] == "Complete")
].drop_duplicates("Project key")[["Project key", enrol_col]]

hc_keys = set(enrolled_complete[
    enrolled_complete[enrol_col].str.contains("Healthy", na=False)
]["Project key"])

clin_labels = clin[["Project key", diag_col]].drop_duplicates("Project key").copy()
clin_labels["det_code"] = pd.to_numeric(clin_labels[diag_col], errors="coerce")
clin_labels.loc[clin_labels["Project key"].isin(hc_keys) & clin_labels["det_code"].isna(), "det_code"] = 10
clin_labels["label"] = clin_labels["det_code"].map(DIAG_LABEL)

labels = enrolled_complete.merge(clin_labels[["Project key","det_code","label"]], on="Project key", how="left")

print("Engineering features…")
feats = labels[["Project key"]].copy()

# 1. Clinical
c1 = clin.drop_duplicates("Project key").copy()
for fn, col in [("disease_duration", first_col(c1,"duration of disease at the time")),
                ("age_at_onset",      first_col(c1,"age at onset")),
                ("bmi",               first_col(c1,"bmi")),
                ("height_cm",         first_col(c1,"height of the participant (cm)")),
                ("weight_kg",         first_col(c1,"weight of the participant (kg)"))]:
    if col: feats[fn] = feats["Project key"].map(c1.set_index("Project key")[col].apply(pd.to_numeric, errors="coerce"))

for fn, col in {"has_dyskinesia":first_col(c1,"dyskinesia","currently"),
                "has_freezing":  first_col(c1,"freezing of gait"),
                "has_falls":     first_col(c1,"fallen in the last 3 months"),
                "has_dementia":  first_col(c1,"does the patient have dementia"),
                "has_dbs":       first_col(c1,"surgery for parkinson"),
                "has_motor_fluctuations":first_col(c1,"motor fluctuations"),
                "has_remission": first_col(c1,"complete remission"),
                "bilateral_onset":first_col(c1,"both sides of your body")}.items():
    if col: feats[fn] = feats["Project key"].map(c1.set_index("Project key")[col].apply(yn))

hc = first_col(c1,"dominant hand")
if hc: feats["right_handed"] = feats["Project key"].map(
    c1.set_index("Project key")[hc].apply(lambda v: 1 if "right" in str(v).lower() else (0 if "left" in str(v).lower() else np.nan)))

# 2. MoCA
m1 = moca.drop_duplicates("Project key").copy()
for fn, col in {"moca_visuospatial":first_col(m1,"visuospatial"),
                "moca_naming":      first_col(m1,"naming","score"),
                "moca_attention":   first_col(m1,"attention","score"),
                "moca_language":    first_col(m1,"language","score"),
                "moca_abstraction": first_col(m1,"abstraction","score"),
                "moca_recall":      first_col(m1,"recall","score"),
                "moca_orientation": first_col(m1,"orientation","score"),
                "moca_total":       first_col(m1,"total score")}.items():
    if col: feats[fn] = feats["Project key"].map(m1.set_index("Project key")[col].apply(pd.to_numeric, errors="coerce"))

# 3. UPDRS part totals
u1 = updrs.drop_duplicates("Project key").copy()
for fn, col in {"updrs_part1":first_col(u1,"part i"),
                "updrs_part2":first_col(u1,"part ii"),
                "updrs_part3":first_col(u1,"part iii"),
                "updrs_part4":first_col(u1,"part iv")}.items():
    if col: feats[fn] = feats["Project key"].map(u1.set_index("Project key")[col].apply(pd.to_numeric, errors="coerce"))

# 4. SCOPA
s1 = scopa.drop_duplicates("Project key").copy()
for fn, col in {"scopa_gi":        first_col(s1,"gastrointestinal"),
                "scopa_urinary":   first_col(s1,"urinary"),
                "scopa_cardiovasc":first_col(s1,"cardiovascular"),
                "scopa_thermoreg": first_col(s1,"thermoregulatory"),
                "scopa_pupilmotor":first_col(s1,"pupillomotor")}.items():
    if col: feats[fn] = feats["Project key"].map(s1.set_index("Project key")[col].apply(pd.to_numeric, errors="coerce"))

# 5. Demographics
d1 = demo.drop_duplicates("Project key").copy()
age_c = first_col(d1,"age at study visit","automatic") or first_col(d1,"age at study visit")
for fn, col in [("age", age_c), ("edu_years", first_col(d1,"years of education"))]:
    if col: feats[fn] = feats["Project key"].map(d1.set_index("Project key")[col].apply(pd.to_numeric, errors="coerce"))
sx = first_col(d1,"gender")
if sx: feats["sex_male"] = feats["Project key"].map(
    d1.set_index("Project key")[sx].apply(
        lambda v: 1 if "male" in str(v).lower() and "female" not in str(v).lower()
        else (0 if "female" in str(v).lower() else np.nan)))

# 6. Neuropsychological
n1 = neuro.drop_duplicates("Project key").copy()
for fn, col in {"neuro_hvlt_total":  first_col(n1,"trial total 1,2,3","raw"),
                "neuro_hvlt_delayed":first_col(n1,"trial 4 delayed"),
                "neuro_digit_fwd":   first_col(n1,"digit span forward","total correct"),
                "neuro_digit_bwd":   first_col(n1,"digit span backward","total correct"),
                "neuro_trail_a":     first_col(n1,"trail a raw score"),
                "neuro_trail_b":     first_col(n1,"trail b raw score") or first_col(n1,"trail b")}.items():
    if col: feats[fn] = feats["Project key"].map(n1.set_index("Project key")[col].apply(pd.to_numeric, errors="coerce"))

# 7. UPDRS 1.2 Part 3 subscores
u3 = updrs3.drop_duplicates("Project key").copy()
for fn, col in {"u3_tremor_total":      first_col(u3,"tremor total"),
                "u3_rigidity_total":    first_col(u3,"rigidity total"),
                "u3_tremor_rigid_ratio":first_col(u3,"ratio tremor"),
                "u3_laterality_index":  first_col(u3,"laterality index")}.items():
    if col: feats[fn] = feats["Project key"].map(u3.set_index("Project key")[col].apply(pd.to_numeric, errors="coerce"))

# 8. MDS-UPDRS PIGD / Tremor-Dominant
_pigd   = [c for c in u1.columns if any(f"Updrs_3_{n}" in c for n in ("10","11","12","13","14"))]
_tremor = [c for c in u1.columns if any(f"Updrs_3_{n}" in c for n in ("15","16","17","18"))]
if _pigd:   feats["updrs_pigd"]         = feats["Project key"].map(u1.set_index("Project key")[_pigd].apply(pd.to_numeric, errors="coerce").mean(axis=1))
if _tremor: feats["updrs_tremor_items"] = feats["Project key"].map(u1.set_index("Project key")[_tremor].apply(pd.to_numeric, errors="coerce").mean(axis=1))
if _pigd and _tremor:
    _t = feats["updrs_tremor_items"].fillna(0); _p = feats["updrs_pigd"].fillna(0)
    feats["updrs_td_score"] = np.where((_t+_p)>0, _t/(_t+_p), np.nan)

# 9. Medication
me = med.drop_duplicates("Project key").copy()
lrc = first_col(me,"significant reduction")
lcc = first_col(me,"Select all current")
ldc = [c for c in me.columns if "levodopa dosage" in c.lower() or "lévodopa" in c.lower()]
if lrc: feats["levo_response"] = feats["Project key"].map(
    me.set_index("Project key")[lrc].apply(
        lambda v: 1.0 if "yes" in str(v).lower() or "oui" in str(v).lower()
        else (0.0 if "no/" in str(v).lower() or "non/" in str(v).lower()
              else (0.5 if "uncertain" in str(v).lower() else np.nan))))
if lcc: feats["on_levodopa"] = feats["Project key"].map(
    me.set_index("Project key")[lcc].apply(lambda v: 1 if "levodopa" in str(v).lower() else (0 if pd.notna(v) else np.nan)))
if ldc:
    feats["total_levo_dose"] = feats["Project key"].map(
        me.set_index("Project key")[ldc].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1))

FEAT_COLS = [c for c in feats.columns if c != "Project key" and feats[c].notna().any()]
df = feats.merge(labels[["Project key","det_code","label"]], on="Project key", how="inner")
print(f"  Master table: {len(df):,} rows × {len(FEAT_COLS)} features")


# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------

TASKS = {
    "Task A — PD vs HC":           df[df["label"].isin(["PD","HC"])].copy(),
    "Task B — PD vs AP":           df[df["det_code"].isin({0}|AP_CODES_NO_ET)].assign(
                                       task_label=lambda d: d["det_code"].apply(
                                           lambda c: "PD" if c==0 else ("AP" if c in AP_CODES_NO_ET else np.nan))),
    "Task C — PD vs HC vs AP":     df[df["label"].isin(["PD","HC"]) | df["det_code"].isin(AP_CODES)].copy(),
    "Task D — AP subtype":         df[df["det_code"].isin(AP_CODES)].copy(),
    "Task E — HC vs Non-HC":       df[df["label"].notna()].copy(),
}

TASK_LABEL_COL = {
    "Task A — PD vs HC":           "label",
    "Task B — PD vs AP":           "task_label",
    "Task C — PD vs HC vs AP":     "label",
    "Task D — AP subtype":         None,   # built below
    "Task E — HC vs Non-HC":       None,
}

# Build label cols for D and E
TASKS["Task D — AP subtype"]["task_label"] = TASKS["Task D — AP subtype"]["det_code"].apply(
    lambda c: "PSP" if c==1 else ("MSA" if c==2 else "DLB+other"))
TASKS["Task E — HC vs Non-HC"]["task_label"] = TASKS["Task E — HC vs Non-HC"]["det_code"].apply(
    lambda c: "HC" if c==10 else ("Non-HC" if pd.notna(c) else np.nan))
TASK_LABEL_COL["Task D — AP subtype"] = "task_label"
TASK_LABEL_COL["Task E — HC vs Non-HC"] = "task_label"

# Also fix Task C label col
TASKS["Task C — PD vs HC vs AP"]["label"] = TASKS["Task C — PD vs HC vs AP"]["det_code"].apply(
    lambda c: "PD" if c==0 else ("HC" if c==10 else ("AP" if c in AP_CODES else np.nan)))


shap_results = {}   # task_name → Series of mean |SHAP| indexed by feature

for task_name, df_task in TASKS.items():
    lbl_col = TASK_LABEL_COL[task_name]
    sub = df_task[FEAT_COLS + [lbl_col]].dropna(subset=[lbl_col]).copy()
    if sub[lbl_col].nunique() < 2 or len(sub) < 20:
        print(f"  {task_name}: skipping (n={len(sub)})")
        continue

    le = LabelEncoder()
    y  = le.fit_transform(sub[lbl_col])
    X  = sub[FEAT_COLS].values.astype(float)

    n_pd = (sub[lbl_col] == "PD").sum() if "PD" in sub[lbl_col].values else 0
    n_min = sub[lbl_col].value_counts().min()
    sp = round(n_pd / max(n_min, 1)) if n_pd > 0 and n_min > 0 and sub[lbl_col].nunique() == 2 else None
    if sub[lbl_col].nunique() > 2:
        sp = None

    print(f"  {task_name}: n={len(sub)}, classes={list(le.classes_)}")
    xgb = make_xgb(scale_pos=sp)
    xgb.fit(X, y)

    explainer = shap.TreeExplainer(xgb)
    shap_vals = explainer.shap_values(X)

    if isinstance(shap_vals, list):
        # old SHAP format: list of (n_samples, n_features) per class
        mean_abs = np.abs(np.array(shap_vals)).mean(axis=0).mean(axis=0)
    elif np.ndim(shap_vals) == 3:
        # new SHAP format: (n_samples, n_features, n_classes)
        mean_abs = np.abs(shap_vals).mean(axis=2).mean(axis=0)
    else:
        # binary: (n_samples, n_features)
        mean_abs = np.abs(shap_vals).mean(axis=0)

    ranked = pd.Series(mean_abs, index=FEAT_COLS).sort_values(ascending=False)
    shap_results[task_name] = ranked

    # --- bar chart ---
    top = ranked.head(20)
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = sns.color_palette("viridis_r", len(top))
    top[::-1].plot.barh(ax=ax, color=colors[::-1])
    ax.set_title(f"{task_name}\nTop features by mean |SHAP| (XGBoost)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Mean |SHAP value|", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=8)
    sns.despine()
    plt.tight_layout()
    fname = re.sub(r"[^\w]", "_", task_name).lower().strip("_") + ".png"
    plt.savefig(os.path.join(PLOT_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    plot saved → {fname}")


# ---------------------------------------------------------------------------
# Save to Excel
# ---------------------------------------------------------------------------

with pd.ExcelWriter(os.path.join(OUT_DIR, "shap_values.xlsx"), engine="openpyxl") as writer:
    for task_name, ranked in shap_results.items():
        out = ranked.reset_index()
        out.columns = ["Feature", "Mean |SHAP|"]
        out["Rank"] = range(1, len(out) + 1)
        out = out[["Rank", "Feature", "Mean |SHAP|"]]
        sheet = re.sub(r"[^\w]", "_", task_name)[:31]
        out.to_excel(writer, sheet_name=sheet, index=False)

print("  shap_values.xlsx saved")


# ---------------------------------------------------------------------------
# Cross-task heatmap (top 20 features by max SHAP across tasks)
# ---------------------------------------------------------------------------

if shap_results:
    all_feat_max = pd.concat(shap_results.values(), axis=1).fillna(0).max(axis=1)
    top_feats = all_feat_max.nlargest(20).index.tolist()

    heat_df = pd.DataFrame(
        {t: s.reindex(top_feats).fillna(0) for t, s in shap_results.items()}
    )
    # Shorten task names for axis
    heat_df.columns = [c.replace("Task ", "").replace(" — ", "\n") for c in heat_df.columns]

    fig, ax = plt.subplots(figsize=(max(6, len(heat_df.columns) * 1.6), 8))
    sns.heatmap(
        heat_df, annot=True, fmt=".3f", cmap="YlOrRd",
        linewidths=0.4, linecolor="white",
        cbar_kws={"label": "Mean |SHAP|"},
        ax=ax,
    )
    ax.set_title("Top 20 features — Mean |SHAP| across tasks", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  heatmap.png saved")

print("\nDone.")
