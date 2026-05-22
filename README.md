# C-OPN Parkinsonism Classification Study

Machine learning pipeline for differentiating Parkinson's Disease (PD) from Atypical Parkinsonism (AP: PSP, MSA, DLB, CBS) using the Canadian Open Parkinson Network (C-OPN) cohort.

---

## Project Structure

```
data/                    Raw REDCap CSV exports (one file per form, no extension)
reference/               Data dictionary and study documentation
  COPN_DataDictionary_2025-09-24_annotated.xlsx
pending/                 Working/input files
  WORK-Qnaire-and-feature-clinical-domains.xlsx
  COPN_Selected_Features_v3_11May2026.xlsx
source/                  Encrypted source Excel file
output/
  clean_pipeline/        Filtered CSVs (enrolled + complete rows only)
    full_enrolled/
    enrolled_and_partial/
  comparative_analysis/  Statistical plots and test results per domain
  model/                 XGBoost model outputs
scripts/                 All pipeline scripts
```

---

## Setup

### 1. Place source file

Put `SSC - Full report - UPDATED.xlsx` in `source/`.

### 2. Create `.env`

```
PASSWORD="your_password_here"
```

### 3. Install dependencies

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Dependencies: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, `xgboost`, `shap`, `openpyxl`

---

## Full Run Order

```bash
# Phase 1 — Data extraction and audit
python3 scripts/extract_csv.py
python3 scripts/export_stats.py

# Phase 2 — Clean pipeline + domain splits
python3 scripts/build_domain_pipeline.py

# Phase 3 — Comparative analysis
python3 scripts/build_comparative_by_domain.py

# Phase 4 — ML model
python3 scripts/build_model_v1.py
```

---

## Phase 1 — Data Extraction and Audit

### `extract_csv.py`

Decrypts `source/SSC - Full report - UPDATED.xlsx` using the password in `.env` and writes one CSV per sheet into `data/`. All subsequent scripts read from `data/`.

### `export_stats.py` → `output/cross_reference.xlsx`

Reads all `data/` CSVs and the data dictionary, then builds a full audit workbook:

| Sheet | Contents |
|-------|----------|
| All Variables | Every variable — source file, % missing, sample values |
| Consolidated | One row per base variable, versioned variables grouped |
| Numerical Stats | Mean, median, SD, min, Q1, Q3, max, skew, kurtosis |
| Categorical Stats | Value counts and % per variable |
| Diagnosis Flags | Participants with inconsistencies between enrolment group, determined diagnosis, and "Was diagnosed with PD?" |
| Diagnosis Review | All 3,541 participants with flag status and flag reason |
| Completeness Matrix | Participants × forms heatmap — % fields missing per cell |
| Completeness by Dx | Average % missing per form, by enrolment group |
| Enrollment Coverage | Every participant × every form — Yes/No data present |
| Withdrawn Summary | 246 withdrawn participants with per-form coverage |

**Flags checked:** enrolled as PD but determined Dx ≠ PD; enrolled as AP but determined Dx = PD; "Was diagnosed with PD?" contradicts determined Dx; 1a alternative Dx contradicts determined code.

---

## Phase 2 — Clean Pipeline (`build_domain_pipeline.py`)

**What it does:** Filters raw data to enrolled + complete rows, matches each CSV column to its clinical domain, and produces domain-split CSVs.

**Inputs:**
- `data/` — raw REDCap CSVs
- `reference/COPN_DataDictionary_2025-09-24_annotated.xlsx` — Field Labels (= exact CSV column header text) + Field Type per variable
- `pending/WORK-Qnaire-and-feature-clinical-domains.xlsx` — Variable → General Clinical Domain

**Key logic:**
- **Enrollment filter:** `Study Status = Enrolled/Inscrit` (full_enrolled) or + partially enrolled
- **Row filter:** `Complete? = Complete` per form
- **Column matching:** Jaccard similarity between CSV column headers and Field Labels from the data dictionary. Field Labels are the exact REDCap export text so matches score ~1.0 (near-perfect). Threshold: 0.45.
- **Form mapping:** Auto-derived by counting variable overlap between old file names and new DD form names — no hard-coded mapping.
- **Domain merging:** `"Cognitive Functioning"` → `"Cognitive Function"`, `"Mood / Psychiatric "` → `"Mood / Psychiatric"`

**Outputs:**
```
output/clean_pipeline/full_enrolled/
  {form}.csv                                   one per form, enrolled+complete rows
  by_clinical_domain/{Domain}/{form}.csv       domain-filtered column subsets
  domain_column_mapping.csv                    CSV col → variable → domain → Field Type → match score
  domain_summary.csv                           row/column counts per domain × form × group
```

**Domains:** Admin · Activities of Daily Living · Autonomic Functioning · Behaviour · Clinical History & Diagnosis · Cognitive Function · Composite Scores · Medications · Mood/Psychiatric · Motor · Non-Motor

---

## Phase 3 — Comparative Analysis (`build_comparative_by_domain.py`)

**What it does:** For each clinical domain, compares groups statistically and generates plots.

**Input:** `output/clean_pipeline/full_enrolled/by_clinical_domain/`

**Groups compared:** by enrolment group (PD / AP / HC) and by determined diagnosis (PD / PSP / MSA / DLB / CBS / HC / ET / RBD)

**Column routing via Field Type from `domain_column_mapping.csv`:**
- `radio`, `yesno`, `checkbox`, `dropdown` → **bar plots** (% per response per group) + chi-square test
- `calc`, `text` (numeric) → **box plots** + Kruskal-Wallis test

**Statistics:**
- Numeric: Kruskal-Wallis H statistic + η² effect size + BH FDR correction; pairwise Mann-Whitney U + Bonferroni correction + Cohen's d
- Categorical: chi-square contingency test per variable
- Min group size: 5; max 30 subplots per figure

**Outputs:**
```
output/comparative_analysis/by_clinical_domain/{Domain}/
  by_enrolment_group/    plots/  stats.xlsx  pairwise.xlsx
  by_determined_dx/      plots/  stats.xlsx  pairwise.xlsx
  significant_fdr.csv    all BH-significant findings across all domains
```

---

## Phase 4 — Feature Selection

51 features selected from comparative analysis results, clinical literature, and MDS diagnostic criteria. Documented in `pending/COPN_Selected_Features_v3_11May2026.xlsx`.

**Tiers by implementation difficulty:**

| Tier | Level | n | Description |
|------|-------|---|-------------|
| A | 1–2 | 33 | Self-report only — completable remotely without a clinic visit |
| B | 3 | 17 | Clinic visit required — UPDRS Part 3 motor exam + MoCA |
| C | 4 | 1 | Trail Making B — excluded from models (85% missing data) |

**Tier A:** age, sex, education; UPDRS Part 1 (cognitive, mood, autonomic NMS items 1.1–1.5, 1.10–1.13); UPDRS Part 2 (motor ADL self-report items 2.1, 2.5, 2.7, 2.8, 2.10, 2.12, 2.13); prodromal NMS (smell loss, REM sleep behaviour disorder, constipation); family history (grandparents, aunts/uncles with PD); comorbidities; head injury; pesticide exposure; symptom asymmetry at onset and current; first symptom type; PDQ-39 mobility, communication, summary index

**Tier B:** UPDRS Part 3 composite scores + individual items (gait, freezing, postural stability, posture, body bradykinesia, speech, facial expression, neck rigidity, rest tremor constancy); MoCA total + visuospatial + attention subscores

**Composite score definitions:**

| Composite | Constituent variables | Formula |
|-----------|----------------------|---------|
| `bradykinesia_mean` | UPDRS 3.4–3.8 (R+L) — 10 items | mean |
| `rest_tremor_mean` | UPDRS 3.17 RUE/LUE/RLE/LLE | mean |
| `action_tremor_mean` | UPDRS 3.15–3.16 (R+L) | mean |
| `rigidity_limb_mean` | UPDRS 3.3 RUE/LUE/RLE/LLE | mean |
| `asymmetry_index` | Right vs left bradykinesia means | `\|R − L\| / (R + L + 1)` |

**Leakage exclusions:** diagnosis labels, enrolment group, disease duration, age at symptom onset

---

## Phase 5 — ML Model (`build_model_v1.py`)

### Cohort

| Stage | n |
|-------|---|
| Total C-OPN participants | 3,541 |
| Enrolled (full_enrolled) | 2,286 |
| Confirmed PD/AP diagnosis | **1,704** |
| — PD | 1,651 |
| — PSP | 25 |
| — MSA | 16 |
| — DLB | 9 |
| — CBS | 3 |

### Feature extraction

| Column type | Extraction method |
|-------------|------------------|
| `Updrs_X_Y value` or `Updrs_X_Y` | Direct numeric (pre-computed in REDCap) |
| UPDRS text e.g. `"2: Mild: …"` | Regex extract leading integer (0–4), strips HTML tags |
| Yes/No epidemiological fields | Yes=1, No=0, Uncertain/Unknown=NaN |
| Multi-select (first symptoms, comorbidities) | Binary flags per category |
| Symptom asymmetry Q8 | Unilateral (R or L)=1, Bilateral=0 |
| Current asymmetry Q10 | One side only=1, One side more=0.5, Equal both=0 |
| PDQ-39, MoCA, Trail B | Direct numeric |

### Missing data handling

- Features with **>80% missing** dropped (Trail B: 85% → excluded)
- All remaining NaN values passed **directly to XGBoost** — no imputation, no row dropping
- XGBoost learns the optimal split direction for missing observations at each node
- PDQ-39 (~76% missing) is retained and handled natively

### Model configuration

**Algorithm:** XGBoost (`n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8`)  
**Validation:** 5-fold stratified cross-validation (`random_state=42`)  
**Class balancing:**
- Binary: `scale_pos_weight = n_PD / n_AP`
- Multiclass: per-sample weights = `n_total / (n_classes × n_class_i)`, `minlength=n_classes` to handle absent classes in folds  
**Interpretability:** SHAP TreeExplainer on model trained on full data; multiclass SHAP shape `(n_samples, n_features, n_classes)` averaged over samples and classes

### 8 variants = 2 tiers × 2 tasks × 2 balance settings

| Variant | AUC | F1 macro | F1 weighted |
|---------|-----|----------|-------------|
| TierAB_binary_balanced | **0.923** | 0.333 | 0.962 |
| TierAB_multi_unbalanced | 0.921 | 0.288 | 0.961 |
| TierAB_binary_unbalanced | 0.918 | 0.254 | 0.961 |
| TierAB_multi_balanced | 0.905 | **0.368** | 0.958 |
| TierA_binary_balanced | 0.906 | 0.273 | 0.959 |
| TierA_multi_unbalanced | 0.900 | 0.244 | 0.956 |
| TierA_binary_unbalanced | 0.904 | 0.206 | 0.960 |
| TierA_multi_balanced | 0.882 | 0.337 | 0.957 |

AUC gain Tier A → Tier AB: **+0.017** (binary), **+0.021** (multiclass)

**Targets:**
- Binary: PD=0, PD-plus (PSP+MSA+DLB+CBS)=1
- Multiclass: PD=0, PSP=1, MSA=2, DLB=3, CBS=4

### Outputs

```
output/model/{variant}/
  confusion_matrix.png    cross-validated confusion matrix
  shap_summary.png        mean |SHAP| feature importance (top 30)
  metrics.json            AUC, F1-macro, F1-weighted, per-class precision/recall/F1

output/model/
  feature_matrix.csv      1704 × 53 joined feature table
  results_summary.csv     one row per variant
```

---

## Key Methodological Notes

- **Column matching:** REDCap exports use full bilingual question text as column headers. Matching to the data dictionary uses Field Labels (identical text) via Jaccard — near-perfect scores (~1.0), not manual mapping.
- **No imputation:** XGBoost native NaN handling used throughout. Median imputation was tested but produces ~76% synthetic values for PDQ-39 (76% missing); complete-case drops 90% of samples. Native handling is preferred.
- **Class imbalance:** Severe (97% PD vs 3% AP). Balanced variants up-weight minority classes. Report AUC (discrimination ability) alongside F1-macro (sensitivity to minority class performance) — they capture different things.
- **AP subtype reliability:** PSP=25, MSA=16, DLB=9, CBS=3. Multiclass results for CBS especially should be interpreted with caution.
