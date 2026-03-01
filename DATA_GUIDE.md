# C-OPN Data Guide for SSC 2026 Case Study

> **Purpose:** This guide helps team members quickly understand the extracted C-OPN dataset — what each file contains, what the columns mean, which variables matter most, and how to use them.

---

## 1. Background

This dataset comes from the **Canadian Open Parkinson Network (C-OPN)** — a pan-Canada multi-site study across 11 Movement Disorders centres in 5 provinces.

**Challenge Goal:** Build a classification model to distinguish:
- **PD** — Parkinson's Disease (majority class, ~84% of participants)
- **AP / PD-plus** — Atypical Parkinsonism: MSA, PSP, CBS, DLB, FTD, ET, RBD (~4.6%)
- **HC** — Healthy Controls (~11.4%)

**Total participants:** ~3,541 rows across all files (each row = one participant visit)

**Key constraint:** You are strongly encouraged to prefer models that use **easy-to-implement variables** (see Implementation Levels below).

---

## 2. Implementation Levels

Every predictor variable has an assigned difficulty level for real-world clinical use:

| Level | Description | Examples |
|-------|-------------|---------|
| **Level 1** | Self-reported by patient | Age, gender, education, lifestyle questions |
| **Level 2** | Total/sub-score of a self-report questionnaire | PDQ-39 total score, BAI total score |
| **Level 3** | Requires professional assistance, but routine in clinic | MoCA, MDS-UPDRS motor exam, Timed Up & Go |
| **Level 4** | Requires extra professional resources | Detailed neuropsychological testing |

When two models perform similarly, the one using lower-level variables is preferred.

---

## 3. Outcome Variables

> **Important:** In case of any discrepancy between outcome columns, always use **`diagnosis_determined`** as the ground truth.

These columns live in the **Enrollement** and **Clinical** files:

| Variable | File | Description |
|----------|------|-------------|
| `enrolment_group_v2` | Enrollement | Group label: `1=PD`, `2=PPS (PD-plus)`, `3=Healthy Control` |
| `enrolment_group_v3` | Enrollement | Same as above with updated label: `1=PD`, `2=AP (Atypical Parkinsonism)`, `3=Healthy Control` |
| `diagnosis_determined` | Clinical | **Primary outcome.** Granular diagnosis: `0=PD`, `1=PSP`, `2=MSA`, `3=CBS`, `4=DLB`, `5=FTD`, `6=ET`, `7=RBD` |
| `diagnosis_pd` | Clinical | Was the patient diagnosed with PD? `1=Yes`, `0=No`, `2=Uncertain` |
| `diagnosis_pdplus` | Clinical | If not PD, which specific PD-plus condition? (1–9) |
| `diagnosis_probability` | Clinical | Certainty of diagnosis: `>90%`, `50-89%`, `<50%`, `Unknown` |

**Distribution in the dataset:**
- PD: ~2,852 participants
- Atypical Parkinsonism (AP): ~171 participants
- Healthy Controls (HC): ~410 participants

---

## 4. Linking Key — `Project key`

Every file shares the column **`Project key`** — this is the **participant ID** used to join all files together.

Also watch for:
- **`Event Name`** — represents the visit/timepoint (C-OPN is longitudinal, multiple visits possible)
- The combination of `Project key` + `Event Name` uniquely identifies a row

---

## 5. File-by-File Reference

### 5.1 Enrollement
**Shape:** 3,541 rows × 41 columns
**Purpose:** Administrative enrollment info + primary group labels

| Column | What it means |
|--------|---------------|
| `Project key` | Participant ID |
| `Site:` | Which of the 9 C-OPN sites the participant is from (1=UBC, 2=U of Alberta, 3=U of Calgary, 4=U of Toronto, 5=Ottawa Hospital, 6=QPN/RPQ, 7=Western, 8=TWH, 9=Halifax) |
| `Enrolment Group: ...` | **Key outcome** — PD / AP / Healthy Control |
| `Date of enrollment...` | When participant enrolled |
| `Study Status:` | Active, withdrawn, excluded |
| `Recruitment Source:` | How participant was recruited |

> Most columns here are **Admin** role — not predictors. The main thing you need here is the **outcome label** and the **site** (for external validation splits).

---

### 5.2 Demographic
**Shape:** 3,541 rows × 50 columns
**Purpose:** Basic patient demographics — all **Level 1** (self-reported)

| Column | What it means | Level |
|--------|---------------|-------|
| `Age at study visit...` | Patient age at visit | 1 |
| `1. Gender/Genre:` | Gender | 1 |
| `2. What is your current marital status?` | Marital status | 1 |
| `3. What is your current living situation?` | Living alone, with others, etc. | 1 |
| `4. What is the highest level of education...` | Education level (categorical) | 1 |
| `5. Years of education...` | Education in years (numeric) | 1 |
| `What is your current source of income?` | Income source | 1 |
| `6. What is your current employment status?` | Employment | 1 |
| `7. Do you have a regular caregiver?` | Has caregiver | 1 |
| `Ethnicity` | Self-reported ethnicity | 1 |

> **Tip:** Age, gender, and education are strongly associated with PD risk and should be included in most models.

---

### 5.3 Clinical
**Shape:** 3,541 rows × 75 columns
**Purpose:** Clinical assessment by a movement disorder specialist — contains the **outcome variables** and key clinical features

| Column | What it means | Level |
|--------|---------------|-------|
| `Determined diagnosis:` | **Primary outcome** — PD vs. specific PD-plus condition | Outcome |
| `1. Was the patient diagnosed with PD?` | Binary PD diagnosis | Outcome |
| `1a. If No or Uncertain, is the diagnosis...` | Specific PD-plus type | Outcome |
| `1b. Level of certainty of diagnosis` | Confidence in diagnosis | Outcome |
| `A. Weight (lbs/kg)`, `B. Height (ft/cm)`, `BMI` | Physical measurements | 3 |
| `4. What were the first symptoms?` | Symptom at onset (tremor, gait, etc.) | 1 |
| `5. Date when symptoms first appear` | Symptom onset date | 3 |
| `Duration of disease (years)` | Time since diagnosis | 3 |
| `Age at diagnosis` | Age at diagnosis | 3 |
| `3. Contact with a nurse specializing in parkinsonism` | Nurse follow-up | Admin |

> **Tip:** `duration_disease` and `age_onset` are very informative for distinguishing PD from PD-plus.

---

### 5.4 Epidemiological
**Shape:** 3,541 rows × 97 columns
**Purpose:** Lifestyle, environment, and health history — all **Level 1** (self-reported)

Key topics covered:
- Comorbidities (other diagnoses)
- Hospital admissions in past 18 months
- Head trauma history
- Smoking and alcohol history
- Pesticide / toxin exposure
- Family history of PD
- Sleep problems (REM sleep behavior)
- Falls history
- Sense of smell changes (anosmia — an early PD symptom!)
- Constipation history

> **Tip:** Loss of smell, constipation, REM sleep behavior disorder, and family history are known early PD markers. These Level 1 variables are clinically meaningful and easy to implement.

---

### 5.5 MDS-UPDRS *(and MDS-UPDRS-1)*
**Shape:** 3,541 rows × 133 columns
**Purpose:** The gold-standard Parkinson's motor and non-motor assessment

The MDS-UPDRS has 4 parts:
| Part | What it covers | Level |
|------|----------------|-------|
| Part I (1.1–1.13) | Non-motor experiences of daily living (cognition, hallucinations, mood, apathy, sleep, pain) | 1 |
| Part II (2.1–2.13) | Motor experiences of daily living (speech, tremor, handwriting, dressing, walking) | 1 |
| Part III (3.1–3.18) | Motor exam (rigidity, bradykinesia, tremor, gait, posture) — done by clinician | 3 |
| Part IV (4.1–4.6) | Motor complications (dyskinesias, fluctuations) | 3 |

Key summary scores:
- `updrs_total` — Total score (higher = more severe)
- Part III scores (rigidity, gait, postural instability) are especially important for distinguishing PD from PD-plus

> **Note:** Two versions exist in the data (`MDS-UPDRS` and `MDS-UPDRS-1`). Check `Event Name` — these represent different visit timepoints. Consider merging or using the most recent.

> **Note:** `UPDRS 1.2 part 3` is a legacy version from older C-OPN sites. Treat separately or merge carefully.

---

### 5.6 MoCA *(and MoCA-1, MoCA-2)*
**Shape:** 3,541 rows × 26 columns each
**Purpose:** Montreal Cognitive Assessment — screens for cognitive impairment
**Level:** 3 (administered by a clinician, routine in clinic)

| Column | What it means |
|--------|---------------|
| `Visuospatial/Executive Score` | Drawing, clock test (0–5) |
| `Naming Score` | Animal naming (0–3) |
| `Memory Score` | Word recall (0–2) |
| `Attention Score` | Number span, serial 7s (0–6) |
| `Language Score` | Sentence repetition, fluency (0–3) |
| `Abstraction Score` | Similarities task (0–2) |
| `Delayed Recall Score` | Word recall after delay (0–5) |
| `Orientation Score` | Date, place, time (0–6) |
| `TOTAL SCORE` | Out of 30 (add 1 point if <12 years education) |

> **Tip:** MoCA total < 26 = possible cognitive impairment. DLB and PSP often show larger cognitive deficits than PD.

> **Note:** Three versions exist (`MoCA`, `MoCA-1`, `MoCA-2`) for different visit timepoints or site variations. Consider merging using the most complete or recent.

---

### 5.7 Medication
**Shape:** 3,541 rows × 54 columns
**Purpose:** Current Parkinson's medications

| Column | What it means | Level |
|--------|---------------|-------|
| `1. Select all current medications:` | Medication list (levodopa, dopamine agonists, MAO-B inhibitors, etc.) | 3 |
| `IR levodopa dosage (Sinemet)` | Immediate-release levodopa dose | 3 |
| `IR levodopa dosage (Prolopa)` | Same drug, different brand | 3 |
| `CR levodopa dosage (Sinemet CR)` | Controlled-release levodopa | 3 |
| Levodopa Equivalent Dose (LED) | Often calculated from these — reflects overall dopaminergic therapy load | 3 |

> **Tip:** Response to levodopa is a key diagnostic feature. PD typically responds well; most PD-plus conditions respond poorly or not at all. **Levodopa Equivalent Dose** can be a powerful predictor.

---

### 5.8 PDQ-39 *(and PDQ-8)*
**Shape:** 3,541 rows × 63 columns (PDQ-39); 17 columns (PDQ-8)
**Purpose:** Parkinson's Disease Questionnaire — quality of life measure, self-reported
**Level:** 2 (total/sub-scores of self-report questionnaire)

PDQ-39 covers 8 domains:
1. Mobility (10 items)
2. Activities of Daily Living (6 items)
3. Emotional Well-being (6 items)
4. Stigma (4 items)
5. Social Support (3 items)
6. Cognition (4 items)
7. Communication (3 items)
8. Bodily Discomfort (3 items)

PDQ-8 is the short version with 1 item per domain.

> **Tip:** Use sub-scores rather than individual items. The PDQ-39 Summary Index (SI) = average of 8 domain scores × 100.

---

### 5.9 SCOPA (Scales for Outcomes in Parkinson's Disease — Autonomic)
**Shape:** 3,541 rows × 51 columns
**Purpose:** Autonomic dysfunction symptoms (self-reported)
**Level:** 1–2

Covers:
- Gastrointestinal (swallowing, saliva, constipation)
- Urinary (urgency, leakage)
- Cardiovascular (dizziness on standing)
- Thermoregulatory (sweating)
- Pupillomotor
- Sexual function

> **Tip:** Autonomic features differ by diagnosis. MSA has severe autonomic failure; PD has mild-moderate; PSP less so.

---

### 5.10 BAI (Beck Anxiety Inventory)
**Shape:** 3,541 rows × 29 columns
**Purpose:** Anxiety symptoms (self-reported, 21 items)
**Level:** 2

Scores: 0–63 total. `0–7=minimal`, `8–15=mild`, `16–25=moderate`, `26+=severe`

---

### 5.11 BDI-II (Beck Depression Inventory, called "BDII")
**Shape:** 3,541 rows × 29 columns
**Purpose:** Depression symptoms (self-reported, 21 items)
**Level:** 2

Scores: 0–63 total. `0–13=minimal`, `14–19=mild`, `20–28=moderate`, `29+=severe`

> **Tip:** Depression and anxiety are common non-motor symptoms of PD and PD-plus, with different profiles across conditions.

---

### 5.12 Neuropsychological *(and Neuropsychological CaPRI, Neuropsychological V02)*
**Shape:** 3,541 rows × 130, 105, 126 columns respectively
**Purpose:** Detailed cognitive testing battery administered by a neuropsychologist
**Level:** 4 (requires specialist)

Covers:
- Verbal learning and memory (e.g., CVLT)
- Visual memory (e.g., Rey Figure)
- Attention and processing speed
- Executive function (Trail Making, Stroop)
- Language

> **Note:** Three versions because different C-OPN sites use slightly different batteries. Consider merging where variables overlap.

> **Note:** These are Level 4 — very informative but hard to implement clinically. Good for research but penalized in grading.

---

### 5.13 MDS-UPDRS Part 3 — Legacy (`UPDRS 1.2 part 3`)
**Shape:** 3,541 rows × 40 columns
**Purpose:** Older UPDRS motor exam version from some sites
**Level:** 3

> Merge with MDS-UPDRS Part III data carefully, noting this is the older scale version.

---

### 5.14 FrSBe (Frontal Systems Behavior Scale)
**Shape:** 3,541 rows × 104 columns
**Purpose:** Behavior changes related to frontal lobe dysfunction (informant-rated)
**Level:** 2

Covers: Apathy, Disinhibition, Executive dysfunction

> **Tip:** FrSBe is useful for distinguishing FTD and PSP from PD.

---

### 5.15 MBIC / MBIC (CaPRI) (Mild Behavioral Impairment Checklist)
**Shape:** 3,541 rows × 81 columns each
**Purpose:** Screens for neuropsychiatric symptoms as early dementia markers
**Level:** 2

---

### 5.16 Apathy Evaluation Self / Apathy Evaluation Informant / Apathy Scale
**Shape:** ~3,541 rows × 21–26 columns each
**Purpose:** Measures apathy (loss of motivation) — a key non-motor PD symptom

| File | Who completes it |
|------|-----------------|
| Apathy Evaluation Self | Patient fills it out |
| Apathy Evaluation Informant | Caregiver fills it out |
| Apathy Scale | Standard Starkstein Apathy Scale |

---

### 5.17 EHI (Edinburgh Handedness Inventory)
**Shape:** 3,541 rows × 22 columns
**Purpose:** Determines hand dominance (left/right/mixed)
**Level:** 1

> PD symptoms often start on the non-dominant side. Handedness can be a useful covariate.

---

### 5.18 Fatigue Severity Scale
**Shape:** 3,541 rows × 17 columns
**Purpose:** Measures fatigue impact (9 items, self-reported)
**Level:** 1–2

Scores range from 9 to 63. Score ≥ 36 = significant fatigue.

---

### 5.19 PDQ-39 / PDQ-8 Parkinson Severity Scale
**Shape:** 3,541 rows × 20 columns
**Purpose:** Self-reported Parkinson severity rating
**Level:** 1

---

### 5.20 Schwab & England Activities of Daily Living
**Shape:** 3,541 rows × 8 columns
**Purpose:** Clinician-rated functional independence (0–100% scale)
**Level:** 3

100% = fully independent. Lower scores = more functional impairment.

---

### 5.21 Timed Up and Go (TUG)
**Shape:** 3,541 rows × 11 columns
**Purpose:** Physical mobility test — time to stand up, walk 3m, return, sit
**Level:** 3

> Longer time = worse mobility. Useful for gait-related distinctions between PD and PD-plus.

---

## 6. Recommended Variables for Model Building

### Easy-to-implement baseline (Levels 1–2)
These can be collected without specialists — good for a clinically deployable model:

| Variable | File | Level |
|----------|------|-------|
| Age at diagnosis / disease duration | Clinical | 1 |
| Gender | Demographic | 1 |
| Years of education | Demographic | 1 |
| First symptoms (tremor vs. gait/balance onset) | Clinical | 1 |
| Loss of smell | Epidemiological | 1 |
| Constipation history | Epidemiological | 1 |
| REM sleep behavior | Epidemiological | 1 |
| Family history of PD | Epidemiological | 1 |
| MDS-UPDRS Part I & II scores | MDS-UPDRS | 1 |
| PDQ-39 sub-scores | PDQ 39 | 2 |
| BDI-II / BAI total scores | BDII / BAI | 2 |
| Fatigue Severity Scale | Fatigue Severity Scale | 2 |

### With routine clinical assessment (Level 3)
Adds significant power:

| Variable | File | Level |
|----------|------|-------|
| MoCA total score + sub-scores | MoCA | 3 |
| MDS-UPDRS Part III (motor exam) | MDS-UPDRS | 3 |
| Timed Up and Go (seconds) | Timed Up Go | 3 |
| Schwab & England score | Schwab & England | 3 |
| Levodopa Equivalent Dose (LED) | Medication | 3 |
| BMI | Clinical | 3 |

---

## 7. Important Data Notes

### 7.1 Multi-site study → use `Site` for external validation
C-OPN spans 9 sites. Consider holding out 1–2 sites as an external validation set to test generalizability.

### 7.2 Multiple versions of the same form
Several forms have duplicates for different site protocols or timepoints:

| Canonical Form | Duplicates |
|----------------|-----------|
| MDS-UPDRS | MDS-UPDRS-1, UPDRS 1.2 part 3 |
| MoCA | MoCA-1, MoCA-2 |
| Neuropsychological | Neuropsychological (CaPRI), Neuropsychological V02 |
| MBIC | MBIC (CaPRI) |

**Strategy:** Merge duplicate forms by aligning overlapping variable names, then use the most complete record per participant.

### 7.3 Columns are bilingual
All column headers are in English and French (e.g., `"1. Gender/Genre:"`). This is cosmetic — the values are encoded numerically.

### 7.4 First 3 columns in every file
Every file starts with:
- `Unnamed: 0` — row index (ignore)
- `Project key` — **participant ID for joining files**
- `Event Name` — visit/timepoint identifier

### 7.5 Sparsity
Not every participant completed every assessment. Expect significant missing data — especially for Level 3/4 assessments. Plan accordingly (imputation, complete-case, etc.).

### 7.6 Class imbalance
The dataset is heavily imbalanced: ~84% PD, ~4.6% AP, ~11.4% HC. Use appropriate strategies (SMOTE, class weights, stratified CV) especially for the minority AP class.

---

## 8. Quick Start Merge Example

```python
import pandas as pd

# Load key files
enroll = pd.read_csv("data/Enrollement")
demo   = pd.read_csv("data/Demographic")
clin   = pd.read_csv("data/Clinical")
mds    = pd.read_csv("data/MDS-UPDRS")
moca   = pd.read_csv("data/MoCA")

# Rename join key
key = "Project key"

# Merge on participant ID
df = enroll[[key, "Enrolment Group:    Groupe d'inscription:"]]
df = df.merge(demo,  on=key, how="left", suffixes=("", "_demo"))
df = df.merge(clin,  on=key, how="left", suffixes=("", "_clin"))
df = df.merge(mds,   on=key, how="left", suffixes=("", "_mds"))
df = df.merge(moca,  on=key, how="left", suffixes=("", "_moca"))

# Rename outcome
df = df.rename(columns={"Enrolment Group:...": "group"})
```

> **Tip:** Column names are very long. After merging, consider creating a short-name mapping using the data dictionary variable names (e.g., `gender`, `study_visit_age`).

---

## 9. File Reference Summary

| File | Rows | Cols | Role | Key Variables |
|------|------|------|------|---------------|
| Enrollement | 3,541 | 41 | Admin + Outcome | Group label, site, enrollment date |
| Demographic | 3,541 | 50 | Predictors (Level 1) | Age, gender, education, living situation |
| Clinical | 3,541 | 75 | Outcome + Predictors | Diagnosis, onset symptoms, disease duration, BMI |
| Epidemiological | 3,541 | 97 | Predictors (Level 1) | Smell loss, constipation, sleep, family history, exposures |
| MDS-UPDRS | 3,541 | 133 | Predictors (Level 1–3) | Motor exam scores, non-motor scores |
| MDS-UPDRS-1 | 3,541 | 133 | Same as above (alt timepoint) | |
| MoCA | 3,541 | 26 | Predictors (Level 3) | Cognitive sub-scores, total score |
| MoCA-1 / MoCA-2 | 3,541 | 26 | Same (alt site/timepoint) | |
| Medication | 3,541 | 54 | Predictors (Level 3) | Levodopa dose, medication types |
| PDQ 39 | 3,541 | 63 | Predictors (Level 2) | QoL domains, mobility, cognition |
| PDQ 8 | 3,541 | 17 | Predictors (Level 2) | Short-form QoL |
| SCOPA | 3,541 | 51 | Predictors (Level 1–2) | Autonomic symptoms |
| BAI | 3,541 | 29 | Predictors (Level 2) | Anxiety severity |
| BDII | 3,541 | 29 | Predictors (Level 2) | Depression severity |
| Neuropsychological | 3,541 | 130 | Predictors (Level 4) | Detailed cognitive battery |
| Neuropsychological (CaPRI) | 3,541 | 105 | Same (alt site) | |
| Neuropsychological V02 | 3,541 | 126 | Same (alt version) | |
| FrSBe | 3,541 | 104 | Predictors (Level 2) | Frontal behavior, apathy, disinhibition |
| MBIC | 3,541 | 81 | Predictors (Level 2) | Mild behavioral impairment |
| MBIC (CaPRI) | 3,541 | 81 | Same (alt site) | |
| Apathy Evaluation Self | 3,541 | 25 | Predictors (Level 1) | Self-rated apathy |
| Apathy Evaluation Informant | 3,541 | 26 | Predictors (Level 1) | Caregiver-rated apathy |
| Apathy Scale | 3,541 | 21 | Predictors (Level 1) | Starkstein Apathy Scale |
| EHI | 3,541 | 22 | Predictors (Level 1) | Handedness |
| Fatigue Severity Scale | 3,541 | 17 | Predictors (Level 1–2) | Fatigue impact |
| Parkinson Severity Scale | 3,541 | 20 | Predictors (Level 1) | Self-rated PD severity |
| Schwab & England | 3,541 | 8 | Predictors (Level 3) | Functional independence % |
| Timed Up Go | 3,541 | 11 | Predictors (Level 3) | Mobility test (seconds) |
| UPDRS 1.2 part 3 | 3,541 | 40 | Predictors (Level 3) | Legacy motor exam |

---

*Data source: Canadian Open Parkinson Network (C-OPN). All data de-identified under REB approval. For questions, contact Dr. Juan Li (juli@ohri.ca).*
