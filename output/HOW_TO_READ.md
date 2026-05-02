# How to Read the Output Files

A plain-English guide for the team. No coding required — all files open in Excel or any spreadsheet app.

---

## Folder Overview

```
output/
  clean_pipeline/          Step 1 — who is in the study
  comparative_analysis/    Step 2 — how groups differ statistically
  shap_summary/            Step 3 — which features the ML model uses most
```

Start at Step 1, work down. Each step builds on the previous one.

---

## Step 1 — clean_pipeline/

**What it is:** The filtered participant lists. Every CSV here is a list of participant IDs (Project key) plus all columns from one questionnaire form, for participants who meet the inclusion criteria.

### Folder structure

```
clean_pipeline/
  full_enrolled/                  ← IDs with Study Status = "Enrolled" only (n = 2,286)
    enrollment.csv                   all 2,286 enrolled IDs + admin info
    demographic.csv                  of those, only who completed Demographic (n = 2,005)
    clinical.csv                     only who completed Clinical (n = 1,778)
    moca.csv                         only who completed MoCA (n = 1,825)
    mds_updrs.csv                    ... and so on for every form
    by_enrolment_group/
      PD/   AP/   HC/              same files but split by diagnosis group
    by_determined_dx/
      0_PD/  1_PSP/  2_MSA/ ...   same files but split by final determined diagnosis
  enrolled_and_partial/           ← adds "QPN partially enrolled" IDs (n = 2,896)
    (same structure as above)
  summary.csv                     count of IDs at every filter step
```

### How to read summary.csv

| Column | Meaning |
|--------|---------|
| `label` | Which folder the row describes (e.g. `full_enrolled/by_enrolment_group/PD`) |
| `form` | Which questionnaire form |
| `ids` | Number of participants who passed all filters for that form |
| `note` | `complete` = filtered by the form's own completion status; `no_completion_col` = form has no completion flag, all enrolled IDs kept |

**Key rule:** A participant appears in `demographic.csv` only if they are enrolled AND marked Complete in the Demographic form. Each form has its own completion requirement, so the N differs per form.

---

## Step 2 — comparative_analysis/

This folder has three sub-analyses. Start with `enhanced_stats/` — it is the most complete.

---

### 2a. enhanced_stats/

The main statistical comparison. Two groupings, same file structure for each.

```
enhanced_stats/
  by_enrolment_group/     comparing PD vs AP vs HC (enrolment label)
    stats.xlsx
    pairwise.xlsx
  by_determined_dx/       comparing PD vs PSP vs MSA vs DLB vs ET vs RBD
    stats.xlsx
    pairwise.xlsx
  significant_fdr.csv     shortlist — only variables that survive FDR correction
```

#### stats.xlsx — one sheet per form

Each row is one variable (questionnaire item or computed score). Columns:

| Column | Meaning |
|--------|---------|
| `Variable` | Short name of the questionnaire item |
| `KW H` | Kruskal-Wallis H statistic — larger = more separation between groups |
| `raw p` | Raw p-value before correction. Do NOT use this alone to claim significance |
| `η²` (eta-squared) | **Effect size** — how large the group difference is (0 to 1). See table below |
| `BH adj p` | **Use this for significance.** P-value after Benjamini-Hochberg FDR correction |
| `BH sig` | `Yes` = survives FDR correction at 5% level |
| `PD n / mean / sd / median` | Descriptive stats for the PD group |
| `AP n / mean / sd / median` | Descriptive stats for the AP group |
| `HC n / mean / sd / median` | Descriptive stats for the HC group |

**Effect size guide (η²):**

| η² value | Interpretation |
|----------|---------------|
| < 0.01 | Negligible |
| 0.01 – 0.06 | Small |
| 0.06 – 0.14 | Medium |
| > 0.14 | Large |

**How to find the most important variables:** Filter `BH sig = Yes`, then sort by `η²` descending. The top rows are the variables with the largest group differences that are also statistically reliable.

#### pairwise.xlsx — one sheet per form

Each row is one variable × one group pair (e.g. PD vs HC, PD vs AP). Columns:

| Column | Meaning |
|--------|---------|
| `Variable` | Questionnaire item |
| `Group 1 / Group 2` | Which two groups are being compared |
| `n1 / n2` | Sample size in each group |
| `MWU stat` | Mann-Whitney U statistic (non-parametric, no normality needed) |
| `raw p` | Uncorrected p-value |
| `bonf p` | Bonferroni-corrected p-value (corrects for multiple pairwise tests on the same variable) |
| `bonf sig` | `Yes` = this specific pair is significantly different |
| `Cohen's d` | **Pairwise effect size.** See table below |
| `Effect` | Plain-English label for Cohen's d |

**Cohen's d guide:**

| d value | Interpretation |
|---------|---------------|
| < 0.2 | Negligible |
| 0.2 – 0.5 | Small |
| 0.5 – 0.8 | Medium |
| > 0.8 | Large |

**Example of how to read a row:**

> Variable = `moca_total`, Group 1 = PD, Group 2 = HC, bonf p = 0.003, Cohen's d = 0.72, Effect = medium

This means: MoCA total score is significantly different between PD and HC patients (p = 0.003 after correction), with a medium-sized effect (d = 0.72).

#### significant_fdr.csv — the shortlist

This file collects every row where `BH sig = Yes` from both groupings. **Start here** when preparing the abstract. It contains only the variables that are:
1. Statistically significant after FDR correction
2. Present in at least 2 groups with n ≥ 5

Sort by `η²` to find the strongest findings.

---

### 2b. by_enrolment_group/ and by_determined_dx/

*(Inside comparative_analysis/ — the original analysis before effect sizes were added.)*

Contains one PNG plot per questionnaire form. Each plot shows box plots of numeric variables grouped by PD/AP/HC (or by dx code). P-values and significance stars are shown on each subplot.

- `*` p < 0.05
- `**` p < 0.01
- `***` p < 0.001
- `ns` not significant

---

### 2c. enrolment_x_dx_grouped/ and within_enrolment_by_dx/

Cross-dimensional analysis — shows both the enrolment group AND the determined diagnosis at the same time.

- **enrolment_x_dx_grouped:** Box plots where x-axis = enrolment group (PD/AP/HC) and colour = determined dx. Use this to see whether, within the "PD enrolled" group, participants with different final diagnoses look different.

- **within_enrolment_by_dx/AP/:** Only the AP-enrolled group, split by final dx (PSP / MSA / DLB / ET / RBD). This is the most clinically interesting comparison — it shows whether different atypical Parkinsonian conditions are distinguishable from each other on clinical measures.

---

## Step 3 — shap_summary/

**What it is:** The ML model's answer to "which features matter most for distinguishing groups?" SHAP (SHapley Additive exPlanations) assigns each feature a score reflecting how much it contributed to the model's predictions.

```
shap_summary/
  shap_values.xlsx         ranked feature list per task
  plots/
    task_a___pd_vs_hc.png
    task_b___pd_vs_ap.png
    task_c___pd_vs_hc_vs_ap.png
    task_d___ap_subtype.png
    task_e___hc_vs_non_hc.png
    heatmap.png              overview of all tasks at once
```

### shap_values.xlsx — one sheet per task

| Column | Meaning |
|--------|---------|
| `Rank` | 1 = most important feature for this task |
| `Feature` | Feature name (matches the ML pipeline's engineered feature names) |
| `Mean \|SHAP\|` | Average absolute SHAP value. Higher = model relied on this feature more |

### heatmap.png

The single most useful plot for the abstract. Shows the top 20 features on the y-axis and all 5 classification tasks on the x-axis. The colour (yellow → red) shows how important each feature is for each task. Features that are dark red across multiple tasks are universally important.

### Task definitions

| Task | Description | Groups |
|------|-------------|--------|
| Task A | Can we separate PD from healthy controls? | PD vs HC |
| Task B | Can we separate PD from atypical Parkinsonism? | PD vs AP (ET excluded) |
| Task C | Three-way classification | PD vs HC vs AP |
| Task D | Can we tell AP subtypes apart? (exploratory) | PSP vs MSA vs DLB+other |
| Task E | Can we identify who is NOT a healthy control? | HC vs everyone else |

---

## Suggested Reading Order for the Abstract

1. **significant_fdr.csv** — filter `Grouping = by_enrolment_group`, sort by `η²` descending. These are your key group differences.
2. **pairwise.xlsx (by_enrolment_group)** — for the top variables from step 1, check which specific pairs differ (PD vs HC? PD vs AP?).
3. **shap_summary/heatmap.png** — which of those variables also show up as top SHAP features?
4. Variables that appear in BOTH the FDR-significant list AND the top SHAP features are your strongest candidates for the abstract.

---

## Common Questions

**Q: Why do different forms have different N?**
Each form's completion status is tracked separately. A participant who completed Demographic but not MDS-UPDRS appears in `demographic.csv` but not `mds_updrs.csv`.

**Q: Why use BH adj p instead of raw p?**
We ran hundreds of statistical tests across 30 forms and dozens of variables. Running this many tests by chance alone will produce many false positives. BH correction controls the False Discovery Rate — it adjusts each p-value based on how many tests were run, so a `BH adj p < 0.05` is still a reliable finding.

**Q: What is the difference between `full_enrolled` and `enrolled_and_partial`?**
`full_enrolled` = Study Status is exactly "Enrolled/Inscrit" (2,286 IDs).
`enrolled_and_partial` = adds "QPN partially enrolled" participants (2,896 IDs total). Use `full_enrolled` as the primary cohort and `enrolled_and_partial` as a sensitivity check.

**Q: Why does `by_determined_dx` have fewer forms than `by_enrolment_group`?**
The determined diagnosis is assigned by clinicians and is only available for ~1,764 enrolled participants. Some forms (e.g. BAI, SCOPA) have too few participants with a known dx in each subgroup (< 5 per group) to run meaningful comparisons, so they are excluded.
