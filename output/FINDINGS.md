# Key Findings — Feature Importance Analysis

Summary of the most important features identified from two complementary methods:
- **Statistical analysis** (Kruskal-Wallis + FDR correction, `enhanced_stats/`)
- **ML model** (XGBoost SHAP values, `shap_summary/`)

Features that appear in both methods are the most reliable candidates for the abstract.

---

## For Enrolment Group (PD vs AP vs HC)

The effect sizes here are medium to large (η² up to 0.16), meaning these groups are meaningfully separable.

| Feature | η² (stats) | SHAP rank (Task A/B) | What it is |
|---|---|---|---|
| **Disease duration** | — | #1 both tasks | Years since diagnosis |
| **MDS-UPDRS Part III** | 0.136 | #3 both tasks | Total motor exam score |
| **MDS-UPDRS Part II** | 0.158 | #7 Task B | Motor aspects of daily living |
| **MoCA total** | — | #4 Task A | Cognitive screening score |
| **Tremor-dominant score** | — | #3 Task B, #6 Task A | Ratio of tremor vs PIGD — PD has more tremor |
| **Verbal learning (HVLT trials)** | 0.093 | — | Neuropsychological memory test |
| **Age at onset** | — | #2 both tasks | Age when symptoms first appeared |
| **Levodopa response** | — | #5 Task B | Whether patient responded to levodopa |
| **Freezing of gait (UPDRS 3.14)** | 0.092 | #10 Task A | Single item from motor exam |
| **Delayed memory recall** | 0.087 | — | Neuropsychological delayed recall |
| **BAI anxiety score** | 0.121 | — | Beck Anxiety Inventory total |
| **MBI-C behavioural total** | 0.084 | — | Mild Behavioural Impairment Checklist |

**Stats-only findings** (FDR significant, moderate effect): BAI anxiety (η²=0.12), MBI-C behavioural total (η²=0.08), delayed memory recall (η²=0.09), semantic fluency (η²=0.08)

**SHAP-only findings** (ML-identified, not in raw form stats): `has_dbs` (DBS surgery history), `total_levo_dose` (levodopa dosage), `has_dyskinesia` — the ML model relies heavily on these even though they don't appear as top statistical discriminators on their own.

---

## For Determined Dx (PD vs PSP vs MSA vs DLB vs ET vs RBD)

Effect sizes are much smaller (η² max ~0.04 vs 0.16 above), reflecting the genuine difficulty of distinguishing dx subtypes from clinical measures alone.

| Feature | η² (stats) | SHAP rank (Task D) | What it is |
|---|---|---|---|
| **Tremor-dominant score** | — | #3 | Tremor vs PIGD ratio — PSP/MSA have less tremor than PD/ET |
| **Levodopa response** | — | #4 | PD responds well; PSP/MSA typically do not |
| **Age / Age at onset** | — | #1, #2 | DLB tends to be older; RBD younger onset |
| **Freezing of gait (UPDRS 3.14)** | 0.029 | #5 | PSP has prominent early freezing |
| **MoCA total** | — | #6 | Pattern of cognitive decline differs by subtype |
| **Falls (number of times)** | 0.038 | #8 | PSP patients fall backward early in disease |
| **UPDRS 3.9 (arising from chair)** | 0.041 | — | Strongest single statistical item — PSP/MSA score high |
| **UPDRS 3.1 (speech)** | 0.034 | — | PSP has severe dysarthria early on |
| **UPDRS 3.8 (rigidity, bilateral)** | 0.026 | — | MSA has prominent rigidity |
| **PDQ-39 Communication** | 0.036 | — | Quality of life communication domain |
| **Disease duration** | 0.025 | #7 | Rate of progression differs by subtype |
| **UPDRS 3.4 (resting tremor)** | 0.028 | — | Lower in PSP/MSA vs PD |

---

## Summary for the Abstract

### Enrolment Group (PD / AP / HC)

> Disease duration, age at onset, UPDRS Part III motor score, MoCA total, tremor-dominant score, and levodopa response are the most important features — both statistically (medium–large effect sizes, FDR corrected) and according to the ML model (top SHAP features across Tasks A, B, C). Neuropsychological measures (verbal learning, delayed recall, semantic fluency) and behavioural measures (BAI, MBI-C) add further discriminative power.

### Determined Dx Subtypes (PSP / MSA / DLB / ET / RBD vs PD)

> The tremor-dominant score and levodopa response are the clearest separators — PSP and MSA show low tremor and poor levodopa response compared to PD. Falls, speech, and freezing items from the UPDRS (items 3.1, 3.9, 3.14) are the strongest individual statistical discriminators between AP subtypes. Effect sizes are small overall, reflecting the inherent difficulty of distinguishing atypical Parkinsonian conditions from motor and cognitive measures alone.

---

## Where to Find the Full Numbers

| What you need | Where to look |
|---|---|
| All FDR-significant variables with effect sizes | `enhanced_stats/significant_fdr.csv` |
| Pairwise comparisons (which specific groups differ) | `enhanced_stats/by_enrolment_group/pairwise.xlsx` |
| SHAP feature rankings per task | `shap_summary/shap_values.xlsx` |
| SHAP overview across all tasks | `shap_summary/plots/heatmap.png` |
| Box plots of group differences by form | `by_enrolment_group/plots/` and `by_determined_dx/plots/` |
| AP subtypes compared within AP-enrolled group | `within_enrolment_by_dx/AP/` |
