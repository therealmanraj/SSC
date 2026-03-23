### Reference sheets

| Sheet                 | What it shows                                                                                                                                                                                                            |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **All Variables**     | Every variable in the data dictionary with its base variable name, which CSV file it was found in, % missing, and sample values                                                                                          |
| **Consolidated**      | One row per base variable — groups versioned variables (e.g. `age_onset_2`, `age_onset_3`) together, with aggregated stats across all versions. Variables with different field labels in the same file are kept separate |
| **Numerical Stats**   | Descriptive statistics (mean, median, std, min, Q1, Q3, max, skew, kurtosis) per base variable, pooled across all versions                                                                                               |
| **Categorical Stats** | Value counts and percentages per base variable, pooled across all versions                                                                                                                                               |

### Diagnosis sheets

| Sheet                | What it shows                                                                                                                                                                                                                                                        |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Diagnosis Flags**  | Participants with at least one inconsistency between their Enrolment Group, Determined Diagnosis, and "Was diagnosed with PD?" fields. Sorted by number of flags (most flags first). Red = 2+ flags, orange = 1 flag                                                 |
| **Diagnosis Review** | Every participant (all 3,541) with their diagnosis fields, Study Status, Withdrawal Date, flag status, and — for unflagged participants — the reason they were not flagged (e.g. "AP enrolled, Determined = ET: consistent"). Flagged rows red/orange, OK rows green |

**Flags checked:**

- Enrolled as PD but Determined Dx ≠ PD
- Enrolled as AP but Determined Dx = PD
- "Was diagnosed with PD?" contradicts Determined Dx
- Enrolment Group contradicts "Was diagnosed with PD?"
- 1a alternative dx field contradicts Determined Dx code

### Completeness sheets

| Sheet                   | What it shows                                                                                                                                                                                              |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Completeness Matrix** | Participants × forms heatmap — each cell is the average % of fields missing for that participant in that form. Green = low missing, red = high missing                                                     |
| **Completeness by Dx**  | Summary of average % missing per form, broken down by Enrolment Group (PD / AP / HC)                                                                                                                       |
| **Enrollment Coverage** | Every enrollment key vs every form — `Yes` (green) if the participant has data in that form, `No` (red) if absent. Includes N Forms Present / N Forms Missing. Sorted most-missing first within each group |
| **Withdrawn Summary**   | All 246 withdrawn participants with their Withdrawal Date, per-form data coverage (Yes/No), and N Forms With Data / N Forms Missing. Useful for understanding what data was collected before withdrawal    |

### Data sheets

One sheet per form (e.g. `Clinical`, `MoCA`, `MDS-UPDRS`) containing the cleaned data with:

- Column names renamed from raw labels to standardised variable names from the data dictionary
- Skeleton rows (all fields blank) removed
- `Days since dx` column added — days between the form completion date and the participant's clinical diagnosis date
