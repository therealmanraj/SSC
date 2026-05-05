"""
Builds domain-filtered CSVs grouped by General Clinical Domain.

For each (General Clinical Domain × form) pair:
  - Rows:    enrolled IDs where the form is Complete  (same rule as clean_pipeline)
  - Columns: only columns whose Field Label matches that domain in the data dictionary
  - Saved to by_clinical_domain/{Domain}/{form}.csv

Domain mapping source:
  - reference/COPN_DataDictionary_2025-09-24_annotated.xlsx  (Field Labels = actual CSV column text)
  - pending/WORK-Qnaire-and-feature-clinical-domains.xlsx    (General Clinical Domain per variable)
  Both joined on Variable / Field Name.

Two enrollment groups: full_enrolled and enrolled_and_partial

Also saves:
  output/clean_pipeline/
    domain_column_mapping.csv   which CSV column matched which DD variable and domain
    domain_summary.csv          IDs and column counts per domain × form × group
"""

import os, re, warnings
import pandas as pd
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DD_REFERENCE = os.path.join(os.path.dirname(__file__), "..", "reference",
                             "COPN_DataDictionary_2025-09-24_annotated.xlsx")
DD_DOMAIN    = os.path.join(os.path.dirname(__file__), "..", "pending",
                             "WORK-Qnaire-and-feature-clinical-domains.xlsx")
DATA_DIR     = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_BASE  = os.path.join(os.path.dirname(__file__), "..", "output", "clean_pipeline")

STATUS_COL = "Study Status:    Statut dans l'étude:"

ENROLLMENT_STATUSES = {
    "full_enrolled": ["Enrolled/Inscrit"],
    "enrolled_and_partial": [
        "Enrolled/Inscrit",
        "QPN partially enrolled/RPQ partiellement inscrit",
    ],
}

# Merge duplicate domain names
DOMAIN_MERGE = {
    "Cognitive Functioning": "Cognitive Function",
    "Mood / Psychiatric ":   "Mood / Psychiatric",
}

SKIP_COLS = {"Project key", "Event Name", "Complete?"}
MATCH_THRESHOLD = 0.45


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize(s: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    s = str(s).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def jaccard(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.drop(
        columns=[c for c in df.columns if c.startswith("Unnamed:")], errors="ignore"
    )


def safe_dirname(s: str) -> str:
    return re.sub(r"[^\w\s\-]", "", s).strip().replace(" ", "_")


def safe_filename(s: str) -> str:
    return re.sub(r"[^\w]", "_", s).lower().strip("_") + ".csv"


# ---------------------------------------------------------------------------
# Step 1 — Build merged data dictionary (Field Label + Domain per variable)
# ---------------------------------------------------------------------------
print("Loading data dictionaries…")

dd_ref = pd.read_excel(DD_REFERENCE, sheet_name="COPN_DataDictionary_selected")
dd_dom = pd.read_excel(DD_DOMAIN,    sheet_name="Variables by Form and domain")

dd_dom["General Clinical Domain"] = (
    dd_dom["General Clinical Domain"].str.strip().replace(DOMAIN_MERGE)
)
dd_dom = dd_dom.dropna(subset=["Variable/Field Name", "General Clinical Domain"])

# Join on Variable / Field Name → each row has: Form Name, Field Label, Domain
dd_ref  = dd_ref.rename(columns={"Variable / Field Name": "var_name",
                                  "Field Label": "field_label",
                                  "Form Name":   "new_form_name"})
dd_dom  = dd_dom.rename(columns={"Variable/Field Name":       "var_name",
                                  "File/Qnaire name":          "old_file_name",
                                  "General Clinical Domain":   "domain"})
dd_ref["var_name"] = dd_ref["var_name"].astype(str).str.strip()
dd_dom["var_name"] = dd_dom["var_name"].astype(str).str.strip()

merged_dd = dd_ref[["var_name", "new_form_name", "field_label", "Field Type"]].merge(
    dd_dom[["var_name", "old_file_name", "domain"]],
    on="var_name", how="inner"
)
merged_dd = merged_dd.dropna(subset=["field_label", "domain"])

print(f"  Merged DD: {len(merged_dd):,} rows across "
      f"{merged_dd['new_form_name'].nunique()} forms, "
      f"{merged_dd['domain'].nunique()} domains")
print(f"  Domains: {sorted(merged_dd['domain'].unique())}\n")


# ---------------------------------------------------------------------------
# Step 2 — Auto-derive mapping: old File/Qnaire name → new DD form name
# ---------------------------------------------------------------------------
print("Building form-name mapping (old file names → new form names)…")

old_vars_by_file  = dd_dom.groupby("old_file_name")["var_name"].apply(set)
new_vars_by_form  = dd_ref.groupby("new_form_name")["var_name"].apply(set)

FORM_MAP: dict[str, str] = {}
for old_file, old_vars in old_vars_by_file.items():
    best_form, best_n = None, 0
    for new_form, new_vars in new_vars_by_form.items():
        n = len(old_vars & new_vars)
        if n > best_n:
            best_n, best_form = n, new_form
    if best_form:
        FORM_MAP[old_file] = best_form
        print(f"  {old_file!r:35} → {best_form!r}  (overlap={best_n})")

print()
all_domains = sorted(merged_dd["domain"].unique())


# ---------------------------------------------------------------------------
# Step 3 — Match CSV columns to domains via Field Label
# ---------------------------------------------------------------------------
print("Matching CSV columns to domains via Field Label…")

form_col_domain: dict[str, dict[str, str]] = {}   # {old_file_name: {csv_col: domain}}
match_report: list[dict] = []

for old_file in sorted(old_vars_by_file.index):
    csv_path = os.path.join(DATA_DIR, old_file)
    if not os.path.exists(csv_path):
        print(f"  WARNING: {old_file!r} not found in data/ — skipping")
        continue

    new_form = FORM_MAP.get(old_file)
    if new_form is None:
        print(f"  WARNING: no form mapping for {old_file!r} — skipping")
        continue

    csv_df   = load_csv(csv_path)
    form_dd  = merged_dd[merged_dd["new_form_name"] == new_form].copy()

    # Build {field_label_norm: (domain, field_type, var_name)} — prefer non-Admin
    label_info: dict[str, tuple] = {}
    for _, row in form_dd.iterrows():
        ln  = normalize(str(row["field_label"]))
        dom = row["domain"]
        ft  = str(row.get("Field Type", "")) if pd.notna(row.get("Field Type", "")) else ""
        vn  = str(row["var_name"])
        if ln not in label_info:
            label_info[ln] = (dom, ft, vn)
        elif label_info[ln][0] == "Admin" and dom != "Admin":
            label_info[ln] = (dom, ft, vn)

    # Match each CSV column to best-scoring Field Label
    col_domain: dict[str, str] = {}
    for col in csv_df.columns:
        if col in SKIP_COLS or col.startswith("Unnamed"):
            continue
        cn = normalize(col)
        best_score, best_dom, best_lbl, best_ft, best_vn = 0.0, None, None, "", ""
        for lbl_norm, (dom, ft, vn) in label_info.items():
            s = jaccard(cn, lbl_norm)
            if s > best_score:
                best_score, best_dom, best_lbl, best_ft, best_vn = s, dom, lbl_norm, ft, vn

        if best_score >= MATCH_THRESHOLD:
            col_domain[col] = best_dom
            match_report.append({
                "Form":                  old_file,
                "CSV column":            col[:80],
                "Matched field label":   best_lbl,
                "Variable / Field Name": best_vn,
                "Field Type":            best_ft,
                "Domain":                best_dom,
                "Score":                 round(best_score, 3),
                "Matched":               "Yes",
            })
        else:
            match_report.append({
                "Form":                  old_file,
                "CSV column":            col[:80],
                "Matched field label":   best_lbl or "",
                "Variable / Field Name": best_vn,
                "Field Type":            best_ft,
                "Domain":                "UNMATCHED",
                "Score":                 round(best_score, 3),
                "Matched":               "No",
            })

    form_col_domain[old_file] = col_domain
    n_matched = len(col_domain)
    n_total   = len([c for c in csv_df.columns
                     if c not in SKIP_COLS and not c.startswith("Unnamed")])
    by_dom = {}
    for d in col_domain.values():
        by_dom[d] = by_dom.get(d, 0) + 1
    print(f"  {old_file:<35}  {n_matched:>3}/{n_total:<3}  {by_dom}")

# Save matching report
match_df = pd.DataFrame(match_report)
match_df.to_csv(os.path.join(OUTPUT_BASE, "domain_column_mapping.csv"), index=False)
print(f"\n  → domain_column_mapping.csv saved ({len(match_df)} rows)\n")


# ---------------------------------------------------------------------------
# Step 4 — Build domain CSVs for each enrollment group
# ---------------------------------------------------------------------------
enrol_df     = load_csv(os.path.join(DATA_DIR, "Enrollement"))
summary_rows = []

for group_name, statuses in ENROLLMENT_STATUSES.items():
    enrolled_ids = set(enrol_df[enrol_df[STATUS_COL].isin(statuses)]["Project key"])
    print(f"=== {group_name} ({len(enrolled_ids)} base IDs) ===")

    for domain in all_domains:
        domain_dir = os.path.join(
            OUTPUT_BASE, group_name, "by_clinical_domain", safe_dirname(domain)
        )
        os.makedirs(domain_dir, exist_ok=True)

        domain_forms = sorted(
            merged_dd[merged_dd["domain"] == domain]["old_file_name"].unique()
        )

        for form_name in domain_forms:
            col_domain = form_col_domain.get(form_name, {})
            domain_cols = [col for col, d in col_domain.items() if d == domain]
            if not domain_cols:
                continue

            csv_path = os.path.join(DATA_DIR, form_name)
            if not os.path.exists(csv_path):
                continue

            df = load_csv(csv_path)
            df = df[df["Project key"].isin(enrolled_ids)].copy()

            if "Complete?" in df.columns:
                df = df[df["Complete?"] == "Complete"].copy()

            keep = ["Project key", "Event Name"] + [
                c for c in domain_cols if c in df.columns
            ]
            df = df[keep]

            if len(df) == 0:
                continue

            out_path = os.path.join(domain_dir, safe_filename(form_name))
            df.to_csv(out_path, index=False)

            summary_rows.append({
                "group":         group_name,
                "domain":        domain,
                "form":          form_name,
                "n_ids":         len(df),
                "n_domain_cols": len(domain_cols),
            })
            print(
                f"  {domain:<30} {form_name:<35}"
                f"  {len(df):>5} IDs  {len(domain_cols)} cols"
            )

    print()

# Save summary
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUTPUT_BASE, "domain_summary.csv"), index=False)
print("domain_summary.csv saved")
print("\nDone.")
