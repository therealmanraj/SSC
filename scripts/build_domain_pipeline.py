"""
Builds domain-filtered CSVs grouped by General Clinical Domain.

For each (General Clinical Domain × form) pair:
  - Rows:    enrolled IDs where the form is Complete  (same rule as clean_pipeline)
  - Columns: only columns whose Short Description matches that domain in the data dictionary
  - Saved to by_clinical_domain/{Domain}/{form}.csv

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
DOMAIN_FILE  = os.path.join(os.path.dirname(__file__), "..", "pending",
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
MATCH_THRESHOLD = 0.45   # minimum Jaccard similarity to accept a column match

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize(s: str) -> str:
    """Lowercase, strip Q-prefixes (Q1. → 1.), remove punctuation."""
    s = str(s).lower()
    s = re.sub(r"\bq(\d+)\.", r"\1 ", s)      # Q1. Gender → 1 Gender
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def csv_english(col: str) -> str:
    """Extract the English prefix from a bilingual CSV column name."""
    return normalize(re.split(r"\s{3,}", col)[0].strip())


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
# Step 1 — Load and clean data dictionary
# ---------------------------------------------------------------------------
print("Loading data dictionary…")
dd = pd.read_excel(DOMAIN_FILE, sheet_name="Variables by Form and domain")
dd["General Clinical Domain"] = (
    dd["General Clinical Domain"].str.strip().replace(DOMAIN_MERGE)
)
dd = dd.dropna(subset=["File/Qnaire name", "Short Description", "General Clinical Domain"])

print(f"  {len(dd):,} variable rows across {dd['File/Qnaire name'].nunique()} forms")
print(f"  Domains: {sorted(dd['General Clinical Domain'].unique())}\n")


# ---------------------------------------------------------------------------
# Step 2 — Build column → domain mapping for every form
# ---------------------------------------------------------------------------
print("Matching CSV columns to domains…")

form_col_domain = {}   # {form_name: {csv_col: domain}}
match_report    = []

for form_name in sorted(dd["File/Qnaire name"].unique()):
    csv_path = os.path.join(DATA_DIR, form_name)
    if not os.path.exists(csv_path):
        print(f"  WARNING: {form_name!r} not found in data/ — skipping")
        continue

    csv_df   = load_csv(csv_path)
    form_dd  = dd[dd["File/Qnaire name"] == form_name].copy()

    # Normalised English prefix for each analysable CSV column
    col_norms = {
        col: csv_english(col)
        for col in csv_df.columns
        if col not in SKIP_COLS and not col.startswith("Unnamed")
    }

    # Build {desc_norm: domain} from data dictionary.
    # If the same normalised description appears under two domains,
    # prefer the non-Admin domain.
    desc_domain: dict[str, str] = {}
    for _, row in form_dd.iterrows():
        dn   = normalize(str(row["Short Description"]))
        dom  = row["General Clinical Domain"]
        if dn not in desc_domain:
            desc_domain[dn] = dom
        elif desc_domain[dn] == "Admin" and dom != "Admin":
            desc_domain[dn] = dom

    # Match each CSV column to the best-scoring description
    col_domain: dict[str, str] = {}
    for col, col_norm in col_norms.items():
        best_score, best_dom, best_desc = 0.0, None, None
        for desc_norm, dom in desc_domain.items():
            s = jaccard(col_norm, desc_norm)
            if s > best_score:
                best_score, best_dom, best_desc = s, dom, desc_norm

        if best_score >= MATCH_THRESHOLD:
            col_domain[col] = best_dom
            match_report.append({
                "Form":        form_name,
                "CSV column":  col[:80],
                "Matched description": best_desc,
                "Domain":      best_dom,
                "Score":       round(best_score, 3),
                "Matched":     "Yes",
            })
        else:
            match_report.append({
                "Form":        form_name,
                "CSV column":  col[:80],
                "Matched description": best_desc or "",
                "Domain":      "UNMATCHED",
                "Score":       round(best_score, 3),
                "Matched":     "No",
            })

    form_col_domain[form_name] = col_domain
    n_matched = len(col_domain)
    n_total   = len(col_norms)
    print(f"  {form_name:<35}  {n_matched:>3}/{n_total:<3} columns matched")

# Save matching report
match_df = pd.DataFrame(match_report)
match_df.to_csv(os.path.join(OUTPUT_BASE, "domain_column_mapping.csv"), index=False)
print(f"\n  → domain_column_mapping.csv saved ({len(match_df)} rows)\n")


# ---------------------------------------------------------------------------
# Step 3 — Build domain CSVs for each enrollment group
# ---------------------------------------------------------------------------
enrol_df    = load_csv(os.path.join(DATA_DIR, "Enrollement"))
all_domains = sorted(dd["General Clinical Domain"].unique())
summary_rows = []

for group_name, statuses in ENROLLMENT_STATUSES.items():
    enrolled_ids = set(enrol_df[enrol_df[STATUS_COL].isin(statuses)]["Project key"])
    print(f"=== {group_name} ({len(enrolled_ids)} base IDs) ===")

    for domain in all_domains:
        domain_dir = os.path.join(
            OUTPUT_BASE, group_name, "by_clinical_domain", safe_dirname(domain)
        )
        os.makedirs(domain_dir, exist_ok=True)

        # Forms that have at least one variable in this domain
        domain_forms = sorted(
            dd[dd["General Clinical Domain"] == domain]["File/Qnaire name"].unique()
        )

        for form_name in domain_forms:
            col_domain = form_col_domain.get(form_name, {})

            # Only columns assigned to this domain
            domain_cols = [col for col, d in col_domain.items() if d == domain]
            if not domain_cols:
                continue

            csv_path = os.path.join(DATA_DIR, form_name)
            if not os.path.exists(csv_path):
                continue

            df = load_csv(csv_path)

            # Row filter: enrolled IDs
            df = df[df["Project key"].isin(enrolled_ids)].copy()

            # Completion filter
            if "Complete?" in df.columns:
                df = df[df["Complete?"] == "Complete"].copy()

            # Column filter: domain columns + identifiers
            keep = ["Project key", "Event Name"] + [
                c for c in domain_cols if c in df.columns
            ]
            df = df[keep]

            if len(df) == 0:
                continue

            out_path = os.path.join(domain_dir, safe_filename(form_name))
            df.to_csv(out_path, index=False)

            summary_rows.append({
                "group":  group_name,
                "domain": domain,
                "form":   form_name,
                "n_ids":  len(df),
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
