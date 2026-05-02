"""
Enhanced statistical analysis on full_enrolled data.

Adds to the existing KW tests:
  - Eta-squared (η²) effect size per variable
  - Benjamini-Hochberg FDR correction within each form
  - Pairwise Mann-Whitney U with Bonferroni correction
  - Cohen's d for every group pair

Both groupings: by_enrolment_group (PD/AP/HC) and by_determined_dx.

Output:
  output/comparative_analysis/enhanced_stats/
    by_enrolment_group/
      stats.xlsx     one sheet per form — KW + η² + BH-adj p
      pairwise.xlsx  one sheet per form — MWU + Cohen's d per pair
    by_determined_dx/
      (same)
    significant_fdr.csv   all variables surviving BH FDR across both groupings
"""

import os, re, math, warnings, itertools
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
PIPELINE_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "clean_pipeline", "full_enrolled")
DATA_DIR     = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "output", "comparative_analysis", "enhanced_stats")

ENROLMENT_GROUP_SHORT = {
    "PD (Parkinson's Disease)/(Maladie de Parkinson)": "PD",
    "AP (Atypical Parkinsonism)/(Parkinsonisme Atypique)": "AP",
    "Healthy control/Contrôle": "HC",
}
DX_LABELS = {0.0:"PD",1.0:"PSP",2.0:"MSA",3.0:"CBS",4.0:"DLB",5.0:"FTD",6.0:"ET",7.0:"RBD"}

SKIP_COLS = {"Project key", "Event Name", "Complete?"}
MIN_N     = 5   # minimum group size

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def short_label(col, maxlen=55):
    label = re.split(r"\s{3,}", col)[0].strip()
    label = re.sub(r'^[\*"\s]+', "", label)
    return label[:maxlen]


def load_csv(path):
    df = pd.read_csv(path)
    return df.drop(columns=[c for c in df.columns if c.startswith("Unnamed:")], errors="ignore")


def numeric_cols_of(df):
    cols = []
    for col in df.columns:
        if col in SKIP_COLS or col.startswith("Unnamed"):
            continue
        s = df[col].dropna()
        if len(s) == 0:
            continue
        conv = pd.to_numeric(s, errors="coerce")
        if conv.notna().sum() / len(s) >= 0.5 and conv.std() > 0:
            cols.append(col)
    return cols


# ---------------------------------------------------------------------------
# Statistical functions
# ---------------------------------------------------------------------------

def bh_adjust(pvals):
    """Benjamini-Hochberg FDR adjustment. Returns adjusted p-values."""
    pvals = np.array(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return pvals
    order = np.argsort(pvals)
    ranked = np.arange(1, n + 1)
    adj = pvals[order] * n / ranked
    # enforce monotonicity from right
    for i in range(n - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    adj = np.minimum(adj, 1.0)
    result = np.empty(n)
    result[order] = adj
    return result


def eta_squared_kw(h, k, n):
    """Eta-squared effect size from Kruskal-Wallis H."""
    if n <= k:
        return np.nan
    return max(0.0, (h - k + 1) / (n - k))


def cohens_d(a, b):
    """Cohen's d between two arrays (pooled SD)."""
    a, b = np.array(a), np.array(b)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled_sd = math.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    if pooled_sd == 0:
        return np.nan
    return round(abs(a.mean() - b.mean()) / pooled_sd, 4)


def effect_label(d):
    if np.isnan(d): return ""
    if d < 0.2:     return "negligible"
    if d < 0.5:     return "small"
    if d < 0.8:     return "medium"
    return "large"


def run_kw(df, group_col, col):
    groups = []
    for _, g in df.groupby(group_col):
        vals = pd.to_numeric(g[col], errors="coerce").dropna().values
        if len(vals) >= 3:
            groups.append(vals)
    if len(groups) < 2:
        return np.nan, np.nan, np.nan
    try:
        h, p = kruskal(*groups)
        n_total = sum(len(g) for g in groups)
        eta2 = eta_squared_kw(h, len(groups), n_total)
        return round(h, 4), round(p, 6), round(eta2, 4)
    except Exception:
        return np.nan, np.nan, np.nan


def run_pairwise(df, group_col, col, groups):
    rows = []
    for g1, g2 in itertools.combinations(groups, 2):
        a = pd.to_numeric(df[df[group_col] == g1][col], errors="coerce").dropna().values
        b = pd.to_numeric(df[df[group_col] == g2][col], errors="coerce").dropna().values
        if len(a) < MIN_N or len(b) < MIN_N:
            continue
        try:
            stat, p = mannwhitneyu(a, b, alternative="two-sided")
            d = cohens_d(a, b)
            rows.append({
                "Group 1": g1, "Group 2": g2,
                "n1": len(a), "n2": len(b),
                "MWU stat": round(stat, 2), "raw p": round(p, 6),
                "Cohen's d": d, "Effect": effect_label(d),
            })
        except Exception:
            pass
    # Bonferroni correction within this variable
    if rows:
        m = len(rows)
        for r in rows:
            r["bonf p"] = round(min(r["raw p"] * m, 1.0), 6)
            r["bonf sig"] = "Yes" if r["bonf p"] < 0.05 else "No"
    return rows


# ---------------------------------------------------------------------------
# Per-form analysis
# ---------------------------------------------------------------------------

def analyse_form(df, group_col, fname):
    """Returns (main_df, pairwise_df) for one form."""
    groups = sorted(df[group_col].dropna().unique())
    num_cols = numeric_cols_of(df)

    main_rows = []
    pair_rows = []

    for col in num_cols:
        h, p, eta2 = run_kw(df, group_col, col)
        row = {
            "Variable": short_label(col),
            "KW H": h, "raw p": p, "η²": eta2,
            # BH p filled in after collecting all p-values for this form
            "BH adj p": np.nan, "BH sig": "",
        }
        # per-group descriptives
        for grp in groups:
            vals = pd.to_numeric(df[df[group_col] == grp][col], errors="coerce").dropna()
            row[f"{grp} n"]      = len(vals)
            row[f"{grp} mean"]   = round(vals.mean(),   3) if len(vals) else np.nan
            row[f"{grp} sd"]     = round(vals.std(),    3) if len(vals) else np.nan
            row[f"{grp} median"] = round(vals.median(), 3) if len(vals) else np.nan
        main_rows.append(row)

        # pairwise
        pw = run_pairwise(df, group_col, col, groups)
        for r in pw:
            r["Variable"] = short_label(col)
        pair_rows.extend(pw)

    # Apply BH FDR on this form's KW p-values
    if main_rows:
        raw_ps = np.array([r["raw p"] for r in main_rows], dtype=float)
        adj_ps = bh_adjust(raw_ps)
        for r, ap in zip(main_rows, adj_ps):
            r["BH adj p"] = round(ap, 6) if not np.isnan(ap) else np.nan
            r["BH sig"]   = "Yes" if (not np.isnan(ap) and ap < 0.05) else "No"

    main_df = pd.DataFrame(main_rows)
    pair_df = pd.DataFrame(pair_rows) if pair_rows else pd.DataFrame()
    return main_df, pair_df


# ---------------------------------------------------------------------------
# Grouping runner
# ---------------------------------------------------------------------------

def run_grouping(grouping_name, group_col, lookup):
    out_dir = os.path.join(OUTPUT_DIR, grouping_name)
    os.makedirs(out_dir, exist_ok=True)

    main_sheets = {}
    pair_sheets = {}
    sig_rows    = []

    for fname in sorted(os.listdir(PIPELINE_DIR)):
        if not fname.endswith(".csv"):
            continue
        df = load_csv(os.path.join(PIPELINE_DIR, fname))
        if "Project key" not in df.columns:
            continue

        df = df.merge(lookup[["Project key", group_col]], on="Project key", how="inner")
        df = df.dropna(subset=[group_col])

        # Drop groups below MIN_N
        valid = df[group_col].value_counts()
        valid = valid[valid >= MIN_N].index
        df = df[df[group_col].isin(valid)]
        if df[group_col].nunique() < 2:
            continue

        form_name = fname.replace(".csv", "").replace("_", " ").title()
        print(f"  {form_name:<35} {len(df):>5} IDs  groups: {df[group_col].value_counts().to_dict()}")

        main_df, pair_df = analyse_form(df, group_col, fname)
        sheet = fname.replace(".csv", "")[:31]
        if not main_df.empty:
            main_sheets[sheet] = main_df
            sig = main_df[main_df["BH sig"] == "Yes"].copy()
            if not sig.empty:
                sig.insert(0, "Form", form_name)
                sig.insert(1, "Grouping", grouping_name)
                sig_rows.append(sig)
        if not pair_df.empty:
            pair_sheets[sheet] = pair_df

    # Write Excel
    def write_xl(sheets, path):
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            for sheet, df in sheets.items():
                df.to_excel(w, sheet_name=sheet, index=False)

    write_xl(main_sheets, os.path.join(out_dir, "stats.xlsx"))
    write_xl(pair_sheets, os.path.join(out_dir, "pairwise.xlsx"))
    print(f"  → saved stats.xlsx + pairwise.xlsx\n")
    return sig_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    enrol = load_csv(os.path.join(PIPELINE_DIR, "enrollment.csv"))
    grp_col_full = [c for c in enrol.columns if "Enrolment Group" in c][0]
    enrol["enrolment_group"] = enrol[grp_col_full].map(ENROLMENT_GROUP_SHORT)
    enrol_lookup = enrol[["Project key", "enrolment_group"]].dropna()

    clin_raw = load_csv(os.path.join(DATA_DIR, "Clinical"))
    dx_col   = [c for c in clin_raw.columns if c.startswith("Determined diagnosis:")][0]
    clin_raw["determined_dx"] = pd.to_numeric(clin_raw[dx_col], errors="coerce").map(DX_LABELS)
    dx_lookup = clin_raw[["Project key", "determined_dx"]].dropna()

    all_sig = []

    print("\n=== Enhanced stats: By Enrolment Group ===")
    all_sig += run_grouping("by_enrolment_group", "enrolment_group", enrol_lookup)

    print("=== Enhanced stats: By Determined Dx ===")
    all_sig += run_grouping("by_determined_dx", "determined_dx", dx_lookup)

    if all_sig:
        summary = pd.concat(all_sig, ignore_index=True)
        summary.to_csv(os.path.join(OUTPUT_DIR, "significant_fdr.csv"), index=False)
        print(f"FDR-significant findings: {len(summary)} rows → significant_fdr.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
