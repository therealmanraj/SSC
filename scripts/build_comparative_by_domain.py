"""
Comparative analysis on domain-split CSVs.

For each General Clinical Domain × grouping (enrolment_group / determined_dx):
  - Box plots (PNG) for numeric variables
  - stats.xlsx    KW H + η² + BH-adjusted p + per-group mean/SD/median
  - pairwise.xlsx  MWU + Bonferroni p + Cohen's d for every group pair

Source: output/clean_pipeline/full_enrolled/by_clinical_domain/{domain}/

Output:
  output/comparative_analysis/by_clinical_domain/
    {Domain}/
      by_enrolment_group/
        plots/  stats.xlsx  pairwise.xlsx
      by_determined_dx/
        plots/  stats.xlsx  pairwise.xlsx
    significant_fdr.csv   all FDR-significant findings across all domains
"""

import os, re, math, warnings, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu, chi2_contingency
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DOMAIN_SRC  = os.path.join(
    os.path.dirname(__file__), "..", "output", "clean_pipeline",
    "full_enrolled", "by_clinical_domain"
)
DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR  = os.path.join(
    os.path.dirname(__file__), "..", "output", "comparative_analysis", "by_clinical_domain"
)

# ---------------------------------------------------------------------------
# Mappings
# ---------------------------------------------------------------------------
ENROLMENT_GROUP_SHORT = {
    "PD (Parkinson's Disease)/(Maladie de Parkinson)": "PD",
    "AP (Atypical Parkinsonism)/(Parkinsonisme Atypique)": "AP",
    "Healthy control/Contrôle": "HC",
}
DX_LABELS = {
    0.0: "PD", 1.0: "PSP", 2.0: "MSA", 3.0: "CBS",
    4.0: "DLB", 5.0: "FTD", 6.0: "ET",  7.0: "RBD",
}

SKIP_COLS   = {"Project key", "Event Name", "Complete?"}
MIN_N       = 5
MAX_PLOTS   = 24

# Field types that should be bar plots (categorical/ordinal responses)
BAR_FIELD_TYPES  = {"radio", "yesno", "checkbox", "dropdown"}
# Field types that should be box plots (continuous/calculated)
BOX_FIELD_TYPES  = {"calc", "text"}

# ---------------------------------------------------------------------------
# Load column → field_type mapping produced by build_domain_pipeline.py
# ---------------------------------------------------------------------------
_MAPPING_PATH = os.path.join(
    os.path.dirname(__file__), "..", "output", "clean_pipeline", "domain_column_mapping.csv"
)

def _load_col_field_type_map():
    """Return {normalised_csv_col: field_type} for all matched columns."""
    try:
        dm = pd.read_csv(_MAPPING_PATH)
        dm = dm[dm["Matched"] == "Yes"].dropna(subset=["Field Type"])
        mapping = {}
        for _, row in dm.iterrows():
            key = str(row["CSV column"]).strip()
            mapping[key] = str(row["Field Type"]).strip()
        return mapping
    except Exception:
        return {}

_COL_FIELD_TYPE = _load_col_field_type_map()

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def short_label(col, maxlen=52):
    label = re.split(r"\s{3,}", col)[0].strip()
    label = re.sub(r'^[\*"\s]+', "", label)
    return label[:maxlen]

def load_csv(path):
    df = pd.read_csv(path)
    return df.drop(columns=[c for c in df.columns if c.startswith("Unnamed:")], errors="ignore")

def _field_type(col):
    """Look up the REDCap field type for a CSV column name (truncated to 80 chars in mapping)."""
    # The mapping stores column names truncated to 80 chars
    return _COL_FIELD_TYPE.get(col[:80], _COL_FIELD_TYPE.get(col, None))

def numeric_cols_of(df):
    """Columns suitable for box plots: continuous/calculated numeric values."""
    cols = []
    for col in df.columns:
        if col in SKIP_COLS or col.startswith("Unnamed"):
            continue
        ft = _field_type(col)
        # If field type is known as categorical → skip here (goes to bar plot)
        if ft in BAR_FIELD_TYPES:
            continue
        s = df[col].dropna()
        if len(s) == 0:
            continue
        conv = pd.to_numeric(s, errors="coerce")
        if conv.notna().sum() / len(s) >= 0.5 and conv.std() > 0:
            cols.append(col)
    return cols

def categorical_cols_of(df):
    """Columns suitable for bar plots: radio/yesno/checkbox/dropdown OR text with few distinct values."""
    cols = []
    for col in df.columns:
        if col in SKIP_COLS or col.startswith("Unnamed"):
            continue
        s = df[col].dropna()
        if len(s) == 0:
            continue
        ft = _field_type(col)
        # Explicitly categorical by field type
        if ft in BAR_FIELD_TYPES:
            cols.append(col)
            continue
        # Fallback: non-numeric text with a small number of distinct levels
        conv = pd.to_numeric(s, errors="coerce")
        if conv.notna().sum() / len(s) < 0.5:
            if 2 <= s.nunique() <= 8:
                cols.append(col)
    return cols

def _chi2_p(df, group_col, col):
    """Chi-square p-value for association between group and categorical column."""
    try:
        ct = pd.crosstab(df[group_col], df[col])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            return np.nan
        _, p, _, _ = chi2_contingency(ct)
        return round(p, 6)
    except Exception:
        return np.nan

def plot_categorical_form(df, group_col, cat_cols, form_name, out_path, palette):
    """Grouped bar chart: % of each response level per group, one subplot per column."""
    valid = [c for c in cat_cols if df[c].notna().sum() >= 10][:MAX_PLOTS]
    if not valid:
        return

    groups = sorted(df[group_col].dropna().unique())
    colors = sns.color_palette(palette, len(groups))
    n      = len(valid)
    ncols  = min(3, n)
    nrows  = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.8))
    axes = np.array(axes).flatten()

    for i, col in enumerate(valid):
        ax = axes[i]
        # Use positional indexing to safely extract arrays regardless of column name chars
        col_idx   = list(df.columns).index(col)
        grp_idx   = list(df.columns).index(group_col)
        mask      = df.iloc[:, col_idx].notna() & df.iloc[:, grp_idx].notna()
        val_raw   = df.iloc[mask.values, col_idx].astype(str).str.split("/").str[0].str.strip().values
        grp_raw   = df.iloc[mask.values, grp_idx].values

        response_levels = sorted(set(val_raw))
        x     = np.arange(len(response_levels))
        bar_w = 0.8 / len(groups)

        for gi, (grp, color) in enumerate(zip(groups, colors)):
            grp_mask = grp_raw == grp
            grp_vals = val_raw[grp_mask]
            n_grp    = len(grp_vals)
            if n_grp == 0:
                continue
            counts = {lv: (grp_vals == lv).sum() for lv in response_levels}
            pct    = [counts[lv] / n_grp * 100 for lv in response_levels]
            ax.bar(x + gi * bar_w - 0.4 + bar_w / 2, pct, bar_w * 0.9,
                   label=grp, color=color, alpha=0.85)

        # Chi-square test using a safe crosstab
        try:
            ct = pd.crosstab(grp_raw, val_raw)
            _, p, _, _ = chi2_contingency(ct)
            p = round(float(p), 6)
        except Exception:
            p = np.nan
        sig = "***" if (not np.isnan(p) and p < 0.001) else \
              "**"  if (not np.isnan(p) and p < 0.01)  else \
              "*"   if (not np.isnan(p) and p < 0.05)  else "ns"
        title = f"{short_label(col, 44)}\nχ²  p={p:.4f} {sig}" if not np.isnan(p) else short_label(col, 44)
        ax.set_title(title, fontsize=8, pad=3)
        ax.set_xticks(x)
        ax.set_xticklabels(response_levels, fontsize=7, rotation=20, ha="right")
        ax.set_ylabel("% within group", fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        if i == 0:
            ax.legend(fontsize=7, title=group_col, title_fontsize=7)

    for j in range(len(valid), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"{form_name}  |  by {group_col}  (bar = % per group)", fontsize=11,
                 fontweight="bold", y=1.005)
    sns.despine()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def bh_adjust(pvals):
    pvals = np.array(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return pvals
    order   = np.argsort(pvals)
    adj     = pvals[order] * n / np.arange(1, n + 1)
    for i in range(n - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    adj = np.minimum(adj, 1.0)
    result = np.empty(n)
    result[order] = adj
    return result

def eta_sq(h, k, n):
    if n <= k:
        return np.nan
    return max(0.0, round((h - k + 1) / (n - k), 4))

def cohens_d(a, b):
    a, b = np.array(a), np.array(b)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pool = math.sqrt(
        ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1))
        / (len(a) + len(b) - 2)
    )
    return round(abs(a.mean() - b.mean()) / pool, 4) if pool > 0 else np.nan

def effect_label(d):
    if np.isnan(d): return ""
    return "negligible" if d < 0.2 else "small" if d < 0.5 else "medium" if d < 0.8 else "large"

def run_kw(df, group_col, col):
    groups = [
        pd.to_numeric(g[col], errors="coerce").dropna().values
        for _, g in df.groupby(group_col)
        if len(pd.to_numeric(g[col], errors="coerce").dropna()) >= 3
    ]
    if len(groups) < 2:
        return np.nan, np.nan, np.nan
    try:
        h, p  = kruskal(*groups)
        n_tot = sum(len(g) for g in groups)
        return round(h, 4), round(p, 6), eta_sq(h, len(groups), n_tot)
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
                "Variable": short_label(col),
                "Group 1": g1, "Group 2": g2,
                "n1": len(a), "n2": len(b),
                "MWU stat": round(stat, 2), "raw p": round(p, 6),
                "bonf p":   round(min(p * len(list(itertools.combinations(groups, 2))), 1.0), 6),
                "bonf sig": "",
                "Cohen's d": d, "Effect": effect_label(d),
            })
        except Exception:
            pass
    # Bonferroni flag
    for r in rows:
        r["bonf sig"] = "Yes" if r["bonf p"] < 0.05 else "No"
    return rows

# ---------------------------------------------------------------------------
# Build stats + pairwise DataFrames for one form
# ---------------------------------------------------------------------------

def analyse_form(df, group_col):
    groups    = sorted(df[group_col].dropna().unique())
    num_cols  = numeric_cols_of(df)
    main_rows = []
    pair_rows = []

    for col in num_cols:
        h, p, e2 = run_kw(df, group_col, col)
        row = {"Variable": short_label(col), "KW H": h, "raw p": p,
               "η²": e2, "BH adj p": np.nan, "BH sig": ""}
        for grp in groups:
            vals = pd.to_numeric(df[df[group_col] == grp][col], errors="coerce").dropna()
            row[f"{grp} n"]      = len(vals)
            row[f"{grp} mean"]   = round(vals.mean(),   3) if len(vals) else np.nan
            row[f"{grp} sd"]     = round(vals.std(),    3) if len(vals) else np.nan
            row[f"{grp} median"] = round(vals.median(), 3) if len(vals) else np.nan
        main_rows.append(row)
        pair_rows.extend(run_pairwise(df, group_col, col, groups))

    # BH correction over this form's p-values
    if main_rows:
        raw_ps = np.array([r["raw p"] for r in main_rows], dtype=float)
        adj_ps = bh_adjust(raw_ps)
        for r, ap in zip(main_rows, adj_ps):
            r["BH adj p"] = round(ap, 6) if not np.isnan(ap) else np.nan
            r["BH sig"]   = "Yes" if (not np.isnan(ap) and ap < 0.05) else "No"

    return pd.DataFrame(main_rows), pd.DataFrame(pair_rows) if pair_rows else pd.DataFrame()

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_form(df, group_col, num_cols, form_name, out_path, palette):
    valid = [c for c in num_cols
             if pd.to_numeric(df[c], errors="coerce").notna().sum() >= 10][:MAX_PLOTS]
    if not valid:
        return

    groups  = sorted(df[group_col].dropna().unique())
    colors  = sns.color_palette(palette, len(groups))
    n       = len(valid)
    ncols   = min(4, n)
    nrows   = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.8))
    axes = np.array(axes).flatten()

    for i, col in enumerate(valid):
        ax      = axes[i]
        plot_df = df[[group_col, col]].copy()
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
        plot_df = plot_df.dropna()

        sns.boxplot(
            data=plot_df, x=group_col, y=col,
            order=groups, palette=dict(zip(groups, colors)),
            width=0.55, linewidth=1.1,
            flierprops=dict(marker="o", markersize=2.5, alpha=0.4),
            ax=ax,
        )
        h, p, _ = run_kw(plot_df, group_col, col)
        sig     = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        title   = f"{short_label(col, 44)}\np={p:.4f} {sig}" if not np.isnan(p) else short_label(col, 44)
        ax.set_title(title, fontsize=8, pad=3)
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=8); ax.tick_params(axis="y", labelsize=7)

    for j in range(len(valid), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"{form_name}  |  by {group_col}", fontsize=11, fontweight="bold", y=1.005)
    sns.despine()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Run one grouping for one domain
# ---------------------------------------------------------------------------

def run_grouping(domain_name, domain_dir, group_col, lookup, palette):
    out_dir   = os.path.join(OUTPUT_DIR, domain_name, group_col.replace("_", " ").title().replace(" ", "_").lower())
    # cleaner folder names
    out_dir   = os.path.join(OUTPUT_DIR, domain_name,
                             "by_enrolment_group" if "enrolment" in group_col else "by_determined_dx")
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    main_sheets = {}
    pair_sheets = {}
    sig_rows    = []

    for fname in sorted(os.listdir(domain_dir)):
        if not fname.endswith(".csv"):
            continue
        df = load_csv(os.path.join(domain_dir, fname))
        if "Project key" not in df.columns or len(df) < MIN_N:
            continue

        # Attach group label
        df = df.merge(lookup[["Project key", group_col]], on="Project key", how="inner")
        df = df.dropna(subset=[group_col])

        valid_grps = df[group_col].value_counts()
        valid_grps = valid_grps[valid_grps >= MIN_N].index
        df = df[df[group_col].isin(valid_grps)]
        if df[group_col].nunique() < 2:
            continue

        form_name = fname.replace(".csv", "").replace("_", " ").title()
        num_cols  = numeric_cols_of(df)
        cat_cols  = categorical_cols_of(df)
        print(f"    {form_name:<30} {len(df):>5} IDs  {df[group_col].value_counts().to_dict()}")

        main_df, pair_df = analyse_form(df, group_col)
        sheet = fname.replace(".csv", "")[:31]

        if not main_df.empty:
            main_sheets[sheet] = main_df
            sig = main_df[main_df["BH sig"] == "Yes"].copy()
            if not sig.empty:
                sig.insert(0, "Domain",   domain_name)
                sig.insert(1, "Form",     form_name)
                sig.insert(2, "Grouping", group_col)
                sig_rows.append(sig)

        if not pair_df.empty:
            pair_sheets[sheet] = pair_df

        # Box plot for numeric columns
        plot_path = os.path.join(plots_dir, fname.replace(".csv", ".png"))
        plot_form(df, group_col, num_cols, form_name, plot_path, palette)

        # Bar plot for categorical columns
        bar_path = os.path.join(plots_dir, fname.replace(".csv", "_bar.png"))
        plot_categorical_form(df, group_col, cat_cols, form_name, bar_path, palette)

    # Write Excel files
    def write_xl(sheets, path):
        if not sheets:
            return
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            for sheet, sdf in sheets.items():
                sdf.to_excel(w, sheet_name=sheet, index=False)

    write_xl(main_sheets, os.path.join(out_dir, "stats.xlsx"))
    write_xl(pair_sheets, os.path.join(out_dir, "pairwise.xlsx"))
    return sig_rows

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Build group label lookups
    enrol    = load_csv(os.path.join(DATA_DIR, "Enrollement"))
    grp_col  = [c for c in enrol.columns if "Enrolment Group" in c][0]
    enrol["enrolment_group"] = enrol[grp_col].map(ENROLMENT_GROUP_SHORT)
    enrol_lookup = enrol[["Project key", "enrolment_group"]].dropna()

    clin_raw = load_csv(os.path.join(DATA_DIR, "Clinical"))
    dx_col   = [c for c in clin_raw.columns if c.startswith("Determined diagnosis:")][0]
    clin_raw["determined_dx"] = pd.to_numeric(clin_raw[dx_col], errors="coerce").map(DX_LABELS)
    dx_lookup = clin_raw[["Project key", "determined_dx"]].dropna()

    all_sig = []

    # Iterate over every domain folder
    for domain_name in sorted(os.listdir(DOMAIN_SRC)):
        domain_dir = os.path.join(DOMAIN_SRC, domain_name)
        if not os.path.isdir(domain_dir):
            continue

        clean_domain = domain_name.replace("_", " ")
        print(f"\n{'='*60}")
        print(f"  Domain: {clean_domain}")
        print(f"{'='*60}")

        print("  → by enrolment group")
        rows = run_grouping(domain_name, domain_dir, "enrolment_group", enrol_lookup, "Set2")
        all_sig.extend(rows)

        print("  → by determined dx")
        rows = run_grouping(domain_name, domain_dir, "determined_dx",   dx_lookup,    "tab10")
        all_sig.extend(rows)

    # Global FDR-significant summary
    if all_sig:
        summary = pd.concat(all_sig, ignore_index=True)
        out_path = os.path.join(OUTPUT_DIR, "significant_fdr.csv")
        summary.to_csv(out_path, index=False)
        print(f"\nFDR-significant findings: {len(summary)} rows → significant_fdr.csv")

    print("\nDone.")

if __name__ == "__main__":
    main()
