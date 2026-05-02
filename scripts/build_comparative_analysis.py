"""
Comparative analysis of full_enrolled IDs by:
  1. Enrolment group (PD / AP / HC)
  2. Determined dx (PD / PSP / MSA / CBS / DLB / ET / RBD)

For each form, exports:
  - Box plots (PNG) for numeric variables
  - stats.xlsx with descriptive stats + Kruskal-Wallis p-values per variable

Output:
  output/comparative_analysis/
    by_enrolment_group/
      plots/{form}.png
      stats.xlsx
    by_determined_dx/
      plots/{form}.png
      stats.xlsx
    significant_findings.csv   <- all p<0.05 results across both analyses
"""

import os
import re
import warnings
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal, chi2_contingency

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PIPELINE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "output", "clean_pipeline", "full_enrolled"
)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "output", "comparative_analysis"
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
    0.0: "PD",
    1.0: "PSP",
    2.0: "MSA",
    3.0: "CBS",
    4.0: "DLB",
    5.0: "FTD",
    6.0: "ET",
    7.0: "RBD",
}

# Always skip these columns
SKIP_COLS = {"Project key", "Event Name", "Complete?"}

# Minimum group size to include a group in analysis/plots
MIN_GROUP_N = 5

# Max subplots per figure
MAX_PLOTS = 24

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def short_label(col: str, maxlen: int = 52) -> str:
    """Extract short English label from a bilingual column name."""
    label = re.split(r"\s{3,}", col)[0].strip()
    label = re.sub(r'^[\*"\s]+', "", label)
    return label[:maxlen] if len(label) > maxlen else label


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.drop(
        columns=[c for c in df.columns if c.startswith("Unnamed:")], errors="ignore"
    )


def classify_columns(df: pd.DataFrame):
    """Return (numeric_cols, cat_cols) based on column content."""
    numeric_cols, cat_cols = [], []
    for col in df.columns:
        if col in SKIP_COLS or col.startswith("Unnamed"):
            continue
        series = df[col].dropna()
        if len(series) == 0:
            continue
        converted = pd.to_numeric(series, errors="coerce")
        frac_numeric = converted.notna().sum() / len(series)
        if frac_numeric >= 0.5:
            if converted.std() > 0:
                numeric_cols.append(col)
        else:
            if series.nunique() > 1:
                cat_cols.append(col)
    return numeric_cols, cat_cols


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def kw_test(df: pd.DataFrame, group_col: str, col: str):
    groups = []
    for _, gdf in df.groupby(group_col):
        vals = pd.to_numeric(gdf[col], errors="coerce").dropna().values
        if len(vals) >= 3:
            groups.append(vals)
    if len(groups) < 2:
        return np.nan, np.nan
    try:
        h, p = kruskal(*groups)
        return round(h, 4), round(p, 6)
    except Exception:
        return np.nan, np.nan


def chi2_test(df: pd.DataFrame, group_col: str, col: str):
    try:
        ct = pd.crosstab(df[group_col], df[col])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            return np.nan, np.nan
        chi2, p, _, _ = chi2_contingency(ct)
        return round(chi2, 4), round(p, 6)
    except Exception:
        return np.nan, np.nan


def build_stats_df(df: pd.DataFrame, group_col: str, numeric_cols, cat_cols) -> pd.DataFrame:
    groups = sorted(df[group_col].dropna().unique())
    rows = []

    for col in numeric_cols:
        h, p = kw_test(df, group_col, col)
        row = {
            "Variable": short_label(col),
            "Type": "Numeric",
            "Test": "Kruskal-Wallis",
            "Statistic": h,
            "p_value": p,
            "Significant": "Yes" if (not math.isnan(p) if not np.isnan(p) else False) and p < 0.05 else "No",
        }
        for grp in groups:
            vals = pd.to_numeric(
                df[df[group_col] == grp][col], errors="coerce"
            ).dropna()
            row[f"{grp} n"] = len(vals)
            row[f"{grp} mean"] = round(vals.mean(), 3) if len(vals) else np.nan
            row[f"{grp} sd"] = round(vals.std(), 3) if len(vals) else np.nan
            row[f"{grp} median"] = round(vals.median(), 3) if len(vals) else np.nan
        rows.append(row)

    for col in cat_cols:
        chi2, p = chi2_test(df, group_col, col)
        row = {
            "Variable": short_label(col),
            "Type": "Categorical",
            "Test": "Chi-square",
            "Statistic": chi2,
            "p_value": p,
            "Significant": "Yes" if (not np.isnan(p) and p < 0.05) else "No",
        }
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_form(
    df: pd.DataFrame,
    group_col: str,
    numeric_cols: list,
    form_name: str,
    out_path: str,
    palette: str,
):
    # Only keep columns with enough non-null values
    valid = [
        c for c in numeric_cols
        if pd.to_numeric(df[c], errors="coerce").notna().sum() >= 10
    ]
    if not valid:
        return

    cols_to_plot = valid[:MAX_PLOTS]
    n = len(cols_to_plot)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.8))
    axes = np.array(axes).flatten()

    groups = sorted(df[group_col].dropna().unique())
    colors = sns.color_palette(palette, len(groups))

    for i, col in enumerate(cols_to_plot):
        ax = axes[i]
        plot_df = df[[group_col, col]].copy()
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
        plot_df = plot_df.dropna()

        sns.boxplot(
            data=plot_df,
            x=group_col,
            y=col,
            order=groups,
            palette=dict(zip(groups, colors)),
            width=0.55,
            linewidth=1.2,
            flierprops=dict(marker="o", markersize=2.5, alpha=0.4),
            ax=ax,
        )

        h, p = kw_test(plot_df, group_col, col)
        if not np.isnan(p):
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            title = f"{short_label(col, 45)}\np = {p:.4f}  {sig}"
        else:
            title = short_label(col, 45)

        ax.set_title(title, fontsize=8, pad=4)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=7)

    for j in range(len(cols_to_plot), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"{form_name}  |  grouped by: {group_col}",
        fontsize=11,
        fontweight="bold",
        y=1.005,
    )
    sns.despine()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    plot saved → {os.path.basename(out_path)}")


# ---------------------------------------------------------------------------
# Main analysis loop
# ---------------------------------------------------------------------------

def analyze_grouping(grouping_name: str, group_col: str, lookup: pd.DataFrame, palette: str):
    out_dir = os.path.join(OUTPUT_DIR, grouping_name)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    excel_path = os.path.join(out_dir, "stats.xlsx")
    writer = pd.ExcelWriter(excel_path, engine="openpyxl")
    all_sig = []

    for fname in sorted(os.listdir(PIPELINE_DIR)):
        if not fname.endswith(".csv"):
            continue

        df = load_csv(os.path.join(PIPELINE_DIR, fname))
        if "Project key" not in df.columns or len(df) < MIN_GROUP_N:
            continue

        # Attach group label
        df = df.merge(lookup[["Project key", group_col]], on="Project key", how="inner")
        df = df.dropna(subset=[group_col])

        # Drop groups smaller than MIN_GROUP_N
        valid_groups = df[group_col].value_counts()
        valid_groups = valid_groups[valid_groups >= MIN_GROUP_N].index.tolist()
        if len(valid_groups) < 2:
            continue
        df = df[df[group_col].isin(valid_groups)]

        form_name = fname.replace(".csv", "").replace("_", " ").title()
        numeric_cols, cat_cols = classify_columns(df)
        n_per_grp = df[group_col].value_counts().to_dict()
        print(f"  {form_name:<35} {len(df):>5} IDs  groups: {n_per_grp}")

        # --- Stats table ---
        stats_df = build_stats_df(df, group_col, numeric_cols, cat_cols)
        if not stats_df.empty:
            sheet = fname.replace(".csv", "")[:31]
            stats_df.to_excel(writer, sheet_name=sheet, index=False)

            sig = stats_df[stats_df["Significant"] == "Yes"].copy()
            if not sig.empty:
                sig.insert(0, "Form", form_name)
                sig.insert(1, "Grouping", grouping_name)
                all_sig.append(sig)

        # --- Plot ---
        plot_path = os.path.join(plots_dir, fname.replace(".csv", ".png"))
        plot_form(df, group_col, numeric_cols, form_name, plot_path, palette)

    writer.close()
    print(f"  → stats.xlsx saved\n")
    return all_sig


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Enrolment group lookup ---
    enrol = load_csv(os.path.join(PIPELINE_DIR, "enrollment.csv"))
    grp_col_full = [c for c in enrol.columns if "Enrolment Group" in c][0]
    enrol["enrolment_group"] = enrol[grp_col_full].map(ENROLMENT_GROUP_SHORT)
    enrol_lookup = enrol[["Project key", "enrolment_group"]].dropna()

    # --- Determined dx lookup (from raw Clinical — broader coverage than filtered) ---
    clin_raw = load_csv(os.path.join(DATA_DIR, "Clinical"))
    dx_col = [c for c in clin_raw.columns if c.startswith("Determined diagnosis:")][0]
    clin_raw["determined_dx"] = (
        pd.to_numeric(clin_raw[dx_col], errors="coerce").map(DX_LABELS)
    )
    dx_lookup = clin_raw[["Project key", "determined_dx"]].dropna()

    all_sig = []

    print("\n=== By Enrolment Group (PD / AP / HC) ===")
    all_sig += analyze_grouping("by_enrolment_group", "enrolment_group", enrol_lookup, "Set2")

    print("=== By Determined Dx ===")
    all_sig += analyze_grouping("by_determined_dx", "determined_dx", dx_lookup, "tab10")

    # --- Significant findings summary ---
    if all_sig:
        summary = pd.concat(all_sig, ignore_index=True)
        out_path = os.path.join(OUTPUT_DIR, "significant_findings.csv")
        summary.to_csv(out_path, index=False)
        print(f"Significant findings ({len(summary)} rows) → significant_findings.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
