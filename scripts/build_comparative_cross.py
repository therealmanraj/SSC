"""
Cross-dimensional comparative analysis: Enrolment Group × Determined Dx

Option 1  →  output/comparative_analysis/enrolment_x_dx_grouped/
  Grouped box plots: x = enrolment group (PD/AP/HC), hue = determined dx.
  Stats table uses combined group label (e.g. "PD | PSP").

Option 2  →  output/comparative_analysis/within_enrolment_by_dx/
  For each enrolment group separately, compare variables by determined dx.
  Subfolders: PD/  AP/  HC/
"""

import os
import re
import math
import warnings

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
    0.0: "PD", 1.0: "PSP", 2.0: "MSA", 3.0: "CBS",
    4.0: "DLB", 5.0: "FTD", 6.0: "ET", 7.0: "RBD",
}
ENROL_ORDER = ["PD", "AP", "HC"]
DX_ORDER    = ["PD", "PSP", "MSA", "CBS", "DLB", "FTD", "ET", "RBD"]

SKIP_COLS   = {"Project key", "Event Name", "Complete?"}
MIN_N       = 5          # minimum group size to include
MAX_PLOTS   = 24         # subplots per figure


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def short_label(col: str, maxlen: int = 50) -> str:
    label = re.split(r"\s{3,}", col)[0].strip()
    label = re.sub(r'^[\*"\s]+', "", label)
    return label[:maxlen]


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.drop(columns=[c for c in df.columns if c.startswith("Unnamed:")], errors="ignore")


def classify_columns(df: pd.DataFrame):
    numeric_cols, cat_cols = [], []
    for col in df.columns:
        if col in SKIP_COLS or col.startswith("Unnamed"):
            continue
        s = df[col].dropna()
        if len(s) == 0:
            continue
        conv = pd.to_numeric(s, errors="coerce")
        if conv.notna().sum() / len(s) >= 0.5:
            if conv.std() > 0:
                numeric_cols.append(col)
        else:
            if s.nunique() > 1:
                cat_cols.append(col)
    return numeric_cols, cat_cols


def kw_test(df: pd.DataFrame, group_col: str, col: str):
    groups = [
        pd.to_numeric(g[col], errors="coerce").dropna().values
        for _, g in df.groupby(group_col)
        if len(pd.to_numeric(g[col], errors="coerce").dropna()) >= 3
    ]
    if len(groups) < 2:
        return np.nan, np.nan
    try:
        h, p = kruskal(*groups)
        return round(h, 4), round(p, 6)
    except Exception:
        return np.nan, np.nan


def build_stats_df(df: pd.DataFrame, group_col: str, numeric_cols, cat_cols) -> pd.DataFrame:
    groups = sorted(df[group_col].dropna().unique())
    rows = []
    for col in numeric_cols:
        h, p = kw_test(df, group_col, col)
        sig = "Yes" if (not np.isnan(p) and p < 0.05) else "No"
        row = {"Variable": short_label(col), "Type": "Numeric",
               "KW_H": h, "p_value": p, "Significant": sig}
        for grp in groups:
            vals = pd.to_numeric(df[df[group_col] == grp][col], errors="coerce").dropna()
            row[f"{grp} n"]      = len(vals)
            row[f"{grp} mean"]   = round(vals.mean(),   3) if len(vals) else np.nan
            row[f"{grp} sd"]     = round(vals.std(),    3) if len(vals) else np.nan
            row[f"{grp} median"] = round(vals.median(), 3) if len(vals) else np.nan
        rows.append(row)
    for col in cat_cols:
        try:
            ct = pd.crosstab(df[group_col], df[col])
            chi2, p, _, _ = chi2_contingency(ct)
            chi2, p = round(chi2, 4), round(p, 6)
        except Exception:
            chi2, p = np.nan, np.nan
        sig = "Yes" if (not np.isnan(p) and p < 0.05) else "No"
        rows.append({"Variable": short_label(col), "Type": "Categorical",
                     "Chi2": chi2, "p_value": p, "Significant": sig})
    return pd.DataFrame(rows)


def save_excel(dfs: dict, path: str):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet, df in dfs.items():
            if not df.empty:
                df.to_excel(writer, sheet_name=sheet[:31], index=False)


# ---------------------------------------------------------------------------
# Option 1 — Grouped box plots: x = enrolment_group, hue = determined_dx
# ---------------------------------------------------------------------------

def plot_grouped(df: pd.DataFrame, numeric_cols: list, form_name: str, out_path: str):
    valid = [c for c in numeric_cols
             if pd.to_numeric(df[c], errors="coerce").notna().sum() >= 10][:MAX_PLOTS]
    if not valid:
        return

    enrol_order = [g for g in ENROL_ORDER if g in df["enrolment_group"].unique()]
    dx_order    = [d for d in DX_ORDER    if d in df["determined_dx"].unique()]
    palette     = dict(zip(dx_order, sns.color_palette("tab10", len(dx_order))))

    n = len(valid)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axes = np.array(axes).flatten()

    for i, col in enumerate(valid):
        ax = axes[i]
        plot_df = df[["enrolment_group", "determined_dx", col]].copy()
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
        plot_df = plot_df.dropna()

        sns.boxplot(
            data=plot_df, x="enrolment_group", y=col,
            hue="determined_dx",
            order=enrol_order, hue_order=dx_order,
            palette=palette,
            width=0.65, linewidth=1.1,
            flierprops=dict(marker="o", markersize=2, alpha=0.35),
            ax=ax,
        )

        h, p = kw_test(plot_df, "enrolment_group", col)
        sig  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.set_title(f"{short_label(col, 44)}\np={p:.4f} {sig}", fontsize=8, pad=3)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=7)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    # Single shared legend
    handles = [
        plt.matplotlib.patches.Patch(color=palette[d], label=d) for d in dx_order
    ]
    fig.legend(handles=handles, title="Determined Dx", loc="lower center",
               ncol=len(dx_order), fontsize=8, title_fontsize=8,
               bbox_to_anchor=(0.5, -0.02))

    for j in range(len(valid), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"{form_name}  |  x = enrolment group,  colour = determined dx",
        fontsize=11, fontweight="bold", y=1.005,
    )
    sns.despine()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    plot saved → {os.path.basename(out_path)}")


def run_option1(master: pd.DataFrame):
    """Option 1: x=enrolment_group, hue=determined_dx."""
    out_dir   = os.path.join(OUTPUT_DIR, "enrolment_x_dx_grouped")
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    sheets = {}
    for fname in sorted(os.listdir(PIPELINE_DIR)):
        if not fname.endswith(".csv"):
            continue
        df = load_csv(os.path.join(PIPELINE_DIR, fname))
        if "Project key" not in df.columns:
            continue

        df = df.merge(master, on="Project key", how="inner").dropna(
            subset=["enrolment_group", "determined_dx"]
        )
        if len(df) < MIN_N:
            continue

        # Keep only enrolment groups and dx values with n >= MIN_N
        valid_enrol = df["enrolment_group"].value_counts()
        valid_enrol = valid_enrol[valid_enrol >= MIN_N].index
        valid_dx    = df["determined_dx"].value_counts()
        valid_dx    = valid_dx[valid_dx >= MIN_N].index
        df = df[df["enrolment_group"].isin(valid_enrol) & df["determined_dx"].isin(valid_dx)]
        if df["enrolment_group"].nunique() < 2 and df["determined_dx"].nunique() < 2:
            continue

        form_name    = fname.replace(".csv", "").replace("_", " ").title()
        numeric_cols, cat_cols = classify_columns(df)
        combo_counts = df.groupby(["enrolment_group", "determined_dx"]).size().to_dict()
        print(f"  {form_name:<35} {len(df):>5} IDs  combos: {combo_counts}")

        # Stats on combined label
        df["group_combo"] = df["enrolment_group"] + " | " + df["determined_dx"]
        stats_df = build_stats_df(df, "group_combo", numeric_cols, cat_cols)
        sheets[fname.replace(".csv", "")[:31]] = stats_df

        plot_path = os.path.join(plots_dir, fname.replace(".csv", ".png"))
        plot_grouped(df, numeric_cols, form_name, plot_path)

    save_excel(sheets, os.path.join(out_dir, "stats.xlsx"))
    print(f"  → enrolment_x_dx_grouped/stats.xlsx saved\n")


# ---------------------------------------------------------------------------
# Option 2 — Within each enrolment group, compare by determined_dx
# ---------------------------------------------------------------------------

def plot_within(df: pd.DataFrame, group_col: str, numeric_cols: list,
                form_name: str, out_path: str, palette: str):
    valid = [c for c in numeric_cols
             if pd.to_numeric(df[c], errors="coerce").notna().sum() >= 5][:MAX_PLOTS]
    if not valid:
        return

    dx_order = [d for d in DX_ORDER if d in df[group_col].unique()]
    colors   = sns.color_palette(palette, len(dx_order))

    n = len(valid)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.8))
    axes = np.array(axes).flatten()

    for i, col in enumerate(valid):
        ax = axes[i]
        plot_df = df[[group_col, col]].copy()
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
        plot_df = plot_df.dropna()

        sns.boxplot(
            data=plot_df, x=group_col, y=col,
            order=dx_order,
            palette=dict(zip(dx_order, colors)),
            width=0.55, linewidth=1.1,
            flierprops=dict(marker="o", markersize=2.5, alpha=0.4),
            ax=ax,
        )

        h, p = kw_test(plot_df, group_col, col)
        sig  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        title = f"{short_label(col, 44)}\np={p:.4f} {sig}" if not np.isnan(p) else short_label(col, 44)
        ax.set_title(title, fontsize=8, pad=3)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=7)

    for j in range(len(valid), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"{form_name}  |  by determined dx",
        fontsize=11, fontweight="bold", y=1.005,
    )
    sns.despine()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    plot saved → {os.path.basename(out_path)}")


def run_option2_for_group(enrol_group: str, master: pd.DataFrame):
    """Option 2: within one enrolment group, compare by determined_dx."""
    out_dir   = os.path.join(OUTPUT_DIR, "within_enrolment_by_dx", enrol_group)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    group_master = master[master["enrolment_group"] == enrol_group]

    sheets = {}
    for fname in sorted(os.listdir(PIPELINE_DIR)):
        if not fname.endswith(".csv"):
            continue
        df = load_csv(os.path.join(PIPELINE_DIR, fname))
        if "Project key" not in df.columns:
            continue

        df = df.merge(group_master, on="Project key", how="inner").dropna(
            subset=["determined_dx"]
        )
        if len(df) < MIN_N:
            continue

        valid_dx = df["determined_dx"].value_counts()
        valid_dx = valid_dx[valid_dx >= MIN_N].index
        df = df[df["determined_dx"].isin(valid_dx)]
        if df["determined_dx"].nunique() < 2:
            continue

        form_name    = fname.replace(".csv", "").replace("_", " ").title()
        numeric_cols, cat_cols = classify_columns(df)
        dx_counts    = df["determined_dx"].value_counts().to_dict()
        print(f"    {form_name:<35} {len(df):>4} IDs  dx: {dx_counts}")

        stats_df = build_stats_df(df, "determined_dx", numeric_cols, cat_cols)
        sheets[fname.replace(".csv", "")[:31]] = stats_df

        plot_path = os.path.join(plots_dir, fname.replace(".csv", ".png"))
        palette   = "Blues_d" if enrol_group == "HC" else ("Oranges_d" if enrol_group == "AP" else "tab10")
        plot_within(df, "determined_dx", numeric_cols, form_name, plot_path, palette)

    if sheets:
        save_excel(sheets, os.path.join(out_dir, "stats.xlsx"))
        print(f"    → within_enrolment_by_dx/{enrol_group}/stats.xlsx saved")


def run_option2(master: pd.DataFrame):
    """Option 2: run for each enrolment group."""
    for grp in ENROL_ORDER:
        print(f"\n  -- Enrolment group: {grp} --")
        run_option2_for_group(grp, master)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Build master lookup: Project key → enrolment_group + determined_dx
    enrol = load_csv(os.path.join(PIPELINE_DIR, "enrollment.csv"))
    grp_col_full = [c for c in enrol.columns if "Enrolment Group" in c][0]
    enrol["enrolment_group"] = enrol[grp_col_full].map(ENROLMENT_GROUP_SHORT)

    clin_raw = load_csv(os.path.join(DATA_DIR, "Clinical"))
    dx_col = [c for c in clin_raw.columns if c.startswith("Determined diagnosis:")][0]
    clin_raw["determined_dx"] = pd.to_numeric(clin_raw[dx_col], errors="coerce").map(DX_LABELS)

    master = (
        enrol[["Project key", "enrolment_group"]]
        .merge(clin_raw[["Project key", "determined_dx"]], on="Project key", how="left")
    )

    print(f"Master lookup: {len(master)} IDs")
    print("Enrolment × Dx cross-tab:")
    print(pd.crosstab(master["enrolment_group"], master["determined_dx"], dropna=False))

    print("\n=== Option 1: Grouped plots (x=enrolment group, hue=determined dx) ===")
    run_option1(master)

    print("=== Option 2: Within each enrolment group, by determined dx ===")
    run_option2(master)

    print("\nDone.")


if __name__ == "__main__":
    main()
