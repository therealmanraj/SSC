"""
Builds a clean filtered ID pipeline with per-form completion files.

Output layout:
  output/clean_pipeline/
    full_enrolled/          IDs with Study Status == Enrolled only
    enrolled_and_partial/   IDs with Enrolled OR QPN partially enrolled
    summary.csv             ID counts at every filter step for both groups
"""

import os
import re
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "clean_pipeline")

STATUS_COL = "Study Status:    Statut dans l'étude:"

GROUPS = {
    "full_enrolled": ["Enrolled/Inscrit"],
    "enrolled_and_partial": [
        "Enrolled/Inscrit",
        "QPN partially enrolled/RPQ partiellement inscrit",
    ],
}


def safe_filename(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_") + ".csv"


def load_form(fname: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA_DIR, fname))
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed:")], errors="ignore")
    return df


def build_group(group_name: str, enrolled_ids: set, group_dir: str) -> list[dict]:
    os.makedirs(group_dir, exist_ok=True)
    summary_rows = []

    enrol_df = load_form("Enrollement")
    base = enrol_df[enrol_df["Project key"].isin(enrolled_ids)].copy()
    base.to_csv(os.path.join(group_dir, "enrollment.csv"), index=False)
    summary_rows.append({"form": "enrollment", "group": group_name, "ids": len(base)})
    print(f"  enrollment        → {len(base):>5} IDs")

    for fname in sorted(os.listdir(DATA_DIR)):
        if fname == "Enrollement":
            continue
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.isfile(fpath):
            continue

        df = load_form(fname)
        if "Project key" not in df.columns:
            continue

        df = df[df["Project key"].isin(enrolled_ids)].copy()

        if "Complete?" in df.columns:
            df = df[df["Complete?"] == "Complete"].copy()
            note = "complete"
        else:
            # Epidemiological has no Complete? column — keep all enrolled IDs
            note = "no_completion_col"

        out_path = os.path.join(group_dir, safe_filename(fname))
        df.to_csv(out_path, index=False)
        summary_rows.append(
            {"form": fname, "group": group_name, "ids": len(df), "note": note}
        )
        print(f"  {fname:<30} → {len(df):>5} IDs  ({note})")

    return summary_rows


def main():
    enrol_df = load_form("Enrollement")
    all_summary = []

    for group_name, statuses in GROUPS.items():
        enrolled_ids = set(enrol_df[enrol_df[STATUS_COL].isin(statuses)]["Project key"])
        group_dir = os.path.join(OUTPUT_DIR, group_name)
        print(f"\n=== {group_name} ({len(enrolled_ids)} base IDs) ===")
        rows = build_group(group_name, enrolled_ids, group_dir)
        all_summary.extend(rows)

    summary_df = pd.DataFrame(all_summary)[["group", "form", "ids", "note"]]
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)
    print(f"\nSummary written to output/clean_pipeline/summary.csv")


if __name__ == "__main__":
    main()
