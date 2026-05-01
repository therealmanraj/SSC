"""
Builds a clean filtered ID pipeline with per-form completion files.

Output layout:
  output/clean_pipeline/
    full_enrolled/
      *.csv                       all enrolled, per-form completion filter
      by_enrolment_group/
        PD/ AP/ HC/               same form files, filtered to that group
      by_determined_dx/
        0_PD/ 1_PSP/ 2_MSA/ ...  same form files, filtered to that dx
    enrolled_and_partial/
      (same structure)
    summary.csv                   ID counts at every step for all groups/subgroups
"""

import os
import re
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "clean_pipeline")

STATUS_COL = "Study Status:    Statut dans l'étude:"
ENROL_GROUP_COL = "Enrolment Group:    Groupe d'inscription:"
DX_COL = (
    "Determined diagnosis:  If score = 0, Parkinson's Disease (PD)  "
    "If score = 1, Progressive Supranuclear Palsy (PSP)  "
    "If score = 2, Multiple System Atrophy (MSA)  "
    "If score = 3, Corticobasal Syndrome (CBS)  "
    "If score = 4, Dementia with Lewy Bodies (DLB)  If sc"
)

ENROLLMENT_GROUPS = {
    "PD (Parkinson's Disease)/(Maladie de Parkinson)": "PD",
    "AP (Atypical Parkinsonism)/(Parkinsonisme Atypique)": "AP",
    "Healthy control/Contrôle": "HC",
}

DX_LABELS = {
    0: "0_PD",
    1: "1_PSP",
    2: "2_MSA",
    3: "3_CBS",
    4: "4_DLB",
    5: "5_FTD",
    6: "6_ET",
    7: "7_RBD",
}

ENROLLMENT_STATUSES = {
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


def save_forms_for_ids(ids: set, out_dir: str, label: str) -> list[dict]:
    """Save enrollment.csv + one CSV per form for the given ID set."""
    os.makedirs(out_dir, exist_ok=True)
    rows = []

    enrol_df = load_form("Enrollement")
    base = enrol_df[enrol_df["Project key"].isin(ids)].copy()
    base.to_csv(os.path.join(out_dir, "enrollment.csv"), index=False)
    rows.append({"label": label, "form": "enrollment", "ids": len(base), "note": ""})
    print(f"  [{label}] enrollment → {len(base):>5}")

    for fname in sorted(os.listdir(DATA_DIR)):
        if fname == "Enrollement":
            continue
        if not os.path.isfile(os.path.join(DATA_DIR, fname)):
            continue

        df = load_form(fname)
        if "Project key" not in df.columns:
            continue

        df = df[df["Project key"].isin(ids)].copy()

        if "Complete?" in df.columns:
            df = df[df["Complete?"] == "Complete"].copy()
            note = "complete"
        else:
            note = "no_completion_col"

        df.to_csv(os.path.join(out_dir, safe_filename(fname)), index=False)
        rows.append({"label": label, "form": fname, "ids": len(df), "note": note})
        print(f"  [{label}] {fname:<32} → {len(df):>5}  ({note})")

    return rows


def get_enrolment_group_ids(enrol_df: pd.DataFrame) -> dict[str, set]:
    """Return {folder_name: set of Project keys} for each enrolment group."""
    result = {}
    for full_label, folder in ENROLLMENT_GROUPS.items():
        result[folder] = set(
            enrol_df[enrol_df[ENROL_GROUP_COL] == full_label]["Project key"]
        )
    return result


def get_dx_ids(clin_df: pd.DataFrame) -> dict[str, set]:
    """Return {folder_name: set of Project keys} for each determined dx code."""
    result = {}
    dx_col = [c for c in clin_df.columns if c.startswith("Determined diagnosis:")][0]
    for code, folder in DX_LABELS.items():
        result[folder] = set(
            clin_df[clin_df[dx_col] == float(code)]["Project key"]
        )
    return result


def main():
    enrol_df = load_form("Enrollement")
    clin_df = load_form("Clinical")
    all_summary = []

    enrol_group_ids = get_enrolment_group_ids(enrol_df)
    dx_ids = get_dx_ids(clin_df)

    for status_name, statuses in ENROLLMENT_STATUSES.items():
        enrolled_ids = set(enrol_df[enrol_df[STATUS_COL].isin(statuses)]["Project key"])
        group_dir = os.path.join(OUTPUT_DIR, status_name)

        # --- All enrolled IDs (baseline) ---
        print(f"\n=== {status_name} — all ({len(enrolled_ids)} IDs) ===")
        rows = save_forms_for_ids(enrolled_ids, group_dir, label=status_name)
        all_summary.extend(rows)

        # --- By enrolment group ---
        for folder, group_ids in enrol_group_ids.items():
            ids = enrolled_ids & group_ids
            sub_dir = os.path.join(group_dir, "by_enrolment_group", folder)
            label = f"{status_name}/by_enrolment_group/{folder}"
            print(f"\n  -- {label} ({len(ids)} IDs) --")
            rows = save_forms_for_ids(ids, sub_dir, label=label)
            all_summary.extend(rows)

        # --- By determined dx ---
        for folder, d_ids in dx_ids.items():
            ids = enrolled_ids & d_ids
            if not ids:
                continue
            sub_dir = os.path.join(group_dir, "by_determined_dx", folder)
            label = f"{status_name}/by_determined_dx/{folder}"
            print(f"\n  -- {label} ({len(ids)} IDs) --")
            rows = save_forms_for_ids(ids, sub_dir, label=label)
            all_summary.extend(rows)

    summary_df = (
        pd.DataFrame(all_summary)[["label", "form", "ids", "note"]]
    )
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)
    print(f"\nSummary written to output/clean_pipeline/summary.csv")


if __name__ == "__main__":
    main()
