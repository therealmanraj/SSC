"""
export_stats.py
Builds cross_reference.xlsx with 4 sheets from the data/ directory:
  - All Variables      (data dictionary cross-reference)
  - Consolidated       (versioned variable consolidation)
  - Numerical Stats    (descriptive stats for every numeric column)
  - Categorical Stats  (value counts for every categorical column)
"""

import os
import re
import pandas as pd
import numpy as np
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment
from openpyxl.utils import get_column_letter

DATA_DIR   = os.path.join(os.path.dirname(__file__), 'data')
EXCEL_PATH = os.path.join(os.path.dirname(__file__), 'cross_reference.xlsx')
SKIP_COLS  = {'Project key', 'Event Name', 'Complete?'}


# ── Load datasets ─────────────────────────────────────────────────────────────

def load_dataset(name: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA_DIR, name))
    if df.columns[0].startswith('Unnamed'):
        df = df.iloc[:, 1:]
    return df


datasets = {name: load_dataset(name) for name in os.listdir(DATA_DIR)}
print(f'Loaded {len(datasets)} datasets from {DATA_DIR}')


# ── Helpers ───────────────────────────────────────────────────────────────────

def col_dtype(series: pd.Series) -> str:
    n_unique = series.nunique()
    if pd.api.types.is_numeric_dtype(series) and n_unique > 12:
        return 'Numerical'
    if n_unique <= 30 or pd.api.types.is_object_dtype(series):
        return 'Categorical'
    return 'Text/Free'


def num_stats(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors='coerce').dropna()
    if len(s) == 0:
        return {}
    return {
        'Count':  int(len(s)),
        'Mean':   round(s.mean(),         3),
        'Median': round(s.median(),       3),
        'Std':    round(s.std(),          3),
        'Min':    round(s.min(),          3),
        'Q1':     round(s.quantile(0.25), 3),
        'Q3':     round(s.quantile(0.75), 3),
        'Max':    round(s.max(),          3),
        'Skew':   round(s.skew(),         3),
        'Kurt':   round(s.kurtosis(),     3),
    }


def cat_stats(series: pd.Series) -> list[dict]:
    vc = series.value_counts(dropna=True)
    total = vc.sum()
    return [
        {
            'Value':   str(v).split('/')[0].strip(),
            'Count':   int(c),
            'Pct (%)': round(100 * c / total, 1),
        }
        for v, c in vc.items()
    ]


# ── Build stats tables ────────────────────────────────────────────────────────

num_rows: list[dict] = []
cat_rows: list[dict] = []

for ds_name, df in sorted(datasets.items()):
    n_total = len(df)
    for col in df.columns:
        if col in SKIP_COLS:
            continue
        series      = df[col]
        n_null      = int(series.isna().sum())
        n_valid     = n_total - n_null
        missing_pct = round(100 * n_null / n_total, 1)
        dtype       = col_dtype(series)
        col_short   = col[:80]

        if dtype == 'Numerical':
            stats = num_stats(series)
            if stats:
                num_rows.append({'Dataset': ds_name, 'Column': col_short,
                                 'Non-null': n_valid, 'Missing %': missing_pct, **stats})

        elif dtype == 'Categorical':
            for row in cat_stats(series):
                cat_rows.append({'Dataset': ds_name, 'Column': col_short,
                                 'Non-null': n_valid, 'Missing %': missing_pct, **row})

num_df = pd.DataFrame(num_rows)
cat_df = pd.DataFrame(cat_rows)
print(f'Numerical  : {len(num_df):,} rows across {num_df["Dataset"].nunique()} datasets')
print(f'Categorical: {len(cat_df):,} rows across {cat_df["Dataset"].nunique()} datasets')


# ── Stub All Variables / Consolidated if no dict file present ─────────────────
# These are normally produced by the notebook's cross-reference section.
# If cross_reference.xlsx already exists, we preserve those sheets.
# Otherwise we write empty placeholders so the file is always valid.

_existing_sheets: dict[str, pd.DataFrame] = {}
if os.path.exists(EXCEL_PATH):
    wb_existing = openpyxl.load_workbook(EXCEL_PATH, read_only=True)
    for sn in ['All Variables', 'Consolidated']:
        if sn in wb_existing.sheetnames:
            wb_existing.close()
            _existing_sheets[sn] = pd.read_excel(EXCEL_PATH, sheet_name=sn)
    wb_existing.close()

all_vars_df  = _existing_sheets.get('All Variables',  pd.DataFrame(columns=['(run notebook to populate)']))
consolidated_df = _existing_sheets.get('Consolidated', pd.DataFrame(columns=['(run notebook to populate)']))


# ── Write all 4 sheets in one pass ───────────────────────────────────────────

print(f'Writing {EXCEL_PATH}...')
with pd.ExcelWriter(EXCEL_PATH, engine='openpyxl') as writer:
    all_vars_df.to_excel(writer,     sheet_name='All Variables',    index=False)
    consolidated_df.to_excel(writer, sheet_name='Consolidated',     index=False)
    num_df.to_excel(writer,          sheet_name='Numerical Stats',  index=False)
    cat_df.to_excel(writer,          sheet_name='Categorical Stats', index=False)


# ── Formatting ────────────────────────────────────────────────────────────────

ROLE_FILL = {
    'Predictor':         PatternFill('solid', fgColor='D0E8FF'),
    'Outcome':           PatternFill('solid', fgColor='FFD0D0'),
    'Outcome/Diagnosis': PatternFill('solid', fgColor='FFE8CC'),
    'Admin':             PatternFill('solid', fgColor='F0F0F0'),
}
IMPL_FILL = {
    'Level 1': PatternFill('solid', fgColor='C6EFCE'),
    'Level 2': PatternFill('solid', fgColor='BDD7EE'),
    'Level 3': PatternFill('solid', fgColor='FFEB9C'),
    'Level 4': PatternFill('solid', fgColor='FFC7CE'),
}
HDR_FILL   = PatternFill('solid', fgColor='212529')
HDR_FONT   = Font(color='FFFFFF', bold=True, size=11)
GREEN_FONT = Font(color='155724', bold=True)
RED_FONT   = Font(color='721C24', italic=True)
EVEN_FILL  = PatternFill('solid', fgColor='F8F9FA')
ODD_FILL   = PatternFill('solid', fgColor='FFFFFF')


def miss_fill(v: float) -> PatternFill:
    if v > 70: return PatternFill('solid', fgColor='FFC7CE')
    if v > 40: return PatternFill('solid', fgColor='FFEB9C')
    return         PatternFill('solid', fgColor='C6EFCE')


def autowidth(ws):
    for col in ws.columns:
        w = max((len(str(c.value or '')) for c in col), default=0)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(w + 2, 60)


def style_dict_sheet(ws, role_col, impl_col, found_col, missing_col):
    for cell in ws[1]:
        cell.fill = HDR_FILL
        cell.font = HDR_FONT
        cell.alignment = Alignment(horizontal='left', vertical='center', wrap_text=False)
    ws.freeze_panes = 'A2'
    for row in ws.iter_rows(min_row=2):
        role_val    = row[role_col - 1].value
        impl_val    = row[impl_col - 1].value
        found_val   = row[found_col - 1].value
        missing_val = row[missing_col - 1].value
        for cell in row:
            if role_val in ROLE_FILL:
                cell.fill = ROLE_FILL[role_val]
        if impl_val in IMPL_FILL:
            row[impl_col - 1].fill = IMPL_FILL[impl_val]
        if found_val == 'Yes':
            row[found_col - 1].font = GREEN_FONT
        elif found_val == 'No':
            row[found_col - 1].font = RED_FONT
        try:
            row[missing_col - 1].fill = miss_fill(float(missing_val))
        except (ValueError, TypeError):
            pass
    autowidth(ws)


def style_stats_sheet(ws, missing_col_idx: int):
    for cell in ws[1]:
        cell.fill = HDR_FILL
        cell.font = HDR_FONT
        cell.alignment = Alignment(horizontal='left', vertical='center')
    ws.freeze_panes = 'A2'
    for i, row in enumerate(ws.iter_rows(min_row=2), start=1):
        rf = EVEN_FILL if i % 2 == 0 else ODD_FILL
        for cell in row:
            cell.fill = rf
        try:
            row[missing_col_idx - 1].fill = miss_fill(float(row[missing_col_idx - 1].value or 0))
        except (ValueError, TypeError):
            pass
    autowidth(ws)


wb = openpyxl.load_workbook(EXCEL_PATH)

if 'All Variables' in wb.sheetnames and len(all_vars_df.columns) > 1:
    style_dict_sheet(wb['All Variables'], role_col=3, impl_col=4, found_col=5, missing_col=8)
if 'Consolidated' in wb.sheetnames and len(consolidated_df.columns) > 1:
    style_dict_sheet(wb['Consolidated'],  role_col=5, impl_col=6, found_col=9, missing_col=12)

style_stats_sheet(wb['Numerical Stats'],   missing_col_idx=4)
style_stats_sheet(wb['Categorical Stats'], missing_col_idx=4)

wb.save(EXCEL_PATH)

print(f'\nDone. Sheets in {EXCEL_PATH}:')
for sn in wb.sheetnames:
    ws = wb[sn]
    print(f'  {sn}: {ws.max_row - 1:,} rows × {ws.max_column} cols')
