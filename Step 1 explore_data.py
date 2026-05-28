"""
TriNetX PancPump Study - Step 1: Data Exploration
Study: Glycemic Control in Kidney+Pump (KTP) vs Simultaneous Pancreas-Kidney (SPK) Transplant
PI: Paul Kuo
Dataset path: C:\\Users\\pkmd0\\OneDrive\\Desktop\\TriNetX PancPump

Run this script first to understand the structure and completeness of your data
before building cohorts and running analyses.

Requirements:
    pip install pandas numpy matplotlib seaborn openpyxl
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION — edit this path if needed
# =============================================================================
DATA_DIR   = r"Z:\homes\Rachel Wolansky\Panc Pump\Data In"
OUTPUT_DIR = r"Z:\homes\Rachel Wolansky\Panc Pump\Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# All files are CSVs despite having Excel icons — load as CSV
def load(filename, **kwargs):
    """Load a TriNetX CSV file with standard settings."""
    path = os.path.join(DATA_DIR, filename)
    print(f"  Loading {filename}...")
    df = pd.read_csv(path, low_memory=False, **kwargs)
    print(f"    -> {len(df):,} rows, {len(df.columns)} columns")
    return df

# =============================================================================
# 1. LOAD CORE FILES
# =============================================================================
print("\n" + "="*60)
print("LOADING CORE FILES")
print("="*60)

patient         = load("patient.csv")
patient_cohort  = load("patient_cohort.csv")
cohort_details  = load("cohort_details.csv")
diagnosis       = load("diagnosis.csv")
lab_result      = load("lab_result.csv")
vital_signs     = load("vitals_signs.csv")   # note: filename has typo in dataset
procedure       = load("procedure.csv")
medication_ing  = load("medication_ingredient.csv")
standardized    = load("standardized_terminology.csv")

# =============================================================================
# 2. COHORT ASSIGNMENT
# =============================================================================
print("\n" + "="*60)
print("COHORT STRUCTURE")
print("="*60)

print("\ncohort_details columns:", cohort_details.columns.tolist())
print(cohort_details)

print("\npatient_cohort columns:", patient_cohort.columns.tolist())
print(patient_cohort.head(10))

# Count patients per cohort
cohort_counts = patient_cohort['cohort_name'].value_counts() \
    if 'cohort_name' in patient_cohort.columns \
    else patient_cohort.iloc[:, 1].value_counts()
print("\nPatients per cohort:")
print(cohort_counts)

# Map cohort membership to patient IDs
# TriNetX typically uses cohort_id or cohort_name to distinguish groups
# Adjust column names below if they differ after inspecting output above
if 'cohort_name' in patient_cohort.columns:
    cohort1_ids = set(patient_cohort[patient_cohort['cohort_name'].str.contains('Kidney|KIdney|pump|Pump', case=False, na=False)]['patient_id'])
    cohort2_ids = set(patient_cohort[patient_cohort['cohort_name'].str.contains('Panc|panc|SPK', case=False, na=False)]['patient_id'])
else:
    # Fall back: use cohort number ordering
    cohort_col = patient_cohort.columns[1]
    unique_cohorts = patient_cohort[cohort_col].unique()
    print(f"\nUnique cohort identifiers: {unique_cohorts}")
    cohort1_ids = set(patient_cohort[patient_cohort[cohort_col] == unique_cohorts[0]]['patient_id'])
    cohort2_ids = set(patient_cohort[patient_cohort[cohort_col] == unique_cohorts[1]]['patient_id'])

print(f"\nCohort 1 (Kidney+Pump): {len(cohort1_ids):,} patients")
print(f"Cohort 2 (SPK):         {len(cohort2_ids):,} patients")
print(f"Overlap (should be 0):  {len(cohort1_ids & cohort2_ids):,} patients")

# Add cohort label to patient table
patient['cohort'] = None
patient.loc[patient['patient_id'].isin(cohort1_ids), 'cohort'] = 'KTP'
patient.loc[patient['patient_id'].isin(cohort2_ids), 'cohort'] = 'SPK'
print(f"\nPatients with cohort assigned: {patient['cohort'].notna().sum():,}")
print(f"Patients without cohort:       {patient['cohort'].isna().sum():,}")

# =============================================================================
# 3. PATIENT DEMOGRAPHICS
# =============================================================================
print("\n" + "="*60)
print("PATIENT DEMOGRAPHICS")
print("="*60)

print("\nPatient table columns:", patient.columns.tolist())
print(patient.head(3))

# Standard TriNetX patient fields: patient_id, sex, race, ethnicity,
# year_of_birth, date_of_death (if available)

# Age calculation (TriNetX provides year_of_birth)
if 'year_of_birth' in patient.columns:
    patient['age'] = 2026 - patient['year_of_birth'].astype(float)
    print("\nAge distribution:")
    print(patient.groupby('cohort')['age'].describe().round(1))

# Sex distribution
if 'sex' in patient.columns:
    sex_dist = patient.groupby(['cohort', 'sex']).size().unstack(fill_value=0)
    print("\nSex distribution:")
    print(sex_dist)

# Race distribution
if 'race' in patient.columns:
    race_dist = patient.groupby(['cohort', 'race']).size().unstack(fill_value=0)
    print("\nRace distribution:")
    print(race_dist)

# Ethnicity
if 'ethnicity' in patient.columns:
    eth_dist = patient.groupby(['cohort', 'ethnicity']).size().unstack(fill_value=0)
    print("\nEthnicity distribution:")
    print(eth_dist)

# =============================================================================
# 4. DIAGNOSIS — KEY ICD-10 CODES
# =============================================================================
print("\n" + "="*60)
print("DIAGNOSIS TABLE — KEY CODES")
print("="*60)

print("\nDiagnosis columns:", diagnosis.columns.tolist())

# Standardize code column name
code_col = 'code' if 'code' in diagnosis.columns else diagnosis.columns[2]
date_col = 'date' if 'date' in diagnosis.columns else diagnosis.columns[4]

# Key codes for this study
key_codes = {
    'Z94.0':  'Kidney transplant status',
    'Z94.83': 'Pancreas transplant status',
    'Z96.41': 'Insulin pump',
    'E10':    'Type 1 DM (any)',
    'E11':    'Type 2 DM (any)',
    'E16.0':  'Hypoglycemia - drug-induced',
    'E16.2':  'Hypoglycemia NOS',
    'T86.11': 'Kidney graft rejection',
    'T86.12': 'Kidney graft failure',
    'T86.891':'Pancreas graft rejection',
    'I21':    'Acute MI (any)',
    'I50':    'Heart failure (any)',
    'I63':    'Stroke (any)',
    'N39.0':  'UTI',
    'A41':    'Sepsis (any)',
    'B25':    'CMV (any)',
}

diag_summary = {}
for code, label in key_codes.items():
    mask = diagnosis[code_col].str.startswith(code, na=False)
    pts_with_code = diagnosis[mask]['patient_id'].nunique()
    # Break down by cohort
    c1 = diagnosis[mask & diagnosis['patient_id'].isin(cohort1_ids)]['patient_id'].nunique()
    c2 = diagnosis[mask & diagnosis['patient_id'].isin(cohort2_ids)]['patient_id'].nunique()
    diag_summary[code] = {'label': label, 'total_patients': pts_with_code,
                          'KTP_n': c1, 'SPK_n': c2}

diag_df = pd.DataFrame(diag_summary).T.reset_index()
diag_df.columns = ['ICD10', 'Label', 'Total_Patients', 'KTP_n', 'SPK_n']
print("\nKey diagnosis code counts by cohort:")
print(diag_df.to_string(index=False))
diag_df.to_csv(os.path.join(OUTPUT_DIR, "key_diagnosis_counts.csv"), index=False)

# =============================================================================
# 5. LAB RESULTS — HbA1c AND CREATININE
# =============================================================================
print("\n" + "="*60)
print("LAB RESULTS — HbA1c AND CREATININE")
print("="*60)

print("\nLab result columns:", lab_result.columns.tolist())

# Key LOINC codes
LOINC_CODES = {
    '4548-4':  'HbA1c (Cohort 1)',
    '17856-6': 'HbA1c alternative',
    '59261-8': 'HbA1c alternative 2',
    'TNX:LAB:9037': 'HbA1c (Cohort 2 query code)',
    '2160-0':  'Creatinine serum/plasma',
    '38483-4': 'Creatinine blood',
    'TNX:LAB:9024': 'Creatinine (query code)',
    '62238-1': 'CKD-EPI eGFR (derived)',
    '33914-3': 'MDRD eGFR (derived)',
    '1558-6':  'Fasting glucose',
    '2345-7':  'Glucose serum',
    '39156-5': 'BMI',
    '29463-7': 'Body weight',
}

lab_code_col = 'code' if 'code' in lab_result.columns else lab_result.columns[2]
lab_val_col  = 'lab_result_num_val' if 'lab_result_num_val' in lab_result.columns \
               else [c for c in lab_result.columns if 'num' in c.lower()][0]
lab_date_col = 'date' if 'date' in lab_result.columns else lab_result.columns[4]

lab_summary = {}
for loinc, label in LOINC_CODES.items():
    mask = lab_result[lab_code_col] == loinc
    total = lab_result[mask][lab_val_col].count()
    pts   = lab_result[mask]['patient_id'].nunique()
    c1_pts = lab_result[mask & lab_result['patient_id'].isin(cohort1_ids)]['patient_id'].nunique()
    c2_pts = lab_result[mask & lab_result['patient_id'].isin(cohort2_ids)]['patient_id'].nunique()
    if total > 0:
        vals = lab_result[mask][lab_val_col].dropna()
        lab_summary[loinc] = {
            'Label': label,
            'Total_rows': total,
            'Unique_patients': pts,
            'KTP_patients': c1_pts,
            'SPK_patients': c2_pts,
            'Median': round(vals.median(), 2),
            'Min': round(vals.min(), 2),
            'Max': round(vals.max(), 2),
        }

lab_df = pd.DataFrame(lab_summary).T.reset_index()
lab_df.columns = ['LOINC'] + list(lab_df.columns[1:])
print("\nLab result availability by LOINC code:")
print(lab_df.to_string(index=False))
lab_df.to_csv(os.path.join(OUTPUT_DIR, "lab_code_summary.csv"), index=False)

# =============================================================================
# 6. HbA1c DISTRIBUTION PLOT
# =============================================================================
print("\n" + "="*60)
print("HbA1c DISTRIBUTION")
print("="*60)

# Try all known HbA1c LOINC codes
hba1c_codes = ['4548-4', '17856-6', '59261-8', 'TNX:LAB:9037']
hba1c_mask  = lab_result[lab_code_col].isin(hba1c_codes)
hba1c_data  = lab_result[hba1c_mask].copy()

# Filter to plausible HbA1c range (3–20%)
hba1c_data  = hba1c_data[(hba1c_data[lab_val_col] >= 3) &
                          (hba1c_data[lab_val_col] <= 20)]

# Label cohort
hba1c_data['cohort'] = None
hba1c_data.loc[hba1c_data['patient_id'].isin(cohort1_ids), 'cohort'] = 'KTP'
hba1c_data.loc[hba1c_data['patient_id'].isin(cohort2_ids), 'cohort'] = 'SPK'
hba1c_data = hba1c_data[hba1c_data['cohort'].notna()]

print(f"\nTotal HbA1c measurements: {len(hba1c_data):,}")
print(f"KTP: {len(hba1c_data[hba1c_data['cohort']=='KTP']):,} measurements")
print(f"SPK: {len(hba1c_data[hba1c_data['cohort']=='SPK']):,} measurements")
print("\nHbA1c summary by cohort:")
print(hba1c_data.groupby('cohort')[lab_val_col].describe().round(2))

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('HbA1c Distribution: KTP vs SPK', fontsize=14, fontweight='bold')

# Histogram
for cohort, color in [('KTP', '#2196F3'), ('SPK', '#FF5722')]:
    subset = hba1c_data[hba1c_data['cohort'] == cohort][lab_val_col]
    axes[0].hist(subset, bins=40, alpha=0.6, color=color, label=cohort, density=True)
axes[0].set_xlabel('HbA1c (%)')
axes[0].set_ylabel('Density')
axes[0].set_title('HbA1c Distribution (all time points)')
axes[0].legend()
axes[0].axvline(7.0, color='red', linestyle='--', alpha=0.7, label='Target <7%')

# Boxplot
hba1c_data.boxplot(column=lab_val_col, by='cohort', ax=axes[1],
                   boxprops=dict(color='navy'),
                   medianprops=dict(color='red', linewidth=2))
axes[1].set_xlabel('Cohort')
axes[1].set_ylabel('HbA1c (%)')
axes[1].set_title('HbA1c Boxplot by Cohort')
plt.suptitle('')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hba1c_distribution.png"), dpi=150)
plt.close()
print("  -> Saved: hba1c_distribution.png")

# =============================================================================
# 7. CREATININE / eGFR DISTRIBUTION
# =============================================================================
print("\n" + "="*60)
print("CREATININE / eGFR DISTRIBUTION")
print("="*60)

egfr_codes = ['62238-1', '33914-3']
cr_codes   = ['2160-0', '38483-4']

for label, codes, vmin, vmax in [
    ('eGFR', egfr_codes, 0, 150),
    ('Creatinine', cr_codes, 0, 15)
]:
    mask = lab_result[lab_code_col].isin(codes)
    data = lab_result[mask].copy()
    data = data[(data[lab_val_col] >= vmin) & (data[lab_val_col] <= vmax)]
    data['cohort'] = None
    data.loc[data['patient_id'].isin(cohort1_ids), 'cohort'] = 'KTP'
    data.loc[data['patient_id'].isin(cohort2_ids), 'cohort'] = 'SPK'
    data = data[data['cohort'].notna()]

    print(f"\n{label} summary by cohort:")
    print(data.groupby('cohort')[lab_val_col].describe().round(2))

    fig, ax = plt.subplots(figsize=(8, 5))
    for cohort, color in [('KTP', '#2196F3'), ('SPK', '#FF5722')]:
        subset = data[data['cohort'] == cohort][lab_val_col]
        ax.hist(subset, bins=40, alpha=0.6, color=color, label=cohort, density=True)
    ax.set_xlabel(label)
    ax.set_ylabel('Density')
    ax.set_title(f'{label} Distribution: KTP vs SPK')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{label.lower()}_distribution.png"), dpi=150)
    plt.close()
    print(f"  -> Saved: {label.lower()}_distribution.png")

# =============================================================================
# 8. VITAL SIGNS — BMI
# =============================================================================
print("\n" + "="*60)
print("VITAL SIGNS — BMI & WEIGHT")
print("="*60)

print("\nVital signs columns:", vital_signs.columns.tolist())
vs_code_col = 'code' if 'code' in vital_signs.columns else vital_signs.columns[2]
vs_val_col  = 'observation_value_numeric' if 'observation_value_numeric' in vital_signs.columns \
              else [c for c in vital_signs.columns if 'num' in c.lower() or 'val' in c.lower()][0]

for label, loinc, vmin, vmax in [
    ('BMI', '39156-5', 10, 80),
    ('Weight_kg', '29463-7', 20, 300),
]:
    mask = vital_signs[vs_code_col] == loinc
    data = vital_signs[mask].copy()
    data = data[(data[vs_val_col].astype(float, errors='ignore') >= vmin) &
                (data[vs_val_col].astype(float, errors='ignore') <= vmax)]
    data['cohort'] = None
    data.loc[data['patient_id'].isin(cohort1_ids), 'cohort'] = 'KTP'
    data.loc[data['patient_id'].isin(cohort2_ids), 'cohort'] = 'SPK'
    data = data[data['cohort'].notna()]
    if len(data) > 0:
        print(f"\n{label} summary by cohort:")
        print(data.groupby('cohort')[vs_val_col].describe().round(1))

# =============================================================================
# 9. MISSING DATA SUMMARY
# =============================================================================
print("\n" + "="*60)
print("MISSING DATA SUMMARY — PATIENT TABLE")
print("="*60)

missing = patient.isnull().sum()
missing_pct = (missing / len(patient) * 100).round(1)
missing_df = pd.DataFrame({'missing_n': missing, 'missing_pct': missing_pct})
missing_df = missing_df[missing_df['missing_n'] > 0].sort_values('missing_pct', ascending=False)
print(missing_df)
missing_df.to_csv(os.path.join(OUTPUT_DIR, "missing_data_patient.csv"))

# Also check how many patients are missing key labs
print("\nPatients with at least one HbA1c value:")
pts_hba1c = hba1c_data['patient_id'].nunique()
print(f"  Total: {pts_hba1c:,} / {len(patient_cohort['patient_id'].unique()):,}")
print(f"  KTP:   {hba1c_data[hba1c_data['cohort']=='KTP']['patient_id'].nunique():,}")
print(f"  SPK:   {hba1c_data[hba1c_data['cohort']=='SPK']['patient_id'].nunique():,}")

# =============================================================================
# 10. DATE RANGE OF DATA
# =============================================================================
print("\n" + "="*60)
print("DATE RANGE OF DATA")
print("="*60)

# Parse dates in lab_result
lab_result[lab_date_col] = pd.to_datetime(lab_result[lab_date_col].astype(str),
                                           format='%Y%m%d', errors='coerce')
print(f"\nLab result date range: {lab_result[lab_date_col].min()} to {lab_result[lab_date_col].max()}")

# Parse dates in diagnosis
diagnosis[date_col] = pd.to_datetime(diagnosis[date_col].astype(str),
                                      format='%Y%m%d', errors='coerce')
print(f"Diagnosis date range:  {diagnosis[date_col].min()} to {diagnosis[date_col].max()}")

# =============================================================================
# 11. PROCEDURE — TRANSPLANT VERIFICATION
# =============================================================================
print("\n" + "="*60)
print("PROCEDURE — TRANSPLANT CODES")
print("="*60)

print("\nProcedure columns:", procedure.columns.tolist())
proc_code_col = 'code' if 'code' in procedure.columns else procedure.columns[2]

# Kidney and pancreas transplant CPT/ICD-10-PCS codes
transplant_codes = {
    '50360': 'Kidney transplant (CPT)',
    '50365': 'Kidney transplant with nephrectomy (CPT)',
    '48554': 'Pancreas transplant (CPT)',
    '48556': 'Pancreas transplant without nephrectomy (CPT)',
    '0TY00Z0': 'Kidney transplant (ICD-10-PCS)',
    '0FYG0Z0': 'Pancreas transplant (ICD-10-PCS)',
}

for code, label in transplant_codes.items():
    mask = procedure[proc_code_col].str.startswith(code, na=False)
    n = procedure[mask]['patient_id'].nunique()
    if n > 0:
        c1 = procedure[mask & procedure['patient_id'].isin(cohort1_ids)]['patient_id'].nunique()
        c2 = procedure[mask & procedure['patient_id'].isin(cohort2_ids)]['patient_id'].nunique()
        print(f"  {code} ({label}): total={n}, KTP={c1}, SPK={c2}")

# =============================================================================
# 12. SAVE SUMMARY REPORT
# =============================================================================
print("\n" + "="*60)
print("SAVING SUMMARY REPORT")
print("="*60)

summary_lines = [
    "TriNetX PancPump — Data Exploration Summary",
    f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
    "",
    f"Total patients in dataset: {len(patient):,}",
    f"Cohort 1 (KTP):            {len(cohort1_ids):,}",
    f"Cohort 2 (SPK):            {len(cohort2_ids):,}",
    "",
    "Files saved to: " + OUTPUT_DIR,
    "  - key_diagnosis_counts.csv",
    "  - lab_code_summary.csv",
    "  - missing_data_patient.csv",
    "  - hba1c_distribution.png",
    "  - egfr_distribution.png",
    "  - creatinine_distribution.png",
    "",
    "NEXT STEPS:",
    "  Run 02_build_index_events.py to define index dates per patient",
    "  Run 03_psm_matching.py for propensity score matching",
    "  Run 04_primary_outcome_hba1c.py for HbA1c analysis",
]

with open(os.path.join(OUTPUT_DIR, "exploration_summary.txt"), 'w') as f:
    f.write('\n'.join(summary_lines))

print("\n".join(summary_lines))
print("\n✓ Exploration complete. Check folder:", OUTPUT_DIR)
