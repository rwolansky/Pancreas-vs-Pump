"""
TriNetX PancPump Study - Step 3: Propensity Score Matching (PSM)
1:1 nearest-neighbor matching with caliper = 0.2 SD of logit propensity score

Covariates matched:
  Demographics:  age_at_index, sex, race_white, race_black, race_asian, ethnicity_hispanic
  DM type:       dm_type1, dm_type2
  Comorbidities: hypertension, cad, heart_failure, pvd, dyslipidemia, obesity
  Baseline labs: baseline_hba1c, baseline_creatinine, baseline_egfr
  Medications:   tacrolimus, mycophenolate, prednisone, statin, acei_arb

Outputs:
  - psm_matched_cohort.csv        (matched patient pairs)
  - psm_covariate_balance.csv     (SMD before/after matching)
  - psm_balance_plot.png          (love plot)
  - psm_propensity_score_plot.png (PS distribution before/after)

PI: Paul Kuo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR   = r"Z:\homes\Rachel Wolansky\Panc Pump\Data In"
OUTPUT_DIR = r"Z:\homes\Rachel Wolansky\Panc Pump\Results"
CALIPER    = 0.2   # caliper in SD units of logit PS
RANDOM_SEED = 42

def load(filename):
    import os
    path = os.path.join(DATA_DIR if not filename.startswith(OUTPUT_DIR)
                        else "", filename)
    if not filename.startswith('C:'):
        path = DATA_DIR + "\\" + filename if '\\' not in filename \
               else OUTPUT_DIR + "\\" + filename
    print(f"  Loading {filename}...")
    df = pd.read_csv(path, low_memory=False)
    print(f"    -> {len(df):,} rows")
    return df

# =============================================================================
# LOAD FILES
# =============================================================================
print("\n" + "="*60)
print("LOADING FILES")
print("="*60)

import os
master     = pd.read_csv(os.path.join(OUTPUT_DIR, "master_patient_table.csv"),
                          low_memory=False)
diagnosis  = pd.read_csv(os.path.join(DATA_DIR, "diagnosis.csv"),
                          low_memory=False)
medication = pd.read_csv(os.path.join(DATA_DIR, "medication_ingredient.csv"),
                          low_memory=False)
lab_clean  = pd.read_csv(os.path.join(OUTPUT_DIR, "lab_result_clean.csv"),
                          low_memory=False)

print(f"  Master patient table: {len(master):,} rows")
print(f"  Diagnosis:            {len(diagnosis):,} rows")
print(f"  Medication:           {len(medication):,} rows")
print(f"  Clean labs:           {len(lab_clean):,} rows")

# Parse dates
master['index_date'] = pd.to_datetime(master['index_date'], errors='coerce')
diagnosis['date']    = pd.to_datetime(diagnosis['date'].astype(str),
                                       format='%Y%m%d', errors='coerce')
lab_clean['date']    = pd.to_datetime(lab_clean['date'], errors='coerce')

if 'start_date' in medication.columns:
    medication['start_date'] = pd.to_datetime(
        medication['start_date'].astype(str), format='%Y%m%d', errors='coerce')

patient_ids = set(master['patient_id'])

# =============================================================================
# SECTION 1: BUILD COVARIATE MATRIX
# =============================================================================
print("\n" + "="*60)
print("SECTION 1: BUILDING COVARIATE MATRIX")
print("="*60)

covariates = master[['patient_id', 'cohort', 'index_date',
                      'age_at_index', 'sex', 'race', 'ethnicity']].copy()

# --- Binary demographics ---
covariates['sex_male']           = (covariates['sex'] == 'M').astype(int)
covariates['race_white']         = covariates['race'].str.contains('White', na=False).astype(int)
covariates['race_black']         = covariates['race'].str.contains('Black', na=False).astype(int)
covariates['race_asian']         = covariates['race'].str.contains('Asian', na=False).astype(int)
covariates['ethnicity_hispanic'] = covariates['ethnicity'].str.contains(
    'Hispanic', na=False).astype(int)

# =============================================================================
# HELPER: flag whether a patient has a diagnosis code BEFORE their index date
# =============================================================================
def flag_dx_before_index(code_prefix, col_name, df_dx, df_master):
    """Returns series: 1 if patient has code_prefix diagnosed before index_date."""
    idx_dates = df_master.set_index('patient_id')['index_date'].to_dict()
    hits = df_dx[df_dx['code'].str.startswith(code_prefix, na=False)].copy()
    hits = hits[hits['patient_id'].isin(patient_ids)]
    hits['index_date'] = hits['patient_id'].map(idx_dates)
    hits = hits[hits['date'] < hits['index_date']]
    flagged = set(hits['patient_id'].unique())
    return df_master['patient_id'].isin(flagged).astype(int).rename(col_name)

print("\nFlagging diagnosis-based covariates (pre-index)...")
dx_flags = {
    'dm_type1':     'E10',
    'dm_type2':     'E11',
    'hypertension': 'I10',
    'cad':          'I25',
    'heart_failure':'I50',
    'pvd':          'I73',
    'dyslipidemia': 'E78',
    'obesity':      'E66',
    'ckd':          'N18',
    'anemia':       'D64',
}
for col, prefix in dx_flags.items():
    covariates[col] = flag_dx_before_index(prefix, col, diagnosis, master).values
    n = covariates[col].sum()
    ktp_n = covariates[covariates["cohort"]=="KTP"][col].sum()
    spk_n = covariates[covariates["cohort"]=="SPK"][col].sum()
    print(f"  {col:20s} ({prefix}): {n:,} patients (KTP={ktp_n}, SPK={spk_n})")

# =============================================================================
# HELPER: flag medication before index date
# =============================================================================
def flag_med_before_index(rxnorm_prefixes, col_name, df_med, df_master):
    idx_dates = df_master.set_index('patient_id')['index_date'].to_dict()
    med_code_col = 'code' if 'code' in df_med.columns else df_med.columns[4]
    date_col     = 'start_date' if 'start_date' in df_med.columns else df_med.columns[5]
    if isinstance(rxnorm_prefixes, str):
        rxnorm_prefixes = [rxnorm_prefixes]
    mask = df_med[med_code_col].astype(str).str.startswith(
        tuple(rxnorm_prefixes), na=False)
    hits = df_med[mask & df_med['patient_id'].isin(patient_ids)].copy()
    hits['index_date'] = hits['patient_id'].map(idx_dates)
    hits['med_date']   = pd.to_datetime(hits[date_col].astype(str),
                                         format='%Y%m%d', errors='coerce')
    hits = hits[hits['med_date'] < hits['index_date']]
    flagged = set(hits['patient_id'].unique())
    return df_master['patient_id'].isin(flagged).astype(int).rename(col_name)

print("\nFlagging medication-based covariates (pre-index)...")
# RxNorm codes for immunosuppressants and key medications
med_flags = {
    'tacrolimus':    ['386975', '1001025'],   # tacrolimus
    'mycophenolate': ['41493', '1404154'],    # mycophenolate mofetil/sodium
    'prednisone':    ['8640'],                # prednisone
    'cyclosporine':  ['3008'],                # cyclosporine
    'statin':        ['41127', '42463', '301542', '83367', '301543'],
                                              # atorva, simva, rosuvastatin etc
    'acei_arb':      ['18867', '214354', '54552', '83515', '321064'],
                                              # lisinopril, losartan, ramipril etc
    'beta_blocker':  ['19484', '33815', '41493'],
                                              # metoprolol, carvedilol etc
    'insulin_any':   ['51428'],               # insulin
}
for col, codes in med_flags.items():
    covariates[col] = flag_med_before_index(codes, col, medication, master).values
    n = covariates[col].sum()
    ktp_n = covariates[covariates["cohort"]=="KTP"][col].sum()
    spk_n = covariates[covariates["cohort"]=="SPK"][col].sum()
    print(f"  {col:20s}: {n:,} patients (KTP={ktp_n}, SPK={spk_n})")

# =============================================================================
# BASELINE LAB VALUES (most recent value BEFORE index date, within 1 year)
# =============================================================================
print("\nExtracting baseline lab values (pre-index, within 1 year)...")

def get_baseline_lab(loinc_codes, col_name, df_lab, df_master,
                     window_days=365):
    idx_dates = df_master.set_index('patient_id')['index_date'].to_dict()
    if isinstance(loinc_codes, str):
        loinc_codes = [loinc_codes]
    hits = df_lab[df_lab['code'].isin(loinc_codes) &
                  df_lab['patient_id'].isin(patient_ids)].copy()
    hits['index_date'] = hits['patient_id'].map(idx_dates)
    hits = hits[
        (hits['date'] < hits['index_date']) &
        (hits['date'] >= hits['index_date'] - pd.Timedelta(days=window_days))
    ]
    hits = hits.dropna(subset=['lab_result_num_val'])
    # Most recent value per patient
    latest = (hits.sort_values('date')
                  .groupby('patient_id')['lab_result_num_val']
                  .last()
                  .reset_index()
                  .rename(columns={'lab_result_num_val': col_name}))
    result = df_master[['patient_id']].merge(latest, on='patient_id', how='left')
    return result[col_name]

covariates['baseline_hba1c']    = get_baseline_lab(
    ['4548-4', '17856-6'], 'baseline_hba1c', lab_clean, master).values
covariates['baseline_creatinine']= get_baseline_lab(
    ['2160-0', '38483-4'], 'baseline_creatinine', lab_clean, master).values
covariates['baseline_egfr']     = get_baseline_lab(
    ['62238-1'], 'baseline_egfr', lab_clean, master).values

for col in ['baseline_hba1c', 'baseline_creatinine', 'baseline_egfr']:
    n_avail = covariates[col].notna().sum()
    ktp_med = covariates[covariates['cohort']=='KTP'][col].median()
    spk_med = covariates[covariates['cohort']=='SPK'][col].median()
    print(f"  {col:25s}: {n_avail:,} available  "
          f"(KTP median={ktp_med:.2f}, SPK median={spk_med:.2f})")

# =============================================================================
# SECTION 2: IMPUTE MISSING COVARIATES
# =============================================================================
print("\n" + "="*60)
print("SECTION 2: HANDLING MISSING COVARIATES")
print("="*60)

# PSM covariate columns
psm_cols = [
    'age_at_index', 'sex_male', 'race_white', 'race_black', 'race_asian',
    'ethnicity_hispanic',
    'dm_type1', 'dm_type2',
    'hypertension', 'cad', 'heart_failure', 'pvd', 'dyslipidemia',
    'obesity', 'ckd', 'anemia',
    'tacrolimus', 'mycophenolate', 'prednisone', 'cyclosporine',
    'statin', 'acei_arb', 'beta_blocker', 'insulin_any',
    'baseline_hba1c', 'baseline_creatinine', 'baseline_egfr',
]

print("\nMissing values per covariate:")
for col in psm_cols:
    n_miss = covariates[col].isna().sum()
    pct    = n_miss / len(covariates) * 100
    if n_miss > 0:
        print(f"  {col:25s}: {n_miss:,} missing ({pct:.1f}%)")

# Impute continuous with cohort-stratified median
for col in ['baseline_hba1c', 'baseline_creatinine', 'baseline_egfr', 'age_at_index']:
    for cohort in ['KTP', 'SPK']:
        mask = (covariates['cohort'] == cohort) & covariates[col].isna()
        med  = covariates[covariates['cohort'] == cohort][col].median()
        covariates.loc[mask, col] = med
        if mask.sum() > 0:
            print(f"  Imputed {mask.sum():,} missing {col} in {cohort} with median={med:.2f}")

# Binary missing -> 0
for col in psm_cols:
    n_miss = covariates[col].isna().sum()
    if n_miss > 0:
        covariates[col] = covariates[col].fillna(0)
        print(f"  Imputed {n_miss:,} missing {col} -> 0")

# =============================================================================
# SECTION 3: LOGISTIC REGRESSION — PROPENSITY SCORE
# =============================================================================
print("\n" + "="*60)
print("SECTION 3: PROPENSITY SCORE ESTIMATION")
print("="*60)

X = covariates[psm_cols].values.astype(float)
y = (covariates['cohort'] == 'KTP').astype(int).values  # 1=KTP, 0=SPK

# Scale continuous covariates
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Logistic regression
lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, C=1.0)
lr.fit(X_scaled, y)
ps = lr.predict_proba(X_scaled)[:, 1]  # P(KTP)

covariates['propensity_score'] = ps
covariates['logit_ps']         = np.log(ps / (1 - ps + 1e-10))

print(f"\nLogistic regression converged: {lr.n_iter_[0]} iterations")
print(f"\nPropensity score summary:")
for cohort in ['KTP', 'SPK']:
    sub = covariates[covariates['cohort'] == cohort]['propensity_score']
    print(f"  {cohort}: mean={sub.mean():.4f}, median={sub.median():.4f}, "
          f"min={sub.min():.4f}, max={sub.max():.4f}")

# C-statistic (AUC)
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y, ps)
print(f"\nC-statistic (AUC) of PS model: {auc:.3f}")
print("  (0.5 = no discrimination, >0.7 = good covariate separation)")

# =============================================================================
# SECTION 4: 1:1 NEAREST-NEIGHBOR MATCHING WITH CALIPER
# =============================================================================
print("\n" + "="*60)
print("SECTION 4: 1:1 NEAREST-NEIGHBOR PSM WITH CALIPER")
print("="*60)

logit_sd = covariates['logit_ps'].std()
caliper_value = CALIPER * logit_sd
print(f"\nLogit PS SD: {logit_sd:.4f}")
print(f"Caliper (0.2 x SD): {caliper_value:.4f}")

ktp_df = covariates[covariates['cohort'] == 'KTP'].copy().reset_index(drop=True)
spk_df = covariates[covariates['cohort'] == 'SPK'].copy().reset_index(drop=True)

# Shuffle KTP to randomize matching order
ktp_df = ktp_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

matched_pairs  = []
used_spk_ids   = set()
unmatched_ktp  = []

for _, ktp_row in ktp_df.iterrows():
    ktp_logit = ktp_row['logit_ps']
    ktp_pid   = ktp_row['patient_id']

    # Candidate SPK patients within caliper, not already matched
    candidates = spk_df[~spk_df['patient_id'].isin(used_spk_ids)].copy()
    candidates['logit_diff'] = abs(candidates['logit_ps'] - ktp_logit)
    within_caliper = candidates[candidates['logit_diff'] <= caliper_value]

    if len(within_caliper) == 0:
        unmatched_ktp.append(ktp_pid)
        continue

    # Select closest match
    best_match = within_caliper.nsmallest(1, 'logit_diff').iloc[0]
    used_spk_ids.add(best_match['patient_id'])
    matched_pairs.append({
        'ktp_patient_id':    ktp_pid,
        'spk_patient_id':    best_match['patient_id'],
        'ktp_logit_ps':      ktp_logit,
        'spk_logit_ps':      best_match['logit_ps'],
        'logit_diff':        abs(ktp_logit - best_match['logit_ps']),
        'ktp_ps':            ktp_row['propensity_score'],
        'spk_ps':            best_match['propensity_score'],
    })

matched_df = pd.DataFrame(matched_pairs)
print(f"\nMatching results:")
print(f"  KTP patients:         {len(ktp_df):,}")
print(f"  KTP matched:          {len(matched_df):,}")
print(f"  KTP unmatched:        {len(unmatched_ktp):,}")
print(f"  SPK pool:             {len(spk_df):,}")
print(f"  SPK matched:          {len(matched_df):,}")
print(f"\nLogit PS difference in matched pairs:")
print(f"  Mean:   {matched_df['logit_diff'].mean():.4f}")
print(f"  Median: {matched_df['logit_diff'].median():.4f}")
print(f"  Max:    {matched_df['logit_diff'].max():.4f}")

# =============================================================================
# SECTION 5: COVARIATE BALANCE — SMD BEFORE AND AFTER
# =============================================================================
print("\n" + "="*60)
print("SECTION 5: COVARIATE BALANCE (SMD)")
print("="*60)

def smd(x1, x2):
    """Standardized mean difference."""
    x1, x2 = np.array(x1, dtype=float), np.array(x2, dtype=float)
    x1, x2 = x1[~np.isnan(x1)], x2[~np.isnan(x2)]
    if len(x1) == 0 or len(x2) == 0:
        return np.nan
    pooled_sd = np.sqrt((np.var(x1, ddof=1) + np.var(x2, ddof=1)) / 2)
    if pooled_sd == 0:
        return 0.0
    return (np.mean(x1) - np.mean(x2)) / pooled_sd

# Matched cohort subsets
matched_ktp_ids = set(matched_df['ktp_patient_id'])
matched_spk_ids = set(matched_df['spk_patient_id'])

ktp_before  = covariates[covariates['cohort'] == 'KTP']
spk_before  = covariates[covariates['cohort'] == 'SPK']
ktp_after   = covariates[covariates['patient_id'].isin(matched_ktp_ids)]
spk_after   = covariates[covariates['patient_id'].isin(matched_spk_ids)]

balance_rows = []
for col in psm_cols:
    smd_before = smd(ktp_before[col], spk_before[col])
    smd_after  = smd(ktp_after[col],  spk_after[col])
    ktp_mean   = ktp_after[col].mean()
    spk_mean   = spk_after[col].mean()
    balance_rows.append({
        'Covariate':   col,
        'KTP_mean_pre':  round(ktp_before[col].mean(), 3),
        'SPK_mean_pre':  round(spk_before[col].mean(), 3),
        'SMD_pre':       round(abs(smd_before), 3),
        'KTP_mean_post': round(ktp_mean, 3),
        'SPK_mean_post': round(spk_mean, 3),
        'SMD_post':      round(abs(smd_after), 3),
        'Balanced':      abs(smd_after) < 0.1,
    })

balance_df = pd.DataFrame(balance_rows).sort_values('SMD_post', ascending=False)
print("\nCovariate balance (SMD < 0.1 = well balanced):")
print(balance_df[['Covariate','KTP_mean_pre','SPK_mean_pre','SMD_pre',
                   'KTP_mean_post','SPK_mean_post','SMD_post','Balanced']]
      .to_string(index=False))

n_balanced = balance_df['Balanced'].sum()
print(f"\nBalanced covariates (SMD < 0.1): {n_balanced}/{len(balance_df)}")
print(f"Unbalanced covariates (SMD ≥ 0.1):")
unbal = balance_df[~balance_df['Balanced']]
if len(unbal) > 0:
    print(unbal[['Covariate','SMD_pre','SMD_post']].to_string(index=False))
else:
    print("  None — all covariates balanced!")

balance_df.to_csv(os.path.join(OUTPUT_DIR, "psm_covariate_balance.csv"), index=False)
print("\n  -> Saved: psm_covariate_balance.csv")

# =============================================================================
# SECTION 6: PLOTS — LOVE PLOT AND PS DISTRIBUTION
# =============================================================================
print("\n" + "="*60)
print("SECTION 6: GENERATING BALANCE PLOTS")
print("="*60)

# --- Love Plot ---
fig, ax = plt.subplots(figsize=(10, 12))
y_pos  = range(len(balance_df))
labels = balance_df['Covariate'].tolist()

ax.scatter(balance_df['SMD_pre'],  list(y_pos), color='#E74C3C',
           marker='o', s=80, label='Before matching', zorder=3)
ax.scatter(balance_df['SMD_post'], list(y_pos), color='#2196F3',
           marker='D', s=80, label='After matching', zorder=4)

for i, row in enumerate(balance_df.itertuples()):
    ax.plot([row.SMD_pre, row.SMD_post], [i, i],
            color='gray', linewidth=0.8, alpha=0.5, zorder=2)

ax.axvline(0.1, color='black', linestyle='--', linewidth=1.5,
           alpha=0.7, label='SMD = 0.1 threshold')
ax.axvline(0.0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
ax.set_yticks(list(y_pos))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('Absolute Standardized Mean Difference', fontsize=11)
ax.set_title('Covariate Balance: Before vs After PSM\n(KTP vs SPK)',
             fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(axis='x', alpha=0.3)
ax.set_xlim(-0.02, max(balance_df['SMD_pre'].max(), 0.5) + 0.05)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "psm_balance_plot.png"), dpi=150)
plt.close()
print("  -> Saved: psm_balance_plot.png")

# --- PS Distribution Plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Propensity Score Distribution: Before and After Matching',
             fontsize=13, fontweight='bold')

# Before matching
for cohort, color in [('KTP','#2196F3'), ('SPK','#FF5722')]:
    sub = covariates[covariates['cohort'] == cohort]['propensity_score']
    axes[0].hist(sub, bins=40, alpha=0.6, color=color, label=cohort, density=True)
axes[0].set_title('Before Matching')
axes[0].set_xlabel('Propensity Score P(KTP)')
axes[0].set_ylabel('Density')
axes[0].legend()

# After matching
ktp_ps_after = covariates[covariates['patient_id'].isin(matched_ktp_ids)]['propensity_score']
spk_ps_after = covariates[covariates['patient_id'].isin(matched_spk_ids)]['propensity_score']
axes[1].hist(ktp_ps_after, bins=40, alpha=0.6, color='#2196F3',
             label='KTP', density=True)
axes[1].hist(spk_ps_after, bins=40, alpha=0.6, color='#FF5722',
             label='SPK', density=True)
axes[1].set_title('After Matching')
axes[1].set_xlabel('Propensity Score P(KTP)')
axes[1].set_ylabel('Density')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "psm_propensity_score_plot.png"), dpi=150)
plt.close()
print("  -> Saved: psm_propensity_score_plot.png")

# =============================================================================
# SECTION 7: SAVE MATCHED COHORT FILES
# =============================================================================
print("\n" + "="*60)
print("SECTION 7: SAVING MATCHED COHORT FILES")
print("="*60)

# Save matched pairs table
matched_df.to_csv(os.path.join(OUTPUT_DIR, "psm_matched_pairs.csv"), index=False)
print(f"  -> Saved: psm_matched_pairs.csv ({len(matched_df):,} pairs)")

# Save matched patient master (long format — one row per patient)
all_matched_ids = matched_ktp_ids | matched_spk_ids
matched_master  = master[master['patient_id'].isin(all_matched_ids)].copy()
matched_master  = matched_master.merge(
    covariates[['patient_id'] + psm_cols + ['propensity_score', 'logit_ps']],
    on='patient_id', how='left'
)
matched_master.to_csv(os.path.join(OUTPUT_DIR, "psm_matched_master.csv"), index=False)
print(f"  -> Saved: psm_matched_master.csv ({len(matched_master):,} patients)")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*60)
print("PSM SUMMARY")
print("="*60)

print(f"""
Propensity Score Matching Results:
------------------------------------
PS model C-statistic (AUC):    {auc:.3f}
Caliper (0.2 x logit PS SD):   {caliper_value:.4f}

                    Before PSM     After PSM
KTP patients:          {len(ktp_df):>5,}          {len(matched_df):>5,}
SPK patients:          {len(spk_df):>5,}          {len(matched_df):>5,}
Total:                 {len(ktp_df)+len(spk_df):>5,}          {len(matched_df)*2:>5,}

KTP unmatched:         {len(unmatched_ktp):>5,}
Covariates balanced (SMD<0.1): {n_balanced}/{len(balance_df)}

Output files:
  psm_matched_pairs.csv
  psm_matched_master.csv
  psm_covariate_balance.csv
  psm_balance_plot.png
  psm_propensity_score_plot.png
""")

print("✓ Script 03 complete.")
print("  Next step: Run 04_primary_outcome_hba1c.py")
