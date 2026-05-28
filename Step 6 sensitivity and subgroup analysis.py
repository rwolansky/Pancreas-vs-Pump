"""
TriNetX PancPump Study - Step 6: Sensitivity & Subgroup Analyses

Sensitivity analyses:
  S1. Exclude outcomes occurring within 90 days of index (avoid peri-operative confounding)
  S2. Require >= 2 pump-related codes for KTP definition (stricter pump verification)
  S3. Restrict to patients with >= 1 post-index HbA1c (complete-case HbA1c analysis)
  S4. Alternative PSM caliper (0.1 SD instead of 0.2 SD)
  S5. Exclude 4 patients with death before index date

Subgroup analyses (primary HbA1c outcome + key secondary outcomes):
  G1. DM Type 1 vs Type 2
  G2. Age < 50 vs >= 50 at index
  G3. Sex (male vs female)
  G4. Race (White vs non-White)
  G5. Baseline eGFR < 30 vs >= 30 (severe vs moderate CKD at index)
  G6. Baseline HbA1c < 8.0 vs >= 8.0 (glycemic control at index)

For each sensitivity analysis: report matched pairs, primary HbA1c HR, key secondary HRs
For each subgroup: test interaction term (cohort × subgroup) in LME and Cox models

Outputs:
  sensitivity_results.csv
  subgroup_hba1c_results.csv
  subgroup_secondary_results.csv
  sensitivity_forest_plot.png
  subgroup_forest_plot.png

PI: Paul Kuo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import statsmodels.formula.api as smf
import os, warnings
warnings.filterwarnings('ignore')

DATA_DIR   = r"Z:\homes\Rachel Wolansky\Panc Pump\Data In"
OUTPUT_DIR = r"Z:\homes\Rachel Wolansky\Panc Pump\Results"
RANDOM_SEED = 42

# =============================================================================
# LOAD FILES
# =============================================================================
print("\n" + "="*60)
print("LOADING FILES")
print("="*60)

master_full    = pd.read_csv(os.path.join(OUTPUT_DIR, "master_patient_table.csv"),   low_memory=False)
matched_master = pd.read_csv(os.path.join(OUTPUT_DIR, "psm_matched_master.csv"),     low_memory=False)
matched_pairs  = pd.read_csv(os.path.join(OUTPUT_DIR, "psm_matched_pairs.csv"),      low_memory=False)
lab_clean      = pd.read_csv(os.path.join(OUTPUT_DIR, "lab_result_clean.csv"),       low_memory=False)
diagnosis      = pd.read_csv(os.path.join(DATA_DIR,   "diagnosis.csv"),              low_memory=False)

master_full['index_date']    = pd.to_datetime(master_full['index_date'],    errors='coerce')
matched_master['index_date'] = pd.to_datetime(matched_master['index_date'], errors='coerce')
lab_clean['date']            = pd.to_datetime(lab_clean['date'],            errors='coerce')
diagnosis['date']            = pd.to_datetime(diagnosis['date'].astype(str), format='%Y%m%d', errors='coerce')

def parse_death_date(val):
    if pd.isna(val): return pd.NaT
    try: val = int(float(val))
    except: pass
    s = str(val).strip()
    if len(s)==6 and s.isdigit():
        return pd.to_datetime(s, format='%Y%m', errors='coerce')
    return pd.to_datetime(s, errors='coerce')

master_full['death_date'] = master_full['month_year_death'].apply(parse_death_date)

# Pump-related ICD codes for S2
PUMP_CODES = ['Z96.41', 'Z96.64', '99.10', 'V45.85', 'V53.91',
              'Y84.1', 'K86.81', '97.21', 'Z45.89']

print(f"  Master full: {len(master_full):,}  Matched: {len(matched_master):,}")

# =============================================================================
# SHARED HELPERS
# =============================================================================
HBAC1C_LOINCS = ['4548-4','17856-6','59261-8']
EGFR_LOINCS   = ['62238-1']
PSM_DX_FLAGS  = {
    'dm_type1':'E10','dm_type2':'E11','hypertension':'I10','cad':'I25',
    'heart_failure':'I50','pvd':'I73','dyslipidemia':'E78',
    'obesity':'E66','ckd':'N18','anemia':'D64',
}
INTERVALS = [
    ('3 months',91,45),('6 months',182,60),('1 year',365,90),
    ('2 years',730,120),('3 years',1095,150),('5 years',1825,180),
]

def flag_dx_before(code_prefix, df_dx, df_pts):
    idx_map = df_pts.set_index('patient_id')['index_date'].to_dict()
    pid_set = set(df_pts['patient_id'])
    hits = df_dx[df_dx['code'].str.startswith(code_prefix,na=False) &
                 df_dx['patient_id'].isin(pid_set)].copy()
    hits['idx'] = hits['patient_id'].map(idx_map)
    hits = hits[hits['date'] < hits['idx']]
    return df_pts['patient_id'].isin(set(hits['patient_id'])).astype(int).values

def get_baseline_lab_val(loinc_list, df_lab, df_pts, window=365):
    idx_map = df_pts.set_index('patient_id')['index_date'].to_dict()
    pid_set = set(df_pts['patient_id'])
    hits = df_lab[df_lab['code'].isin(loinc_list) &
                  df_lab['patient_id'].isin(pid_set) &
                  df_lab['lab_result_num_val'].notna()].copy()
    hits['idx']  = hits['patient_id'].map(idx_map)
    hits['days'] = (hits['date'] - hits['idx']).dt.days
    hits = hits[(hits['days']>=-window) & (hits['days']<=0)]
    latest = hits.sort_values('days').groupby('patient_id')['lab_result_num_val'].last().reset_index()
    result = df_pts[['patient_id']].merge(latest, on='patient_id', how='left')
    return result['lab_result_num_val'].values

def build_covariates(df_pts):
    cov = df_pts[['patient_id','cohort','index_date','age_at_index','sex','race','ethnicity']].copy()
    cov['sex_male']           = (cov['sex']=='M').astype(int)
    cov['race_white']         = cov['race'].str.contains('White', na=False).astype(int)
    cov['race_black']         = cov['race'].str.contains('Black', na=False).astype(int)
    cov['race_asian']         = cov['race'].str.contains('Asian', na=False).astype(int)
    cov['ethnicity_hispanic'] = cov['ethnicity'].str.contains('Hispanic', na=False).astype(int)
    for col, prefix in PSM_DX_FLAGS.items():
        cov[col] = flag_dx_before(prefix, diagnosis, df_pts)
    cov['baseline_hba1c']      = get_baseline_lab_val(HBAC1C_LOINCS, lab_clean, df_pts)
    cov['baseline_creatinine'] = get_baseline_lab_val(['2160-0','38483-4'], lab_clean, df_pts)
    cov['baseline_egfr']       = get_baseline_lab_val(EGFR_LOINCS, lab_clean, df_pts)
    for col in ['baseline_hba1c','baseline_creatinine','baseline_egfr','age_at_index']:
        for cohort in ['KTP','SPK']:
            mask = (cov['cohort']==cohort) & cov[col].isna()
            cov.loc[mask, col] = cov[cov['cohort']==cohort][col].median()
    return cov.fillna(0)

def run_psm(cov_df, caliper_sd=0.2, seed=RANDOM_SEED):
    psm_cols = [
        'age_at_index','sex_male','race_white','race_black','race_asian',
        'ethnicity_hispanic','dm_type1','dm_type2','hypertension','cad',
        'heart_failure','pvd','dyslipidemia','obesity','ckd','anemia',
        'baseline_hba1c','baseline_creatinine','baseline_egfr',
    ]
    X  = cov_df[psm_cols].values.astype(float)
    y  = (cov_df['cohort']=='KTP').astype(int).values
    Xs = StandardScaler().fit_transform(X)
    lr = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
    lr.fit(Xs, y)
    ps = lr.predict_proba(Xs)[:,1]
    cov_df = cov_df.copy()
    cov_df['logit_ps'] = np.log(ps/(1-ps+1e-10))
    caliper = caliper_sd * cov_df['logit_ps'].std()
    ktp_df  = cov_df[cov_df['cohort']=='KTP'].sample(frac=1, random_state=seed).reset_index(drop=True)
    spk_df  = cov_df[cov_df['cohort']=='SPK'].reset_index(drop=True)
    matched, used = [], set()
    for _, row in ktp_df.iterrows():
        cands  = spk_df[~spk_df['patient_id'].isin(used)].copy()
        cands['diff'] = abs(cands['logit_ps']-row['logit_ps'])
        within = cands[cands['diff']<=caliper]
        if len(within)==0: continue
        best = within.nsmallest(1,'diff').iloc[0]
        used.add(best['patient_id'])
        matched.append({'ktp_patient_id':row['patient_id'],
                        'spk_patient_id':best['patient_id']})
    return pd.DataFrame(matched)

def run_hba1c_km_cox(pid_set, pairs_df, idx_map, c_map,
                      exclude_days=0, label=''):
    """KM+Cox for time-to-HbA1c <7.0%. exclude_days: censor events before this day."""
    hba1c = lab_clean[lab_clean['code'].isin(HBAC1C_LOINCS) &
                       lab_clean['patient_id'].isin(pid_set) &
                       lab_clean['lab_result_num_val'].notna()].copy()
    hba1c['idx']      = hba1c['patient_id'].map(idx_map)
    hba1c['days_post'] = (hba1c['date'] - hba1c['idx']).dt.days
    hba1c_post = hba1c[(hba1c['days_post']>exclude_days)].copy()

    tte_rows = []
    for pid, grp in hba1c_post.groupby('patient_id'):
        grp    = grp.sort_values('days_post')
        events = grp[grp['lab_result_num_val']<7.0]
        last_t = grp['days_post'].max()
        if len(events)>0:
            tte_rows.append({'patient_id':pid,'time':int(events['days_post'].iloc[0]),'event':1})
        else:
            tte_rows.append({'patient_id':pid,'time':max(int(last_t),1),'event':0})

    tte_df = pd.DataFrame(tte_rows)
    tte_df['cohort']     = tte_df['patient_id'].map(c_map)
    tte_df['cohort_bin'] = (tte_df['cohort']=='KTP').astype(int)
    tte_df = tte_df[tte_df['time']>0]

    ktp_t = tte_df[tte_df['cohort']=='KTP']
    spk_t = tte_df[tte_df['cohort']=='SPK']
    if len(ktp_t)<5 or len(spk_t)<5:
        return dict(hr=np.nan, ci_lo=np.nan, ci_hi=np.nan, cox_p=np.nan,
                    logrank_p=np.nan, n_ktp=len(ktp_t), n_spk=len(spk_t))

    lr = logrank_test(ktp_t['time'], spk_t['time'],
                       event_observed_A=ktp_t['event'], event_observed_B=spk_t['event'])
    hr=ci_lo=ci_hi=cox_p=np.nan
    try:
        cph = CoxPHFitter()
        cph.fit(tte_df[['time','event','cohort_bin']], duration_col='time', event_col='event')
        hr    = float(np.exp(cph.params_['cohort_bin']))
        ci_lo = float(np.exp(cph.confidence_intervals_['95% lower-bound']['cohort_bin']))
        ci_hi = float(np.exp(cph.confidence_intervals_['95% upper-bound']['cohort_bin']))
        cox_p = float(cph.summary['p']['cohort_bin'])
    except:
        cox_p = lr.p_value
    return dict(hr=round(hr,3) if not np.isnan(hr) else np.nan,
                ci_lo=round(ci_lo,3) if not np.isnan(ci_lo) else np.nan,
                ci_hi=round(ci_hi,3) if not np.isnan(ci_hi) else np.nan,
                cox_p=round(cox_p,4) if not np.isnan(cox_p) else np.nan,
                logrank_p=round(lr.p_value,4),
                n_ktp=len(ktp_t), n_spk=len(spk_t))

def run_secondary_km_cox(code_prefixes, pid_set, idx_map, c_map, exclude_days=0):
    if isinstance(code_prefixes, str): code_prefixes=[code_prefixes]
    mask = diagnosis['patient_id'].isin(pid_set)
    cm   = diagnosis['code'].str.startswith(code_prefixes[0], na=False)
    for p in code_prefixes[1:]: cm = cm | diagnosis['code'].str.startswith(p, na=False)
    hits = diagnosis[mask & cm].copy()
    hits['idx']       = hits['patient_id'].map(idx_map)
    hits['days_post'] = (hits['date'] - hits['idx']).dt.days
    hits = hits[hits['days_post']>exclude_days]
    event_dict = hits.groupby('patient_id')['days_post'].min().to_dict()

    dx_m = diagnosis[diagnosis['patient_id'].isin(pid_set)].copy()
    dx_m['idx'] = dx_m['patient_id'].map(idx_map)
    dx_m['dp']  = (dx_m['date'] - dx_m['idx']).dt.days
    last_obs = dx_m[dx_m['dp']>0].groupby('patient_id')['dp'].max().to_dict()

    rows = []
    for pid in pid_set:
        if pid in event_dict:
            rows.append({'patient_id':pid,'time':min(event_dict[pid],3650),'event':1})
        else:
            rows.append({'patient_id':pid,'time':max(int(last_obs.get(pid,365)),1),'event':0})
    tte = pd.DataFrame(rows)
    tte['cohort']     = tte['patient_id'].map(c_map)
    tte['cohort_bin'] = (tte['cohort']=='KTP').astype(int)
    tte = tte[tte['time']>0]

    ktp_t = tte[tte['cohort']=='KTP']
    spk_t = tte[tte['cohort']=='SPK']
    if len(ktp_t)<5 or len(spk_t)<5:
        return dict(rate_ktp=np.nan, rate_spk=np.nan, hr=np.nan,
                    ci_lo=np.nan, ci_hi=np.nan, cox_p=np.nan)
    lr = logrank_test(ktp_t['time'], spk_t['time'],
                      event_observed_A=ktp_t['event'], event_observed_B=spk_t['event'])
    hr=ci_lo=ci_hi=cox_p=np.nan
    try:
        cph = CoxPHFitter()
        cph.fit(tte[['time','event','cohort_bin']], duration_col='time', event_col='event')
        hr    = float(np.exp(cph.params_['cohort_bin']))
        ci_lo = float(np.exp(cph.confidence_intervals_['95% lower-bound']['cohort_bin']))
        ci_hi = float(np.exp(cph.confidence_intervals_['95% upper-bound']['cohort_bin']))
        cox_p = float(cph.summary['p']['cohort_bin'])
    except:
        cox_p = lr.p_value
    return dict(
        rate_ktp=round(ktp_t['event'].mean()*100,1),
        rate_spk=round(spk_t['event'].mean()*100,1),
        hr=round(hr,3) if not np.isnan(hr) else np.nan,
        ci_lo=round(ci_lo,3) if not np.isnan(ci_lo) else np.nan,
        ci_hi=round(ci_hi,3) if not np.isnan(ci_hi) else np.nan,
        cox_p=round(cox_p,4) if not np.isnan(cox_p) else np.nan,
    )

# Base matched cohort maps
base_ids   = set(matched_master['patient_id'])
base_idx   = matched_master.set_index('patient_id')['index_date'].to_dict()
base_cmap  = matched_master.set_index('patient_id')['cohort'].to_dict()
base_pairs = matched_pairs.copy()
n_base     = len(matched_pairs)

KEY_SECONDARY = {
    'Hypoglycemia':       ['E16.0','E16.2','T38.3'],
    'Heart Failure':      ['I50'],
    'MACE':               ['I21','I22','I63','I64','I50'],
    'Kidney Rejection':   ['T86.11'],
}

# =============================================================================
# SENSITIVITY ANALYSES
# =============================================================================
print("\n" + "="*60)
print("SENSITIVITY ANALYSES")
print("="*60)

sensitivity_results = []

def run_sensitivity(label, pid_set, pairs_df, idx_map, c_map,
                    exclude_days=0, note=''):
    print(f"\n  {label} (n={len(pairs_df)} pairs, exclude <{exclude_days}d)")
    hba1c_res = run_hba1c_km_cox(pid_set, pairs_df, idx_map, c_map,
                                   exclude_days=exclude_days)
    row = {'Analysis': label, 'Note': note,
           'N_pairs': len(pairs_df),
           'HbA1c_HR': hba1c_res.get('hr'),
           'HbA1c_CI_lo': hba1c_res.get('ci_lo'),
           'HbA1c_CI_hi': hba1c_res.get('ci_hi'),
           'HbA1c_p': hba1c_res.get('cox_p')}
    sig = '*' if (hba1c_res.get('cox_p') or 1) < 0.05 else ''
    print(f"    HbA1c: HR={hba1c_res.get('hr','?')} "
          f"[{hba1c_res.get('ci_lo','?')},{hba1c_res.get('ci_hi','?')}] "
          f"p={hba1c_res.get('cox_p','?')}{sig}")
    for sec_label, prefixes in KEY_SECONDARY.items():
        res = run_secondary_km_cox(prefixes, pid_set, idx_map, c_map,
                                    exclude_days=exclude_days)
        row[f'{sec_label}_HR']  = res.get('hr')
        row[f'{sec_label}_p']   = res.get('cox_p')
        sig2 = '*' if (res.get('cox_p') or 1) < 0.05 else ''
        print(f"    {sec_label:20s}: HR={res.get('hr','?')} p={res.get('cox_p','?')}{sig2}")
    sensitivity_results.append(row)

# S0: Primary analysis (reference)
run_sensitivity('S0: Primary (reference)',
                base_ids, base_pairs, base_idx, base_cmap,
                note='Script 04 primary analysis result')

# S1: Exclude outcomes < 90 days
run_sensitivity('S1: Exclude outcomes <90d',
                base_ids, base_pairs, base_idx, base_cmap,
                exclude_days=90,
                note='Avoids peri-operative confounding')

# S2: Stricter KTP definition — require >= 2 pump codes
print("\n  Building S2: strict pump definition (>=2 pump codes)...")
pump_mask = diagnosis['code'].isin(PUMP_CODES)
pump_counts = (diagnosis[pump_mask & diagnosis['patient_id'].isin(
    matched_master[matched_master['cohort']=='KTP']['patient_id'])]
    .groupby('patient_id')['code'].count())
ktp_strict = set(pump_counts[pump_counts >= 2].index)
ktp_drop   = set(matched_master[matched_master['cohort']=='KTP']['patient_id']) - ktp_strict
print(f"    KTP with >=2 pump codes: {len(ktp_strict)} "
      f"(dropped {len(ktp_drop)} with only 1 code)")

# Remove pairs where KTP patient is dropped
pairs_s2 = base_pairs[base_pairs['ktp_patient_id'].isin(ktp_strict)].copy()
ids_s2   = set(pairs_s2['ktp_patient_id']) | set(pairs_s2['spk_patient_id'])
run_sensitivity('S2: Strict pump (>=2 codes)',
                ids_s2, pairs_s2, base_idx, base_cmap,
                note='Requires >=2 insulin pump ICD codes for KTP')

# S3: Complete-case HbA1c (require >= 1 post-index HbA1c)
hba1c_post_pts = set(
    lab_clean[lab_clean['code'].isin(HBAC1C_LOINCS) &
              lab_clean['patient_id'].isin(base_ids)].assign(
        days_post=lambda d: (d['date'] - d['patient_id'].map(base_idx)).dt.days
    ).query('days_post > 0')['patient_id']
)
pairs_s3 = base_pairs[
    base_pairs['ktp_patient_id'].isin(hba1c_post_pts) &
    base_pairs['spk_patient_id'].isin(hba1c_post_pts)
].copy()
ids_s3 = set(pairs_s3['ktp_patient_id']) | set(pairs_s3['spk_patient_id'])
print(f"\n  S3: {len(pairs_s3)} pairs with >=1 post-index HbA1c in both")
run_sensitivity('S3: Complete-case HbA1c',
                ids_s3, pairs_s3, base_idx, base_cmap,
                note='Both matched patients have >=1 post-index HbA1c')

# S4: Tighter PSM caliper (0.1 SD)
print("\n  S4: Re-running PSM with caliper=0.1 SD...")
cov_s4   = build_covariates(master_full[master_full['patient_id'].isin(base_ids)])
pairs_s4 = run_psm(cov_s4, caliper_sd=0.1)
ids_s4   = set(pairs_s4['ktp_patient_id']) | set(pairs_s4['spk_patient_id'])
cmap_s4  = {p:'KTP' for p in pairs_s4['ktp_patient_id']}
cmap_s4.update({p:'SPK' for p in pairs_s4['spk_patient_id']})
idx_s4   = {p: base_idx[p] for p in ids_s4 if p in base_idx}
run_sensitivity('S4: Tight caliper (0.1 SD)',
                ids_s4, pairs_s4, idx_s4, cmap_s4,
                note='Caliper=0.1xSD logit PS (vs 0.2 primary)')

# S5: Exclude deaths before index
pts_death_before = set(
    master_full[master_full['patient_id'].isin(base_ids) &
                (master_full['death_date'] < master_full['index_date'])]['patient_id']
)
pairs_s5 = base_pairs[
    ~base_pairs['ktp_patient_id'].isin(pts_death_before) &
    ~base_pairs['spk_patient_id'].isin(pts_death_before)
].copy()
ids_s5 = set(pairs_s5['ktp_patient_id']) | set(pairs_s5['spk_patient_id'])
print(f"\n  S5: Excluding {len(pts_death_before)} patients with death before index")
run_sensitivity('S5: Exclude pre-index deaths',
                ids_s5, pairs_s5, base_idx, base_cmap,
                note=f'Excludes {len(pts_death_before)} patients with death before index date')

sens_df = pd.DataFrame(sensitivity_results)
sens_df.to_csv(os.path.join(OUTPUT_DIR, "sensitivity_results.csv"), index=False)
print(f"\n  -> Saved: sensitivity_results.csv")

# =============================================================================
# SUBGROUP ANALYSES
# =============================================================================
print("\n" + "="*60)
print("SUBGROUP ANALYSES")
print("="*60)

# Build subgroup flags on matched cohort
# Auto-detect column names (psm_matched_master may use 'age' not 'age_at_index')
_age_col = 'age_at_index' if 'age_at_index' in matched_master.columns else 'age'
_sex_col = 'sex' if 'sex' in matched_master.columns else 'gender'
sg_base = matched_master[['patient_id','cohort','index_date',_age_col,_sex_col,'race']].copy()
sg_base = sg_base.rename(columns={_age_col:'age_at_index', _sex_col:'sex'})
# Merge baseline HbA1c and eGFR from covariates
cov_full = build_covariates(master_full[master_full['patient_id'].isin(base_ids)])
sg_base  = sg_base.merge(
    cov_full[['patient_id','dm_type1','dm_type2','baseline_hba1c','baseline_egfr']],
    on='patient_id', how='left')

SUBGROUPS = {
    'DM Type 1':         sg_base['dm_type1']==1,
    'DM Type 2':         sg_base['dm_type2']==1,
    'Age <50':           sg_base['age_at_index']<50,
    'Age >=50':          sg_base['age_at_index']>=50,
    'Male':              sg_base['sex']=='M',
    'Female':            sg_base['sex']!='M',
    'White':             sg_base['race'].str.contains('White',na=False),
    'Non-White':         ~sg_base['race'].str.contains('White',na=False),
    'Baseline eGFR <30': sg_base['baseline_egfr']<30,
    'Baseline eGFR >=30':sg_base['baseline_egfr']>=30,
    'Baseline HbA1c <8': sg_base['baseline_hba1c']<8.0,
    'Baseline HbA1c >=8':sg_base['baseline_hba1c']>=8.0,
}

subgroup_hba1c_results   = []
subgroup_secondary_results = []

for sg_label, sg_mask in SUBGROUPS.items():
    sg_pts = set(sg_base[sg_mask]['patient_id'])

    # Restrict pairs to those where BOTH members are in subgroup
    sg_pairs = base_pairs[
        base_pairs['ktp_patient_id'].isin(sg_pts) &
        base_pairs['spk_patient_id'].isin(sg_pts)
    ].copy()
    sg_ids  = set(sg_pairs['ktp_patient_id']) | set(sg_pairs['spk_patient_id'])
    n_ktp   = len(sg_pairs)
    n_spk   = len(sg_pairs)

    if n_ktp < 10:
        print(f"\n  {sg_label}: too few pairs ({n_ktp}) — skipping")
        continue

    print(f"\n  Subgroup: {sg_label} (n={n_ktp} pairs)")

    # HbA1c
    hba1c_res = run_hba1c_km_cox(sg_ids, sg_pairs, base_idx, base_cmap)
    sig = '*' if (hba1c_res.get('cox_p') or 1) < 0.05 else ''
    print(f"    HbA1c: HR={hba1c_res.get('hr','?')} "
          f"[{hba1c_res.get('ci_lo','?')},{hba1c_res.get('ci_hi','?')}] "
          f"p={hba1c_res.get('cox_p','?')}{sig}")
    subgroup_hba1c_results.append({
        'Subgroup': sg_label, 'N_pairs': n_ktp,
        'HR': hba1c_res.get('hr'), 'CI_lo': hba1c_res.get('ci_lo'),
        'CI_hi': hba1c_res.get('ci_hi'), 'p': hba1c_res.get('cox_p'),
        'Sig': '*' if (hba1c_res.get('cox_p') or 1)<0.05 else '',
    })

    # Key secondary
    for sec_label, prefixes in KEY_SECONDARY.items():
        res = run_secondary_km_cox(prefixes, sg_ids, base_idx, base_cmap)
        sig2 = '*' if (res.get('cox_p') or 1) < 0.05 else ''
        print(f"    {sec_label:20s}: HR={res.get('hr','?')} p={res.get('cox_p','?')}{sig2}")
        subgroup_secondary_results.append({
            'Subgroup': sg_label, 'Outcome': sec_label, 'N_pairs': n_ktp,
            'Rate_KTP': res.get('rate_ktp'), 'Rate_SPK': res.get('rate_spk'),
            'HR': res.get('hr'), 'CI_lo': res.get('ci_lo'),
            'CI_hi': res.get('ci_hi'), 'p': res.get('cox_p'),
            'Sig': '*' if (res.get('cox_p') or 1)<0.05 else '',
        })

sg_hba1c_df = pd.DataFrame(subgroup_hba1c_results)
sg_sec_df   = pd.DataFrame(subgroup_secondary_results)
sg_hba1c_df.to_csv(os.path.join(OUTPUT_DIR, "subgroup_hba1c_results.csv"),      index=False)
sg_sec_df.to_csv(os.path.join(OUTPUT_DIR,   "subgroup_secondary_results.csv"),   index=False)
print(f"\n  -> Saved: subgroup_hba1c_results.csv")
print(f"  -> Saved: subgroup_secondary_results.csv")

# =============================================================================
# INTERACTION TESTS (cohort x subgroup in LME for HbA1c)
# =============================================================================
print("\n" + "="*60)
print("INTERACTION TESTS")
print("="*60)

# Binary subgroup variables for interaction testing
BINARY_SG = {
    'DM Type 1 vs 2':       ('dm_type1', 1),
    'Age <50 vs >=50':      ('age_lt50', None),
    'Male vs Female':       ('sex_male', None),
    'White vs Non-White':   ('race_white', None),
    'eGFR <30 vs >=30':     ('egfr_lt30', None),
    'HbA1c <8 vs >=8':      ('hba1c_lt8', None),
}

hba1c_all = lab_clean[lab_clean['code'].isin(HBAC1C_LOINCS) &
                       lab_clean['patient_id'].isin(base_ids) &
                       lab_clean['lab_result_num_val'].notna()].copy()
hba1c_all['idx']       = hba1c_all['patient_id'].map(base_idx)
hba1c_all['days_post'] = (hba1c_all['date'] - hba1c_all['idx']).dt.days
hba1c_all['years_post'] = hba1c_all['days_post']/365.25
hba1c_all = hba1c_all[hba1c_all['days_post']>0].copy()
hba1c_all = hba1c_all.merge(
    cov_full[['patient_id','dm_type1','dm_type2','baseline_hba1c','baseline_egfr']],
    on='patient_id', how='left')
age_col2 = 'age_at_index' if 'age_at_index' in matched_master.columns else 'age'
sex_col2 = 'sex' if 'sex' in matched_master.columns else 'gender'
mm_cols  = matched_master[['patient_id','cohort', age_col2, sex_col2,'race']].rename(
    columns={age_col2:'age_at_index', sex_col2:'sex'})
hba1c_all = hba1c_all.merge(mm_cols, on='patient_id', how='left')
hba1c_all['is_ktp']    = (hba1c_all['cohort']=='KTP').astype(int)
hba1c_all['age_lt50']  = (hba1c_all['age_at_index']<50).astype(int)
hba1c_all['sex_male']  = (hba1c_all['sex']=='M').astype(int)
hba1c_all['race_white'] = hba1c_all['race'].str.contains('White',na=False).astype(int)
hba1c_all['egfr_lt30'] = (hba1c_all['baseline_egfr']<30).astype(int)
hba1c_all['hba1c_lt8'] = (hba1c_all['baseline_hba1c']<8.0).astype(int)
hba1c_all = hba1c_all.rename(columns={'lab_result_num_val':'hba1c'})

print("\nLME interaction tests (cohort × subgroup on HbA1c trajectory):")
interaction_results = []
for sg_label, (sg_var, _) in BINARY_SG.items():
    if sg_var not in hba1c_all.columns:
        continue
    try:
        sub = hba1c_all[hba1c_all[sg_var].notna() & hba1c_all['hba1c'].notna()].copy()
        sub[sg_var] = sub[sg_var].astype(int)
        formula = f"hba1c ~ years_post * is_ktp * {sg_var}"
        lme = smf.mixedlm(formula, sub, groups=sub['patient_id']).fit(reml=True)
        int_term = f'years_post:is_ktp:{sg_var}'
        if int_term in lme.fe_params:
            coef = float(lme.fe_params[int_term])
            p    = float(lme.pvalues[int_term])
            sig  = '*' if p<0.05 else ''
            print(f"  {sg_label:25s}: interaction coef={coef:.4f}  p={p:.4f}{sig}")
            interaction_results.append({'Subgroup':sg_label,'Interaction_coef':round(coef,4),'p':round(p,4),'Sig':sig})
        else:
            print(f"  {sg_label:25s}: interaction term not estimable")
    except Exception as e:
        print(f"  {sg_label:25s}: model failed ({e.__class__.__name__})")

# =============================================================================
# PLOTS
# =============================================================================
print("\n" + "="*60)
print("GENERATING PLOTS")
print("="*60)

# Plot 1 — Sensitivity analysis forest plot (HbA1c HR)
fig, ax = plt.subplots(figsize=(10, 7))
colors  = ['#1565C0','#2E86AB','#A23B72','#F18F01','#C73E1D','#3B1F2B']
for i, row in sens_df.iterrows():
    hr    = row['HbA1c_HR']
    ci_lo = row['HbA1c_CI_lo']
    ci_hi = row['HbA1c_CI_hi']
    p     = row['HbA1c_p']
    if np.isnan(hr): continue
    color = colors[i % len(colors)]
    y     = len(sens_df) - i
    ax.plot(hr, y, 'o', color=color, markersize=9, zorder=3)
    ax.plot([ci_lo, ci_hi], [y, y], '-', color=color, linewidth=2, zorder=2)
    if not np.isnan(p) and p < 0.05:
        ax.plot(hr, y, 'o', color=color, markersize=14,
                markerfacecolor='none', markeredgewidth=2, zorder=4)
    ax.text(ci_hi+0.03, y,
            f"HR={hr:.2f} [{ci_lo:.2f},{ci_hi:.2f}] p={p:.3f}",
            va='center', fontsize=8.5, color=color)
ax.axvline(1.0, color='black', linewidth=1.2, linestyle='--', alpha=0.6)
ax.set_yticks(list(range(1, len(sens_df)+1)))
ax.set_yticklabels(list(reversed(sens_df['Analysis'].tolist())), fontsize=9)
ax.set_xlabel('Hazard Ratio — Time to HbA1c <7.0% (KTP vs SPK)', fontsize=10)
ax.set_title('Sensitivity Analyses: Primary HbA1c Outcome\n(HR >1 favors SPK)',
             fontsize=12, fontweight='bold')
ax.set_xlim(0.3, 2.0)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sensitivity_forest_plot.png"), dpi=150)
plt.close()
print("  -> Saved: sensitivity_forest_plot.png")

# Plot 2 — Subgroup forest plot (HbA1c HR)
if len(sg_hba1c_df) > 0:
    fig, ax = plt.subplots(figsize=(11, 9))
    sg_hba1c_df = sg_hba1c_df.reset_index(drop=True)
    n = len(sg_hba1c_df)
    for i, row in sg_hba1c_df.iterrows():
        hr    = row['HR']
        ci_lo = row['CI_lo']
        ci_hi = row['CI_hi']
        p     = row['p']
        if pd.isna(hr): continue
        y     = n - i
        color = '#1565C0' if not pd.isna(p) and p<0.05 else '#888888'
        ax.plot(hr, y, 'o', color=color, markersize=8, zorder=3)
        ax.plot([ci_lo,ci_hi],[y,y],'-',color=color,linewidth=2,zorder=2)
        ax.text(max(ci_hi,2.5)+0.05, y,
                f"n={int(row['N_pairs'])}  HR={hr:.2f} [{ci_lo:.2f},{ci_hi:.2f}]  p={p:.3f}",
                va='center', fontsize=8)
    ax.axvline(1.0, color='black', linewidth=1.2, linestyle='--', alpha=0.6)
    ax.set_yticks(list(range(1,n+1)))
    ax.set_yticklabels(list(reversed(sg_hba1c_df['Subgroup'].tolist())), fontsize=9)
    ax.set_xlabel('Hazard Ratio — Time to HbA1c <7.0% (KTP vs SPK)', fontsize=10)
    ax.set_title('Subgroup Analyses: Primary HbA1c Outcome\n(HR >1 = slower time to target in KTP)',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(0.2, 4.0)
    ax.grid(axis='x', alpha=0.3)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#1565C0', label='p<0.05'),
                       Patch(facecolor='#888888', label='p≥0.05')]
    ax.legend(handles=legend_elements, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "subgroup_forest_plot.png"), dpi=150)
    plt.close()
    print("  -> Saved: subgroup_forest_plot.png")

# =============================================================================
# PRINT FINAL SUMMARY
# =============================================================================
print("\n" + "="*60)
print("SENSITIVITY ANALYSIS SUMMARY")
print("="*60)
print(sens_df[['Analysis','N_pairs','HbA1c_HR','HbA1c_CI_lo',
               'HbA1c_CI_hi','HbA1c_p']].to_string(index=False))

print("\n" + "="*60)
print("SUBGROUP HbA1c SUMMARY")
print("="*60)
print(sg_hba1c_df[['Subgroup','N_pairs','HR','CI_lo','CI_hi','p','Sig']].to_string(index=False))

print("\n✓ Script 06 complete.")
print("  Outputs: sensitivity_results.csv")
print("           subgroup_hba1c_results.csv")
print("           subgroup_secondary_results.csv")
print("           sensitivity_forest_plot.png")
print("           subgroup_forest_plot.png")
print("  Next step: Run 07_manuscript_tables.py")
