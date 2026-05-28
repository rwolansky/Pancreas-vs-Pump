"""
TriNetX PancPump Study - Step 5c: Era Sensitivity Analysis — Secondary Outcomes
Repeats all secondary outcomes from Scripts 05 + 05b restricted to:
  - Past 10 years: index date >= 2016-01-01
  - Past 5 years:  index date >= 2021-01-01
  - Full cohort:   all dates (reference)

For each era: re-run PSM, then run all secondary outcomes:
  A. eGFR trajectory (serial means + LME)
  B. Graft survival (kidney rejection, failure, any graft event)
  C. All-cause mortality
  D. MACE (MI, stroke, heart failure, composite)
  E. Hypoglycemia
  F. Infections (UTI, sepsis, CMV, BK virus, composite)
  G. BMI trajectory
  H. Immunosuppression use (tacrolimus, mycophenolate, prednisone, etc.)

Outputs:
  era_secondary_all_results.csv   — full cross-era comparison table
  era_secondary_egfr_plot.png     — eGFR trajectory by era
  era_secondary_outcomes_plot.png — HR forest plot by era

PI: Paul Kuo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
# LOAD ALL SOURCE FILES ONCE
# =============================================================================
print("\n" + "="*60)
print("LOADING FILES")
print("="*60)

master_full = pd.read_csv(os.path.join(OUTPUT_DIR, "master_patient_table.csv"), low_memory=False)
lab_clean   = pd.read_csv(os.path.join(OUTPUT_DIR, "lab_result_clean.csv"),     low_memory=False)
diagnosis   = pd.read_csv(os.path.join(DATA_DIR,   "diagnosis.csv"),             low_memory=False)
med_drug    = pd.read_csv(os.path.join(DATA_DIR,   "medication_drug.csv"),       low_memory=False)

vs_path     = os.path.join(DATA_DIR, "vital_signs.csv")
if not os.path.exists(vs_path):
    vs_path = os.path.join(DATA_DIR, "vitals_signs.csv")
vital_signs = pd.read_csv(vs_path, low_memory=False)

master_full['index_date'] = pd.to_datetime(master_full['index_date'], errors='coerce')
lab_clean['date']         = pd.to_datetime(lab_clean['date'],         errors='coerce')
diagnosis['date']         = pd.to_datetime(diagnosis['date'].astype(str), format='%Y%m%d', errors='coerce')
med_drug['start_date']    = pd.to_datetime(med_drug['start_date'].astype(str), format='%Y%m%d', errors='coerce')
vital_signs['date']       = pd.to_datetime(vital_signs['date'].astype(str), format='%Y%m%d', errors='coerce')

# Parse death dates (stored as float YYYYMM)
def parse_death_date(val):
    if pd.isna(val): return pd.NaT
    try: val = int(float(val))
    except: pass
    s = str(val).strip()
    if len(s) == 6 and s.isdigit():
        return pd.to_datetime(s, format='%Y%m', errors='coerce')
    return pd.to_datetime(s, errors='coerce')

master_full['death_date'] = master_full['month_year_death'].apply(parse_death_date)

print(f"  Master: {len(master_full):,}  Labs: {len(lab_clean):,}  "
      f"Dx: {len(diagnosis):,}  Meds: {len(med_drug):,}  VS: {len(vital_signs):,}")

# =============================================================================
# CONSTANTS
# =============================================================================
ERAS = {
    'Full cohort':   pd.Timestamp('2000-01-01'),
    'Past 10 years': pd.Timestamp('2016-01-01'),
    'Past 5 years':  pd.Timestamp('2021-01-01'),
}

INTERVALS = [
    ('3 months',  91,  45),
    ('6 months', 182,  60),
    ('1 year',   365,  90),
    ('2 years',  730, 120),
    ('3 years', 1095, 150),
    ('5 years', 1825, 180),
]

PSM_DX_FLAGS = {
    'dm_type1':      'E10', 'dm_type2':      'E11',
    'hypertension':  'I10', 'cad':           'I25',
    'heart_failure': 'I50', 'pvd':           'I73',
    'dyslipidemia':  'E78', 'obesity':       'E66',
    'ckd':           'N18', 'anemia':        'D64',
}

HBAC1C_LOINCS   = ['4548-4', '17856-6', '59261-8']
EGFR_LOINCS     = ['62238-1']
CREAT_LOINCS    = ['2160-0', '38483-4']
BMI_LOINCS      = ['39156-5']

IMMUNO_CODES = {
    'Tacrolimus':    ['242120','205085','349093','1049502','1049518','1437700','1437702'],
    'Mycophenolate': ['313782','313783','582391','1249746','1545146','904589'],
    'Prednisone':    ['198377','312617','313002','197361','312615','312616','198334'],
    'Cyclosporine':  ['308416','311700','313190','205483','197463'],
    'Sirolimus':     ['847630','865098'],
    'Everolimus':    ['1049621','1049622','1049623','1049624'],
}

# =============================================================================
# PSM HELPERS
# =============================================================================
def flag_dx_before_index(code_prefix, df_dx, df_pts):
    idx_map = df_pts.set_index('patient_id')['index_date'].to_dict()
    pid_set = set(df_pts['patient_id'])
    hits    = df_dx[df_dx['code'].str.startswith(code_prefix, na=False) &
                    df_dx['patient_id'].isin(pid_set)].copy()
    hits['idx'] = hits['patient_id'].map(idx_map)
    hits = hits[hits['date'] < hits['idx']]
    flagged = set(hits['patient_id'])
    return df_pts['patient_id'].isin(flagged).astype(int).values

def get_baseline_lab(loinc_list, df_lab, df_pts, window=365):
    idx_map = df_pts.set_index('patient_id')['index_date'].to_dict()
    pid_set = set(df_pts['patient_id'])
    hits    = df_lab[df_lab['code'].isin(loinc_list) &
                     df_lab['patient_id'].isin(pid_set) &
                     df_lab['lab_result_num_val'].notna()].copy()
    hits['idx']  = hits['patient_id'].map(idx_map)
    hits['days'] = (hits['date'] - hits['idx']).dt.days
    hits = hits[(hits['days'] >= -window) & (hits['days'] <= 0)]
    latest = hits.sort_values('days').groupby('patient_id')['lab_result_num_val'].last().reset_index()
    result = df_pts[['patient_id']].merge(latest, on='patient_id', how='left')
    return result['lab_result_num_val'].values

def build_covariates(df_pts):
    cov = df_pts[['patient_id','cohort','index_date','age_at_index',
                  'sex','race','ethnicity']].copy()
    cov['sex_male']           = (cov['sex']=='M').astype(int)
    cov['race_white']         = cov['race'].str.contains('White',    na=False).astype(int)
    cov['race_black']         = cov['race'].str.contains('Black',    na=False).astype(int)
    cov['race_asian']         = cov['race'].str.contains('Asian',    na=False).astype(int)
    cov['ethnicity_hispanic'] = cov['ethnicity'].str.contains('Hispanic', na=False).astype(int)
    for col, prefix in PSM_DX_FLAGS.items():
        cov[col] = flag_dx_before_index(prefix, diagnosis, df_pts)
    cov['baseline_hba1c']      = get_baseline_lab(HBAC1C_LOINCS, lab_clean, df_pts)
    cov['baseline_creatinine'] = get_baseline_lab(CREAT_LOINCS,  lab_clean, df_pts)
    cov['baseline_egfr']       = get_baseline_lab(EGFR_LOINCS,   lab_clean, df_pts)
    for col in ['baseline_hba1c','baseline_creatinine','baseline_egfr','age_at_index']:
        for cohort in ['KTP','SPK']:
            mask = (cov['cohort']==cohort) & cov[col].isna()
            med  = cov[cov['cohort']==cohort][col].median()
            cov.loc[mask, col] = med
    return cov.fillna(0)

def run_psm(cov_df, seed=RANDOM_SEED):
    psm_cols = [
        'age_at_index','sex_male','race_white','race_black','race_asian',
        'ethnicity_hispanic','dm_type1','dm_type2','hypertension','cad',
        'heart_failure','pvd','dyslipidemia','obesity','ckd','anemia',
        'baseline_hba1c','baseline_creatinine','baseline_egfr',
    ]
    X = cov_df[psm_cols].values.astype(float)
    y = (cov_df['cohort']=='KTP').astype(int).values
    X_sc = StandardScaler().fit_transform(X)
    lr   = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
    lr.fit(X_sc, y)
    ps   = lr.predict_proba(X_sc)[:,1]
    cov_df = cov_df.copy()
    cov_df['ps']       = ps
    cov_df['logit_ps'] = np.log(ps / (1 - ps + 1e-10))
    caliper = 0.2 * cov_df['logit_ps'].std()
    ktp_df  = cov_df[cov_df['cohort']=='KTP'].sample(frac=1, random_state=seed).reset_index(drop=True)
    spk_df  = cov_df[cov_df['cohort']=='SPK'].reset_index(drop=True)
    matched, used_spk = [], set()
    for _, row in ktp_df.iterrows():
        cands  = spk_df[~spk_df['patient_id'].isin(used_spk)].copy()
        cands['diff'] = abs(cands['logit_ps'] - row['logit_ps'])
        within = cands[cands['diff'] <= caliper]
        if len(within) == 0: continue
        best = within.nsmallest(1,'diff').iloc[0]
        used_spk.add(best['patient_id'])
        matched.append({'ktp_patient_id': row['patient_id'],
                        'spk_patient_id': best['patient_id']})
    return pd.DataFrame(matched), cov_df

# =============================================================================
# OUTCOME HELPERS
# =============================================================================
def flag_events_post_index(code_prefixes, pid_set, idx_map):
    """Return {patient_id: days_to_first_event} for any of code_prefixes after index."""
    if isinstance(code_prefixes, str):
        code_prefixes = [code_prefixes]
    mask = diagnosis['patient_id'].isin(pid_set)
    code_mask = diagnosis['code'].str.startswith(code_prefixes[0], na=False)
    for pfx in code_prefixes[1:]:
        code_mask = code_mask | diagnosis['code'].str.startswith(pfx, na=False)
    hits = diagnosis[mask & code_mask].copy()
    hits['idx']      = hits['patient_id'].map(idx_map)
    hits['days_post'] = (hits['date'] - hits['idx']).dt.days
    hits = hits[hits['days_post'] > 0]
    return hits.groupby('patient_id')['days_post'].min().to_dict()

def build_tte(event_dict, pid_set, idx_map, max_days=3650):
    dx_m = diagnosis[diagnosis['patient_id'].isin(pid_set)].copy()
    dx_m['idx'] = dx_m['patient_id'].map(idx_map)
    dx_m['dp']  = (dx_m['date'] - dx_m['idx']).dt.days
    last_obs = dx_m[dx_m['dp']>0].groupby('patient_id')['dp'].max().to_dict()
    rows = []
    for pid in pid_set:
        if pid in event_dict:
            rows.append({'patient_id':pid, 'time': min(event_dict[pid], max_days), 'event':1})
        else:
            rows.append({'patient_id':pid, 'time': max(int(last_obs.get(pid,365)),1), 'event':0})
    return pd.DataFrame(rows)

def run_km_cox(tte_df, cohort_map):
    tte_df = tte_df[tte_df['time']>0].copy()
    tte_df['cohort']     = tte_df['patient_id'].map(cohort_map)
    tte_df['cohort_bin'] = (tte_df['cohort']=='KTP').astype(int)
    ktp_t = tte_df[tte_df['cohort']=='KTP']
    spk_t = tte_df[tte_df['cohort']=='SPK']
    if len(ktp_t)==0 or len(spk_t)==0:
        return dict(events_ktp=0, events_spk=0,
                    rate_ktp=np.nan, rate_spk=np.nan,
                    logrank_p=np.nan, hr=np.nan, ci_lo=np.nan, ci_hi=np.nan, cox_p=np.nan)
    lr = logrank_test(ktp_t['time'], spk_t['time'],
                      event_observed_A=ktp_t['event'], event_observed_B=spk_t['event'])
    hr = ci_lo = ci_hi = cox_p = np.nan
    try:
        cph = CoxPHFitter()
        cph.fit(tte_df[['time','event','cohort_bin']], duration_col='time', event_col='event')
        hr    = float(np.exp(cph.params_['cohort_bin']))
        ci_lo = float(np.exp(cph.confidence_intervals_['95% lower-bound']['cohort_bin']))
        ci_hi = float(np.exp(cph.confidence_intervals_['95% upper-bound']['cohort_bin']))
        cox_p = float(cph.summary['p']['cohort_bin'])
    except:
        cox_p = lr.p_value
    return dict(
        n_ktp=len(ktp_t), n_spk=len(spk_t),
        events_ktp=int(ktp_t['event'].sum()), events_spk=int(spk_t['event'].sum()),
        rate_ktp=round(ktp_t['event'].mean()*100,1),
        rate_spk=round(spk_t['event'].mean()*100,1),
        logrank_p=round(lr.p_value,4),
        hr=round(hr,3) if not np.isnan(hr) else np.nan,
        ci_lo=round(ci_lo,3) if not np.isnan(ci_lo) else np.nan,
        ci_hi=round(ci_hi,3) if not np.isnan(ci_hi) else np.nan,
        cox_p=round(cox_p,4) if not np.isnan(cox_p) else np.nan,
    )

def get_serial_lab_era(loinc_list, pid_set, idx_map, cohort_map_local):
    lab = lab_clean[lab_clean['code'].isin(loinc_list) &
                    lab_clean['patient_id'].isin(pid_set) &
                    lab_clean['lab_result_num_val'].notna()].copy()
    lab['idx']      = lab['patient_id'].map(idx_map)
    lab['days_post'] = (lab['date'] - lab['idx']).dt.days
    lab['cohort']   = lab['patient_id'].map(cohort_map_local)
    lab_post = lab[lab['days_post']>0].copy()
    rows = []
    for lbl, center, hw in INTERVALS:
        lo  = max(center-hw, 1)
        hi  = center+hw
        sub = lab_post[(lab_post['days_post']>=lo) & (lab_post['days_post']<=hi)].copy()
        if len(sub)==0:
            rows.append({'Interval':lbl,'N_KTP':0,'N_SPK':0,
                         'Mean_KTP':np.nan,'Mean_SPK':np.nan,'Diff':np.nan,'p_value':np.nan})
            continue
        sub['dist'] = abs(sub['days_post']-center)
        best = sub.sort_values('dist').groupby('patient_id', as_index=False).first()
        vk = best[best['cohort']=='KTP']['lab_result_num_val']
        vs = best[best['cohort']=='SPK']['lab_result_num_val']
        t,p = (stats.ttest_ind(vk,vs) if (len(vk)>=5 and len(vs)>=5) else (np.nan,np.nan))
        rows.append({'Interval':lbl,'N_KTP':len(vk),'N_SPK':len(vs),
                     'Mean_KTP': round(float(vk.mean()),2) if len(vk)>0 else np.nan,
                     'SD_KTP':   round(float(vk.std()),2)  if len(vk)>0 else np.nan,
                     'Mean_SPK': round(float(vs.mean()),2) if len(vs)>0 else np.nan,
                     'SD_SPK':   round(float(vs.std()),2)  if len(vs)>0 else np.nan,
                     'Diff':     round(float(vk.mean()-vs.mean()),2) if (len(vk)>0 and len(vs)>0) else np.nan,
                     'p_value':  round(float(p),4) if not np.isnan(p) else np.nan})
    return pd.DataFrame(rows), lab_post

def get_serial_vital_era(loinc_list, pid_set, idx_map, cohort_map_local):
    vs = vital_signs[vital_signs['patient_id'].isin(pid_set) &
                     vital_signs['code'].astype(str).isin(loinc_list)].copy()
    if len(vs)==0:
        return pd.DataFrame()
    vs['value']     = pd.to_numeric(vs['value'], errors='coerce')
    vs['idx']       = vs['patient_id'].map(idx_map)
    vs['days_post'] = (vs['date'] - vs['idx']).dt.days
    vs['cohort']    = vs['patient_id'].map(cohort_map_local)
    vs_post = vs[(vs['days_post']>0) & vs['value'].notna()].copy()
    rows = []
    for lbl, center, hw in INTERVALS:
        lo  = max(center-hw, 1)
        hi  = center+hw
        sub = vs_post[(vs_post['days_post']>=lo) & (vs_post['days_post']<=hi)].copy()
        if len(sub)==0:
            rows.append({'Interval':lbl,'N_KTP':0,'N_SPK':0,
                         'Mean_KTP':np.nan,'Mean_SPK':np.nan,'Diff':np.nan,'p_value':np.nan})
            continue
        sub['dist'] = abs(sub['days_post']-center)
        best = sub.sort_values('dist').groupby('patient_id', as_index=False).first()
        vk = best[best['cohort']=='KTP']['value']
        vs2= best[best['cohort']=='SPK']['value']
        t,p = (stats.ttest_ind(vk,vs2) if (len(vk)>=5 and len(vs2)>=5) else (np.nan,np.nan))
        rows.append({'Interval':lbl,'N_KTP':len(vk),'N_SPK':len(vs2),
                     'Mean_KTP': round(float(vk.mean()),2) if len(vk)>0 else np.nan,
                     'Mean_SPK': round(float(vs2.mean()),2) if len(vs2)>0 else np.nan,
                     'Diff':     round(float(vk.mean()-vs2.mean()),2) if (len(vk)>0 and len(vs2)>0) else np.nan,
                     'p_value':  round(float(p),4) if not np.isnan(p) else np.nan})
    return pd.DataFrame(rows)

def get_immuno_era(pid_set, idx_map, cohort_map_local, n_ktp, n_spk):
    m = med_drug[med_drug['patient_id'].isin(pid_set)].copy()
    m['idx']       = m['patient_id'].map(idx_map)
    m['days_post'] = (m['start_date'] - m['idx']).dt.days
    m_post = m[m['days_post']>0].copy()
    m_post['cohort'] = m_post['patient_id'].map(cohort_map_local)
    rows = []
    for drug, codes in IMMUNO_CODES.items():
        sub  = m_post[m_post['code'].astype(str).isin(codes)]
        kn   = sub[sub['cohort']=='KTP']['patient_id'].nunique()
        sn   = sub[sub['cohort']=='SPK']['patient_id'].nunique()
        kpct = round(kn/n_ktp*100,1) if n_ktp>0 else np.nan
        spct = round(sn/n_spk*100,1) if n_spk>0 else np.nan
        p    = np.nan
        if (kn+sn)>0:
            try:
                _, p, _, _ = chi2_contingency([[kn, n_ktp-kn],[sn, n_spk-sn]])
            except:
                pass
        rows.append({'Drug':drug,'KTP_pct':kpct,'SPK_pct':spct,
                     'p':round(p,4) if not np.isnan(p) else np.nan,
                     'sig': p<0.05 if not np.isnan(p) else False})
    return pd.DataFrame(rows)

def run_mortality_era(pid_set, idx_map, cohort_map_local):
    mort = master_full[master_full['patient_id'].isin(pid_set)][
        ['patient_id','death_date','index_date']].copy()
    mort['index_date'] = mort['patient_id'].map(idx_map)
    mort['days_to_death'] = (mort['death_date'] - mort['index_date']).dt.days
    dx_m = diagnosis[diagnosis['patient_id'].isin(pid_set)].copy()
    dx_m['idx'] = dx_m['patient_id'].map(idx_map)
    dx_m['dp']  = (dx_m['date'] - dx_m['idx']).dt.days
    last_obs = dx_m[dx_m['dp']>0].groupby('patient_id')['dp'].max().to_dict()
    rows = []
    for pid in pid_set:
        row  = mort[mort['patient_id']==pid]
        dtd  = row['days_to_death'].iloc[0] if len(row)>0 else np.nan
        if pd.notna(dtd) and dtd > 0:
            rows.append({'patient_id':pid,'time':min(int(dtd),3650),'event':1})
        else:
            rows.append({'patient_id':pid,'time':max(int(last_obs.get(pid,365)),1),'event':0})
    df = pd.DataFrame(rows)
    df['cohort']     = df['patient_id'].map(cohort_map_local)
    df['cohort_bin'] = (df['cohort']=='KTP').astype(int)
    return run_km_cox(df[['patient_id','time','event']], cohort_map_local)

# =============================================================================
# MAIN LOOP
# =============================================================================
all_era_results   = []
era_egfr_data     = {}
era_bmi_data      = {}
era_immuno_data   = {}

OUTCOMES = [
    ('kidney_rejection',  ['T86.11'],             'Kidney Graft Rejection'),
    ('kidney_failure',    ['T86.12','T86.13'],     'Kidney Graft Failure'),
    ('any_graft_event',   ['T86.11','T86.12','T86.13'], 'Any Kidney Graft Event'),
    ('mi',                ['I21','I22'],            'MI'),
    ('stroke',            ['I63','I64'],            'Stroke'),
    ('heart_failure',     ['I50'],                  'Heart Failure'),
    ('mace',              ['I21','I22','I63','I64','I50'], 'MACE composite'),
    ('hypoglycemia',      ['E16.0','E16.2','T38.3'], 'Hypoglycemia'),
    ('uti',               ['N39.0'],                'UTI'),
    ('sepsis',            ['A41'],                  'Sepsis'),
    ('cmv',               ['B25'],                  'CMV'),
    ('bk_virus',          ['B34.2','B97.89'],        'BK Virus'),
    ('serious_infection', ['A41','B25','B34.2','B97.89'], 'Any serious infection'),
]

for era_label, cutoff in ERAS.items():
    print("\n" + "="*60)
    print(f"ERA: {era_label.upper()}  (index >= {cutoff.date()})")
    print("="*60)

    era_pts = master_full[master_full['index_date'] >= cutoff].copy()
    n_ktp_raw = (era_pts['cohort']=='KTP').sum()
    n_spk_raw = (era_pts['cohort']=='SPK').sum()
    print(f"  Raw: KTP={n_ktp_raw}, SPK={n_spk_raw}")

    if n_ktp_raw < 20:
        print(f"  Skipping — too few KTP patients")
        continue

    # PSM
    print("  Running PSM...")
    cov_df = build_covariates(era_pts)
    pairs_df, _ = run_psm(cov_df)
    n_pairs = len(pairs_df)
    print(f"  Matched pairs: {n_pairs}")

    matched_ktp = set(pairs_df['ktp_patient_id'])
    matched_spk = set(pairs_df['spk_patient_id'])
    matched_ids = matched_ktp | matched_spk
    idx_map     = era_pts.set_index('patient_id')['index_date'].to_dict()
    c_map       = {p:'KTP' for p in matched_ktp}
    c_map.update({p:'SPK' for p in matched_spk})

    # eGFR
    print("  eGFR trajectory...")
    egfr_df, egfr_post = get_serial_lab_era(EGFR_LOINCS, matched_ids, idx_map, c_map)
    era_egfr_data[era_label] = egfr_df

    # LME for eGFR
    egfr_lme_res = {}
    try:
        lme_d = egfr_post.rename(columns={'lab_result_num_val':'egfr'}).copy()
        lme_d['years_post'] = lme_d['days_post']/365.25
        lme_d['is_ktp']     = (lme_d['cohort']=='KTP').astype(int)
        lme_d = lme_d[lme_d['egfr'].notna()]
        res   = smf.mixedlm("egfr ~ years_post * is_ktp", lme_d,
                              groups=lme_d['patient_id']).fit(reml=True)
        egfr_lme_res = {
            'intercept':   round(float(res.fe_params['Intercept']),2),
            'slope_spk':   round(float(res.fe_params['years_post']),3),
            'ktp_offset':  round(float(res.fe_params['is_ktp']),3),
            'interaction': round(float(res.fe_params['years_post:is_ktp']),3),
            'p_interact':  round(float(res.pvalues['years_post:is_ktp']),4),
        }
    except Exception as e:
        egfr_lme_res = {'error': str(e)}

    # BMI
    print("  BMI trajectory...")
    bmi_df = get_serial_vital_era(BMI_LOINCS, matched_ids, idx_map, c_map)
    era_bmi_data[era_label] = bmi_df

    # Mortality
    print("  Mortality...")
    mort_res = run_mortality_era(matched_ids, idx_map, c_map)

    # Event outcomes
    print("  Event outcomes...")
    outcome_results = {}
    for key, prefixes, label in OUTCOMES:
        event_dict = flag_events_post_index(prefixes, matched_ids, idx_map)
        tte        = build_tte(event_dict, matched_ids, idx_map)
        res        = run_km_cox(tte, c_map)
        outcome_results[key] = res
        print(f"    {label:30s}: KTP={res.get('rate_ktp','?')}%  "
              f"SPK={res.get('rate_spk','?')}%  "
              f"HR={res.get('hr','?')}  p={res.get('cox_p','?')}")

    # Immunosuppression
    print("  Immunosuppression...")
    immuno_df = get_immuno_era(matched_ids, idx_map, c_map, n_pairs, n_pairs)
    era_immuno_data[era_label] = immuno_df

    # Print eGFR summary
    print(f"\n  eGFR serial means (LME interaction p={egfr_lme_res.get('p_interact','?')}):")
    print(egfr_df[['Interval','N_KTP','N_SPK','Mean_KTP','Mean_SPK','Diff','p_value']].to_string(index=False))

    # Collect all results for cross-era table
    for key, prefixes, label in OUTCOMES:
        r = outcome_results[key]
        all_era_results.append({
            'Era': era_label, 'N_pairs': n_pairs,
            'Outcome': label,
            'Rate_KTP': r.get('rate_ktp'), 'Rate_SPK': r.get('rate_spk'),
            'HR': r.get('hr'), 'CI_lo': r.get('ci_lo'), 'CI_hi': r.get('ci_hi'),
            'p': r.get('cox_p'),
            'Sig': '*' if (r.get('cox_p') or 1) < 0.05 else '',
        })
    all_era_results.append({
        'Era': era_label, 'N_pairs': n_pairs,
        'Outcome': 'All-cause mortality',
        'Rate_KTP': mort_res.get('rate_ktp'), 'Rate_SPK': mort_res.get('rate_spk'),
        'HR': mort_res.get('hr'), 'CI_lo': mort_res.get('ci_lo'), 'CI_hi': mort_res.get('ci_hi'),
        'p': mort_res.get('cox_p'),
        'Sig': '*' if (mort_res.get('cox_p') or 1) < 0.05 else '',
    })

# =============================================================================
# CROSS-ERA COMPARISON TABLE
# =============================================================================
print("\n" + "="*60)
print("CROSS-ERA COMPARISON TABLE")
print("="*60)

results_df = pd.DataFrame(all_era_results)
results_df.to_csv(os.path.join(OUTPUT_DIR, "era_secondary_all_results.csv"), index=False)

# Print pivoted view — one row per outcome, columns per era
pivot = results_df.pivot_table(
    index='Outcome', columns='Era',
    values=['Rate_KTP','Rate_SPK','HR','p'],
    aggfunc='first'
)
print("\nHR by outcome and era:")
hr_pivot = results_df.pivot_table(index='Outcome', columns='Era', values='HR', aggfunc='first')
p_pivot  = results_df.pivot_table(index='Outcome', columns='Era', values='p',  aggfunc='first')
for outcome in hr_pivot.index:
    print(f"\n  {outcome}")
    for era in ERAS.keys():
        if era in hr_pivot.columns:
            hr = hr_pivot.loc[outcome, era]
            p  = p_pivot.loc[outcome, era]
            if not np.isnan(hr):
                ci_row = results_df[(results_df['Outcome']==outcome) & (results_df['Era']==era)]
                ci_lo = ci_row['CI_lo'].iloc[0] if len(ci_row)>0 else np.nan
                ci_hi = ci_row['CI_hi'].iloc[0] if len(ci_row)>0 else np.nan
                sig = '*' if p < 0.05 else ''
                print(f"    {era:15s}: HR={hr:.3f} [{ci_lo:.3f},{ci_hi:.3f}]  p={p:.4f}{sig}")

print(f"\n  -> Saved: era_secondary_all_results.csv")

# =============================================================================
# IMMUNOSUPPRESSION CROSS-ERA TABLE
# =============================================================================
print("\n" + "="*60)
print("IMMUNOSUPPRESSION BY ERA")
print("="*60)

for era_label, imm_df in era_immuno_data.items():
    print(f"\n  {era_label}:")
    print(imm_df.to_string(index=False))

# =============================================================================
# PLOTS
# =============================================================================
print("\n" + "="*60)
print("GENERATING PLOTS")
print("="*60)

era_list   = list(era_egfr_data.keys())
n_eras     = len(era_list)
ck = ['#1565C0','#2196F3','#90CAF9']
cs = ['#BF360C','#FF5722','#FFCCBC']

# Plot 1 — eGFR serial means by era
fig, axes = plt.subplots(1, n_eras, figsize=(6*n_eras, 6), sharey=True)
if n_eras==1: axes=[axes]
fig.suptitle('eGFR Trajectory by Era: KTP vs SPK', fontsize=14, fontweight='bold')
for i, era_label in enumerate(era_list):
    ax     = axes[i]
    df     = era_egfr_data[era_label]
    plot_df = df[df['Mean_KTP'].notna() & df['Mean_SPK'].notna()]
    if len(plot_df)==0: continue
    x    = list(range(len(plot_df)))
    semk = (plot_df['SD_KTP']/np.sqrt(plot_df['N_KTP'])).tolist()
    sems = (plot_df['SD_SPK']/np.sqrt(plot_df['N_SPK'])).tolist()
    ax.errorbar(x, plot_df['Mean_KTP'], yerr=semk, color=ck[i], marker='o',
                linewidth=2, markersize=7, capsize=4, label='KTP')
    ax.errorbar(x, plot_df['Mean_SPK'], yerr=sems, color=cs[i], marker='s',
                linewidth=2, markersize=7, capsize=4, label='SPK')
    ax.axhline(60, color='green', linestyle='--', alpha=0.4, linewidth=1)
    ax.axhline(30, color='red',   linestyle='--', alpha=0.4, linewidth=1)
    n_pairs_era = results_df[results_df['Era']==era_label]['N_pairs'].iloc[0] \
                  if len(results_df[results_df['Era']==era_label])>0 else '?'
    ax.set_title(f'{era_label}\n(n={n_pairs_era} pairs)', fontsize=11, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(plot_df['Interval'].tolist(), rotation=40, ha='right', fontsize=8)
    ax.set_xlabel('Time post-index'); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    for j, row in enumerate(plot_df.itertuples()):
        pv = row.p_value
        if isinstance(pv,float) and not np.isnan(pv) and pv<0.05:
            ax.text(j, max(row.Mean_KTP,row.Mean_SPK)+1.5,
                    'p<0.001' if pv<0.001 else f'p={pv:.3f}',
                    ha='center', fontsize=7, style='italic')
axes[0].set_ylabel('Mean eGFR ± SEM (mL/min/1.73m²)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "era_secondary_egfr_plot.png"), dpi=150)
plt.close()
print("  -> Saved: era_secondary_egfr_plot.png")

# Plot 2 — Forest plot: HR by outcome, grouped by era
key_outcomes = [
    'Hypoglycemia', 'Heart Failure', 'MACE composite',
    'Kidney Graft Rejection', 'Kidney Graft Failure',
    'MI', 'Stroke', 'Sepsis', 'All-cause mortality',
]
era_colors = {'Full cohort':'#1565C0', 'Past 10 years':'#2196F3', 'Past 5 years':'#90CAF9'}
fig, ax = plt.subplots(figsize=(11, 9))
y_pos = 0
yticks, ylabels = [], []
offsets = {'Full cohort':-0.22, 'Past 10 years':0, 'Past 5 years':0.22}
for outcome in reversed(key_outcomes):
    y_pos += 1
    for era_label in ERAS.keys():
        row = results_df[(results_df['Outcome']==outcome) & (results_df['Era']==era_label)]
        if len(row)==0: continue
        hr    = row['HR'].iloc[0]
        ci_lo = row['CI_lo'].iloc[0]
        ci_hi = row['CI_hi'].iloc[0]
        p     = row['p'].iloc[0]
        if np.isnan(hr): continue
        yy    = y_pos + offsets[era_label]
        color = era_colors[era_label]
        ax.plot(hr, yy, 'o', color=color, markersize=7, zorder=3)
        ax.plot([ci_lo, ci_hi], [yy, yy], '-', color=color, linewidth=1.5, zorder=2)
        if p < 0.05:
            ax.plot(hr, yy, 'o', color=color, markersize=11,
                    markerfacecolor='none', markeredgewidth=1.5, zorder=4)
    yticks.append(y_pos)
    ylabels.append(outcome)

ax.axvline(1.0, color='black', linewidth=1.2, linestyle='--', alpha=0.6)
ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=10)
ax.set_xlabel('Hazard Ratio (KTP vs SPK)', fontsize=11)
ax.set_title('Secondary Outcomes by Era — Forest Plot\n(KTP vs SPK, PSM-Matched)',
             fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
from matplotlib.lines import Line2D
legend_elements = [Line2D([0],[0], color=era_colors[e], marker='o', linewidth=2,
                           markersize=7, label=e) for e in ERAS.keys()]
legend_elements.append(Line2D([0],[0], color='gray', marker='o', linewidth=0,
                                markersize=11, markerfacecolor='none',
                                markeredgewidth=1.5, label='p<0.05'))
ax.legend(handles=legend_elements, fontsize=9, loc='lower right')
ax.set_xlim(0.3, 3.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "era_secondary_forest_plot.png"), dpi=150)
plt.close()
print("  -> Saved: era_secondary_forest_plot.png")

# Plot 3 — BMI by era
fig, axes = plt.subplots(1, n_eras, figsize=(6*n_eras, 6), sharey=True)
if n_eras==1: axes=[axes]
fig.suptitle('BMI Trajectory by Era: KTP vs SPK', fontsize=14, fontweight='bold')
for i, era_label in enumerate(era_list):
    ax     = axes[i]
    df     = era_bmi_data.get(era_label, pd.DataFrame())
    if len(df)==0: ax.set_title(f'{era_label}\nNo data'); continue
    plot_df = df[df['Mean_KTP'].notna() & df['Mean_SPK'].notna()]
    if len(plot_df)==0: continue
    x = list(range(len(plot_df)))
    ax.plot(x, plot_df['Mean_KTP'], color=ck[i], marker='o', linewidth=2, markersize=7, label='KTP')
    ax.plot(x, plot_df['Mean_SPK'], color=cs[i], marker='s', linewidth=2, markersize=7, label='SPK')
    n_pairs_era = results_df[results_df['Era']==era_label]['N_pairs'].iloc[0] \
                  if len(results_df[results_df['Era']==era_label])>0 else '?'
    ax.set_title(f'{era_label}\n(n={n_pairs_era} pairs)', fontsize=11, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(plot_df['Interval'].tolist(), rotation=40, ha='right', fontsize=8)
    ax.set_xlabel('Time post-index'); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    for j, row in enumerate(plot_df.itertuples()):
        pv = row.p_value
        if isinstance(pv,float) and not np.isnan(pv) and pv<0.05:
            ax.text(j, max(row.Mean_KTP,row.Mean_SPK)+0.3,
                    'p<0.001' if pv<0.001 else f'p={pv:.3f}',
                    ha='center', fontsize=7, style='italic')
axes[0].set_ylabel('Mean BMI (kg/m²)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "era_secondary_bmi_plot.png"), dpi=150)
plt.close()
print("  -> Saved: era_secondary_bmi_plot.png")

print("\n✓ Script 05c complete.")
print("  Outputs: era_secondary_all_results.csv")
print("           era_secondary_egfr_plot.png")
print("           era_secondary_forest_plot.png")
print("           era_secondary_bmi_plot.png")
print("  Next step: Run 06_sensitivity_subgroup.py")
