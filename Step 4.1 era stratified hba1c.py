"""
TriNetX PancPump Study - Step 4b: Era Sensitivity Analysis
Repeats primary HbA1c analysis restricted to:
  - Past 5 years:  index date >= 2021-01-01
  - Past 10 years: index date >= 2016-01-01
  - Full cohort:   all dates (reference, from Script 04)

For each era:
  1. Re-run PSM on era-restricted cohort
  2. Serial mean HbA1c at defined intervals
  3. Time-to-HbA1c <7.0% KM + Cox
  4. LME trajectory model
  5. Side-by-side comparison table across all three eras

PI: Paul Kuo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, proportional_hazard_test
import statsmodels.formula.api as smf
import os, warnings
warnings.filterwarnings('ignore')

DATA_DIR   = r"Z:\homes\Rachel Wolansky\Panc Pump\Data In"
OUTPUT_DIR = r"Z:\homes\Rachel Wolansky\Panc Pump\Results"
RANDOM_SEED = 42

# =============================================================================
# LOAD BASE FILES
# =============================================================================
print("\n" + "="*60)
print("LOADING FILES")
print("="*60)

master         = pd.read_csv(os.path.join(OUTPUT_DIR, "master_patient_table.csv"),    low_memory=False)
lab_clean      = pd.read_csv(os.path.join(OUTPUT_DIR, "lab_result_clean.csv"),        low_memory=False)
psm_balance_full = pd.read_csv(os.path.join(OUTPUT_DIR, "psm_covariate_balance.csv"), low_memory=False)
diagnosis      = pd.read_csv(os.path.join(DATA_DIR,   "diagnosis.csv"),               low_memory=False)

master['index_date']   = pd.to_datetime(master['index_date'],   errors='coerce')
lab_clean['date']      = pd.to_datetime(lab_clean['date'],      errors='coerce')
diagnosis['date']      = pd.to_datetime(diagnosis['date'].astype(str), format='%Y%m%d', errors='coerce')

print(f"  Master: {len(master):,} patients  (KTP={(master['cohort']=='KTP').sum()}, SPK={(master['cohort']=='SPK').sum()})")
print(f"  Labs  : {len(lab_clean):,} rows")

# Era definitions
ERAS = {
    'Full cohort':    pd.Timestamp('2000-01-01'),
    'Past 10 years':  pd.Timestamp('2016-01-01'),
    'Past 5 years':   pd.Timestamp('2021-01-01'),
}

# PSM covariates (same as Script 03 — diagnosis-based + labs)
PSM_DX_FLAGS = {
    'dm_type1':      'E10',
    'dm_type2':      'E11',
    'hypertension':  'I10',
    'cad':           'I25',
    'heart_failure': 'I50',
    'pvd':           'I73',
    'dyslipidemia':  'E78',
    'obesity':       'E66',
    'ckd':           'N18',
    'anemia':        'D64',
}

HBAC1C_LOINCS = ['4548-4', '17856-6', '59261-8']
INTERVALS = [
    ('3 months',  91,  45),
    ('6 months', 182,  60),
    ('1 year',   365,  90),
    ('2 years',  730, 120),
    ('3 years', 1095, 150),
    ('5 years', 1825, 180),
]

# =============================================================================
# HELPER FUNCTIONS
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
    hits['idx'] = hits['patient_id'].map(idx_map)
    hits['days'] = (hits['date'] - hits['idx']).dt.days
    hits = hits[(hits['days'] >= -window) & (hits['days'] <= 0)]
    latest = hits.sort_values('days').groupby('patient_id')['lab_result_num_val'].last().reset_index()
    result = df_pts[['patient_id']].merge(latest, on='patient_id', how='left')
    return result['lab_result_num_val'].values

def build_covariates(df_pts, df_dx, df_lab):
    cov = df_pts[['patient_id','cohort','index_date',
                  'age_at_index','sex','race','ethnicity']].copy()
    cov['sex_male']           = (cov['sex']=='M').astype(int)
    cov['race_white']         = cov['race'].str.contains('White',    na=False).astype(int)
    cov['race_black']         = cov['race'].str.contains('Black',    na=False).astype(int)
    cov['race_asian']         = cov['race'].str.contains('Asian',    na=False).astype(int)
    cov['ethnicity_hispanic'] = cov['ethnicity'].str.contains('Hispanic', na=False).astype(int)
    for col, prefix in PSM_DX_FLAGS.items():
        cov[col] = flag_dx_before_index(prefix, df_dx, df_pts)
    cov['baseline_hba1c']     = get_baseline_lab(HBAC1C_LOINCS, df_lab, df_pts)
    cov['baseline_creatinine']= get_baseline_lab(['2160-0','38483-4'], df_lab, df_pts)
    cov['baseline_egfr']      = get_baseline_lab(['62238-1'], df_lab, df_pts)
    # Impute missing with cohort median
    for col in ['baseline_hba1c','baseline_creatinine','baseline_egfr','age_at_index']:
        for cohort in ['KTP','SPK']:
            mask = (cov['cohort']==cohort) & cov[col].isna()
            med  = cov[cov['cohort']==cohort][col].median()
            cov.loc[mask, col] = med
    cov = cov.fillna(0)
    return cov

def run_psm(cov_df, caliper=0.2, seed=RANDOM_SEED):
    psm_cols = [
        'age_at_index','sex_male','race_white','race_black','race_asian',
        'ethnicity_hispanic','dm_type1','dm_type2','hypertension','cad',
        'heart_failure','pvd','dyslipidemia','obesity','ckd','anemia',
        'baseline_hba1c','baseline_creatinine','baseline_egfr',
    ]
    X = cov_df[psm_cols].values.astype(float)
    y = (cov_df['cohort']=='KTP').astype(int).values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
    lr.fit(X_scaled, y)
    ps = lr.predict_proba(X_scaled)[:,1]
    cov_df = cov_df.copy()
    cov_df['ps']       = ps
    cov_df['logit_ps'] = np.log(ps / (1-ps+1e-10))

    logit_sd      = cov_df['logit_ps'].std()
    caliper_value = caliper * logit_sd

    ktp_df = cov_df[cov_df['cohort']=='KTP'].sample(frac=1, random_state=seed).reset_index(drop=True)
    spk_df = cov_df[cov_df['cohort']=='SPK'].reset_index(drop=True)

    matched_pairs = []
    used_spk      = set()
    for _, row in ktp_df.iterrows():
        cands = spk_df[~spk_df['patient_id'].isin(used_spk)].copy()
        cands['diff'] = abs(cands['logit_ps'] - row['logit_ps'])
        within = cands[cands['diff'] <= caliper_value]
        if len(within) == 0:
            continue
        best = within.nsmallest(1,'diff').iloc[0]
        used_spk.add(best['patient_id'])
        matched_pairs.append({
            'ktp_patient_id': row['patient_id'],
            'spk_patient_id': best['patient_id'],
        })
    pairs_df = pd.DataFrame(matched_pairs)
    return pairs_df, cov_df

def smd(x1, x2):
    x1, x2 = np.array(x1,dtype=float), np.array(x2,dtype=float)
    x1, x2 = x1[~np.isnan(x1)], x2[~np.isnan(x2)]
    if len(x1)==0 or len(x2)==0: return np.nan
    pooled = np.sqrt((np.var(x1,ddof=1)+np.var(x2,ddof=1))/2)
    return 0.0 if pooled==0 else (np.mean(x1)-np.mean(x2))/pooled

def get_interval_vals(df, center, hw):
    lo  = max(center-hw, 1)
    hi  = center+hw
    sub = df[(df['days_post']>=lo) & (df['days_post']<=hi)].copy()
    if len(sub)==0:
        return pd.DataFrame(columns=['patient_id','lab_result_num_val'])
    sub['dist'] = abs(sub['days_post']-center)
    return sub.sort_values('dist').groupby('patient_id', as_index=False).first()[['patient_id','lab_result_num_val']]

def run_hba1c_analysis(matched_ids, pairs_df, cohort_map, idx_dates, df_lab, era_label):
    """Full HbA1c analysis for a given matched cohort."""
    hba1c = df_lab[
        df_lab['code'].isin(HBAC1C_LOINCS) &
        df_lab['patient_id'].isin(matched_ids) &
        df_lab['lab_result_num_val'].notna()
    ].copy()
    hba1c['index_date'] = hba1c['patient_id'].map(idx_dates)
    hba1c['cohort']     = hba1c['patient_id'].map(cohort_map)
    hba1c['days_post']  = (hba1c['date'] - hba1c['index_date']).dt.days
    hba1c_post = hba1c[hba1c['days_post']>0].copy()

    n_ktp_hba1c = (hba1c_post['cohort']=='KTP').sum()
    n_spk_hba1c = (hba1c_post['cohort']=='SPK').sum()
    print(f"  Post-index HbA1c: {len(hba1c_post):,}  (KTP={n_ktp_hba1c:,}, SPK={n_spk_hba1c:,})")

    # Serial means
    serial_rows = []
    for label, center, hw in INTERVALS:
        v_ktp = get_interval_vals(hba1c_post[hba1c_post['cohort']=='KTP'], center, hw)
        v_spk = get_interval_vals(hba1c_post[hba1c_post['cohort']=='SPK'], center, hw)
        n_ktp, n_spk = len(v_ktp), len(v_spk)
        if n_ktp==0 and n_spk==0:
            serial_rows.append({'Interval':label,'N_KTP':0,'N_SPK':0,'N_paired':0,
                                 'Mean_KTP':np.nan,'Mean_SPK':np.nan,
                                 'Diff':np.nan,'p_ttest':np.nan,'Sig':False,
                                 'Pct_KTP':np.nan,'Pct_SPK':np.nan})
            continue
        mean_ktp = v_ktp['lab_result_num_val'].mean() if n_ktp>0 else np.nan
        mean_spk = v_spk['lab_result_num_val'].mean() if n_spk>0 else np.nan
        sd_ktp   = v_ktp['lab_result_num_val'].std()  if n_ktp>0 else np.nan
        sd_spk   = v_spk['lab_result_num_val'].std()  if n_spk>0 else np.nan
        pct_ktp  = (v_ktp['lab_result_num_val']<7.0).mean()*100 if n_ktp>0 else np.nan
        pct_spk  = (v_spk['lab_result_num_val']<7.0).mean()*100 if n_spk>0 else np.nan

        pk = pairs_df.merge(v_ktp.rename(columns={'lab_result_num_val':'ktp_hba1c','patient_id':'ktp_patient_id'}), on='ktp_patient_id', how='inner')
        pm = pk.merge(v_spk.rename(columns={'lab_result_num_val':'spk_hba1c','patient_id':'spk_patient_id'}), on='spk_patient_id', how='inner')
        n_paired = len(pm)

        t_stat = p_val = np.nan
        if n_paired>=10:
            t_stat, p_val = stats.ttest_rel(pm['ktp_hba1c'], pm['spk_hba1c'])
        elif n_ktp>=5 and n_spk>=5:
            t_stat, p_val = stats.ttest_ind(v_ktp['lab_result_num_val'], v_spk['lab_result_num_val'])

        serial_rows.append({
            'Interval': label, 'N_KTP': n_ktp, 'N_SPK': n_spk, 'N_paired': n_paired,
            'Mean_KTP': round(float(mean_ktp),2) if not np.isnan(mean_ktp) else np.nan,
            'SD_KTP':   round(float(sd_ktp),2)   if not np.isnan(sd_ktp)   else np.nan,
            'Mean_SPK': round(float(mean_spk),2) if not np.isnan(mean_spk) else np.nan,
            'SD_SPK':   round(float(sd_spk),2)   if not np.isnan(sd_spk)   else np.nan,
            'Diff':     round(float(mean_ktp-mean_spk),2) if (not np.isnan(mean_ktp) and not np.isnan(mean_spk)) else np.nan,
            'Pct_KTP':  round(float(pct_ktp),1) if not np.isnan(pct_ktp) else np.nan,
            'Pct_SPK':  round(float(pct_spk),1) if not np.isnan(pct_spk) else np.nan,
            'p_ttest':  round(float(p_val),4)   if not np.isnan(p_val)  else np.nan,
            'Sig':      bool(p_val<0.05)         if not np.isnan(p_val)  else False,
        })

    serial_df = pd.DataFrame(serial_rows)

    # KM + Cox
    tte_rows = []
    for pid, grp in hba1c_post.groupby('patient_id'):
        grp    = grp.sort_values('days_post')
        events = grp[grp['lab_result_num_val']<7.0]
        last_t = grp['days_post'].max()
        if len(events)>0:
            tte_rows.append({'patient_id':pid,'cohort':cohort_map.get(pid),
                             'time':int(events['days_post'].iloc[0]),'event':1})
        else:
            tte_rows.append({'patient_id':pid,'cohort':cohort_map.get(pid),
                             'time':max(int(last_t),1),'event':0})
    tte_df = pd.DataFrame(tte_rows)
    tte_df = tte_df[tte_df['patient_id'].isin(matched_ids)].copy()
    tte_df['cohort_bin'] = (tte_df['cohort']=='KTP').astype(int)
    tte_df = tte_df[tte_df['time']>0]

    ktp_tte = tte_df[tte_df['cohort']=='KTP']
    spk_tte = tte_df[tte_df['cohort']=='SPK']

    lr_result = logrank_test(ktp_tte['time'], spk_tte['time'],
                              event_observed_A=ktp_tte['event'],
                              event_observed_B=spk_tte['event'])

    cph = CoxPHFitter()
    cph.fit(tte_df[['time','event','cohort_bin']], duration_col='time', event_col='event')
    hr    = float(np.exp(cph.params_['cohort_bin']))
    ci_lo = float(np.exp(cph.confidence_intervals_['95% lower-bound']['cohort_bin']))
    ci_hi = float(np.exp(cph.confidence_intervals_['95% upper-bound']['cohort_bin']))
    cox_p = float(cph.summary['p']['cohort_bin'])

    # PH assumption
    ph_ok = True
    try:
        ph_test = proportional_hazard_test(cph, tte_df[['time','event','cohort_bin']], time_transform='rank')
        ph_ok   = float(ph_test.summary['p'].iloc[0]) > 0.05
    except:
        pass

    # LME
    lme_df = hba1c_post.copy()
    lme_df['years_post'] = lme_df['days_post']/365.25
    lme_df['is_ktp']     = (lme_df['cohort']=='KTP').astype(int)
    lme_df = lme_df.rename(columns={'lab_result_num_val':'hba1c'})
    lme_df = lme_df[lme_df['hba1c'].notna() & lme_df['years_post'].notna()]

    lme_results = {}
    try:
        model  = smf.mixedlm("hba1c ~ years_post * is_ktp", lme_df, groups=lme_df['patient_id'])
        result = model.fit(reml=True)
        coefs  = result.fe_params
        pvals  = result.pvalues
        lme_results = {
            'intercept':      round(float(coefs['Intercept']),3),
            'slope_spk':      round(float(coefs['years_post']),3),
            'ktp_offset':     round(float(coefs['is_ktp']),3),
            'interaction':    round(float(coefs['years_post:is_ktp']),3),
            'p_interaction':  round(float(pvals['years_post:is_ktp']),4),
            'p_ktp_offset':   round(float(pvals['is_ktp']),4),
        }
    except Exception as e:
        lme_results = {'error': str(e)}

    return {
        'serial_df':    serial_df,
        'lr_p':         lr_result.p_value,
        'hr':           hr,
        'ci_lo':        ci_lo,
        'ci_hi':        ci_hi,
        'cox_p':        cox_p,
        'ph_ok':        ph_ok,
        'lme':          lme_results,
        'tte_df':       tte_df,
        'hba1c_post':   hba1c_post,
    }

# =============================================================================
# MAIN LOOP — RUN FOR EACH ERA
# =============================================================================
era_results  = {}
era_pairs    = {}
era_masters  = {}

for era_label, cutoff in ERAS.items():
    print("\n" + "="*60)
    print(f"ERA: {era_label.upper()}  (index date >= {cutoff.date()})")
    print("="*60)

    # Restrict master to era
    era_master = master[master['index_date'] >= cutoff].copy()
    n_ktp = (era_master['cohort']=='KTP').sum()
    n_spk = (era_master['cohort']=='SPK').sum()
    print(f"  Patients: KTP={n_ktp}, SPK={n_spk}")

    if n_ktp < 20:
        print(f"  ⚠ Too few KTP patients ({n_ktp}) — skipping this era")
        continue

    # Build covariates
    print("  Building covariates...")
    cov_df = build_covariates(era_master, diagnosis, lab_clean)

    # PSM
    print("  Running PSM...")
    pairs_df, cov_with_ps = run_psm(cov_df)
    n_pairs = len(pairs_df)
    print(f"  Matched pairs: {n_pairs}")

    if n_pairs < 10:
        print(f"  ⚠ Too few matched pairs ({n_pairs}) — skipping")
        continue

    # Balance check
    matched_ktp = set(pairs_df['ktp_patient_id'])
    matched_spk = set(pairs_df['spk_patient_id'])
    all_matched = matched_ktp | matched_spk
    ktp_after   = cov_with_ps[cov_with_ps['patient_id'].isin(matched_ktp)]
    spk_after   = cov_with_ps[cov_with_ps['patient_id'].isin(matched_spk)]
    bal_cols    = ['age_at_index','dm_type1','dm_type2','hypertension',
                   'heart_failure','baseline_hba1c','baseline_egfr']
    smd_vals    = {c: abs(smd(ktp_after[c], spk_after[c])) for c in bal_cols}
    max_smd     = max(smd_vals.values())
    n_unbal     = sum(1 for v in smd_vals.values() if v >= 0.1)
    print(f"  Post-match SMD: max={max_smd:.3f}, unbalanced (>=0.1): {n_unbal}/7")

    # Cohort maps
    era_idx_dates  = era_master.set_index('patient_id')['index_date'].to_dict()
    era_cohort_map = era_master.set_index('patient_id')['cohort'].to_dict()

    # HbA1c analysis
    print("  Running HbA1c analysis...")
    results = run_hba1c_analysis(all_matched, pairs_df, era_cohort_map,
                                  era_idx_dates, lab_clean, era_label)

    era_results[era_label] = results
    era_pairs[era_label]   = pairs_df
    era_masters[era_label] = era_master[era_master['patient_id'].isin(all_matched)].copy()

    # Print serial results
    print(f"\n  Serial HbA1c (n={n_pairs} pairs):")
    df = results['serial_df']
    print(df[['Interval','N_KTP','N_SPK','N_paired','Mean_KTP','Mean_SPK','Diff','p_ttest','Sig']].to_string(index=False))
    print(f"\n  Log-rank p = {results['lr_p']:.4f}")
    print(f"  HR (KTP vs SPK) = {results['hr']:.3f}  [{results['ci_lo']:.3f}, {results['ci_hi']:.3f}]  p={results['cox_p']:.4f}")
    print(f"  PH assumption: {'OK' if results['ph_ok'] else 'VIOLATED'}")
    if 'error' not in results['lme']:
        lme = results['lme']
        print(f"  LME: KTP offset={lme['ktp_offset']:.3f} (p={lme['p_ktp_offset']:.4f}), "
              f"interaction={lme['interaction']:.3f} (p={lme['p_interaction']:.4f})")

# =============================================================================
# CROSS-ERA COMPARISON TABLE
# =============================================================================
print("\n" + "="*60)
print("CROSS-ERA COMPARISON TABLE")
print("="*60)

comp_rows = []
for era_label, results in era_results.items():
    pairs_df   = era_pairs[era_label]
    serial_df  = results['serial_df']
    lme        = results['lme']
    for _, row in serial_df.iterrows():
        if np.isnan(row['Mean_KTP']):
            continue
        comp_rows.append({
            'Era':          era_label,
            'N_pairs':      len(pairs_df),
            'Interval':     row['Interval'],
            'Mean_KTP':     row['Mean_KTP'],
            'Mean_SPK':     row['Mean_SPK'],
            'Diff':         row['Diff'],
            'p_ttest':      row['p_ttest'],
            'Sig':          '*' if row['Sig'] else '',
            'Pct_KTP<7':    row['Pct_KTP'],
            'Pct_SPK<7':    row['Pct_SPK'],
        })

comp_df = pd.DataFrame(comp_rows)
print("\n" + comp_df.to_string(index=False))
comp_df.to_csv(os.path.join(OUTPUT_DIR, "era_comparison_hba1c.csv"), index=False)
print("\n  -> Saved: era_comparison_hba1c.csv")

# Cox + LME summary across eras
print("\nCox + LME summary by era:")
print(f"{'Era':<20} {'N_pairs':>8} {'HR':>6} {'95%CI':>18} {'Cox_p':>7} {'LME_KTP_offset':>15} {'LME_interaction':>16} {'LME_p_interact':>15}")
for era_label, results in era_results.items():
    lme = results['lme']
    n   = len(era_pairs[era_label])
    if 'error' not in lme:
        print(f"{era_label:<20} {n:>8} {results['hr']:>6.3f} "
              f"[{results['ci_lo']:.3f},{results['ci_hi']:.3f}] "
              f"{results['cox_p']:>7.4f} {lme['ktp_offset']:>15.3f} "
              f"{lme['interaction']:>16.3f} {lme['p_interaction']:>15.4f}")
    else:
        print(f"{era_label:<20} {n:>8} {results['hr']:>6.3f}  LME failed")

# =============================================================================
# COMBINED VISUALIZATION — 3 ERAS SIDE BY SIDE
# =============================================================================
print("\n" + "="*60)
print("GENERATING ERA COMPARISON PLOTS")
print("="*60)

era_list   = list(era_results.keys())
n_eras     = len(era_list)
colors_ktp = ['#1565C0', '#2196F3', '#90CAF9']   # dark -> light blue
colors_spk = ['#BF360C', '#FF5722', '#FFCCBC']   # dark -> light orange

# Plot 1 — Serial HbA1c means, one panel per era
fig, axes = plt.subplots(1, n_eras, figsize=(6*n_eras, 6), sharey=True)
if n_eras == 1:
    axes = [axes]
fig.suptitle('Serial HbA1c by Era: KTP vs SPK (PSM-Matched)',
             fontsize=14, fontweight='bold')

for i, era_label in enumerate(era_list):
    ax       = axes[i]
    serial   = era_results[era_label]['serial_df']
    plot_df  = serial[serial['Mean_KTP'].notna() & serial['Mean_SPK'].notna()].copy()
    x_pos    = list(range(len(plot_df)))
    n_pairs  = len(era_pairs[era_label])
    ck       = colors_ktp[i % len(colors_ktp)]
    cs       = colors_spk[i % len(colors_spk)]

    sem_ktp  = (plot_df['SD_KTP']/np.sqrt(plot_df['N_KTP'])).tolist()
    sem_spk  = (plot_df['SD_SPK']/np.sqrt(plot_df['N_SPK'])).tolist()

    ax.errorbar(x_pos, plot_df['Mean_KTP'], yerr=sem_ktp,
                color=ck, marker='o', linewidth=2, markersize=7,
                capsize=4, label='KTP')
    ax.errorbar(x_pos, plot_df['Mean_SPK'], yerr=sem_spk,
                color=cs, marker='s', linewidth=2, markersize=7,
                capsize=4, label='SPK')
    ax.axhline(7.0, color='green', linestyle='--', alpha=0.6, linewidth=1.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(plot_df['Interval'].tolist(), rotation=40, ha='right', fontsize=8)
    ax.set_title(f'{era_label}\n(n={n_pairs} pairs)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time post-index')
    ax.set_ylim(4, 10.5)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=9)

    # significance markers
    for j, row in enumerate(plot_df.itertuples()):
        pv = row.p_ttest
        if isinstance(pv, float) and not np.isnan(pv) and pv < 0.05:
            pt = 'p<0.001' if pv<0.001 else f'p={pv:.3f}'
            ax.text(j, max(row.Mean_KTP, row.Mean_SPK)+0.35,
                    pt, ha='center', fontsize=7, style='italic', color='black')

axes[0].set_ylabel('Mean HbA1c ± SEM (%)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "era_hba1c_serial_plot.png"), dpi=150)
plt.close()
print("  -> Saved: era_hba1c_serial_plot.png")

# Plot 2 — KM curves overlaid across eras
fig, axes = plt.subplots(1, n_eras, figsize=(6*n_eras, 6))
if n_eras == 1:
    axes = [axes]
fig.suptitle('Time to HbA1c <7.0% by Era: KTP vs SPK',
             fontsize=14, fontweight='bold')

for i, era_label in enumerate(era_list):
    ax      = axes[i]
    tte_df  = era_results[era_label]['tte_df']
    n_pairs = len(era_pairs[era_label])
    lr_p    = era_results[era_label]['lr_p']
    hr      = era_results[era_label]['hr']
    ck      = colors_ktp[i % len(colors_ktp)]
    cs      = colors_spk[i % len(colors_spk)]

    ktp_t = tte_df[tte_df['cohort']=='KTP']
    spk_t = tte_df[tte_df['cohort']=='SPK']
    kmf_k = KaplanMeierFitter(label='KTP')
    kmf_s = KaplanMeierFitter(label='SPK')
    kmf_k.fit(ktp_t['time'], event_observed=ktp_t['event'])
    kmf_s.fit(spk_t['time'], event_observed=spk_t['event'])
    kmf_k.plot_survival_function(ax=ax, color=ck, linewidth=2, ci_show=True, ci_alpha=0.12)
    kmf_s.plot_survival_function(ax=ax, color=cs, linewidth=2, ci_show=True, ci_alpha=0.12)

    p_str = 'p<0.001' if lr_p<0.001 else f'p={lr_p:.4f}'
    ax.text(0.55, 0.88,
            f'Log-rank {p_str}\nHR={hr:.2f}',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_title(f'{era_label}\n(n={n_pairs} pairs)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Days Post-Index')
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

axes[0].set_ylabel('P(Not achieving HbA1c <7.0%)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "era_km_plot.png"), dpi=150)
plt.close()
print("  -> Saved: era_km_plot.png")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*60)
print("ERA SENSITIVITY ANALYSIS SUMMARY")
print("="*60)

for era_label, results in era_results.items():
    n = len(era_pairs[era_label])
    lme = results['lme']
    print(f"\n{era_label} (n={n} pairs):")
    print(f"  Log-rank p = {results['lr_p']:.4f}")
    print(f"  HR = {results['hr']:.3f}  [{results['ci_lo']:.3f}, {results['ci_hi']:.3f}]  p={results['cox_p']:.4f}")
    if 'error' not in lme:
        print(f"  LME KTP offset    = {lme['ktp_offset']:.3f}  (p={lme['p_ktp_offset']:.4f})")
        print(f"  LME interaction   = {lme['interaction']:.3f}  (p={lme['p_interaction']:.4f})")
    serial = results['serial_df']
    for _, row in serial.iterrows():
        if not np.isnan(row['Mean_KTP']):
            sig = '*' if row['Sig'] else ''
            print(f"    {row['Interval']:10s}: KTP={row['Mean_KTP']:.2f}  "
                  f"SPK={row['Mean_SPK']:.2f}  diff={row['Diff']:+.2f}  "
                  f"p={row['p_ttest']:.4f}{sig}")

print("\nOutput files:")
print("  era_comparison_hba1c.csv")
print("  era_hba1c_serial_plot.png")
print("  era_km_plot.png")
print("\n✓ Script 04b complete.")
print("  Next step: Run 05_secondary_outcomes.py")
