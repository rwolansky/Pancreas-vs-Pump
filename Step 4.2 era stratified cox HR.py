"""
TriNetX PancPump Study - Step 4c: Era-Stratified HbA1c Cox HR at Multiple Thresholds

Fills the gap in Script 04b: that script printed era-stratified Cox HRs for
HbA1c <7.0% to console but never saved them to disk. This script:
  1. Re-runs era-stratified PSM (identical methodology to 04b)
  2. Computes Cox HR at TWO thresholds for EACH era:
       - <7.0%  diabetic target
       - <5.7%  normoglycemia
  3. Saves results to a structured CSV so the abstract table has a paper trail
  4. Generates a forest plot and KM plots

Why <5.7% matters: it's the upper bound of normal glycemia (HbA1c <5.7% = no
diabetes by ADA criteria). SPK eliminates exogenous insulin, so a true
normoglycemia analysis isolates the "cure" effect rather than just glycemic
target attainment.

ERAS:
  Full cohort:    all dates
  Past 10 years:  index >= 2016-01-01 (post hybrid closed-loop FDA approval)
  Past 5 years:   index >= 2021-01-01 (Control-IQ / Omnipod 5 era)

Outputs (all to OUTPUT_DIR):
  era_hba1c_thresholds_cox.csv      — main results table (abstract-ready)
  era_hba1c_thresholds_forest.png   — forest plot, two thresholds x three eras
  era_hba1c_km_threshold_7_0.png    — KM by era for <7.0%
  era_hba1c_km_threshold_5_7.png    — KM by era for <5.7%

PI: Paul Kuo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import os, warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR    = r"Z:\homes\Rachel Wolansky\Panc Pump\Data In"
OUTPUT_DIR  = r"Z:\homes\Rachel Wolansky\Panc Pump\Results"
RANDOM_SEED = 42

# Thresholds: (cutoff, descriptive label)
THRESHOLDS = [
    (7.0, 'Diabetic target'),
    (5.7, 'Normoglycemia'),
]

ERAS = {
    'Full cohort':   pd.Timestamp('2000-01-01'),
    'Past 10 years': pd.Timestamp('2016-01-01'),
    'Past 5 years':  pd.Timestamp('2021-01-01'),
}

PSM_DX_FLAGS = {
    'dm_type1':'E10', 'dm_type2':'E11', 'hypertension':'I10', 'cad':'I25',
    'heart_failure':'I50', 'pvd':'I73', 'dyslipidemia':'E78',
    'obesity':'E66', 'ckd':'N18', 'anemia':'D64',
}

HBAC1C_LOINCS = ['4548-4', '17856-6', '59261-8']
CREAT_LOINCS  = ['2160-0', '38483-4']
EGFR_LOINCS   = ['62238-1']

# =============================================================================
# LOAD FILES
# =============================================================================
print("\n" + "="*60)
print("LOADING FILES")
print("="*60)

master    = pd.read_csv(os.path.join(OUTPUT_DIR, "master_patient_table.csv"), low_memory=False)
lab_clean = pd.read_csv(os.path.join(OUTPUT_DIR, "lab_result_clean.csv"),     low_memory=False)
diagnosis = pd.read_csv(os.path.join(DATA_DIR,   "diagnosis.csv"),             low_memory=False)

master['index_date'] = pd.to_datetime(master['index_date'], errors='coerce')
lab_clean['date']    = pd.to_datetime(lab_clean['date'],    errors='coerce')
diagnosis['date']    = pd.to_datetime(diagnosis['date'].astype(str),
                                       format='%Y%m%d', errors='coerce')

print(f"  Master: {len(master):,}  Labs: {len(lab_clean):,}  Dx: {len(diagnosis):,}")

# =============================================================================
# PSM HELPERS  (identical to Script 04b — keeps methodology consistent)
# =============================================================================
def flag_dx_before_index(code_prefix, df_dx, df_pts):
    idx_map = df_pts.set_index('patient_id')['index_date'].to_dict()
    pid_set = set(df_pts['patient_id'])
    hits = df_dx[df_dx['code'].str.startswith(code_prefix, na=False) &
                 df_dx['patient_id'].isin(pid_set)].copy()
    hits['idx'] = hits['patient_id'].map(idx_map)
    hits = hits[hits['date'] < hits['idx']]
    return df_pts['patient_id'].isin(set(hits['patient_id'])).astype(int).values

def get_baseline_lab(loinc_list, df_lab, df_pts, window=365):
    idx_map = df_pts.set_index('patient_id')['index_date'].to_dict()
    pid_set = set(df_pts['patient_id'])
    hits = df_lab[df_lab['code'].isin(loinc_list) &
                  df_lab['patient_id'].isin(pid_set) &
                  df_lab['lab_result_num_val'].notna()].copy()
    hits['idx']  = hits['patient_id'].map(idx_map)
    hits['days'] = (hits['date'] - hits['idx']).dt.days
    hits = hits[(hits['days'] >= -window) & (hits['days'] <= 0)]
    latest = (hits.sort_values('days')
                  .groupby('patient_id')['lab_result_num_val']
                  .last().reset_index())
    result = df_pts[['patient_id']].merge(latest, on='patient_id', how='left')
    return result['lab_result_num_val'].values

def build_covariates(df_pts):
    cov = df_pts[['patient_id','cohort','index_date','age_at_index',
                  'sex','race','ethnicity']].copy()
    cov['sex_male']           = (cov['sex']=='M').astype(int)
    cov['race_white']         = cov['race'].str.contains('White', na=False).astype(int)
    cov['race_black']         = cov['race'].str.contains('Black', na=False).astype(int)
    cov['race_asian']         = cov['race'].str.contains('Asian', na=False).astype(int)
    cov['ethnicity_hispanic'] = cov['ethnicity'].str.contains('Hispanic', na=False).astype(int)
    for col, prefix in PSM_DX_FLAGS.items():
        cov[col] = flag_dx_before_index(prefix, diagnosis, df_pts)
    cov['baseline_hba1c']      = get_baseline_lab(HBAC1C_LOINCS, lab_clean, df_pts)
    cov['baseline_creatinine'] = get_baseline_lab(CREAT_LOINCS,  lab_clean, df_pts)
    cov['baseline_egfr']       = get_baseline_lab(EGFR_LOINCS,   lab_clean, df_pts)
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
    X    = cov_df[psm_cols].values.astype(float)
    y    = (cov_df['cohort']=='KTP').astype(int).values
    X_sc = StandardScaler().fit_transform(X)
    lr   = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
    lr.fit(X_sc, y)
    ps   = lr.predict_proba(X_sc)[:,1]
    cov_df = cov_df.copy()
    cov_df['logit_ps'] = np.log(ps / (1 - ps + 1e-10))
    caliper = caliper_sd * cov_df['logit_ps'].std()
    ktp_df  = cov_df[cov_df['cohort']=='KTP'].sample(frac=1, random_state=seed).reset_index(drop=True)
    spk_df  = cov_df[cov_df['cohort']=='SPK'].reset_index(drop=True)
    matched, used = [], set()
    for _, row in ktp_df.iterrows():
        cands  = spk_df[~spk_df['patient_id'].isin(used)].copy()
        cands['diff'] = abs(cands['logit_ps'] - row['logit_ps'])
        within = cands[cands['diff'] <= caliper]
        if len(within) == 0: continue
        best = within.nsmallest(1,'diff').iloc[0]
        used.add(best['patient_id'])
        matched.append({'ktp_patient_id': row['patient_id'],
                        'spk_patient_id': best['patient_id']})
    return pd.DataFrame(matched)

# =============================================================================
# THRESHOLD-PARAMETERISED COX/KM
# =============================================================================
def run_cox_km(threshold, matched_ids, idx_map, c_map):
    """Time-to-first HbA1c < threshold. Returns dict or None if too few events."""
    hba1c = lab_clean[
        lab_clean['code'].isin(HBAC1C_LOINCS) &
        lab_clean['patient_id'].isin(matched_ids) &
        lab_clean['lab_result_num_val'].notna()
    ].copy()
    hba1c['idx']       = hba1c['patient_id'].map(idx_map)
    hba1c['days_post'] = (hba1c['date'] - hba1c['idx']).dt.days
    hba1c_post = hba1c[hba1c['days_post'] > 0].copy()

    tte_rows = []
    for pid, grp in hba1c_post.groupby('patient_id'):
        grp    = grp.sort_values('days_post')
        events = grp[grp['lab_result_num_val'] < threshold]
        last_t = grp['days_post'].max()
        if len(events) > 0:
            tte_rows.append({'patient_id': pid, 'cohort': c_map.get(pid),
                             'time': int(events['days_post'].iloc[0]), 'event': 1})
        else:
            tte_rows.append({'patient_id': pid, 'cohort': c_map.get(pid),
                             'time': max(int(last_t), 1), 'event': 0})

    tte_df = pd.DataFrame(tte_rows)
    tte_df = tte_df[tte_df['patient_id'].isin(matched_ids)].copy()
    tte_df['cohort_bin'] = (tte_df['cohort'] == 'KTP').astype(int)
    tte_df = tte_df[tte_df['time'] > 0]

    ktp_t = tte_df[tte_df['cohort']=='KTP']
    spk_t = tte_df[tte_df['cohort']=='SPK']

    if len(ktp_t) < 5 or len(spk_t) < 5:
        return None

    lr = logrank_test(ktp_t['time'], spk_t['time'],
                      event_observed_A=ktp_t['event'],
                      event_observed_B=spk_t['event'])

    hr = ci_lo = ci_hi = cox_p = np.nan
    try:
        cph = CoxPHFitter()
        cph.fit(tte_df[['time','event','cohort_bin']],
                duration_col='time', event_col='event')
        hr    = float(np.exp(cph.params_['cohort_bin']))
        ci_lo = float(np.exp(cph.confidence_intervals_['95% lower-bound']['cohort_bin']))
        ci_hi = float(np.exp(cph.confidence_intervals_['95% upper-bound']['cohort_bin']))
        cox_p = float(cph.summary['p']['cohort_bin'])
    except Exception as e:
        print(f"      Cox failed for threshold {threshold}: "
              f"{e.__class__.__name__} ({e})")
        cox_p = lr.p_value

    return {
        'tte_df':     tte_df,
        'n_ktp':      len(ktp_t),
        'n_spk':      len(spk_t),
        'events_ktp': int(ktp_t['event'].sum()),
        'events_spk': int(spk_t['event'].sum()),
        'rate_ktp':   round(ktp_t['event'].mean()*100, 1),
        'rate_spk':   round(spk_t['event'].mean()*100, 1),
        'logrank_p':  round(lr.p_value, 4),
        'hr':         round(hr,    3) if not np.isnan(hr)    else np.nan,
        'ci_lo':      round(ci_lo, 3) if not np.isnan(ci_lo) else np.nan,
        'ci_hi':      round(ci_hi, 3) if not np.isnan(ci_hi) else np.nan,
        'cox_p':      round(cox_p, 4) if not np.isnan(cox_p) else np.nan,
    }

# =============================================================================
# MAIN — RUN BY ERA, THEN BY THRESHOLD
# =============================================================================
all_results = []
era_data    = {}

for era_label, cutoff in ERAS.items():
    print("\n" + "="*60)
    print(f"ERA: {era_label.upper()}  (index >= {cutoff.date()})")
    print("="*60)

    era_pts   = master[master['index_date'] >= cutoff].copy()
    n_ktp_raw = (era_pts['cohort']=='KTP').sum()
    n_spk_raw = (era_pts['cohort']=='SPK').sum()
    print(f"  Raw: KTP={n_ktp_raw}, SPK={n_spk_raw}")

    if n_ktp_raw < 20:
        print(f"  Skipping — too few KTP patients ({n_ktp_raw})")
        continue

    print("  Building covariates and running PSM...")
    cov_df   = build_covariates(era_pts)
    pairs_df = run_psm(cov_df)
    n_pairs  = len(pairs_df)
    print(f"  Matched pairs: {n_pairs}")

    matched_ktp = set(pairs_df['ktp_patient_id'])
    matched_spk = set(pairs_df['spk_patient_id'])
    matched_ids = matched_ktp | matched_spk
    idx_map     = era_pts.set_index('patient_id')['index_date'].to_dict()
    c_map       = {p:'KTP' for p in matched_ktp}
    c_map.update({p:'SPK' for p in matched_spk})

    era_data[era_label] = {'pairs': n_pairs, 'tte_by_thr': {}}

    for threshold, thr_label in THRESHOLDS:
        print(f"\n  Threshold: HbA1c < {threshold}%  ({thr_label})")
        result = run_cox_km(threshold, matched_ids, idx_map, c_map)

        if result is None:
            print(f"    Too few events — skipping")
            continue

        sig = '*' if (result['cox_p'] or 1) < 0.05 else ''
        print(f"    KTP: n={result['n_ktp']}  events={result['events_ktp']} ({result['rate_ktp']}%)")
        print(f"    SPK: n={result['n_spk']}  events={result['events_spk']} ({result['rate_spk']}%)")
        print(f"    Log-rank p = {result['logrank_p']:.4f}")
        print(f"    Cox HR (KTP vs SPK) = {result['hr']:.3f}  "
              f"[{result['ci_lo']:.3f}, {result['ci_hi']:.3f}]  p={result['cox_p']:.4f}{sig}")

        era_data[era_label]['tte_by_thr'][threshold] = result['tte_df']

        all_results.append({
            'Era':              era_label,
            'Threshold':        f'<{threshold}%',
            'Threshold_label':  thr_label,
            'N_pairs':          n_pairs,
            'N_KTP':            result['n_ktp'],
            'N_SPK':            result['n_spk'],
            'Events_KTP':       result['events_ktp'],
            'Events_SPK':       result['events_spk'],
            'Pct_KTP_reached':  result['rate_ktp'],
            'Pct_SPK_reached':  result['rate_spk'],
            'HR':               result['hr'],
            'CI_lo':            result['ci_lo'],
            'CI_hi':            result['ci_hi'],
            'Cox_p':            result['cox_p'],
            'Logrank_p':        result['logrank_p'],
            'Significant':      '*' if (result['cox_p'] or 1) < 0.05 else '',
        })

# =============================================================================
# RESULTS TABLE
# =============================================================================
print("\n" + "="*60)
print("RESULTS TABLE")
print("="*60)

results_df = pd.DataFrame(all_results)
out_path   = os.path.join(OUTPUT_DIR, "era_hba1c_thresholds_cox.csv")
results_df.to_csv(out_path, index=False)

print("\n" + results_df[['Era','Threshold','N_pairs',
                          'Pct_KTP_reached','Pct_SPK_reached',
                          'HR','CI_lo','CI_hi','Cox_p','Significant']].to_string(index=False))
print(f"\n  -> Saved: era_hba1c_thresholds_cox.csv")

# =============================================================================
# ABSTRACT-READY FORMAT (paste directly into Table 1)
# =============================================================================
print("\n" + "="*60)
print("ABSTRACT TABLE FORMAT — paste these rows into Table 1")
print("="*60)

for threshold, thr_label in THRESHOLDS:
    pretty = f"<{threshold}%"
    label  = f"HbA1c Cox HR {pretty} ({thr_label.lower()})"
    print(f"\n  {label}:")
    for era_label in ERAS.keys():
        row = results_df[(results_df['Era']==era_label) &
                          (results_df['Threshold']==pretty)]
        if len(row)==0: continue
        r = row.iloc[0]
        p_str = '<0.001' if r['Cox_p'] < 0.001 else f"{r['Cox_p']:.3f}"
        sig   = '*' if r['Significant'] == '*' else ''
        print(f"    {era_label:15s}: {r['HR']:.3f} (p={p_str}{sig})  "
              f"KTP {r['Pct_KTP_reached']}% vs SPK {r['Pct_SPK_reached']}%")

# =============================================================================
# FOREST PLOT — HR by era and threshold
# =============================================================================
print("\n" + "="*60)
print("GENERATING FOREST PLOT")
print("="*60)

fig, ax = plt.subplots(figsize=(12, 7))
era_list = [e for e in ERAS.keys() if e in era_data]
offsets  = {7.0: -0.18, 5.7: 0.18}
colors   = {7.0: '#2196F3', 5.7: '#FF5722'}

for i, era_label in enumerate(era_list):
    for threshold, thr_label in THRESHOLDS:
        row = results_df[(results_df['Era']==era_label) &
                          (results_df['Threshold']==f'<{threshold}%')]
        if len(row)==0: continue
        r = row.iloc[0]
        y = (len(era_list) - i) + offsets[threshold]
        ax.plot(r['HR'], y, 'o', color=colors[threshold], markersize=10, zorder=3)
        ax.plot([r['CI_lo'], r['CI_hi']], [y, y], '-',
                color=colors[threshold], linewidth=2, zorder=2)
        if r['Significant'] == '*':
            ax.plot(r['HR'], y, 'o', color=colors[threshold], markersize=14,
                    markerfacecolor='none', markeredgewidth=2, zorder=4)
        p_str = '<0.001' if r['Cox_p'] < 0.001 else f"{r['Cox_p']:.3f}"
        ax.text(max(r['CI_hi'], 1.5) + 0.04, y,
                f"HR={r['HR']:.2f} [{r['CI_lo']:.2f},{r['CI_hi']:.2f}]  p={p_str}",
                va='center', fontsize=8.5, color=colors[threshold])

ax.axvline(1.0, color='black', linewidth=1.2, linestyle='--', alpha=0.6)
ax.set_yticks(list(range(1, len(era_list)+1)))
ax.set_yticklabels(list(reversed(era_list)), fontsize=10)
ax.set_xlabel('Hazard Ratio — Time to HbA1c Threshold (KTP vs SPK)', fontsize=11)
ax.set_title('Era-Stratified HbA1c Cox HR\nDiabetic Target (<7.0%) vs Normoglycemia (<5.7%)\n'
             'HR <1 = KTP slower to reach target',
             fontsize=12, fontweight='bold')
ax.set_xlim(0.05, 2.5)
ax.grid(axis='x', alpha=0.3)

legend_elements = [
    Line2D([0],[0], color=colors[7.0], marker='o', linewidth=2, markersize=8,
            label='<7.0% diabetic target'),
    Line2D([0],[0], color=colors[5.7], marker='o', linewidth=2, markersize=8,
            label='<5.7% normoglycemia'),
    Line2D([0],[0], color='gray', marker='o', linewidth=0, markersize=12,
            markerfacecolor='none', markeredgewidth=2, label='p<0.05'),
]
ax.legend(handles=legend_elements, fontsize=9, loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "era_hba1c_thresholds_forest.png"), dpi=150)
plt.close()
print("  -> Saved: era_hba1c_thresholds_forest.png")

# =============================================================================
# KM PLOTS — one figure per threshold, eras side by side
# =============================================================================
ck = ['#1565C0','#2196F3','#90CAF9']
cs = ['#BF360C','#FF5722','#FFCCBC']

for threshold, thr_label in THRESHOLDS:
    fig, axes = plt.subplots(1, len(era_list), figsize=(6*len(era_list), 6))
    if len(era_list) == 1:
        axes = [axes]
    fig.suptitle(f'Time to HbA1c <{threshold}% by Era: KTP vs SPK',
                  fontsize=14, fontweight='bold')

    for i, era_label in enumerate(era_list):
        ax = axes[i]
        if (era_label not in era_data or
            threshold not in era_data[era_label]['tte_by_thr']):
            ax.set_title(f'{era_label}\nNo data', fontsize=11)
            continue
        tte   = era_data[era_label]['tte_by_thr'][threshold]
        ktp_t = tte[tte['cohort']=='KTP']
        spk_t = tte[tte['cohort']=='SPK']
        row   = results_df[(results_df['Era']==era_label) &
                            (results_df['Threshold']==f'<{threshold}%')].iloc[0]

        kmf_k = KaplanMeierFitter(label='KTP')
        kmf_s = KaplanMeierFitter(label='SPK')
        kmf_k.fit(ktp_t['time'], event_observed=ktp_t['event'])
        kmf_s.fit(spk_t['time'], event_observed=spk_t['event'])
        kmf_k.plot_survival_function(ax=ax, color=ck[i], linewidth=2,
                                      ci_show=True, ci_alpha=0.12)
        kmf_s.plot_survival_function(ax=ax, color=cs[i], linewidth=2,
                                      ci_show=True, ci_alpha=0.12)

        p_str = '<0.001' if row['Logrank_p'] < 0.001 else f"={row['Logrank_p']:.4f}"
        ax.text(0.50, 0.86,
                f"Log-rank p{p_str}\nHR={row['HR']:.2f} "
                f"[{row['CI_lo']:.2f},{row['CI_hi']:.2f}]",
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        ax.set_title(f'{era_label}\n(n={era_data[era_label]["pairs"]} pairs)',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Days Post-Index')
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    axes[0].set_ylabel(f'P(Not reaching HbA1c <{threshold}%)')
    plt.tight_layout()
    fn = f"era_hba1c_km_threshold_{str(threshold).replace('.','_')}.png"
    plt.savefig(os.path.join(OUTPUT_DIR, fn), dpi=150)
    plt.close()
    print(f"  -> Saved: {fn}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*60)
print("SCRIPT 04c COMPLETE")
print("="*60)
print(f"""
Outputs:
  era_hba1c_thresholds_cox.csv      <- main results, abstract-ready
  era_hba1c_thresholds_forest.png   <- forest plot
  era_hba1c_km_threshold_7_0.png    <- KM curves <7.0% by era
  era_hba1c_km_threshold_5_7.png    <- KM curves <5.7% by era

Use the printed "ABSTRACT TABLE FORMAT" block above to replace the
HbA1c <7.0% and HbA1c <5.7% rows in your Table 1.

Note: this script independently re-runs PSM for each era (matching
Script 04b methodology). The Full Cohort pair count may differ
slightly from Script 04 (306) because Script 04 used the saved
psm_matched_master from Script 03 rather than re-matching.
""")
