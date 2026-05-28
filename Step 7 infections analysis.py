"""
TriNetX PancPump Study - Script 07: Expanded Opportunistic Infections

Era-stratified Cox HRs for the expanded opportunistic infection composite,
replacing the prior sepsis+CMV+BK composite (which used B34.2, actually
coronavirus in ICD-10-CM). Components:
  Sepsis                A41
  CMV                   B25
  BK/polyomavirus       B97.89
  PJP                   B59
  Aspergillosis         B44
  Invasive candidiasis  B37.5, B37.7, B37.81
  Tuberculosis          A15-A19
  Cryptococcosis        B45
  Herpes zoster         B02

Unbounded Cox follow-up (chronic immunosuppression risk).

Also saves matched pair lists per era to OUTPUT_DIR so Script 08 (LOS)
and Script 09 (mortality) can skip the expensive PSM step.

PI: Paul Kuo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
import os, warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR    = r"Z:\homes\Rachel Wolansky\Panc Pump\Data In"
OUTPUT_DIR  = r"Z:\homes\Rachel Wolansky\Panc Pump\Results"
RANDOM_SEED = 42

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

EXPANDED_SERIOUS_INFECTIONS = {
    'Sepsis':              ['A41'],
    'CMV':                 ['B25'],
    'BK/polyomavirus':     ['B97.89'],
    'PJP':                 ['B59'],
    'Aspergillosis':       ['B44'],
    'Invasive candidiasis':['B37.5', 'B37.7', 'B37.81'],
    'Tuberculosis':        ['A15', 'A16', 'A17', 'A18', 'A19'],
    'Cryptococcosis':      ['B45'],
    'Herpes zoster':       ['B02'],
}
SERIOUS_INFECTION_COMPOSITE = sorted(
    {c for codes in EXPANDED_SERIOUS_INFECTIONS.values() for c in codes}
)

def era_slug(era_label):
    return era_label.lower().replace(' ', '_')

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
diagnosis['date']    = pd.to_datetime(diagnosis['date'].astype(str), format='%Y%m%d', errors='coerce')

print(f"  Master: {len(master):,}  Labs: {len(lab_clean):,}  Dx: {len(diagnosis):,}")

# =============================================================================
# PSM HELPERS
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
# OUTCOME EXTRACTORS
# =============================================================================
def flag_dx_events_post_index(code_list, pid_set, idx_map):
    if not code_list:
        return {}
    mask = diagnosis['patient_id'].isin(pid_set)
    code_mask = diagnosis['code'].str.startswith(code_list[0], na=False)
    for code in code_list[1:]:
        code_mask = code_mask | diagnosis['code'].str.startswith(code, na=False)
    hits = diagnosis[mask & code_mask].copy()
    hits['idx']       = hits['patient_id'].map(idx_map)
    hits['days_post'] = (hits['date'] - hits['idx']).dt.days
    hits = hits[hits['days_post'] > 0]
    return hits.groupby('patient_id')['days_post'].min().to_dict()

def build_tte_unbounded(event_dict, pid_set, idx_map, max_days=3650):
    dx_m = diagnosis[diagnosis['patient_id'].isin(pid_set)].copy()
    dx_m['idx'] = dx_m['patient_id'].map(idx_map)
    dx_m['dp']  = (dx_m['date'] - dx_m['idx']).dt.days
    last_obs = dx_m[dx_m['dp']>0].groupby('patient_id')['dp'].max().to_dict()
    rows = []
    for pid in pid_set:
        if pid in event_dict:
            rows.append({'patient_id': pid,
                         'time': min(int(event_dict[pid]), max_days),
                         'event': 1})
        else:
            rows.append({'patient_id': pid,
                         'time': max(int(last_obs.get(pid, 365)), 1),
                         'event': 0})
    return pd.DataFrame(rows)

def run_km_cox(tte_df, cohort_map):
    tte_df = tte_df[tte_df['time']>0].copy()
    tte_df['cohort']     = tte_df['patient_id'].map(cohort_map)
    tte_df['cohort_bin'] = (tte_df['cohort']=='KTP').astype(int)
    ktp_t = tte_df[tte_df['cohort']=='KTP']
    spk_t = tte_df[tte_df['cohort']=='SPK']
    if len(ktp_t)==0 or len(spk_t)==0 or (ktp_t['event'].sum() + spk_t['event'].sum()) < 5:
        return dict(events_ktp=int(ktp_t['event'].sum()) if len(ktp_t)>0 else 0,
                    events_spk=int(spk_t['event'].sum()) if len(spk_t)>0 else 0,
                    rate_ktp=np.nan, rate_spk=np.nan, logrank_p=np.nan,
                    hr=np.nan, ci_lo=np.nan, ci_hi=np.nan, cox_p=np.nan,
                    sparse=True)
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
    except Exception:
        cox_p = lr.p_value
    return dict(
        n_ktp=len(ktp_t), n_spk=len(spk_t),
        events_ktp=int(ktp_t['event'].sum()),
        events_spk=int(spk_t['event'].sum()),
        rate_ktp=round(ktp_t['event'].mean()*100, 1),
        rate_spk=round(spk_t['event'].mean()*100, 1),
        logrank_p=round(lr.p_value, 4),
        hr=round(hr, 3) if not np.isnan(hr) else np.nan,
        ci_lo=round(ci_lo, 3) if not np.isnan(ci_lo) else np.nan,
        ci_hi=round(ci_hi, 3) if not np.isnan(ci_hi) else np.nan,
        cox_p=round(cox_p, 4) if not np.isnan(cox_p) else np.nan,
        sparse=False,
    )

# =============================================================================
# MAIN LOOP
# =============================================================================
all_results = []

for era_label, cutoff in ERAS.items():
    print("\n" + "="*60)
    print(f"ERA: {era_label.upper()}  (index >= {cutoff.date()})")
    print("="*60)

    era_pts = master[master['index_date'] >= cutoff].copy()
    n_ktp_raw = (era_pts['cohort']=='KTP').sum()
    n_spk_raw = (era_pts['cohort']=='SPK').sum()
    print(f"  Raw: KTP={n_ktp_raw}, SPK={n_spk_raw}")

    if n_ktp_raw < 20:
        print(f"  Skipping — too few KTP patients")
        continue

    print("  Building covariates and running PSM...")
    cov_df   = build_covariates(era_pts)
    pairs_df = run_psm(cov_df)
    n_pairs  = len(pairs_df)
    print(f"  Matched pairs: {n_pairs}")

    # SAVE PAIRS CACHE for Scripts 08 and 09
    cache_path = os.path.join(OUTPUT_DIR, f"psm_pairs_{era_slug(era_label)}.csv")
    pairs_df.to_csv(cache_path, index=False)
    print(f"  -> Saved pair cache: {os.path.basename(cache_path)}")

    matched_ktp = set(pairs_df['ktp_patient_id'])
    matched_spk = set(pairs_df['spk_patient_id'])
    matched_ids = matched_ktp | matched_spk
    idx_map     = era_pts.set_index('patient_id')['index_date'].to_dict()
    c_map       = {p:'KTP' for p in matched_ktp}
    c_map.update({p:'SPK' for p in matched_spk})

    # Infection components
    print("\n  ----- Expanded serious infection components -----")
    for label, codes in EXPANDED_SERIOUS_INFECTIONS.items():
        event_dict = flag_dx_events_post_index(codes, matched_ids, idx_map)
        tte        = build_tte_unbounded(event_dict, matched_ids, idx_map)
        res        = run_km_cox(tte, c_map)
        all_results.append({'Era': era_label, 'N_pairs': n_pairs,
                            'Outcome_group': 'Infection component',
                            'Outcome': label, **res})
        if not res.get('sparse'):
            sig = '*' if (res.get('cox_p') or 1) < 0.05 else ''
            print(f"    {label:24s}: KTP={res.get('rate_ktp','?')}%  "
                  f"SPK={res.get('rate_spk','?')}%  HR={res.get('hr','?')}  "
                  f"p={res.get('cox_p','?')}{sig}")
        else:
            print(f"    {label:24s}: too few events "
                  f"(KTP={res['events_ktp']}, SPK={res['events_spk']})")

    # Composite
    print("\n  ----- EXPANDED SERIOUS INFECTION COMPOSITE -----")
    event_dict = flag_dx_events_post_index(SERIOUS_INFECTION_COMPOSITE, matched_ids, idx_map)
    tte        = build_tte_unbounded(event_dict, matched_ids, idx_map)
    res        = run_km_cox(tte, c_map)
    all_results.append({'Era': era_label, 'N_pairs': n_pairs,
                        'Outcome_group': 'Composite',
                        'Outcome': 'Serious infections (expanded)', **res})
    sig = '*' if (res.get('cox_p') or 1) < 0.05 else ''
    print(f"    EXPANDED COMPOSITE: KTP={res.get('rate_ktp','?')}%  "
          f"SPK={res.get('rate_spk','?')}%  HR={res.get('hr','?')} "
          f"[{res.get('ci_lo','?')},{res.get('ci_hi','?')}]  "
          f"p={res.get('cox_p','?')}{sig}")

# =============================================================================
# SAVE + REPORT
# =============================================================================
print("\n" + "="*60)
print("SAVING")
print("="*60)

results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(OUTPUT_DIR, "expanded_infections_results.csv"),
                  index=False)
print("  -> Saved: expanded_infections_results.csv")

print("\n" + "="*60)
print("ABSTRACT TABLE FORMAT")
print("="*60)

composites = results_df[results_df['Outcome_group']=='Composite']
print("\n  Serious infections (expanded) HR:")
for era_label in ERAS.keys():
    row = composites[(composites['Outcome']=='Serious infections (expanded)') &
                     (composites['Era']==era_label)]
    if len(row)==0: continue
    r = row.iloc[0]
    if pd.isna(r['hr']):
        print(f"    {era_label:15s}: sparse"); continue
    p_str = '<0.001' if r['cox_p'] < 0.001 else f"{r['cox_p']:.3f}"
    sig   = '*' if r['cox_p'] < 0.05 else ''
    print(f"    {era_label:15s}: {r['hr']:.3f} (p={p_str}{sig})  "
          f"KTP {r['rate_ktp']}% vs SPK {r['rate_spk']}%")

# Forest plot
plot_outcomes = ['Serious infections (expanded)', 'Sepsis', 'CMV',
                  'BK/polyomavirus', 'PJP', 'Aspergillosis',
                  'Invasive candidiasis', 'Herpes zoster']
era_colors = {'Full cohort':'#1565C0', 'Past 10 years':'#2196F3', 'Past 5 years':'#90CAF9'}
offsets    = {'Full cohort':-0.22, 'Past 10 years':0, 'Past 5 years':0.22}

fig, ax = plt.subplots(figsize=(12, 9))
yticks, ylabels = [], []
y_pos = 0
for outcome in reversed(plot_outcomes):
    y_pos += 1; plotted = False
    for era_label in ERAS.keys():
        row = results_df[(results_df['Outcome']==outcome) &
                          (results_df['Era']==era_label)]
        if len(row)==0: continue
        r = row.iloc[0]
        if pd.isna(r.get('hr', np.nan)): continue
        yy = y_pos + offsets[era_label]; color = era_colors[era_label]
        ax.plot(r['hr'], yy, 'o', color=color, markersize=7, zorder=3)
        ax.plot([r['ci_lo'], r['ci_hi']], [yy, yy], '-',
                color=color, linewidth=1.5, zorder=2)
        if r['cox_p'] < 0.05:
            ax.plot(r['hr'], yy, 'o', color=color, markersize=11,
                    markerfacecolor='none', markeredgewidth=1.5, zorder=4)
        plotted = True
    if plotted: yticks.append(y_pos); ylabels.append(outcome)

ax.axvline(1.0, color='black', linewidth=1.2, linestyle='--', alpha=0.6)
ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=10)
ax.set_xlabel('Hazard Ratio (KTP vs SPK)', fontsize=11)
ax.set_title('Expanded Serious Infections by Era\n(HR<1 favors KTP; circles = p<0.05)',
             fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
legend_elements = [Line2D([0],[0], color=era_colors[e], marker='o', linewidth=2,
                           markersize=7, label=e) for e in ERAS.keys()]
legend_elements.append(Line2D([0],[0], color='gray', marker='o', linewidth=0,
                                markersize=11, markerfacecolor='none',
                                markeredgewidth=1.5, label='p<0.05'))
ax.legend(handles=legend_elements, fontsize=9, loc='lower right')
ax.set_xlim(0.1, 5); ax.set_xscale('log')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "expanded_infections_forest.png"), dpi=150)
plt.close()
print("  -> Saved: expanded_infections_forest.png")

print("\n" + "="*60)
print("SCRIPT 07 COMPLETE")
print("="*60)
print("""
Outputs:
  expanded_infections_results.csv
  expanded_infections_forest.png
  psm_pairs_full_cohort.csv     <- pair cache for Scripts 08 and 09
  psm_pairs_past_10_years.csv
  psm_pairs_past_5_years.csv

Scripts 08 (LOS) and 09 (mortality) will load these pair caches and
skip the PSM step entirely, making iteration much faster.
""")
