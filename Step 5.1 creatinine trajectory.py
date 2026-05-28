"""
TriNetX PancPump Study - Script 05d: Serum Creatinine Analysis
Serial means (mirrors eGFR in Script 05 Section A) + Cox HR for time-to-Cr ≥1.5 mg/dL
Era-stratified: Full cohort, Past 10 years, Past 5 years
PI: Paul Kuo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import datetime, os, warnings
warnings.filterwarnings('ignore')

DATA_DIR   = r"Z:\homes\Rachel Wolansky\Panc Pump\Data In"
OUTPUT_DIR = r"Z:\homes\Rachel Wolansky\Panc Pump\Results"

# Creatinine LOINC codes
# 2160-0  = Creatinine [Mass/volume] in Serum or Plasma  (primary)
# 38483-4 = Creatinine [Mass/volume] in Blood            (fallback)
CREATININE_LOINCS = ['2160-0', '38483-4']

# Cox threshold
CR_THRESHOLD = 2.0   # mg/dL

# Era cutoffs
TODAY    = datetime.date.today()
CUT_10YR = pd.Timestamp(TODAY) - pd.DateOffset(years=10)
CUT_5YR  = pd.Timestamp(TODAY) - pd.DateOffset(years=5)

INTERVALS = [
    ('3 months',   91,  45),
    ('6 months',  182,  60),
    ('1 year',    365,  90),
    ('2 years',   730, 120),
    ('3 years',  1095, 150),
    ('5 years',  1825, 180),
]

# =============================================================================
# LOAD FILES
# =============================================================================
print("\n" + "="*60)
print("LOADING FILES")
print("="*60)

matched_master = pd.read_csv(os.path.join(OUTPUT_DIR, "psm_matched_master.csv"), low_memory=False)
matched_pairs  = pd.read_csv(os.path.join(OUTPUT_DIR, "psm_matched_pairs.csv"),  low_memory=False)
lab_clean      = pd.read_csv(os.path.join(OUTPUT_DIR, "lab_result_clean.csv"),   low_memory=False)
diagnosis      = pd.read_csv(os.path.join(DATA_DIR,   "diagnosis.csv"),           low_memory=False)

matched_master['index_date'] = pd.to_datetime(matched_master['index_date'], errors='coerce')
lab_clean['date']            = pd.to_datetime(lab_clean['date'],            errors='coerce')
diagnosis['date']            = pd.to_datetime(diagnosis['date'].astype(str), format='%Y%m%d', errors='coerce')

matched_ids = set(matched_master['patient_id'])
idx_dates   = matched_master.set_index('patient_id')['index_date'].to_dict()
cohort_map  = matched_master.set_index('patient_id')['cohort'].to_dict()

print(f"  Matched patients: {len(matched_master):,}  "
      f"(KTP={(matched_master['cohort']=='KTP').sum()}, "
      f"SPK={(matched_master['cohort']=='SPK').sum()})")

# =============================================================================
# FILTER CREATININE LABS
# =============================================================================
print("\n" + "="*60)
print("CREATININE LAB AUDIT")
print("="*60)

# Check what creatinine codes exist in the dataset
cr_codes_found = lab_clean[lab_clean['code'].isin(CREATININE_LOINCS)]['code'].value_counts()
print(f"\nCreatinine LOINC codes found:")
print(cr_codes_found.to_string())

# If primary LOINC not found, search by description
if len(cr_codes_found) == 0:
    print("\n  Primary LOINCs not found — searching for creatinine by description...")
    if 'code_description' in lab_clean.columns:
        cr_desc = lab_clean[lab_clean['code_description'].str.contains(
            'creatinine', case=False, na=False)]['code'].value_counts().head(10)
        print(cr_desc.to_string())
        CREATININE_LOINCS = list(cr_desc.index)

cr_all = lab_clean[
    lab_clean['code'].isin(CREATININE_LOINCS) &
    lab_clean['patient_id'].isin(matched_ids) &
    lab_clean['lab_result_num_val'].notna()
].copy()

cr_all['index_date'] = cr_all['patient_id'].map(idx_dates)
cr_all['cohort']     = cr_all['patient_id'].map(cohort_map)
cr_all['days_post']  = (cr_all['date'] - cr_all['index_date']).dt.days
cr_post = cr_all[cr_all['days_post'] > 0].copy()

print(f"\nPost-index creatinine rows: {len(cr_post):,}")
print(f"  KTP: {(cr_post['cohort']=='KTP').sum():,}  "
      f"SPK: {(cr_post['cohort']=='SPK').sum():,}")
print(f"  Unique patients: {cr_post['patient_id'].nunique():,}")
print(f"  Value range: {cr_post['lab_result_num_val'].min():.2f} – "
      f"{cr_post['lab_result_num_val'].max():.2f} mg/dL")
print(f"  Median: {cr_post['lab_result_num_val'].median():.2f} mg/dL")

# Flag implausible values (>20 mg/dL likely unit error)
n_outliers = (cr_post['lab_result_num_val'] > 20).sum()
if n_outliers > 0:
    print(f"\n  WARNING: {n_outliers} values >20 mg/dL — excluding as likely unit errors")
    cr_post = cr_post[cr_post['lab_result_num_val'] <= 20].copy()

# =============================================================================
# HELPERS (mirrors Script 05 structure)
# =============================================================================

def get_interval_vals(df_cohort, center, hw):
    lo  = max(center - hw, 1)
    hi  = center + hw
    sub = df_cohort[(df_cohort['days_post'] >= lo) & (df_cohort['days_post'] <= hi)].copy()
    if len(sub) == 0:
        return pd.Series(dtype=float)
    sub['dist'] = abs(sub['days_post'] - center)
    return sub.sort_values('dist').groupby('patient_id').first()['lab_result_num_val']


def era_ids(cut):
    return set(matched_master[matched_master['index_date'] >= cut]['patient_id'])


def era_pairs(pairs, era_set):
    return pairs[
        pairs['ktp_patient_id'].isin(era_set) &
        pairs['spk_patient_id'].isin(era_set)
    ]


def build_tte_cr(cr_sub, threshold, patient_set):
    """Time-to-first creatinine >= threshold. Censored at last creatinine date."""
    rows = []
    last_obs = cr_sub[cr_sub['patient_id'].isin(patient_set)].groupby(
        'patient_id')['days_post'].max().to_dict()

    for pid in patient_set:
        cohort = cohort_map.get(pid)
        grp = cr_sub[cr_sub['patient_id'] == pid].sort_values('days_post')
        events = grp[grp['lab_result_num_val'] >= threshold]
        if len(events) > 0:
            rows.append({'patient_id': pid, 'cohort': cohort,
                         'time': int(events['days_post'].iloc[0]), 'event': 1})
        else:
            rows.append({'patient_id': pid, 'cohort': cohort,
                         'time': max(int(last_obs.get(pid, 365)), 1), 'event': 0})
    df = pd.DataFrame(rows)
    df['cohort_bin'] = (df['cohort'] == 'KTP').astype(int)
    return df


def run_cox(tte_df):
    """Return hr, ci_lo, ci_hi, p."""
    cox_df = tte_df[['time', 'event', 'cohort_bin']].copy()
    cox_df = cox_df[cox_df['time'] > 0]
    if cox_df['event'].sum() < 5:
        return np.nan, np.nan, np.nan, np.nan
    try:
        cph = CoxPHFitter()
        cph.fit(cox_df, duration_col='time', event_col='event')
        hr    = float(np.exp(cph.params_['cohort_bin']))
        ci_lo = float(np.exp(cph.confidence_intervals_['95% lower-bound']['cohort_bin']))
        ci_hi = float(np.exp(cph.confidence_intervals_['95% upper-bound']['cohort_bin']))
        p     = float(cph.summary['p']['cohort_bin'])
        return hr, ci_lo, ci_hi, p
    except Exception as e:
        print(f"    Cox failed: {e}")
        return np.nan, np.nan, np.nan, np.nan


# Era definitions
ids_full = set(matched_master['patient_id'])
ids_10yr = era_ids(CUT_10YR)
ids_5yr  = era_ids(CUT_5YR)

pairs_full = matched_pairs.copy()
pairs_10yr = era_pairs(matched_pairs, ids_10yr)
pairs_5yr  = era_pairs(matched_pairs, ids_5yr)

ERAS = [
    ('Full cohort',   ids_full,  pairs_full),
    ('Past 10 years', ids_10yr,  pairs_10yr),
    ('Past 5 years',  ids_5yr,   pairs_5yr),
]

# =============================================================================
# SECTION A: SERIAL MEAN CREATININE
# =============================================================================
print("\n" + "="*60)
print("SECTION A: SERIAL MEAN CREATININE (Full Cohort)")
print("="*60)

serial_rows = []
for label, center, hw in INTERVALS:
    v_ktp = get_interval_vals(cr_post[cr_post['cohort'] == 'KTP'], center, hw)
    v_spk = get_interval_vals(cr_post[cr_post['cohort'] == 'SPK'], center, hw)
    n_k, n_s = len(v_ktp), len(v_spk)

    mean_k = v_ktp.mean() if n_k > 0 else np.nan
    mean_s = v_spk.mean() if n_s > 0 else np.nan
    sd_k   = v_ktp.std()  if n_k > 0 else np.nan
    sd_s   = v_spk.std()  if n_s > 0 else np.nan

    t_stat = p_val = np.nan
    if n_k >= 5 and n_s >= 5:
        t_stat, p_val = stats.ttest_ind(v_ktp, v_spk)

    sig = '*' if (not np.isnan(p_val) and p_val < 0.05) else ''
    print(f"  {label:10s}: KTP {mean_k:.2f} (n={n_k})  SPK {mean_s:.2f} (n={n_s})  "
          f"Diff={mean_k-mean_s:+.2f}  p={p_val:.3f}{sig}"
          if (not np.isnan(p_val)) else
          f"  {label:10s}: KTP {mean_k:.2f} (n={n_k})  SPK {mean_s:.2f} (n={n_s})  p=NA")

    serial_rows.append({
        'Interval': label, 'Center_days': center,
        'N_KTP': n_k, 'N_SPK': n_s,
        'Mean_KTP': round(float(mean_k), 2) if not np.isnan(mean_k) else np.nan,
        'SD_KTP':   round(float(sd_k),   2) if not np.isnan(sd_k)   else np.nan,
        'Mean_SPK': round(float(mean_s), 2) if not np.isnan(mean_s) else np.nan,
        'SD_SPK':   round(float(sd_s),   2) if not np.isnan(sd_s)   else np.nan,
        'Diff_KTP_SPK': round(float(mean_k - mean_s), 2)
                        if (not np.isnan(mean_k) and not np.isnan(mean_s)) else np.nan,
        'p_value': round(float(p_val), 4) if not np.isnan(p_val) else np.nan,
        'Sig': bool(p_val < 0.05) if not np.isnan(p_val) else False,
    })

serial_df = pd.DataFrame(serial_rows)
serial_df.to_csv(os.path.join(OUTPUT_DIR, "secondary_creatinine_serial.csv"), index=False)
print("  -> Saved: secondary_creatinine_serial.csv")

# =============================================================================
# SECTION B: LME TRAJECTORY (mirrors eGFR LME in Script 05)
# =============================================================================
print("\n" + "="*60)
print("SECTION B: LME CREATININE TRAJECTORY")
print("="*60)

cr_lme = cr_post.copy()
cr_lme['years_post'] = cr_lme['days_post'] / 365.25
cr_lme['is_ktp']     = (cr_lme['cohort'] == 'KTP').astype(int)
cr_lme = cr_lme.rename(columns={'lab_result_num_val': 'creatinine'})
cr_lme = cr_lme[cr_lme['creatinine'].notna()]

print(f"\nLME dataset: {len(cr_lme):,} obs, {cr_lme['patient_id'].nunique():,} patients")
try:
    model  = smf.mixedlm("creatinine ~ years_post * is_ktp", cr_lme,
                          groups=cr_lme['patient_id'])
    result = model.fit(reml=True)
    coefs  = result.fe_params
    pvals  = result.pvalues
    print(f"\n  Intercept (SPK at t=0):       {coefs['Intercept']:.3f} mg/dL")
    print(f"  Slope SPK (per yr):           {coefs['years_post']:.4f}  p={pvals['years_post']:.4f}")
    print(f"  KTP offset at index:          {coefs['is_ktp']:.3f}  p={pvals['is_ktp']:.4f}")
    print(f"  KTP×time interaction (per yr):{coefs['years_post:is_ktp']:.4f}  "
          f"p={pvals['years_post:is_ktp']:.4f}")
    lme_summary = {
        'SPK_intercept': round(float(coefs['Intercept']), 3),
        'SPK_slope_per_yr': round(float(coefs['years_post']), 4),
        'KTP_offset': round(float(coefs['is_ktp']), 3),
        'KTP_offset_p': round(float(pvals['is_ktp']), 4),
        'KTP_x_time': round(float(coefs['years_post:is_ktp']), 4),
        'KTP_x_time_p': round(float(pvals['years_post:is_ktp']), 4),
    }
except Exception as e:
    print(f"  LME failed: {e}")
    lme_summary = {}

# =============================================================================
# SECTION C: COX — TIME TO CREATININE >= 1.5 mg/dL (ALL ERAS)
# =============================================================================
print("\n" + "="*60)
print(f"SECTION C: COX HR — TIME TO CREATININE >= {CR_THRESHOLD} mg/dL BY ERA")
print("="*60)

cox_results = []

for era_label, era_set, era_pairs_df in ERAS:
    cr_era = cr_post[cr_post['patient_id'].isin(era_set)].copy()
    n_ktp  = (matched_master[matched_master['patient_id'].isin(era_set)]['cohort'] == 'KTP').sum()
    n_spk  = (matched_master[matched_master['patient_id'].isin(era_set)]['cohort'] == 'SPK').sum()
    n_pairs = len(era_pairs_df)

    tte = build_tte_cr(cr_era, CR_THRESHOLD, era_set)
    ktp_tte = tte[tte['cohort'] == 'KTP']
    spk_tte = tte[tte['cohort'] == 'SPK']

    ktp_events = int(ktp_tte['event'].sum())
    spk_events = int(spk_tte['event'].sum())
    ktp_pct    = round(ktp_tte['event'].mean() * 100, 1)
    spk_pct    = round(spk_tte['event'].mean() * 100, 1)

    print(f"\n  {era_label} (n={n_pairs} pairs):")
    print(f"    KTP: n={len(ktp_tte)}, events={ktp_events} ({ktp_pct}%)")
    print(f"    SPK: n={len(spk_tte)}, events={spk_events} ({spk_pct}%)")

    hr, ci_lo, ci_hi, p = run_cox(tte)
    if not np.isnan(hr):
        sig = '*' if p < 0.05 else ''
        print(f"    HR (KTP vs SPK) = {hr:.3f}  [{ci_lo:.3f}, {ci_hi:.3f}]  p={p:.4f}{sig}")
    else:
        print(f"    Cox: insufficient events")

    cox_results.append({
        'Era': era_label,
        'N_pairs': n_pairs,
        'KTP_events': ktp_events,
        'SPK_events': spk_events,
        'KTP_event_pct': ktp_pct,
        'SPK_event_pct': spk_pct,
        'HR':    round(float(hr),    3) if not np.isnan(hr)    else np.nan,
        'CI_lo': round(float(ci_lo), 3) if not np.isnan(ci_lo) else np.nan,
        'CI_hi': round(float(ci_hi), 3) if not np.isnan(ci_hi) else np.nan,
        'p':     round(float(p),     4) if not np.isnan(p)     else np.nan,
    })

cox_df = pd.DataFrame(cox_results)
cox_df.to_csv(os.path.join(OUTPUT_DIR, "secondary_creatinine_cox.csv"), index=False)
print(f"\n  -> Saved: secondary_creatinine_cox.csv")

# =============================================================================
# SECTION D: KM PLOT — FULL COHORT
# =============================================================================
print("\n" + "="*60)
print("SECTION D: KM PLOT (Full cohort)")
print("="*60)

tte_full = build_tte_cr(cr_post, CR_THRESHOLD, ids_full)
ktp_full = tte_full[tte_full['cohort'] == 'KTP']
spk_full = tte_full[tte_full['cohort'] == 'SPK']

lr = logrank_test(ktp_full['time'], spk_full['time'],
                  event_observed_A=ktp_full['event'],
                  event_observed_B=spk_full['event'])

fig, ax = plt.subplots(figsize=(10, 7))
kmf_k = KaplanMeierFitter(label=f"KTP (n={len(ktp_full)}, events={ktp_full['event'].sum()})")
kmf_s = KaplanMeierFitter(label=f"SPK (n={len(spk_full)}, events={spk_full['event'].sum()})")
kmf_k.fit(ktp_full['time'], event_observed=ktp_full['event'])
kmf_s.fit(spk_full['time'], event_observed=spk_full['event'])
kmf_k.plot_survival_function(ax=ax, color='#2196F3', linewidth=2, ci_show=True, ci_alpha=0.12)
kmf_s.plot_survival_function(ax=ax, color='#FF5722', linewidth=2, ci_show=True, ci_alpha=0.12)

hr0 = cox_df[cox_df['Era'] == 'Full cohort'].iloc[0]
p_str = 'p<0.001' if lr.p_value < 0.001 else f'p={lr.p_value:.4f}'
ax.text(0.58, 0.85,
        f'Log-rank {p_str}\nHR(KTP/SPK)={hr0["HR"]:.3f} [{hr0["CI_lo"]:.3f},{hr0["CI_hi"]:.3f}]',
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
ax.set_xlabel('Days Post-Index', fontsize=12)
ax.set_ylabel(f'Probability of NOT reaching Cr ≥{CR_THRESHOLD} mg/dL', fontsize=11)
ax.set_title(f'Time to Creatinine ≥{CR_THRESHOLD} mg/dL: KTP vs SPK\n(PSM-Matched, n=306 pairs)',
             fontsize=13, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "secondary_creatinine_km.png"), dpi=150)
plt.close()
print(f"  -> Saved: secondary_creatinine_km.png")

# =============================================================================
# SECTION E: SERIAL MEANS PLOT
# =============================================================================
plot_df = serial_df[serial_df['Mean_KTP'].notna() & serial_df['Mean_SPK'].notna()]
if len(plot_df) > 0:
    x = list(range(len(plot_df)))
    fig, ax = plt.subplots(figsize=(10, 6))
    sem_k = (plot_df['SD_KTP'] / np.sqrt(plot_df['N_KTP'])).tolist()
    sem_s = (plot_df['SD_SPK'] / np.sqrt(plot_df['N_SPK'])).tolist()
    ax.errorbar(x, plot_df['Mean_KTP'], yerr=sem_k, color='#2196F3',
                marker='o', linewidth=2, markersize=7, capsize=4, label='KTP')
    ax.errorbar(x, plot_df['Mean_SPK'], yerr=sem_s, color='#FF5722',
                marker='s', linewidth=2, markersize=7, capsize=4, label='SPK')
    ax.axhline(CR_THRESHOLD, color='red', linestyle='--', alpha=0.6,
               label=f'Threshold: {CR_THRESHOLD} mg/dL')
    ax.axhline(1.2, color='green', linestyle=':', alpha=0.5, label='Normal upper limit (~1.2 mg/dL)')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df['Interval'].tolist(), rotation=30, ha='right')
    ax.set_ylabel('Mean Serum Creatinine ± SEM (mg/dL)')
    ax.set_title('Serum Creatinine Trajectory: KTP vs SPK\n(PSM-Matched, n=306 pairs)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    for i, row in enumerate(plot_df.itertuples()):
        pv = row.p_value
        if not np.isnan(pv) and pv < 0.05:
            pt = 'p<0.001' if pv < 0.001 else f'p={pv:.3f}'
            ax.text(i, max(row.Mean_KTP, row.Mean_SPK) + 0.05, pt,
                    ha='center', fontsize=7, style='italic')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "secondary_creatinine_serial_plot.png"), dpi=150)
    plt.close()
    print("  -> Saved: secondary_creatinine_serial_plot.png")

# =============================================================================
# SUMMARY FOR ABSTRACT / TABLE
# =============================================================================
print("\n" + "="*60)
print("ABSTRACT TABLE FORMAT — Creatinine")
print("="*60)

print(f"\nSerial Mean Creatinine (Full Cohort):")
for _, row in serial_df.iterrows():
    sig = '*' if row['Sig'] else ''
    print(f"  {row['Interval']:10s}: KTP {row['Mean_KTP']:.2f}  SPK {row['Mean_SPK']:.2f}  "
          f"Diff={row['Diff_KTP_SPK']:+.2f}  p={row['p_value']:.3f}{sig}"
          if not np.isnan(row['p_value']) else
          f"  {row['Interval']:10s}: KTP {row['Mean_KTP']:.2f}  SPK {row['Mean_SPK']:.2f}  p=NA")

print(f"\nCox HR — Time to Creatinine ≥{CR_THRESHOLD} mg/dL:")
print(cox_df[['Era', 'N_pairs', 'KTP_event_pct', 'SPK_event_pct', 'HR', 'CI_lo', 'CI_hi', 'p']].to_string(index=False))

print("\nDone. Outputs for abstract table:")
print(f"  - Serial creatinine means: secondary_creatinine_serial.csv")
print(f"  - Cox HR ≥{CR_THRESHOLD} mg/dL by era: secondary_creatinine_cox.csv")
print(f"  - KM plot: secondary_creatinine_km.png")
print(f"  - Serial plot: secondary_creatinine_serial_plot.png")
print("\nNext: add creatinine row to abstract table after eGFR row.")
