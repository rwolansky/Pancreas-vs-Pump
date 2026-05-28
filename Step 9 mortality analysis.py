"""
TriNetX PancPump Study - Script 09: All-Cause Mortality

Loads matched pair lists from Script 07 (psm_pairs_*.csv) and skips PSM.
Only requires:
  - master_patient_table.csv (for index dates)
  - patient.csv (for month_year_death)
  - diagnosis.csv (for censoring on last observed encounter)

Death dates parsed from patient.csv month_year_death column (YYYYMM
integer; e.g., 202208 = August 2022). TriNetX deidentifies day, so we
use the 15th of the month as a midpoint estimate. All time points
carry ~15-day uncertainty.

Cox HRs at:
  30-day (hard censor at 30d)
  90-day (hard censor at 90d)
  1-year (hard censor at 365d)
  Overall (unbounded)

In-hospital mortality NOT computed (requires day-level precision).

PI: Paul Kuo
"""

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
import os, warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR    = r"Z:\homes\Rachel Wolansky\Panc Pump\Data In"
OUTPUT_DIR  = r"Z:\homes\Rachel Wolansky\Panc Pump\Results"

ERAS = {
    'Full cohort':   pd.Timestamp('2000-01-01'),
    'Past 10 years': pd.Timestamp('2016-01-01'),
    'Past 5 years':  pd.Timestamp('2021-01-01'),
}

def era_slug(era_label):
    return era_label.lower().replace(' ', '_')

# =============================================================================
# LOAD MASTER + PAIRS
# =============================================================================
print("\n" + "="*60)
print("LOADING MASTER + PAIRS")
print("="*60)

master = pd.read_csv(os.path.join(OUTPUT_DIR, "master_patient_table.csv"), low_memory=False)
master['index_date'] = pd.to_datetime(master['index_date'], errors='coerce')
print(f"  Master: {len(master):,}")

pairs_per_era = {}
for era_label in ERAS.keys():
    cache_path = os.path.join(OUTPUT_DIR, f"psm_pairs_{era_slug(era_label)}.csv")
    if os.path.exists(cache_path):
        pairs_per_era[era_label] = pd.read_csv(cache_path)
        print(f"  {era_label:15s}: loaded {len(pairs_per_era[era_label])} pairs")
    else:
        print(f"  {era_label:15s}: MISSING — run Script 07 first")

if len(pairs_per_era) == 0:
    print("\n  No pair caches found. Run Script 07 first to generate them.")
    raise SystemExit

# =============================================================================
# LOAD DEATHS
# =============================================================================
print("\n" + "="*60)
print("LOADING DEATHS (patient.csv)")
print("="*60)

patient = pd.read_csv(os.path.join(DATA_DIR, 'patient.csv'), low_memory=False)
print(f"  Rows: {len(patient):,}")

dd = patient[['patient_id', 'month_year_death']].copy()
dd['myd'] = pd.to_numeric(dd['month_year_death'], errors='coerce')
dd = dd[dd['myd'].notna() & (dd['myd'] > 190000) & (dd['myd'] < 210000)]
dd['year']  = (dd['myd'] // 100).astype(int)
dd['month'] = (dd['myd'] % 100).astype(int)
dd = dd[(dd['month'] >= 1) & (dd['month'] <= 12)]
dd['death_dt'] = pd.to_datetime(
    dict(year=dd['year'], month=dd['month'], day=[15]*len(dd)),
    errors='coerce')
dd = dd[dd['death_dt'].notna()]
death_date_map = dict(zip(dd['patient_id'], dd['death_dt']))
print(f"  Patients with recorded death: {len(death_date_map):,}")

# =============================================================================
# LOAD DIAGNOSIS  (for censoring)
# =============================================================================
print("\n" + "="*60)
print("LOADING DIAGNOSIS (for censoring last-seen)")
print("="*60)

diagnosis = pd.read_csv(os.path.join(DATA_DIR, "diagnosis.csv"), low_memory=False)
diagnosis['date'] = pd.to_datetime(diagnosis['date'].astype(str),
                                     format='%Y%m%d', errors='coerce')
# Precompute last seen per patient (faster than per-patient queries)
last_seen = diagnosis.groupby('patient_id')['date'].max().to_dict()
print(f"  Diagnosis rows: {len(diagnosis):,}")
print(f"  Patients with at least one dx: {len(last_seen):,}")

# =============================================================================
# MORTALITY HELPER
# =============================================================================
def mortality_tte(pid, index_date, max_days=3650):
    if pid in death_date_map:
        dd = death_date_map[pid]
        d  = (dd - index_date).days
        if d > 0:
            return (1, min(d, max_days))
    last_dx = last_seen.get(pid)
    if last_dx is None or pd.isna(last_dx):
        return (0, max_days)
    d = (last_dx - index_date).days
    return (0, max(min(d, max_days), 1))

# =============================================================================
# MAIN LOOP
# =============================================================================
all_results = []
idx_map = master.set_index('patient_id')['index_date'].to_dict()

for era_label in ERAS.keys():
    if era_label not in pairs_per_era:
        continue
    print("\n" + "="*60)
    print(f"ERA: {era_label.upper()}")
    print("="*60)
    pairs_df = pairs_per_era[era_label]
    n_pairs  = len(pairs_df)
    print(f"  Matched pairs: {n_pairs}")

    matched_ktp = list(pairs_df['ktp_patient_id'])
    matched_spk = list(pairs_df['spk_patient_id'])
    all_pids    = matched_ktp + matched_spk
    cohort      = ['KTP']*len(matched_ktp) + ['SPK']*len(matched_spk)
    mort        = [mortality_tte(p, idx_map.get(p, pd.NaT)) for p in all_pids]

    mort_df = pd.DataFrame({
        'patient_id': all_pids, 'cohort': cohort,
        'event': [m[0] for m in mort],
        'time':  [m[1] for m in mort],
    })
    mort_df['cohort_bin'] = (mort_df['cohort']=='KTP').astype(int)

    for window_label, window in [('30-day', 30), ('90-day', 90),
                                   ('1-year', 365), ('Overall', None)]:
        df = mort_df.copy()
        if window is not None:
            df['event'] = ((df['event']==1) & (df['time'] <= window)).astype(int)
            df['time']  = df['time'].clip(upper=window)
        df = df[df['time'] > 0]

        n_ev_ktp = int(df[df['cohort']=='KTP']['event'].sum())
        n_ev_spk = int(df[df['cohort']=='SPK']['event'].sum())
        n_ktp    = int((df['cohort']=='KTP').sum())
        n_spk    = int((df['cohort']=='SPK').sum())

        row = dict(Era=era_label, N_pairs=n_pairs,
                   Outcome=f'{window_label} mortality',
                   n_ktp=n_ktp, n_spk=n_spk,
                   events_ktp=n_ev_ktp, events_spk=n_ev_spk,
                   rate_ktp=round(100*n_ev_ktp/max(n_ktp,1), 2),
                   rate_spk=round(100*n_ev_spk/max(n_spk,1), 2))

        if (n_ev_ktp + n_ev_spk) < 5:
            print(f"  {window_label:8s}: too few events (KTP={n_ev_ktp}, SPK={n_ev_spk})")
            row.update(hr=np.nan, ci_lo=np.nan, ci_hi=np.nan, cox_p=np.nan)
        else:
            try:
                cph = CoxPHFitter()
                cph.fit(df[['time','event','cohort_bin']],
                        duration_col='time', event_col='event')
                hr    = float(np.exp(cph.params_['cohort_bin']))
                ci_lo = float(np.exp(cph.confidence_intervals_['95% lower-bound']['cohort_bin']))
                ci_hi = float(np.exp(cph.confidence_intervals_['95% upper-bound']['cohort_bin']))
                cox_p = float(cph.summary['p']['cohort_bin'])
                row.update(hr=round(hr,3), ci_lo=round(ci_lo,3),
                           ci_hi=round(ci_hi,3), cox_p=round(cox_p,4))
                sig = '*' if cox_p < 0.05 else ''
                print(f"  {window_label:8s}: KTP {n_ev_ktp}/{n_ktp} ({row['rate_ktp']}%)  "
                      f"SPK {n_ev_spk}/{n_spk} ({row['rate_spk']}%)  "
                      f"HR={row['hr']} [{row['ci_lo']},{row['ci_hi']}]  "
                      f"p={row['cox_p']}{sig}")
            except Exception as e:
                row.update(hr=np.nan, ci_lo=np.nan, ci_hi=np.nan, cox_p=np.nan)
                print(f"  {window_label:8s}: Cox failed ({e})")
        all_results.append(row)

# =============================================================================
# SAVE + REPORT
# =============================================================================
print("\n" + "="*60)
print("SAVING")
print("="*60)
results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(OUTPUT_DIR, "mortality_results.csv"), index=False)
print("  -> Saved: mortality_results.csv")

print("\n" + "="*60)
print("ABSTRACT TABLE FORMAT")
print("="*60)
for outcome in ['30-day mortality', '90-day mortality',
                '1-year mortality', 'Overall mortality']:
    rows = results_df[results_df['Outcome']==outcome]
    if len(rows) == 0: continue
    print(f"\n  {outcome}:")
    for _, r in rows.iterrows():
        if pd.isna(r.get('hr', np.nan)):
            print(f"    {r['Era']:15s}: sparse "
                  f"(KTP={r['events_ktp']}, SPK={r['events_spk']})")
            continue
        p_str = '<0.001' if r['cox_p'] < 0.001 else f"{r['cox_p']:.3f}"
        sig   = '*' if r['cox_p'] < 0.05 else ''
        print(f"    {r['Era']:15s}: HR={r['hr']:.3f} [{r['ci_lo']:.3f},{r['ci_hi']:.3f}]  "
              f"p={p_str}{sig}  KTP {r['rate_ktp']}% vs SPK {r['rate_spk']}%")

print("\n" + "="*60)
print("SCRIPT 09 COMPLETE")
print("="*60)
