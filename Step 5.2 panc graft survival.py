"""
TriNetX PancPump Study - Step 05e: Pancreas Graft Survival (SPK Cohort)

Two endpoints:
  A. Time-to-pancreas-graft-failure (ICD T86.22) — Kaplan-Meier + event rate
  B. Time-to-return-to-insulin (functional surrogate) — insulin prescription
     post-index in SPK patients, interpreted as loss of graft function

Also reports:
  - Pancreas rejection (T86.21) rate and KM
  - Pancreas graft infection (T86.23) rate
  - Composite pancreas graft event (T86.21 + T86.22 + T86.23 + T86.298)
  - Cross-tabulation: rejection -> failure sequence
  - Stratified by era (full, past 10yr, past 5yr)

Outputs:
  secondary_panc_graft_survival.csv
  secondary_panc_km_failure.png
  secondary_panc_km_composite.png
  secondary_panc_return_to_insulin_km.png
  secondary_panc_graft_summary.csv

PI: Paul Kuo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import os, warnings
warnings.filterwarnings('ignore')

DATA_DIR   = r"Z:\homes\Rachel Wolansky\Panc Pump\Data In"
OUTPUT_DIR = r"Z:\homes\Rachel Wolansky\Panc Pump\Results"

# =============================================================================
# LOAD
# =============================================================================
print("\n" + "="*60)
print("LOADING FILES")
print("="*60)

matched_master = pd.read_csv(os.path.join(OUTPUT_DIR, "psm_matched_master.csv"), low_memory=False)
master_full    = pd.read_csv(os.path.join(OUTPUT_DIR, "master_patient_table.csv"), low_memory=False)
diagnosis      = pd.read_csv(os.path.join(DATA_DIR,   "diagnosis.csv"),            low_memory=False)
med_drug       = pd.read_csv(os.path.join(DATA_DIR,   "medication_drug.csv"),      low_memory=False)

matched_master['index_date'] = pd.to_datetime(matched_master['index_date'], errors='coerce')
diagnosis['date']            = pd.to_datetime(diagnosis['date'].astype(str), format='%Y%m%d', errors='coerce')
med_drug['start_date']       = pd.to_datetime(med_drug['start_date'].astype(str), format='%Y%m%d', errors='coerce')

# SPK cohort only
spk = matched_master[matched_master['cohort'] == 'SPK'].copy()
spk_ids  = set(spk['patient_id'])
idx_map  = spk.set_index('patient_id')['index_date'].to_dict()

print(f"  SPK matched patients: {len(spk)}")
print(f"  Index date range: {spk['index_date'].min().date()} to {spk['index_date'].max().date()}")

# Last observation date from diagnosis (for censoring)
dx_spk = diagnosis[diagnosis['patient_id'].isin(spk_ids)].copy()
dx_spk['idx']      = dx_spk['patient_id'].map(idx_map)
dx_spk['days_post'] = (dx_spk['date'] - dx_spk['idx']).dt.days
last_obs = dx_spk[dx_spk['days_post'] > 0].groupby('patient_id')['days_post'].max().to_dict()

def make_tte(event_dict, pid_set, max_days=3650, label=''):
    rows = []
    for pid in pid_set:
        censor = max(int(last_obs.get(pid, 365)), 1)
        if pid in event_dict and event_dict[pid] > 0:
            rows.append({'patient_id': pid,
                         'time': min(int(event_dict[pid]), max_days),
                         'event': 1})
        else:
            rows.append({'patient_id': pid,
                         'time': min(censor, max_days),
                         'event': 0})
    return pd.DataFrame(rows)

def get_first_event(code_list, pid_set, idx_map_local, prefix_match=False):
    """Return {patient_id: days_to_first_event} for given ICD codes."""
    dx = diagnosis[diagnosis['patient_id'].isin(pid_set)].copy()
    if prefix_match:
        mask = dx['code'].str.startswith(tuple(code_list), na=False)
    else:
        mask = dx['code'].isin(code_list)
    hits = dx[mask].copy()
    hits['idx']       = hits['patient_id'].map(idx_map_local)
    hits['days_post'] = (hits['date'] - hits['idx']).dt.days
    hits = hits[hits['days_post'] > 0]
    return hits.groupby('patient_id')['days_post'].min().to_dict()

def km_plot(tte_df, title, filename, color='#1565C0', threshold_line=None):
    kmf = KaplanMeierFitter()
    kmf.fit(tte_df['time'], event_observed=tte_df['event'])
    fig, ax = plt.subplots(figsize=(9, 6))
    kmf.plot_survival_function(ax=ax, color=color, linewidth=2.5,
                                ci_show=True, ci_alpha=0.15,
                                label=f'SPK (n={len(tte_df)}, events={tte_df["event"].sum()})')
    if threshold_line:
        ax.axhline(threshold_line, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Days Post-Index', fontsize=11)
    ax.set_ylabel('Graft Survival Probability', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)

    # Annotate survival estimates at key timepoints
    for t in [365, 730, 1825, 3650]:
        try:
            s = kmf.survival_function_at_times([t]).values[0]
            if 0 < s < 1:
                ax.annotate(f'{s*100:.1f}%',
                            xy=(t, s), xytext=(t+60, s+0.03),
                            fontsize=8, color=color,
                            arrowprops=dict(arrowstyle='->', color=color, lw=0.8))
        except:
            pass

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()
    print(f"  -> Saved: {filename}")
    return kmf

def summarize_tte(tte_df, label):
    n       = len(tte_df)
    events  = int(tte_df['event'].sum())
    rate    = round(events / n * 100, 1)
    med_fu  = round(tte_df['time'].median(), 0)
    print(f"\n  {label}:")
    print(f"    N={n}, Events={events} ({rate}%), Median follow-up={med_fu:.0f}d")
    return {'Label': label, 'N': n, 'Events': events, 'Event_rate_pct': rate,
            'Median_followup_days': med_fu}

# =============================================================================
# PANCREAS GRAFT FAILURE CODES
# =============================================================================
PANC_REJECTION  = ['T86.21', 'T86.810', 'T86.811', 'T86.812', 'T86.818', 'T86.819']
PANC_FAILURE    = ['T86.22', 'T86.820', 'T86.821', 'T86.822', 'T86.828', 'T86.829']
PANC_INFECTION  = ['T86.23', 'T86.830', 'T86.831', 'T86.832', 'T86.838', 'T86.839']
PANC_OTHER      = ['T86.298', 'T86.890', 'T86.891', 'T86.892', 'T86.898', 'T86.899']
PANC_COMPOSITE  = PANC_REJECTION + PANC_FAILURE + PANC_INFECTION + PANC_OTHER

# Also check ICD-9 equivalents
PANC_ICD9 = ['996.86', '996.87', 'V42.83']

# =============================================================================
# AUDIT — what pancreas codes actually exist in our data
# =============================================================================
print("\n" + "="*60)
print("PANCREAS GRAFT CODE AUDIT")
print("="*60)

dx_spk_all = diagnosis[diagnosis['patient_id'].isin(spk_ids)].copy()
panc_prefix = dx_spk_all[dx_spk_all['code'].str.startswith('T86', na=False)]
print(f"\nT86.xx codes in SPK cohort (all time):")
print(panc_prefix['code'].value_counts().head(30).to_string())

icd9_panc = dx_spk_all[dx_spk_all['code'].isin(PANC_ICD9)]
print(f"\nICD-9 pancreas codes: {len(icd9_panc)} rows")

# =============================================================================
# SECTION A — PANCREAS GRAFT EVENTS (ICD-CODED)
# =============================================================================
print("\n" + "="*60)
print("SECTION A: PANCREAS GRAFT EVENTS (ICD-CODED)")
print("="*60)

summary_rows = []

# A1 — Rejection
rej_dict = get_first_event(PANC_REJECTION + PANC_ICD9[:1], spk_ids, idx_map)
rej_tte  = make_tte(rej_dict, spk_ids, label='Pancreas Rejection')
summary_rows.append(summarize_tte(rej_tte, 'Pancreas Rejection (T86.21)'))
kmf_rej = km_plot(rej_tte,
                   'Pancreas Graft Rejection-Free Survival (SPK)',
                   'secondary_panc_km_rejection.png',
                   color='#C62828')

# A2 — Failure
fail_dict = get_first_event(PANC_FAILURE, spk_ids, idx_map)
fail_tte  = make_tte(fail_dict, spk_ids, label='Pancreas Failure')
summary_rows.append(summarize_tte(fail_tte, 'Pancreas Graft Failure (T86.22)'))
kmf_fail = km_plot(fail_tte,
                    'Pancreas Graft Failure-Free Survival (SPK)',
                    'secondary_panc_km_failure.png',
                    color='#B71C1C')

# A3 — Infection
inf_dict = get_first_event(PANC_INFECTION, spk_ids, idx_map)
inf_tte  = make_tte(inf_dict, spk_ids, label='Pancreas Infection')
summary_rows.append(summarize_tte(inf_tte, 'Pancreas Graft Infection (T86.23)'))

# A4 — Composite
comp_dict = get_first_event(PANC_COMPOSITE, spk_ids, idx_map)
comp_tte  = make_tte(comp_dict, spk_ids, label='Composite Pancreas Graft Event')
summary_rows.append(summarize_tte(comp_tte, 'Composite Pancreas Graft Event'))
kmf_comp = km_plot(comp_tte,
                    'Composite Pancreas Graft Event-Free Survival (SPK)',
                    'secondary_panc_km_composite.png',
                    color='#4A148C')

# A5 — Rejection -> Failure sequence
pts_rejected = set(rej_dict.keys())
pts_failed   = set(fail_dict.keys())
both = pts_rejected & pts_failed
print(f"\n  Rejection -> Failure sequence:")
print(f"    Rejected: {len(pts_rejected)} ({len(pts_rejected)/len(spk_ids)*100:.1f}%)")
print(f"    Failed:   {len(pts_failed)} ({len(pts_failed)/len(spk_ids)*100:.1f}%)")
print(f"    Both rejection AND failure: {len(both)} ({len(both)/len(spk_ids)*100:.1f}%)")
if len(pts_rejected) > 0:
    print(f"    Of those rejected, % who also failed: {len(both)/len(pts_rejected)*100:.1f}%")

# =============================================================================
# SECTION B — RETURN TO INSULIN (FUNCTIONAL SURROGATE)
# =============================================================================
print("\n" + "="*60)
print("SECTION B: RETURN TO INSULIN (FUNCTIONAL SURROGATE)")
print("="*60)

# Insulin RxNorm codes — broad search for any insulin product
INSULIN_CODES = [
    # Insulin glargine
    '274783','261542','847232','847239','847241','1551291','1551297',
    # Insulin aspart
    '1008501','1008505','1008507','865098',
    # Insulin lispro
    '1008507','311025','311026',
    # Insulin detemir
    '349094','351297',
    # Insulin NPH
    '311028','311030',
    # Insulin regular
    '311026','311027',
    # Insulin degludec
    '1534390','1534391',
    # Generic insulin
    '253182','253183','253184','86009','86010',
]

# Also search by drug name pattern in text if available
med_spk = med_drug[med_drug['patient_id'].isin(spk_ids)].copy()
med_spk['idx']       = med_spk['patient_id'].map(idx_map)
med_spk['days_post'] = (med_spk['start_date'] - med_spk['idx']).dt.days
med_spk_post = med_spk[med_spk['days_post'] > 90].copy()  # exclude first 90d peri-op

# Check what insulin codes are in the data
print(f"\nPost-index (>90d) medication rows for SPK: {len(med_spk_post):,}")
print(f"Unique drug codes: {med_spk_post['code'].nunique():,}")

# Match insulin codes
insulin_hits = med_spk_post[med_spk_post['code'].astype(str).isin(INSULIN_CODES)]
print(f"Insulin code matches (exact): {len(insulin_hits):,} rows, "
      f"{insulin_hits['patient_id'].nunique()} patients")

# Also try text description if available
text_col = next((c for c in med_drug.columns
                 if any(x in c.lower() for x in ['name','description','drug','text'])), None)
if text_col and text_col != 'code':
    insulin_text = med_spk_post[med_spk_post[text_col].astype(str).str.lower().str.contains(
        'insulin', na=False)]
    print(f"Insulin text matches ('{text_col}' contains 'insulin'): "
          f"{len(insulin_text):,} rows, {insulin_text['patient_id'].nunique()} patients")
    # Combine
    insulin_all = pd.concat([insulin_hits, insulin_text]).drop_duplicates()
else:
    # Fallback: check all codes in med_drug for any insulin-related entries
    print(f"\nNo text description column found. Checking top post-index SPK drug codes:")
    top_codes = med_spk_post['code'].value_counts().head(30)
    print(top_codes.to_string())
    insulin_all = insulin_hits

print(f"\nTotal insulin events (combined): {len(insulin_all):,} rows, "
      f"{insulin_all['patient_id'].nunique()} patients "
      f"({insulin_all['patient_id'].nunique()/len(spk_ids)*100:.1f}% of SPK)")

# Build TTE for return to insulin
if len(insulin_all) > 0:
    insulin_first = insulin_all.groupby('patient_id')['days_post'].min().to_dict()
    ins_tte = make_tte(insulin_first, spk_ids, label='Return to Insulin')
    summary_rows.append(summarize_tte(ins_tte, 'Return to Insulin (functional surrogate, >90d post-index)'))
    kmf_ins = km_plot(ins_tte,
                       'Return to Insulin Post-SPK\n(Functional Pancreas Graft Loss Surrogate)',
                       'secondary_panc_return_to_insulin_km.png',
                       color='#1565C0')

    # Cross-tab: insulin use vs ICD-coded failure
    ins_pts   = set(insulin_first.keys())
    fail_pts  = set(fail_dict.keys())
    print(f"\n  Cross-tabulation:")
    print(f"    Return to insulin only (no T86.22): {len(ins_pts - fail_pts)}")
    print(f"    T86.22 failure only (no insulin):   {len(fail_pts - ins_pts)}")
    print(f"    Both insulin AND T86.22:             {len(ins_pts & fail_pts)}")
    print(f"    Neither:                             {len(spk_ids - ins_pts - fail_pts)}")
else:
    print("\n  No insulin events found with current codes.")
    print("  Consider expanding RxNorm code list or using text search.")

# =============================================================================
# SECTION C — ERA STRATIFICATION
# =============================================================================
print("\n" + "="*60)
print("SECTION C: ERA STRATIFICATION")
print("="*60)

ERAS = {
    'Full cohort':   pd.Timestamp('2000-01-01'),
    'Past 10 years': pd.Timestamp('2016-01-01'),
    'Past 5 years':  pd.Timestamp('2021-01-01'),
}

era_summary = []
for era_label, cutoff in ERAS.items():
    era_spk = spk[spk['index_date'] >= cutoff]
    era_ids = set(era_spk['patient_id'])
    era_idx = era_spk.set_index('patient_id')['index_date'].to_dict()
    n = len(era_ids)
    if n < 10:
        continue

    # Composite pancreas graft event
    comp_d  = get_first_event(PANC_COMPOSITE, era_ids, era_idx)
    comp_t  = make_tte(comp_d, era_ids)
    n_ev    = int(comp_t['event'].sum())
    rate    = round(n_ev/n*100, 1)

    # Return to insulin
    ins_d   = med_drug[med_drug['patient_id'].isin(era_ids)].copy()
    ins_d['idx']       = ins_d['patient_id'].map(era_idx)
    ins_d['days_post'] = (ins_d['start_date'] - ins_d['idx']).dt.days
    ins_d_post = ins_d[ins_d['days_post'] > 90]
    ins_hits = ins_d_post[ins_d_post['code'].astype(str).isin(INSULIN_CODES)]
    if text_col and text_col != 'code':
        ins_txt = ins_d_post[ins_d_post[text_col].astype(str).str.lower().str.contains('insulin',na=False)]
        ins_hits = pd.concat([ins_hits, ins_txt]).drop_duplicates()
    ins_first_era = ins_hits.groupby('patient_id')['days_post'].min().to_dict() if len(ins_hits)>0 else {}
    ins_t   = make_tte(ins_first_era, era_ids)
    ins_ev  = int(ins_t['event'].sum())
    ins_rate = round(ins_ev/n*100,1)

    print(f"\n  {era_label} (n={n} SPK):")
    print(f"    Composite pancreas graft event: {n_ev}/{n} ({rate}%)")
    print(f"    Return to insulin (>90d):        {ins_ev}/{n} ({ins_rate}%)")
    era_summary.append({
        'Era': era_label, 'N_SPK': n,
        'Composite_graft_event_n': n_ev, 'Composite_graft_event_pct': rate,
        'Return_to_insulin_n': ins_ev, 'Return_to_insulin_pct': ins_rate,
    })

era_df = pd.DataFrame(era_summary)

# =============================================================================
# COMBINED KM FIGURE — Rejection + Failure + Composite side by side
# =============================================================================
print("\n" + "="*60)
print("COMBINED KM FIGURE")
print("="*60)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Pancreas Graft Outcomes — SPK Cohort (n=306)',
             fontsize=14, fontweight='bold')

plots = [
    (rej_tte,  'Rejection-Free Survival\n(T86.21)', '#C62828'),
    (fail_tte, 'Failure-Free Survival\n(T86.22)',   '#B71C1C'),
    (comp_tte, 'Composite Event-Free\nSurvival',    '#4A148C'),
]
for ax, (tte, title, color) in zip(axes, plots):
    kmf = KaplanMeierFitter()
    kmf.fit(tte['time'], event_observed=tte['event'])
    kmf.plot_survival_function(ax=ax, color=color, linewidth=2.5, ci_show=True,
                                ci_alpha=0.15,
                                label=f'n={len(tte)}, events={tte["event"].sum()}')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Days Post-Index')
    ax.set_ylabel('Survival Probability')
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    for t in [365, 1825]:
        try:
            s = kmf.survival_function_at_times([t]).values[0]
            ax.annotate(f'{s*100:.1f}%', xy=(t,s),
                        xytext=(t+80, s+0.04), fontsize=8, color=color,
                        arrowprops=dict(arrowstyle='->', color=color, lw=0.8))
        except:
            pass

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'secondary_panc_combined_km.png'), dpi=150)
plt.close()
print("  -> Saved: secondary_panc_combined_km.png")

# =============================================================================
# SAVE SUMMARY
# =============================================================================
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUTPUT_DIR, 'secondary_panc_graft_summary.csv'), index=False)
era_df.to_csv(os.path.join(OUTPUT_DIR, 'secondary_panc_era_summary.csv'), index=False)
print("\n  -> Saved: secondary_panc_graft_summary.csv")
print("  -> Saved: secondary_panc_era_summary.csv")

# =============================================================================
# PRINT FINAL SUMMARY
# =============================================================================
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(summary_df.to_string(index=False))
print("\nEra Stratification:")
print(era_df.to_string(index=False))
print("\n✓ Script 05e complete.")
print("  Outputs: secondary_panc_graft_summary.csv")
print("           secondary_panc_era_summary.csv")
print("           secondary_panc_combined_km.png")
print("           secondary_panc_km_rejection.png")
print("           secondary_panc_km_failure.png")
print("           secondary_panc_km_composite.png")
print("           secondary_panc_return_to_insulin_km.png")
