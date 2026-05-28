"""
TriNetX PancPump Study - Step 4: Primary Outcome — HbA1c Analysis
Rewritten clean version. PI: Paul Kuo
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chi2
import os, warnings
warnings.filterwarnings('ignore')

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, proportional_hazard_test
import statsmodels.formula.api as smf

DATA_DIR   = r"Z:\homes\Rachel Wolansky\Panc Pump\Data In"
OUTPUT_DIR = r"Z:\homes\Rachel Wolansky\Panc Pump\Results"

print("\n" + "="*60)
print("LOADING FILES")
print("="*60)

matched_master = pd.read_csv(os.path.join(OUTPUT_DIR, "psm_matched_master.csv"), low_memory=False)
matched_pairs  = pd.read_csv(os.path.join(OUTPUT_DIR, "psm_matched_pairs.csv"),  low_memory=False)
lab_clean      = pd.read_csv(os.path.join(OUTPUT_DIR, "lab_result_clean.csv"),   low_memory=False)

matched_master['index_date'] = pd.to_datetime(matched_master['index_date'], errors='coerce')
lab_clean['date']            = pd.to_datetime(lab_clean['date'], errors='coerce')

matched_ids = set(matched_master['patient_id'])
idx_dates   = matched_master.set_index('patient_id')['index_date'].to_dict()
cohort_map  = matched_master.set_index('patient_id')['cohort'].to_dict()

print(f"  Matched patients: {len(matched_master):,}  (KTP={(matched_master['cohort']=='KTP').sum()}, SPK={(matched_master['cohort']=='SPK').sum()})")

HBAC1C_LOINCS = ['4548-4', '17856-6', '59261-8']

hba1c = lab_clean[
    lab_clean['code'].isin(HBAC1C_LOINCS) &
    lab_clean['patient_id'].isin(matched_ids) &
    lab_clean['lab_result_num_val'].notna()
].copy()

hba1c['index_date'] = hba1c['patient_id'].map(idx_dates)
hba1c['cohort']     = hba1c['patient_id'].map(cohort_map)
hba1c['days_post']  = (hba1c['date'] - hba1c['index_date']).dt.days
hba1c_post = hba1c[hba1c['days_post'] > 0].copy()

print(f"\nPost-index HbA1c: {len(hba1c_post):,}  (KTP={(hba1c_post['cohort']=='KTP').sum():,}, SPK={(hba1c_post['cohort']=='SPK').sum():,})")

def get_interval_vals(df, center_days, half_window):
    lo  = max(center_days - half_window, 1)
    hi  = center_days + half_window
    sub = df[(df['days_post'] >= lo) & (df['days_post'] <= hi)].copy()
    if len(sub) == 0:
        return pd.DataFrame(columns=['patient_id', 'lab_result_num_val'])
    sub = sub.copy()
    sub['dist'] = abs(sub['days_post'] - center_days)
    best = sub.sort_values('dist').groupby('patient_id', as_index=False).first()[['patient_id', 'lab_result_num_val']]
    return best

INTERVALS = [
    ('3 months',  91,  45),
    ('6 months', 182,  60),
    ('1 year',   365,  90),
    ('2 years',  730, 120),
    ('3 years', 1095, 150),
    ('5 years', 1825, 180),
]

print("\n" + "="*60)
print("SECTION 2: SERIAL MEAN HbA1c")
print("="*60)

serial_rows = []
for label, center, hw in INTERVALS:
    v_ktp = get_interval_vals(hba1c_post[hba1c_post['cohort']=='KTP'], center, hw)
    v_spk = get_interval_vals(hba1c_post[hba1c_post['cohort']=='SPK'], center, hw)
    n_ktp, n_spk = len(v_ktp), len(v_spk)

    mean_ktp = v_ktp['lab_result_num_val'].mean() if n_ktp>0 else np.nan
    mean_spk = v_spk['lab_result_num_val'].mean() if n_spk>0 else np.nan
    sd_ktp   = v_ktp['lab_result_num_val'].std()  if n_ktp>0 else np.nan
    sd_spk   = v_spk['lab_result_num_val'].std()  if n_spk>0 else np.nan
    pct_ktp  = (v_ktp['lab_result_num_val']<7.0).mean()*100 if n_ktp>0 else np.nan
    pct_spk  = (v_spk['lab_result_num_val']<7.0).mean()*100 if n_spk>0 else np.nan

    pk = matched_pairs.merge(v_ktp.rename(columns={'lab_result_num_val':'ktp_hba1c','patient_id':'ktp_patient_id'}), on='ktp_patient_id', how='inner')
    pm = pk.merge(v_spk.rename(columns={'lab_result_num_val':'spk_hba1c','patient_id':'spk_patient_id'}), on='spk_patient_id', how='inner')
    n_paired = len(pm)

    t_stat = p_val = mc_p = np.nan
    if n_paired >= 10:
        t_stat, p_val = stats.ttest_rel(pm['ktp_hba1c'], pm['spk_hba1c'])
        kt = (pm['ktp_hba1c']<7.0).astype(int)
        st = (pm['spk_hba1c']<7.0).astype(int)
        b  = ((kt==1)&(st==0)).sum()
        c  = ((kt==0)&(st==1)).sum()
        if (b+c)>0:
            mc_stat = (abs(b-c)-1)**2/(b+c)
            mc_p    = 1 - chi2.cdf(mc_stat, df=1)
    elif n_ktp>=5 and n_spk>=5:
        t_stat, p_val = stats.ttest_ind(v_ktp['lab_result_num_val'], v_spk['lab_result_num_val'])

    serial_rows.append({
        'Interval': label, 'Center_days': center,
        'N_KTP': n_ktp, 'N_SPK': n_spk, 'N_paired': n_paired,
        'Mean_KTP': round(float(mean_ktp),2) if not np.isnan(mean_ktp) else np.nan,
        'SD_KTP':   round(float(sd_ktp),2)   if not np.isnan(sd_ktp)   else np.nan,
        'Mean_SPK': round(float(mean_spk),2) if not np.isnan(mean_spk) else np.nan,
        'SD_SPK':   round(float(sd_spk),2)   if not np.isnan(sd_spk)   else np.nan,
        'Diff_KTP_SPK': round(float(mean_ktp-mean_spk),2) if (not np.isnan(mean_ktp) and not np.isnan(mean_spk)) else np.nan,
        'Pct_target_KTP': round(float(pct_ktp),1) if not np.isnan(pct_ktp) else np.nan,
        'Pct_target_SPK': round(float(pct_spk),1) if not np.isnan(pct_spk) else np.nan,
        'p_value_ttest':   round(float(p_val),4)  if not np.isnan(p_val)  else np.nan,
        'p_value_mcnemar': round(float(mc_p),4)   if not np.isnan(mc_p)   else np.nan,
        'Sig_ttest': bool(p_val<0.05) if not np.isnan(p_val) else False,
    })

serial_df = pd.DataFrame(serial_rows)
print(serial_df[['Interval','N_KTP','N_SPK','N_paired','Mean_KTP','Mean_SPK','Diff_KTP_SPK','p_value_ttest','Sig_ttest']].to_string(index=False))
serial_df.to_csv(os.path.join(OUTPUT_DIR, "hba1c_serial_means.csv"), index=False)
print("  -> Saved: hba1c_serial_means.csv")

# Plot
plot_df = serial_df[serial_df['Mean_KTP'].notna() & serial_df['Mean_SPK'].notna()].copy()
x_pos   = list(range(len(plot_df)))
fig, axes = plt.subplots(1, 2, figsize=(16,6))
fig.suptitle('HbA1c Over Time: KTP vs SPK (PSM-Matched, n=306 pairs)', fontsize=13, fontweight='bold')

ax = axes[0]
sem_ktp = (plot_df['SD_KTP']/np.sqrt(plot_df['N_KTP'])).tolist()
sem_spk = (plot_df['SD_SPK']/np.sqrt(plot_df['N_SPK'])).tolist()
ax.errorbar(x_pos, plot_df['Mean_KTP'], yerr=sem_ktp, color='#2196F3', marker='o', linewidth=2, markersize=8, capsize=5, label='KTP')
ax.errorbar(x_pos, plot_df['Mean_SPK'], yerr=sem_spk, color='#FF5722', marker='s', linewidth=2, markersize=8, capsize=5, label='SPK')
ax.axhline(7.0, color='green', linestyle='--', alpha=0.7, linewidth=1.5, label='Target <7.0%')
ax.set_xticks(x_pos); ax.set_xticklabels(plot_df['Interval'].tolist(), rotation=30, ha='right')
ax.set_ylabel('Mean HbA1c ± SEM (%)'); ax.set_title('A. Mean HbA1c Over Time')
ax.legend(fontsize=9); ax.set_ylim(4,10); ax.grid(axis='y', alpha=0.3)
for i, row in enumerate(plot_df.itertuples()):
    pv = row.p_value_ttest
    if isinstance(pv, float) and not np.isnan(pv):
        pt = 'p<0.001' if pv<0.001 else (f'p={pv:.3f}' if pv<0.05 else f'p={pv:.2f}')
        ax.text(i, max(row.Mean_KTP, row.Mean_SPK)+0.3, pt, ha='center', fontsize=7, style='italic')

ax2 = axes[1]
w   = 0.35; xa = np.array(x_pos)
b1  = ax2.bar(xa-w/2, plot_df['Pct_target_KTP'], w, color='#2196F3', alpha=0.8, label='KTP')
b2  = ax2.bar(xa+w/2, plot_df['Pct_target_SPK'], w, color='#FF5722', alpha=0.8, label='SPK')
ax2.set_xticks(x_pos); ax2.set_xticklabels(plot_df['Interval'].tolist(), rotation=30, ha='right')
ax2.set_ylabel('Proportion with HbA1c <7.0% (%)'); ax2.set_title('B. Proportion Achieving HbA1c <7.0%')
ax2.legend(fontsize=9); ax2.set_ylim(0,100); ax2.grid(axis='y', alpha=0.3)
for bar in list(b1)+list(b2):
    h = bar.get_height()
    if not np.isnan(h):
        ax2.text(bar.get_x()+bar.get_width()/2., h+1, f'{h:.0f}%', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hba1c_serial_means_plot.png"), dpi=150)
plt.close()
print("  -> Saved: hba1c_serial_means_plot.png")

# KM
print("\n" + "="*60)
print("SECTION 4: KAPLAN-MEIER")
print("="*60)

tte_rows = []
for pid, grp in hba1c_post.groupby('patient_id'):
    grp    = grp.sort_values('days_post')
    events = grp[grp['lab_result_num_val'] < 7.0]
    last_t = grp['days_post'].max()
    if len(events) > 0:
        tte_rows.append({'patient_id':pid,'cohort':cohort_map.get(pid),'time':int(events['days_post'].iloc[0]),'event':1})
    else:
        tte_rows.append({'patient_id':pid,'cohort':cohort_map.get(pid),'time':max(int(last_t),1),'event':0})

tte_df = pd.DataFrame(tte_rows)
tte_df = tte_df[tte_df['patient_id'].isin(matched_ids)].copy()
tte_df['cohort_bin'] = (tte_df['cohort']=='KTP').astype(int)

for c in ['KTP','SPK']:
    sub = tte_df[tte_df['cohort']==c]
    print(f"  {c}: n={len(sub):,}, events={sub['event'].sum():,} ({sub['event'].mean()*100:.1f}%), median follow-up={sub['time'].median():.0f}d")

ktp_tte = tte_df[tte_df['cohort']=='KTP']
spk_tte = tte_df[tte_df['cohort']=='SPK']
kmf_ktp = KaplanMeierFitter(label='KTP (Kidney+Pump)')
kmf_spk = KaplanMeierFitter(label='SPK')
kmf_ktp.fit(ktp_tte['time'], event_observed=ktp_tte['event'])
kmf_spk.fit(spk_tte['time'], event_observed=spk_tte['event'])
lr = logrank_test(ktp_tte['time'], spk_tte['time'], event_observed_A=ktp_tte['event'], event_observed_B=spk_tte['event'])
print(f"\nLog-rank p = {lr.p_value:.4f}")

fig, ax = plt.subplots(figsize=(10,7))
kmf_ktp.plot_survival_function(ax=ax, color='#2196F3', linewidth=2, ci_show=True, ci_alpha=0.15)
kmf_spk.plot_survival_function(ax=ax, color='#FF5722', linewidth=2, ci_show=True, ci_alpha=0.15)
p_str = 'Log-rank p < 0.001' if lr.p_value<0.001 else f'Log-rank p = {lr.p_value:.4f}'
ax.text(0.62, 0.85, p_str, transform=ax.transAxes, fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.set_xlabel('Days Post-Index Event', fontsize=12)
ax.set_ylabel('Probability of NOT Achieving HbA1c <7.0%', fontsize=11)
ax.set_title('Time to First HbA1c <7.0%: KTP vs SPK\n(PSM-Matched Cohort)', fontsize=13, fontweight='bold')
ax.set_ylim(0,1.05); ax.legend(fontsize=11); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hba1c_km_time_to_target.png"), dpi=150)
plt.close()
print("  -> Saved: hba1c_km_time_to_target.png")

# Cox
print("\n" + "="*60)
print("SECTION 5: COX PH MODEL")
print("="*60)

cox_df = tte_df[['time','event','cohort_bin']].copy()
cox_df = cox_df[cox_df['time']>0]
cph    = CoxPHFitter()
cph.fit(cox_df, duration_col='time', event_col='event')
cph.print_summary()

hr    = float(np.exp(cph.params_['cohort_bin']))
ci_lo = float(np.exp(cph.confidence_intervals_['95% lower-bound']['cohort_bin']))
ci_hi = float(np.exp(cph.confidence_intervals_['95% upper-bound']['cohort_bin']))
cox_p = float(cph.summary['p']['cohort_bin'])
print(f"\nHR (KTP vs SPK) = {hr:.3f}  95% CI [{ci_lo:.3f}, {ci_hi:.3f}]  p={cox_p:.4f}")

ph_assumption_ok = True
try:
    ph_test = proportional_hazard_test(cph, cox_df, time_transform='rank')
    ph_p    = float(ph_test.summary['p'].iloc[0])
    ph_assumption_ok = ph_p > 0.05
    print(f"Schoenfeld PH test: p={ph_p:.4f}  -> {'SATISFIED' if ph_assumption_ok else 'VIOLATED'}")
except Exception as e:
    print(f"PH test skipped: {e}")

cox_summary = cph.summary.copy()
cox_summary['ph_ok'] = ph_assumption_ok
cox_summary.to_csv(os.path.join(OUTPUT_DIR, "hba1c_cox_results.csv"))
print("  -> Saved: hba1c_cox_results.csv")

# LME
print("\n" + "="*60)
print("SECTION 6: LINEAR MIXED-EFFECTS MODEL")
print("="*60)

lme_df = hba1c_post[hba1c_post['patient_id'].isin(matched_ids)].copy()
lme_df['years_post'] = lme_df['days_post']/365.25
lme_df['is_ktp']     = (lme_df['cohort']=='KTP').astype(int)
lme_df = lme_df.rename(columns={'lab_result_num_val':'hba1c'})
lme_df = lme_df[lme_df['hba1c'].notna() & lme_df['years_post'].notna()]

print(f"\nLME: {len(lme_df):,} obs, {lme_df['patient_id'].nunique():,} patients")
try:
    model  = smf.mixedlm("hba1c ~ years_post * is_ktp", lme_df, groups=lme_df['patient_id'])
    result = model.fit(reml=True)
    print(result.summary())
    coefs = result.fe_params
    print(f"\n  Intercept (SPK at t=0):     {coefs['Intercept']:.3f}")
    print(f"  Time slope (SPK, per yr):   {coefs['years_post']:.3f}")
    print(f"  KTP offset at index:        {coefs['is_ktp']:.3f}")
    print(f"  KTP x time interaction:     {coefs['years_post:is_ktp']:.3f}")
except Exception as e:
    print(f"  LME failed: {e}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"\nMatched pairs: 306")
print(f"Log-rank p = {lr.p_value:.4f}")
print(f"HR (KTP vs SPK) = {hr:.3f}  [{ci_lo:.3f}, {ci_hi:.3f}]  p={cox_p:.4f}")
for _, row in serial_df.iterrows():
    if not (isinstance(row['Mean_KTP'], float) and np.isnan(row['Mean_KTP'])):
        sig = '*' if row['Sig_ttest'] else ''
        print(f"  {row['Interval']:10s}: KTP={row['Mean_KTP']:.2f}  SPK={row['Mean_SPK']:.2f}  diff={row['Diff_KTP_SPK']:+.2f}  p={row['p_value_ttest']:.4f}{sig}")

serial_df.to_csv(os.path.join(OUTPUT_DIR, "hba1c_full_results_table.csv"), index=False)
print("\n  -> Saved: hba1c_full_results_table.csv")
print("\nDone. Next: Run 05_secondary_outcomes.py")
