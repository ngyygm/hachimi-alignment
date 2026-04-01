#!/usr/bin/env python3
"""
Step 6: Comparison experiments based on matched segment data.

Experiments:
1. Duration-alignment correlation
2. Quality threshold analysis (high vs low z-score)
3. Cross-audio comparison within matched pairs
4. Effect size comparison: full songs vs matched segments
5. MS-CLAP cross-audio comparison
"""
import os, sys, warnings, json, time
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import numpy as np
from pathlib import Path
from scipy import stats

SEED = 42
np.random.seed(SEED)

# ─── Paths (relative to project root) ────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = PROJECT_ROOT / "results"

# Load data
with open(RESULTS_DIR / "matched_segment_results.json") as f:
    seg_results = json.load(f)
with open(RESULTS_DIR / "cleaned_results.json") as f:
    full_results = json.load(f)
with open(RESULTS_DIR / "segment_match_aligned.json") as f:
    aligned = json.load(f)
with open(RESULTS_DIR / "conditions.json") as f:
    conditions = json.load(f)

song_names = seg_results['song_list']

# Build lookup: song -> alignment scores (full songs)
full_laion = full_results['laion_clap']['hach_audio']
full_per_song = full_laion['per_song']
full_c0 = {n: s for n, s in zip(full_per_song['song_names'], full_per_song['C0'])}
full_c1 = {n: s for n, s in zip(full_per_song['song_names'], full_per_song['C1'])}

seg_laion = seg_results['laion_clap']['hach_seg_audio']
seg_per_song = seg_laion['per_song']
seg_c0 = {n: s for n, s in zip(seg_per_song['song_names'], seg_per_song['C0'])}
seg_c1 = {n: s for n, s in zip(seg_per_song['song_names'], seg_per_song['C1'])}

# Build aligned metadata lookup
aligned_meta = {s['song']: s for s in aligned if s['song'] in conditions}

# Only use songs in both datasets
common_songs = [n for n in song_names if n in full_c0 and n in seg_c0]
print(f"Common songs: {len(common_songs)}")

# ============================================================
# Experiment 1: Duration-Alignment Correlation
# ============================================================
print("\n" + "="*60)
print("EXPERIMENT 1: Duration-Alignment Correlation")
print("="*60)

durations = []
c0_changes = []
c1_changes = []
c0c1_gaps_full = []
c0c1_gaps_seg = []

for s in aligned:
    if s['song'] not in common_songs:
        continue
    n = s['song']
    dur = s['matched_duration']
    fc0, fc1 = full_c0[n], full_c1[n]
    sc0, sc1 = seg_c0[n], seg_c1[n]

    durations.append(dur)
    c0_changes.append(sc0 - fc0)
    c1_changes.append(sc1 - fc1)
    c0c1_gaps_full.append(fc1 - fc0)
    c0c1_gaps_seg.append(sc1 - sc0)

durations = np.array(durations)
c0_changes = np.array(c0_changes)
c1_changes = np.array(c1_changes)
c0c1_gaps_full = np.array(c0c1_gaps_full)
c0c1_gaps_seg = np.array(c0c1_gaps_seg)

# Correlations
r_c0, p_c0 = stats.pearsonr(durations, c0_changes)
r_c1, p_c1 = stats.pearsonr(durations, c1_changes)
r_gap_full, p_gap_full = stats.pearsonr(durations, c0c1_gaps_full)
r_gap_seg, p_gap_seg = stats.pearsonr(durations, c0c1_gaps_seg)

print(f"\nSegment duration stats: mean={durations.mean():.1f}s, std={durations.std():.1f}s, range=[{durations.min():.1f}, {durations.max():.1f}]")
print(f"\nCorrelations with segment duration (N={len(durations)}):")
print(f"  C0 alignment change:    r={r_c0:.3f}, p={p_c0:.4f}")
print(f"  C1 alignment change:    r={r_c1:.3f}, p={p_c1:.4f}")
print(f"  C0-C1 gap (full songs): r={r_gap_full:.3f}, p={p_gap_full:.4f}")
print(f"  C0-C1 gap (segments):   r={r_gap_seg:.3f}, p={p_gap_seg:.4f}")

# Short vs long segments split
median_dur = np.median(durations)
short_mask = durations <= median_dur
long_mask = durations > median_dur
print(f"\nShort segments (<= {median_dur:.0f}s, N={short_mask.sum()}):")
print(f"  C0-C1 gap (seg): {c0c1_gaps_seg[short_mask].mean():.4f} +/- {c0c1_gaps_seg[short_mask].std():.4f}")
print(f"  Mean C0 change: {c0_changes[short_mask].mean():.4f}")
print(f"  Mean C1 change: {c1_changes[short_mask].mean():.4f}")

print(f"\nLong segments (> {median_dur:.0f}s, N={long_mask.sum()}):")
print(f"  C0-C1 gap (seg): {c0c1_gaps_seg[long_mask].mean():.4f} +/- {c0c1_gaps_seg[long_mask].std():.4f}")
print(f"  Mean C0 change: {c0_changes[long_mask].mean():.4f}")
print(f"  Mean C1 change: {c1_changes[long_mask].mean():.4f}")

t_gap, p_gap = stats.ttest_ind(c0c1_gaps_seg[short_mask], c0c1_gaps_seg[long_mask])
print(f"  Short vs Long gap diff: t={t_gap:.2f}, p={p_gap:.4f}")

# ============================================================
# Experiment 2: Quality Threshold Analysis
# ============================================================
print("\n" + "="*60)
print("EXPERIMENT 2: Match Quality Threshold Analysis")
print("="*60)

z_scores = []
for s in aligned:
    if s['song'] in common_songs:
        z_scores.append((s['song'], s['z_score'], s['fm_sim']))

z_scores.sort(key=lambda x: x[1])
songs_z = [x[0] for x in z_scores]
zs = np.array([x[1] for x in z_scores])
fms = np.array([x[2] for x in z_scores])

high_mask = zs >= 4.0
low_mask = (zs >= 2.0) & (zs < 4.0)

seg_c0_arr = np.array([seg_c0[n] for n in songs_z])
seg_c1_arr = np.array([seg_c1[n] for n in songs_z])
full_c0_arr = np.array([full_c0[n] for n in songs_z])
full_c1_arr = np.array([full_c1[n] for n in songs_z])

print(f"\nHigh quality matches (z>=4, N={high_mask.sum()}):")
print(f"  Mean z-score: {zs[high_mask].mean():.2f}, fm_sim: {fms[high_mask].mean():.3f}")
h_gap = seg_c1_arr[high_mask] - seg_c0_arr[high_mask]
print(f"  C0-C1 gap (seg): {h_gap.mean():.4f} +/- {h_gap.std():.4f}")
t_h, p_h = stats.ttest_rel(seg_c1_arr[high_mask], seg_c0_arr[high_mask])
d_h = h_gap.mean() / (h_gap.std() + 1e-9)
print(f"  C0 vs C1: d={d_h:.2f}, p={p_h:.4f}")

print(f"\nLow quality matches (2<=z<4, N={low_mask.sum()}):")
print(f"  Mean z-score: {zs[low_mask].mean():.2f}, fm_sim: {fms[low_mask].mean():.3f}")
l_gap = seg_c1_arr[low_mask] - seg_c0_arr[low_mask]
print(f"  C0-C1 gap (seg): {l_gap.mean():.4f} +/- {l_gap.std():.4f}")
t_l, p_l = stats.ttest_rel(seg_c1_arr[low_mask], seg_c0_arr[low_mask])
d_l = l_gap.mean() / (l_gap.std() + 1e-9)
print(f"  C0 vs C1: d={d_l:.2f}, p={p_l:.4f}")

t_hl, p_hl = stats.ttest_ind(h_gap, l_gap)
print(f"\nHigh vs Low quality gap diff: t={t_hl:.2f}, p={p_hl:.4f}")

# ============================================================
# Experiment 3: Cross-Audio Comparison
# ============================================================
print("\n" + "="*60)
print("EXPERIMENT 3: Cross-Audio Comparison")
print("="*60)

seg_orig_per_song = seg_results['laion_clap']['orig_seg_audio']['per_song']
orig_seg_c0 = {n: s for n, s in zip(seg_orig_per_song['song_names'], seg_orig_per_song['C0'])}
orig_seg_c1 = {n: s for n, s in zip(seg_orig_per_song['song_names'], seg_orig_per_song['C1'])}

hach_audio_c0 = np.array([seg_c0[n] for n in common_songs])
hach_audio_c1 = np.array([seg_c1[n] for n in common_songs])
orig_audio_c0 = np.array([orig_seg_c0[n] for n in common_songs if n in orig_seg_c0])
orig_audio_c1 = np.array([orig_seg_c1[n] for n in common_songs if n in orig_seg_c1])

n_cross = min(len(hach_audio_c0), len(orig_audio_c0))
hach_audio_c0 = hach_audio_c0[:n_cross]
hach_audio_c1 = hach_audio_c1[:n_cross]
orig_audio_c0 = orig_audio_c0[:n_cross]
orig_audio_c1 = orig_audio_c1[:n_cross]

print(f"\nN={n_cross} songs with both audio types")

print(f"\nHachimi segment audio:")
print(f"  C0: {hach_audio_c0.mean():.4f} +/- {hach_audio_c0.std():.4f}")
print(f"  C1: {hach_audio_c1.mean():.4f} +/- {hach_audio_c1.std():.4f}")
gap_hach = hach_audio_c1 - hach_audio_c0
print(f"  C0-C1 gap: {gap_hach.mean():.4f} +/- {gap_hach.std():.4f}")
t_ha, p_ha = stats.ttest_rel(hach_audio_c1, hach_audio_c0)
print(f"  C0 vs C1: d={gap_hach.mean()/(gap_hach.std()+1e-9):.2f}, p={p_ha:.4f}")

print(f"\nOriginal segment audio:")
print(f"  C0: {orig_audio_c0.mean():.4f} +/- {orig_audio_c0.std():.4f}")
print(f"  C1: {orig_audio_c1.mean():.4f} +/- {orig_audio_c1.std():.4f}")
gap_orig = orig_audio_c1 - orig_audio_c0
print(f"  C0-C1 gap: {gap_orig.mean():.4f} +/- {gap_orig.std():.4f}")
t_oa, p_oa = stats.ttest_rel(orig_audio_c1, orig_audio_c0)
print(f"  C0 vs C1: d={gap_orig.mean()/(gap_orig.std()+1e-9):.2f}, p={p_oa:.4f}")

t_cross, p_cross = stats.ttest_rel(gap_hach, gap_orig)
print(f"\nHach vs Orig audio C0-C1 gap: d_diff={gap_hach.mean()-gap_orig.mean():.4f}, t={t_cross:.2f}, p={p_cross:.4f}")

t_c0_audio, p_c0_audio = stats.ttest_rel(hach_audio_c0, orig_audio_c0)
t_c1_audio, p_c1_audio = stats.ttest_rel(hach_audio_c1, orig_audio_c1)
print(f"\nSame text conditions, different audio:")
print(f"  C0 (hach vs orig audio): d={hach_audio_c0.mean()-orig_audio_c0.mean():.4f}, p={p_c0_audio:.4f}")
print(f"  C1 (hach vs orig audio): d={hach_audio_c1.mean()-orig_audio_c1.mean():.4f}, p={p_c1_audio:.4f}")

# ============================================================
# Experiment 4: Effect Size Comparison
# ============================================================
print("\n" + "="*60)
print("EXPERIMENT 4: Effect Size Comparison")
print("="*60)

fs_c0 = np.array([full_c0[n] for n in common_songs])
fs_c1 = np.array([full_c1[n] for n in common_songs])
ss_c0 = np.array([seg_c0[n] for n in common_songs])
ss_c1 = np.array([seg_c1[n] for n in common_songs])

fs_gap = fs_c1 - fs_c0
ss_gap = ss_c1 - ss_c0

print(f"\nN={len(common_songs)} songs (common to both datasets)")

print(f"\nFull songs:")
print(f"  C0: {fs_c0.mean():.4f} +/- {fs_c0.std():.4f}")
print(f"  C1: {fs_c1.mean():.4f} +/- {fs_c1.std():.4f}")
print(f"  C0-C1 gap: {fs_gap.mean():.4f} +/- {fs_gap.std():.4f}")
d_fs = fs_gap.mean() / (fs_gap.std() + 1e-9)
t_fs, p_fs = stats.ttest_rel(fs_c1, fs_c0)
print(f"  C0 vs C1: d={d_fs:.2f}, p={p_fs:.4f}")

print(f"\nMatched segments:")
print(f"  C0: {ss_c0.mean():.4f} +/- {ss_c0.std():.4f}")
print(f"  C1: {ss_c1.mean():.4f} +/- {ss_c1.std():.4f}")
print(f"  C0-C1 gap: {ss_gap.mean():.4f} +/- {ss_gap.std():.4f}")
d_ss = ss_gap.mean() / (ss_gap.std() + 1e-9)
t_ss, p_ss = stats.ttest_rel(ss_c1, ss_c0)
print(f"  C0 vs C1: d={d_ss:.2f}, p={p_ss:.4f}")

t_gap_comp, p_gap_comp = stats.ttest_rel(ss_gap, fs_gap)
print(f"\nGap comparison (seg vs full): diff={ss_gap.mean()-fs_gap.mean():.4f}, t={t_gap_comp:.2f}, p={p_gap_comp:.4f}")

agreement = np.mean(np.sign(ss_gap) == np.sign(fs_gap))
print(f"Gap direction agreement: {agreement:.1%}")

r_gap_corr, p_gap_corr = stats.pearsonr(fs_gap, ss_gap)
print(f"Gap correlation (full vs seg): r={r_gap_corr:.3f}, p={p_gap_corr:.4f}")

# ============================================================
# Experiment 5: MS-CLAP cross-audio
# ============================================================
print("\n" + "="*60)
print("EXPERIMENT 5: MS-CLAP Cross-Audio Comparison")
print("="*60)

if 'msclap' in seg_results and 'orig_seg_audio' in seg_results.get('msclap', {}):
    ms_orig = seg_results['msclap']['orig_seg_audio']
    ms_hach = seg_results['msclap']['hach_seg_audio']

    print("\nMS-CLAP alignment with matched segments:")
    print(f"\n  vs hachimi segment audio:")
    for c in ['C0_orig_lyrics', 'C1_original', 'C8_paraphrase']:
        if c in ms_hach:
            print(f"    {c}: {ms_hach[c]['mean']:.4f} [{ms_hach[c]['ci_lo']:.4f}, {ms_hach[c]['ci_hi']:.4f}]")

    print(f"\n  vs original segment audio:")
    for c in ['C0_orig_lyrics', 'C1_original', 'C8_paraphrase']:
        if c in ms_orig:
            print(f"    {c}: {ms_orig[c]['mean']:.4f} [{ms_orig[c]['ci_lo']:.4f}, {ms_orig[c]['ci_hi']:.4f}]")

    if 'C0_vs_C1' in ms_orig:
        print(f"\n  C0 vs C1 (orig audio): d={ms_orig['C0_vs_C1']['d']:.2f}, p={ms_orig['C0_vs_C1']['p']:.4f}")
    if 'C0_vs_C1' in ms_hach:
        print(f"  C0 vs C1 (hach audio): d={ms_hach['C0_vs_C1']['d']:.2f}, p={ms_hach['C0_vs_C1']['p']:.4f}")

    ms_full = full_results.get('msclap', {}).get('hach_audio', {})
    print(f"\n  Full songs comparison:")
    for c in ['C0_orig_lyrics', 'C1_original']:
        if c in ms_full and c in ms_hach:
            print(f"    {c}: full={ms_full[c]['mean']:.4f}, seg_hach={ms_hach[c]['mean']:.4f}, seg_orig={ms_orig.get(c, {}).get('mean', 'N/A')}")
else:
    print("  MS-CLAP cross-audio data not available")

# ============================================================
# Summary
# ============================================================
print("\n" + "="*60)
print("SUMMARY OF ALL COMPARISON EXPERIMENTS")
print("="*60)

results_summary = {
    'experiment_1_duration': {
        'n': len(durations),
        'duration_range': [float(durations.min()), float(durations.max())],
        'duration_mean': float(durations.mean()),
        'r_c0_change': float(r_c0), 'p_c0_change': float(p_c0),
        'r_c1_change': float(r_c1), 'p_c1_change': float(p_c1),
        'r_gap_full': float(r_gap_full), 'p_gap_full': float(p_gap_full),
        'r_gap_seg': float(r_gap_seg), 'p_gap_seg': float(p_gap_seg),
        'short_gap_mean': float(c0c1_gaps_seg[short_mask].mean()),
        'long_gap_mean': float(c0c1_gaps_seg[long_mask].mean()),
        'short_vs_long_p': float(p_gap),
    },
    'experiment_2_quality': {
        'n_high': int(high_mask.sum()),
        'n_low': int(low_mask.sum()),
        'high_gap_mean': float(h_gap.mean()), 'high_d': float(d_h), 'high_p': float(p_h),
        'low_gap_mean': float(l_gap.mean()), 'low_d': float(d_l), 'low_p': float(p_l),
        'high_vs_low_p': float(p_hl),
    },
    'experiment_3_cross_audio': {
        'n': n_cross,
        'hach_c0': float(hach_audio_c0.mean()), 'hach_c1': float(hach_audio_c1.mean()),
        'hach_gap': float(gap_hach.mean()), 'hach_p': float(p_ha),
        'orig_c0': float(orig_audio_c0.mean()), 'orig_c1': float(orig_audio_c1.mean()),
        'orig_gap': float(gap_orig.mean()), 'orig_p': float(p_oa),
        'gap_diff_p': float(p_cross),
    },
    'experiment_4_effect_size': {
        'n': len(common_songs),
        'full_c0': float(fs_c0.mean()), 'full_c1': float(fs_c1.mean()),
        'full_gap': float(fs_gap.mean()), 'full_d': float(d_fs), 'full_p': float(p_fs),
        'seg_c0': float(ss_c0.mean()), 'seg_c1': float(ss_c1.mean()),
        'seg_gap': float(ss_gap.mean()), 'seg_d': float(d_ss), 'seg_p': float(p_ss),
        'gap_diff': float(ss_gap.mean() - fs_gap.mean()),
        'gap_diff_p': float(p_gap_comp),
        'gap_agreement': float(agreement),
        'gap_correlation': float(r_gap_corr),
    },
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
}

out_file = RESULTS_DIR / "comparison_experiments_results.json"
with open(out_file, "w") as f:
    json.dump(results_summary, f, indent=2)
print(f"\nSaved to {out_file}")
print("DONE!")
