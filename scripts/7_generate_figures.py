#!/usr/bin/env python3
"""
Step 7: Generate paper figures from experiment results.

Produces:
- fig1_alignment.pdf: LAION CLAP alignment across all conditions
- fig7_cross_model.pdf: Cross-model comparison (LAION CLAP vs MS-CLAP)
- fig8_vocal_only.pdf: Vocal-only comparison (from old results)
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Noto Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ─── Paths (relative to project root) ────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUT = PROJECT_ROOT / "paper" / "figures"

# Load results
CLEAN = PROJECT_ROOT / "results" / "cleaned_results.json"
with open(CLEAN) as f:
    R = json.load(f)

PS = PROJECT_ROOT / "results" / "per_song_all_conditions.json"
with open(PS) as f:
    ps_data = json.load(f)

# Load old results (for vocal-only comparison)
OLD_RES = PROJECT_ROOT / "results" / "full_results_v2.json"
if OLD_RES.exists():
    with open(OLD_RES) as f:
        OLD = json.load(f)
else:
    OLD = None
    print("Warning: full_results_v2.json not found, skipping fig8")

CONDS_ORIG = ['C0_orig_lyrics', 'C1_original', 'C2_char_shuffle', 'C3_reversed',
              'C4_english_nonsense', 'C5_random_phonemes', 'C6_semantic_inversion',
              'C6b_semantic_negation', 'C7_random_chinese']

# 10 conditions with C8 inserted
CONDS_ALL = ['C0_orig_lyrics', 'C8_paraphrase'] + CONDS_ORIG[1:]

# Descriptive short names
CONDS_SHORT = [
    'Orig.\nLyrics', 'Para-\nphrase', 'Hachimi', 'Char-\nShuffle',
    'Reversed', 'English\nNonsense', 'Random\nPhonemes',
    'Sem.\nInversion', 'Sem.\nNegation', 'Random\nChinese',
]
CONDS_ABBREV = [
    'Orig.', 'Paraphr.', 'Hachimi', 'Shuffle', 'Reverse',
    'English', 'Rand.Phon.', 'Sem.Inv.', 'Sem.Neg.', 'Rand.Ch.'
]

colors = ['#4C72B0', '#E377C2', '#DD8452', '#55A868', '#C44E52', '#8172B3',
          '#CCB974', '#64B5CD', '#B07AA1', '#8C564B']

# Get cleaned LAION CLAP data
laion = R['laion_clap']['hach_audio']
msclap = R['msclap']['hach_audio']

# ============================================================
# Fig 1: LAION CLAP alignment across all conditions (bar chart)
# ============================================================
print("Generating Fig 1: alignment bar chart...")
fig, ax = plt.subplots(figsize=(7.5, 3.8))

means = [laion['C0_orig_lyrics']['mean']]
ci_lo = [laion['C0_orig_lyrics']['ci_lo']]
ci_hi = [laion['C0_orig_lyrics']['ci_hi']]
means.append(laion['C8']['mean'])
ci_lo.append(laion['C8']['ci_lo'])
ci_hi.append(laion['C8']['ci_hi'])
for c in CONDS_ORIG[1:]:
    means.append(laion[c]['mean'])
    ci_lo.append(laion[c]['ci_lo'])
    ci_hi.append(laion[c]['ci_hi'])

xerr = [[m - lo for m, lo in zip(means, ci_lo)],
        [hi - m for m, hi in zip(means, ci_hi)]]

bars = ax.bar(range(len(CONDS_ALL)), means, yerr=xerr,
              color=colors, alpha=0.85, edgecolor='black', linewidth=0.5, width=0.7)

for i, (bar, m) in enumerate(zip(bars, means)):
    ax.text(bar.get_x() + bar.get_width()/2., m + xerr[1][i] + 0.002,
            f'{m:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xticks(range(len(CONDS_ALL)))
ax.set_xticklabels(CONDS_SHORT, fontsize=8)
ax.set_ylabel('Cosine Similarity')
ax.set_title('LAION CLAP Audio-Text Alignment (vs. Hachimi Audio)', fontsize=12, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(OUT / 'fig1_alignment.pdf')
plt.savefig(OUT / 'fig1_alignment.png')
plt.close()
print("  Saved fig1_alignment")

# ============================================================
# Fig 7: Cross-model comparison (LAION CLAP vs MS-CLAP)
# ============================================================
print("Generating Fig 7: cross-model comparison...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Build means arrays
laion_means = [laion['C0_orig_lyrics']['mean'], laion['C8']['mean']]
for c in CONDS_ORIG[1:]:
    laion_means.append(laion[c]['mean'])

ms_means = [msclap['C0_orig_lyrics']['mean'], msclap['C8']['mean']]
for c in CONDS_ORIG[1:]:
    ms_means.append(msclap[c]['mean'])

# Left panel: all 10 conditions
x = np.arange(len(CONDS_ALL))
w = 0.35
b1 = ax1.bar(x - w/2, laion_means, w, label='LAION CLAP', color='#4C72B0', alpha=0.8, edgecolor='black', linewidth=0.5)
b2 = ax1.bar(x + w/2, ms_means, w, label='Microsoft CLAP', color='#DD8452', alpha=0.8, edgecolor='black', linewidth=0.5)

ax1.set_xticks(x)
ax1.set_xticklabels(CONDS_ABBREV, fontsize=7.5, rotation=30, ha='right')
ax1.set_ylabel('Cosine Similarity')
ax1.set_title('(a) All Conditions', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Right panel: C0 vs C8 vs C1 focused comparison
conditions_3 = ['Original\nLyrics', 'Paraphrased\nLyrics', 'Hachimi\nNonsense']
laion_3 = [laion_means[0], laion_means[1], laion_means[2]]
ms_3 = [ms_means[0], ms_means[1], ms_means[2]]
x3 = np.arange(3)
b3 = ax2.bar(x3 - w/2, laion_3, w, label='LAION CLAP', color='#4C72B0', alpha=0.8, edgecolor='black', linewidth=0.5)
b4 = ax2.bar(x3 + w/2, ms_3, w, label='Microsoft CLAP', color='#DD8452', alpha=0.8, edgecolor='black', linewidth=0.5)

# Add value labels
for bar_group, vals in [(b3, laion_3), (b4, ms_3)]:
    for bar, v in zip(bar_group, vals):
        ax2.text(bar.get_x() + bar.get_width()/2., v + 0.005,
                 f'{v:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax2.set_xticks(x3)
ax2.set_xticklabels(conditions_3, fontsize=9)
ax2.set_ylabel('Cosine Similarity')
ax2.set_title('(b) Semantic Control: C0 vs C8 vs C1', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Annotate: C0 ≈ C8 in LAION
ax2.annotate('C0 $\\approx$ C8\n(d=$-$0.03)',
            xy=(0 - w/2, laion_3[0]), xytext=(0.5, min(laion_3) - 0.02),
            fontsize=9, color='#4C72B0', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#4C72B0', lw=1.5))

plt.tight_layout()
plt.savefig(OUT / 'fig7_cross_model.pdf')
plt.savefig(OUT / 'fig7_cross_model.png')
plt.close()
print("  Saved fig7_cross_model")

# ============================================================
# Fig 8: Vocal-only comparison (use old results)
# ============================================================
if OLD:
    print("Generating Fig 8: vocal-only comparison...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    laion_vocal_hach = OLD['complete_analysis']['laion_clap']['vocal_only']['vs_hach_vocals']
    ms_vocal_hach = OLD['complete_analysis']['msclap']['vocal_only']['vs_hach_vocals']
    laion_full_hach = OLD['complete_analysis']['laion_clap']['full_mix']['vs_hach_audio']

    for ax_obj, (full_data, vocal_data, title) in zip(
        [ax1, ax2],
        [(laion_full_hach, laion_vocal_hach, 'LAION CLAP'),
         (ms_vocal_hach, ms_vocal_hach, 'Microsoft CLAP')]):

        x = np.arange(len(CONDS_ORIG))
        w = 0.35

        full_means = [full_data[c]['mean'] for c in CONDS_ORIG]
        vocal_means = [vocal_data[c]['mean'] for c in CONDS_ORIG]

        b1 = ax_obj.bar(x - w/2, full_means, w, label='Full Mix', color='#4C72B0', alpha=0.8, edgecolor='black', linewidth=0.5)
        b2 = ax_obj.bar(x + w/2, vocal_means, w, label='Vocal Only', color='#DD8452', alpha=0.8, edgecolor='black', linewidth=0.5)

        ax_obj.set_xticks(x)
        ax_obj.set_xticklabels(CONDS_ABBREV[:1] + CONDS_ABBREV[2:], fontsize=7.5, rotation=30, ha='right')
        ax_obj.set_ylabel('Cosine Similarity')
        ax_obj.set_title(title, fontsize=12, fontweight='bold')
        ax_obj.legend(fontsize=10)
        ax_obj.spines['top'].set_visible(False)
        ax_obj.spines['right'].set_visible(False)

        c0_mean = full_data['C0_orig_lyrics']['mean']
        c1_mean = full_data['C1_original']['mean']
        if c1_mean > c0_mean:
            ax_obj.annotate('Orig. < Hachimi***',
                            xy=(0, c0_mean), xytext=(2, min(full_means) - 0.01),
                            fontsize=9, color='#C44E52', fontweight='bold',
                            arrowprops=dict(arrowstyle='->', color='#C44E52', lw=1.5))

    plt.tight_layout()
    plt.savefig(OUT / 'fig8_vocal_only.pdf')
    plt.savefig(OUT / 'fig8_vocal_only.png')
    plt.close()
    print("  Saved fig8_vocal_only")

print("\nAll figures generated!")
