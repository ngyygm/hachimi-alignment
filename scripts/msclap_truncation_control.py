#!/usr/bin/env python3
"""
Experiment 2: msCLAP Truncation Control

Reruns msCLAP alignment with LENGTH-MATCHED text:
- C0_truncated: original lyrics truncated to same length as C8 paraphrase
- C8_truncated: paraphrase (already short, truncate to same length)

This removes the text-length confound (C0 mean=925 chars vs C8 mean=277 chars).
"""
import os, sys, json, time
import numpy as np
import torch
from pathlib import Path
from scipy import stats

SEED = 42
np.random.seed(SEED)

from pathlib import Path as _Path
SCRIPT_DIR = _Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data" / "哈基米音乐和原曲对照合集"

TRUNC = 500

print("=" * 60)
print("EXPERIMENT 2: msCLAP TRUNCATION CONTROL")
print("=" * 60)

# Load conditions
with open(RESULTS_DIR / "conditions.json", "r", encoding="utf-8") as f:
    conditions = json.load(f)

# Get valid songs with C8
def get_audio_path(song_name, audio_type):
    d = DATA_DIR / song_name
    if not d.exists(): return None
    if audio_type == 'hach':
        files = [f for f in d.glob("hachimi-*.mp3") if all(x not in f.name for x in ['_vocals','_instrumental','_noreverb'])]
        return str(files[0]) if files else None
    return None

song_list_c8 = []
for name in sorted(conditions.keys()):
    if 'C8_paraphrase' in conditions[name] and get_audio_path(name, 'hach'):
        song_list_c8.append(name)

print(f"Songs with C8 + audio: {len(song_list_c8)}")

# Check text lengths
c0_lens = [len(conditions[n]['C0_orig_lyrics']) for n in song_list_c8]
c8_lens = [len(conditions[n]['C8_paraphrase']) for n in song_list_c8]
print(f"C0 text lengths: mean={np.mean(c0_lens):.0f}, median={np.median(c0_lens):.0f}")
print(f"C8 text lengths: mean={np.mean(c8_lens):.0f}, median={np.median(c8_lens):.0f}")

# Create length-matched versions
# Strategy: truncate both C0 and C8 to min(C0_len, C8_len, 500) per song
# This ensures identical maximum length
matched_texts = {}
for name in song_list_c8:
    c0_text = conditions[name]['C0_orig_lyrics']
    c8_text = conditions[name]['C8_paraphrase']
    c1_text = conditions[name].get('C1_original', '')

    # Length-match: truncate both to the shorter one's length, capped at 500
    max_len = min(len(c0_text), len(c8_text), TRUNC)

    matched_texts[name] = {
        'C0_matched': c0_text[:max_len],
        'C8_matched': c8_text[:max_len],
        'C1_matched': c1_text[:max_len] if c1_text else '',
        'match_length': max_len,
    }

match_lens = [matched_texts[n]['match_length'] for n in song_list_c8]
print(f"Matched length: mean={np.mean(match_lens):.0f}, median={np.median(match_lens):.0f}")

# ================================================================
# msCLAP computation
# ================================================================
print("\nLoading msCLAP...")
import msclap
ms_clap = msclap.CLAP()

def msclap_text_embed(texts, batch_size=1):
    """Process msCLAP text one-by-one to avoid variable-length tensor errors."""
    all_emb = []
    for i, text in enumerate(texts):
        try:
            emb = ms_clap.get_text_embeddings([text])
            if isinstance(emb, torch.Tensor): emb = emb.cpu().numpy()
            if emb.ndim == 1: emb = emb.reshape(1, -1)
            all_emb.append(emb[0])
        except Exception as e:
            if i < 3:
                print(f"  Text embed error at {i}: {e}")
            all_emb.append(np.zeros(1024))
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(texts)} done")
    return np.array(all_emb)

def msclap_audio_embed(paths, batch_size=16):
    all_emb = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i+batch_size]
        try:
            embs = ms_clap.get_audio_embeddings(batch, resample=True)
            if isinstance(embs, torch.Tensor): embs = embs.cpu().numpy()
            all_emb.append(embs)
        except Exception as e:
            print(f"  Error: {e}")
            all_emb.append(np.zeros((len(batch), 1024)))
    return np.vstack(all_emb)

def cos_sim(a, b):
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return np.sum(an * bn, axis=1)

def bootstrap_ci(data, n_boot=10000, alpha=0.05):
    s = []
    for _ in range(n_boot):
        idx = np.random.choice(len(data), len(data), replace=True)
        s.append(np.mean(data[idx]))
    s = np.array(s)
    return float(s.mean()), float(np.percentile(s, 100*alpha/2)), float(np.percentile(s, 100*(1-alpha/2)))

# Compute embeddings
print("Computing msCLAP text embeddings...")
# Standard truncation (original setup)
c0_std = [conditions[n]['C0_orig_lyrics'][:TRUNC] for n in song_list_c8]
c8_std = [conditions[n]['C8_paraphrase'][:TRUNC] for n in song_list_c8]
c1_std = [conditions[n].get('C1_original', '')[:TRUNC] for n in song_list_c8]

c0_std_embs = msclap_text_embed(c0_std)
c8_std_embs = msclap_text_embed(c8_std)
c1_std_embs = msclap_text_embed(c1_std)

# Length-matched truncation
c0_mat = [matched_texts[n]['C0_matched'] for n in song_list_c8]
c8_mat = [matched_texts[n]['C8_matched'] for n in song_list_c8]
c1_mat = [matched_texts[n]['C1_matched'] for n in song_list_c8]

c0_mat_embs = msclap_text_embed(c0_mat)
c8_mat_embs = msclap_text_embed(c8_mat)
c1_mat_embs = msclap_text_embed(c1_mat)

# Audio
print("Computing msCLAP audio embeddings...")
audio_paths = [get_audio_path(n, 'hach') for n in song_list_c8]
audio_embs = msclap_audio_embed(audio_paths)
print(f"Audio shape: {audio_embs.shape}")

# ================================================================
# Results
# ================================================================
print("\n" + "=" * 60)
print("msCLAP TRUNCATION CONTROL RESULTS")
print("=" * 60)

# Standard truncation
c0_std_align = cos_sim(c0_std_embs, audio_embs)
c8_std_align = cos_sim(c8_std_embs, audio_embs)
c1_std_align = cos_sim(c1_std_embs, audio_embs)

print("\n--- Standard 500-char truncation ---")
m0, l0, h0 = bootstrap_ci(c0_std_align)
m8, l8, h8 = bootstrap_ci(c8_std_align)
m1, l1, h1 = bootstrap_ci(c1_std_align)
print(f"  C0: {m0:.4f} [{l0:.4f}, {h0:.4f}]")
print(f"  C8: {m8:.4f} [{l8:.4f}, {h8:.4f}]")
print(f"  C1: {m1:.4f} [{l1:.4f}, {h1:.4f}]")

t, p = stats.ttest_rel(c0_std_align, c8_std_align)
d = (c0_std_align - c8_std_align).mean() / ((c0_std_align - c8_std_align).std() + 1e-9)
sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
print(f"  C0 vs C8: d={d:.3f}, p={p:.4f} ({sig})")

# Length-matched
c0_mat_align = cos_sim(c0_mat_embs, audio_embs)
c8_mat_align = cos_sim(c8_mat_embs, audio_embs)
c1_mat_align = cos_sim(c1_mat_embs, audio_embs)

print("\n--- Length-matched truncation ---")
m0m, l0m, h0m = bootstrap_ci(c0_mat_align)
m8m, l8m, h8m = bootstrap_ci(c8_mat_align)
m1m, l1m, h1m = bootstrap_ci(c1_mat_align)
print(f"  C0_matched: {m0m:.4f} [{l0m:.4f}, {h0m:.4f}]")
print(f"  C8_matched: {m8m:.4f} [{l8m:.4f}, {h8m:.4f}]")
print(f"  C1_matched: {m1m:.4f} [{l1m:.4f}, {h1m:.4f}]")

t_m, p_m = stats.ttest_rel(c0_mat_align, c8_mat_align)
d_m = (c0_mat_align - c8_mat_align).mean() / ((c0_mat_align - c8_mat_align).std() + 1e-9)
sig_m = '***' if p_m < 0.001 else ('**' if p_m < 0.01 else ('*' if p_m < 0.05 else 'n.s.'))
print(f"  C0_matched vs C8_matched: d={d_m:.3f}, p={p_m:.4f} ({sig_m})")

# C1 comparison
t_c1, p_c1 = stats.ttest_rel(c1_mat_align, c0_mat_align)
d_c1 = (c1_mat_align - c0_mat_align).mean() / ((c1_mat_align - c0_mat_align).std() + 1e-9)
sig_c1 = '***' if p_c1 < 0.001 else ('**' if p_c1 < 0.01 else ('*' if p_c1 < 0.05 else 'n.s.'))
print(f"  C1_matched vs C0_matched: d={d_c1:.3f}, p={p_c1:.4f} ({sig_c1})")

# Key question: does the C0-C8 gap survive length matching?
print(f"\n--- KEY COMPARISON ---")
print(f"  Standard truncation: C0-C8 gap = d={d:.3f} (p={p:.4f})")
print(f"  Length-matched:      C0-C8 gap = d={d_m:.3f} (p={p_m:.4f})")
if p_m > 0.05:
    print(f"  → C0-C8 gap DISAPPEARS with length matching → confounded by text length")
else:
    print(f"  → C0-C8 gap SURVIVES length matching → genuine semantic sensitivity")

# Save
output = {
    'description': 'msCLAP truncation control experiment',
    'n_songs': len(song_list_c8),
    'standard_truncation': {
        'C0': {'mean': float(m0), 'ci': [float(l0), float(h0)]},
        'C8': {'mean': float(m8), 'ci': [float(l8), float(h8)]},
        'C1': {'mean': float(m1), 'ci': [float(l1), float(h1)]},
        'C0_vs_C8': {'d': float(d), 'p': float(p)},
    },
    'length_matched': {
        'C0': {'mean': float(m0m), 'ci': [float(l0m), float(h0m)]},
        'C8': {'mean': float(m8m), 'ci': [float(l8m), float(h8m)]},
        'C1': {'mean': float(m1m), 'ci': [float(l1m), float(h1m)]},
        'C0_vs_C8': {'d': float(d_m), 'p': float(p_m)},
        'C1_vs_C0': {'d': float(d_c1), 'p': float(p_c1)},
    },
    'per_song': {
        'c0_std': c0_std_align.tolist(),
        'c8_std': c8_std_align.tolist(),
        'c0_matched': c0_mat_align.tolist(),
        'c8_matched': c8_mat_align.tolist(),
    },
}

out_file = RESULTS_DIR / "msclap_truncation_results.json"
with open(out_file, "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nSaved to {out_file}")
print("DONE!")
