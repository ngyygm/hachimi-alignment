#!/usr/bin/env python3
"""
Experiment 1: Homophone Condition — Phonology-Controlled Semantic Manipulation

For each song's original lyrics, create a "homophone" version:
- Replace each character with a DIFFERENT character that has the SAME pinyin
- This preserves phonology (same sounds) but destroys semantics (different meaning)

If alignment stays the same → phonology dominates
If alignment drops → semantics matters
"""
import os, sys, json, time, random
import numpy as np
import torch
from pathlib import Path
from pypinyin import pinyin, Style
from scipy import stats

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

from pathlib import Path as _Path
SCRIPT_DIR = _Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data" / "哈基米音乐和原曲对照合集"

# ================================================================
# Step 1: Build homophone dictionary
# ================================================================
print("=" * 60)
print("EXPERIMENT 1: HOMOPHONE PHONOLOGY CONTROL")
print("=" * 60)

# Load a Chinese character dictionary
# We'll build a pinyin→chars mapping from the lyrics themselves + common chars
from collections import defaultdict
pinyin_to_chars = defaultdict(set)

# First, collect all characters from conditions.json
with open(RESULTS_DIR / "conditions.json", "r", encoding="utf-8") as f:
    conditions = json.load(f)

all_chars = set()
for song_conds in conditions.values():
    for cond_name, text in song_conds.items():
        all_chars.update(text)

print(f"Total unique characters in lyrics: {len(all_chars)}")

# Map each char to its pinyin
char_to_pinyin = {}
for ch in all_chars:
    if '\u4e00' <= ch <= '\u9fff':  # CJK range
        py = pinyin(ch, style=Style.TONE3, heteronym=False)
        if py:
            char_to_pinyin[ch] = py[0][0]
            pinyin_to_chars[py[0][0]].add(ch)

print(f"Characters with pinyin: {len(char_to_pinyin)}")
print(f"Unique pinyin: {len(pinyin_to_chars)}")

# ================================================================
# Step 2: Generate homophone versions
# ================================================================
def create_homophone(text, seed=42):
    """Replace each Chinese character with a different character having the same pinyin."""
    rng = random.Random(seed)
    result = []
    replaced = 0
    total_cjk = 0

    for ch in text:
        if '\u4e00' <= ch <= '\u9fff':
            total_cjk += 1
            py = char_to_pinyin.get(ch)
            if py:
                candidates = pinyin_to_chars[py] - {ch}
                if candidates:
                    replacement = rng.choice(list(candidates))
                    result.append(replacement)
                    replaced += 1
                else:
                    result.append(ch)  # no alternative, keep original
            else:
                result.append(ch)
        else:
            result.append(ch)

    return ''.join(result), replaced, total_cjk

# Generate for all songs
print("\nGenerating homophone conditions...")
homophone_texts = {}
replaced_stats = []
song_list = sorted(conditions.keys())

# Get valid songs (those with audio)
def get_audio_path(song_name, audio_type):
    d = DATA_DIR / song_name
    if not d.exists(): return None
    if audio_type == 'hach':
        files = [f for f in d.glob("hachimi-*.mp3") if all(x not in f.name for x in ['_vocals','_instrumental','_noreverb'])]
        return str(files[0]) if files else None
    return None

valid_songs = []
for name in song_list:
    if get_audio_path(name, 'hach'):
        valid_songs.append(name)
print(f"Songs with audio: {len(valid_songs)}")

for name in valid_songs:
    orig_text = conditions[name].get('C0_orig_lyrics', '')
    if orig_text:
        homo_text, n_replaced, n_total = create_homophone(orig_text)
        homophone_texts[name] = homo_text
        replaced_stats.append((n_replaced, n_total))

# Report replacement stats
replaced_rates = [r/t if t > 0 else 0 for r, t in replaced_stats]
print(f"Homophone replacement rate: {np.mean(replaced_rates):.1%} (median: {np.median(replaced_rates):.1%})")
print(f"Example: {valid_songs[0]}")
print(f"  Original: {conditions[valid_songs[0]]['C0_orig_lyrics'][:60]}...")
print(f"  Homophone: {homophone_texts[valid_songs[0]][:60]}...")

# ================================================================
# Step 3: Compute LAION CLAP alignment
# ================================================================
print("\n" + "=" * 60)
print("Computing CLAP alignment for homophone condition...")
print("=" * 60)

from laion_clap import CLAP_Module

clap = CLAP_Module(enable_fusion=False)
clap.load_ckpt()
clap = clap.cuda().eval()
print("LAION CLAP loaded")

# Text embeddings
def clap_text_embed(texts, batch_size=16):
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            emb = clap.get_text_embedding(batch)
            if isinstance(emb, torch.Tensor): emb = emb.cpu().numpy()
            if emb.ndim == 1: emb = emb.reshape(1, -1)
            embs.append(emb)
        except Exception as e:
            print(f"  Error: {e}")
            embs.append(np.zeros((len(batch), 512)))
    return np.vstack(embs)

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

# Compute homophone text embeddings
print("  Computing homophone text embeddings...")
homo_texts_list = [homophone_texts[n] for n in valid_songs]
homo_embs = clap_text_embed(homo_texts_list)

# Also compute C0 and C8 embeddings for comparison
print("  Computing C0 (original) text embeddings...")
c0_texts = [conditions[n]['C0_orig_lyrics'] for n in valid_songs]
c0_embs = clap_text_embed(c0_texts)

# C8 if available
c8_songs = [n for n in valid_songs if 'C8_paraphrase' in conditions[n]]
print(f"  Computing C8 (paraphrase) for {len(c8_songs)} songs...")
c8_texts = [conditions[n]['C8_paraphrase'] for n in c8_songs]
c8_embs = clap_text_embed(c8_texts)

# Audio embeddings
print("  Computing audio embeddings...")
audio_paths = [get_audio_path(n, 'hach') for n in valid_songs]
all_audio_embs = []
batch_size = 16
for i in range(0, len(audio_paths), batch_size):
    batch = audio_paths[i:i+batch_size]
    try:
        embs = clap.get_audio_embedding_from_filelist(x=batch, use_tensor=False)
        if isinstance(embs, torch.Tensor): embs = embs.cpu().numpy()
        if isinstance(embs, list): embs = np.array(embs)
        all_audio_embs.append(embs)
    except Exception as e:
        print(f"  Audio error: {e}")
        all_audio_embs.append(np.zeros((len(batch), 512)))
audio_embs = np.vstack(all_audio_embs)

print(f"  Audio embeddings: {audio_embs.shape}")

# ================================================================
# Step 4: Compute alignment scores
# ================================================================
print("\n" + "=" * 60)
print("ALIGNMENT RESULTS")
print("=" * 60)

# Homophone alignment
homo_align = cos_sim(homo_embs, audio_embs)
c0_align = cos_sim(c0_embs, audio_embs)

# Full results
m_homo, lo_homo, hi_homo = bootstrap_ci(homo_align)
m_c0, lo_c0, hi_c0 = bootstrap_ci(c0_align)

print(f"\n  C0 (original):     {m_c0:.4f} [{lo_c0:.4f}, {hi_c0:.4f}]")
print(f"  C_homo (homophone): {m_homo:.4f} [{lo_homo:.4f}, {hi_homo:.4f}]")

# Paired tests
t_homo_c0, p_homo_c0 = stats.ttest_rel(homo_align, c0_align)
d_homo_c0 = (homo_align - c0_align).mean() / ((homo_align - c0_align).std() + 1e-9)
w_homo_c0, wp_homo_c0 = stats.wilcoxon(homo_align, c0_align)
sig = '***' if p_homo_c0 < 0.001 else ('**' if p_homo_c0 < 0.01 else ('*' if p_homo_c0 < 0.05 else 'n.s.'))
print(f"\n  C_homo vs C0: d={d_homo_c0:.3f}, t={t_homo_c0:.3f}, p={p_homo_c0:.4f} ({sig})")
print(f"  Wilcoxon: W={w_homo_c0:.1f}, p={wp_homo_c0:.6f}")

# Win rate
homo_wins = np.sum(homo_align > c0_align)
print(f"  C_homo > C0: {homo_wins}/{len(valid_songs)} ({homo_wins/len(valid_songs):.1%})")

# Also compare C8 subset
if c8_songs:
    c8_idx = [valid_songs.index(n) for n in c8_songs]
    homo_c8 = homo_align[c8_idx]
    c0_c8 = c0_align[c8_idx]
    c8_align = cos_sim(c8_embs, audio_embs[c8_idx])

    m_c8, lo_c8, hi_c8 = bootstrap_ci(c8_align)
    print(f"\n  C8 (paraphrase):    {m_c8:.4f} [{lo_c8:.4f}, {hi_c8:.4f}]")
    print(f"  C_homo (subset):    {np.mean(homo_c8):.4f}")

    t_homo_c8, p_homo_c8 = stats.ttest_rel(homo_c8, c8_align)
    d_homo_c8 = (homo_c8 - c8_align).mean() / ((homo_c8 - c8_align).std() + 1e-9)
    sig8 = '***' if p_homo_c8 < 0.001 else ('**' if p_homo_c8 < 0.01 else ('*' if p_homo_c8 < 0.05 else 'n.s.'))
    print(f"  C_homo vs C8: d={d_homo_c8:.3f}, p={p_homo_c8:.4f} ({sig8})")

# Three-way comparison
print(f"\n{'='*60}")
print("THREE-WAY: C0 (original) vs C_homo (same-sound, diff-meaning) vs C8 (same-meaning, diff-words)")
print(f"{'='*60}")
print("  If phonology dominates: C_homo ≈ C0")
print("  If semantics dominates: C_homo ≈ C8 (both changed)")
print("  If both matter: C_homo between C0 and C8")

# Save results
output = {
    'description': 'Homophone phonology control experiment',
    'n_songs': len(valid_songs),
    'c0_mean': float(m_c0), 'c0_ci': [float(lo_c0), float(hi_c0)],
    'homo_mean': float(m_homo), 'homo_ci': [float(lo_homo), float(hi_homo)],
    'homo_vs_c0': {'d': float(d_homo_c0), 't': float(t_homo_c0), 'p': float(p_homo_c0), 'wilcoxon_p': float(wp_homo_c0), 'homo_wins': int(homo_wins)},
    'replacement_rate': float(np.mean(replaced_rates)),
    'per_song_c0': c0_align.tolist(),
    'per_song_homo': homo_align.tolist(),
    'song_names': valid_songs,
}

out_file = RESULTS_DIR / "homophone_results.json"
with open(out_file, "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nSaved to {out_file}")
print("DONE!")
