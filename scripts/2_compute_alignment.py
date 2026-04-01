#!/usr/bin/env python3
"""
Step 2: Compute CLAP and msCLAP audio-text alignment scores.

Loads conditions.json (text conditions C0-C8) and audio files from data/,
computes cosine similarity between audio and text embeddings for all
conditions using LAION CLAP and Microsoft CLAP.  Saves cleaned_results.json
and per_song_all_conditions.json.
"""
import os, sys, warnings, json, time
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import numpy as np
import torch
from pathlib import Path
from scipy import stats

SEED = 42
np.random.seed(SEED)

# ─── Paths (relative to project root) ────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = PROJECT_ROOT / "data" / "哈基米音乐和原曲对照合集"
RESULTS_DIR = PROJECT_ROOT / "results"
CONDITIONS_FILE = RESULTS_DIR / "conditions.json"

with open(CONDITIONS_FILE, "r", encoding="utf-8") as f:
    conditions = json.load(f)

print(f"Loaded {len(conditions)} songs")

# ============================================================
# [1] Audio paths
# ============================================================
def get_audio_path(song_name, audio_type):
    d = DATA_DIR / song_name
    if not d.exists(): return None
    if audio_type == 'hach':
        files = [f for f in d.glob("hachimi-*.mp3") if all(x not in f.name for x in ['_vocals','_instrumental','_noreverb'])]
        return str(files[0]) if files else None
    elif audio_type == 'orig':
        files = [f for f in d.glob("raw-*.mp3") if all(x not in f.name for x in ['_vocals','_instrumental','_noreverb'])]
        return str(files[0]) if files else None
    return None

hach_paths, orig_paths = {}, {}
for name in sorted(conditions.keys()):
    for atype, store in [('hach', hach_paths), ('orig', orig_paths)]:
        p = get_audio_path(name, atype)
        if p and os.path.exists(p): store[name] = p

valid_full = sorted(set(hach_paths.keys()) & set(orig_paths.keys()))
print(f"Songs with both audio: {len(valid_full)}")

# Also check C8 availability
valid_c8 = [n for n in valid_full if 'C8_paraphrase' in conditions[n]]
print(f"Songs with C8: {len(valid_c8)}")

# ============================================================
# [2] CLAP alignment (LAION)
# ============================================================
print("\n[2] LAION CLAP alignment...")
from laion_clap import CLAP_Module
clap = CLAP_Module(enable_fusion=False)
clap.load_ckpt()
clap = clap.cuda().eval()
print("  CLAP loaded")

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
            print(f"  Text embed error: {e}, filling zeros for {len(batch)} texts")
            embs.append(np.zeros((len(batch), 512)))
    return np.vstack(embs)

def clap_audio_embed_files(paths, batch_size=16):
    all_emb = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i+batch_size]
        try:
            embs = clap.get_audio_embedding_from_filelist(x=batch, use_tensor=False)
            if isinstance(embs, torch.Tensor): embs = embs.cpu().numpy()
            if isinstance(embs, list): embs = np.array(embs)
            all_emb.append(embs)
        except Exception as e:
            print(f"  Audio embed error: {e}, filling zeros for {len(batch)} files")
            all_emb.append(np.zeros((len(batch), 512)))
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

def paired_test(a, b):
    """Paired t-test with Cohen's d and bootstrap CI."""
    delta = a - b
    t, p = stats.ttest_rel(a, b)
    d = delta.mean() / (delta.std() + 1e-9)
    m, ci_lo, ci_hi = bootstrap_ci(delta)
    return {
        'mean_diff': float(delta.mean()),
        'd': float(d),
        'p': float(p),
        'p_bonf': float(min(p * 10, 1.0)),  # Bonferroni for ~10 comparisons
        'ci_lo': float(ci_lo),
        'ci_hi': float(ci_hi),
    }

# Text embeddings for all conditions
CONDS = ['C0_orig_lyrics', 'C1_original', 'C2_char_shuffle', 'C3_reversed',
         'C4_english_nonsense', 'C5_random_phonemes', 'C6_semantic_inversion',
         'C6b_semantic_negation', 'C7_random_chinese']

print("  Computing text embeddings...")
text_embs = {}
for c in CONDS:
    texts = [conditions[n][c] for n in valid_full]
    text_embs[c] = clap_text_embed(texts)
    print(f"    {c}: done")

# C8 paraphrase embeddings (for songs that have it)
if valid_c8:
    c8_texts = [conditions[n]['C8_paraphrase'] for n in valid_c8]
    text_embs['C8_paraphrase'] = clap_text_embed(c8_texts)
    print(f"    C8_paraphrase: done ({len(valid_c8)} songs)")

# Audio embeddings
print("  Computing audio embeddings...")
hach_audio = clap_audio_embed_files([hach_paths[n] for n in valid_full])
orig_audio = clap_audio_embed_files([orig_paths[n] for n in valid_full])
print(f"    hach_audio: {hach_audio.shape}, orig_audio: {orig_audio.shape}")

# Compute alignment
print("\n  === LAION CLAP Results (cleaned data) ===")
laion_results = {}

for audio_name, audio_embs in [("hach_audio", hach_audio), ("orig_audio", orig_audio)]:
    laion_results[audio_name] = {}
    print(f"\n  vs {audio_name}:")
    for c in CONDS:
        align = cos_sim(text_embs[c], audio_embs)
        m, lo, hi = bootstrap_ci(align)
        laion_results[audio_name][c] = {
            'mean': float(align.mean()),
            'std': float(align.std()),
            'ci_lo': lo,
            'ci_hi': hi,
        }
        sig = '***' if laion_results[audio_name][c].get('p_bonf',0) < 0.001 else ''
        print(f"    {c}: mean={align.mean():.4f} (CI: [{lo:.4f}, {hi:.4f}])")

    # C0 vs C1
    c0_align = cos_sim(text_embs['C0_orig_lyrics'], audio_embs)
    c1_align = cos_sim(text_embs['C1_original'], audio_embs)
    test = paired_test(c0_align, c1_align)
    laion_results[audio_name]['C0_vs_C1'] = test
    sig = '***' if test['p'] < 0.001 else ('**' if test['p'] < 0.01 else ('*' if test['p'] < 0.05 else 'n.s.'))
    print(f"    C0 vs C1: d={test['d']:.2f}, p={test['p']:.4f} ({sig})")

    # Store per-song alignment
    laion_results[audio_name]['per_song'] = {
        'C0': c0_align.tolist(),
        'C1': c1_align.tolist(),
        'song_names': valid_full,
    }

# C8 paraphrase results
if valid_c8:
    print("\n  === C8 Paraphrase Results (cleaned data) ===")
    hach_c8 = hach_audio[[valid_full.index(n) for n in valid_c8]]
    orig_c8 = orig_audio[[valid_full.index(n) for n in valid_c8]]

    for audio_name, audio_embs_sub in [("hach_audio", hach_c8), ("orig_audio", orig_c8)]:
        c0_align = cos_sim(text_embs['C0_orig_lyrics'][[valid_full.index(n) for n in valid_c8]], audio_embs_sub)
        c1_align = cos_sim(text_embs['C1_original'][[valid_full.index(n) for n in valid_c8]], audio_embs_sub)
        c8_align = cos_sim(text_embs['C8_paraphrase'], audio_embs_sub)

        m, lo, hi = bootstrap_ci(c8_align)
        laion_results[audio_name]['C8'] = {
            'mean': float(c8_align.mean()),
            'std': float(c8_align.std()),
            'ci_lo': lo,
            'ci_hi': hi,
        }

        # C0 vs C8
        test = paired_test(c0_align, c8_align)
        laion_results[audio_name]['C0_vs_C8'] = test
        sig = '***' if test['p'] < 0.001 else ('**' if test['p'] < 0.01 else ('*' if test['p'] < 0.05 else 'n.s.'))
        print(f"  vs {audio_name}:")
        print(f"    C0: {c0_align.mean():.4f}, C1: {c1_align.mean():.4f}, C8: {c8_align.mean():.4f}")
        print(f"    C0 vs C1: d={paired_test(c0_align,c1_align)['d']:.2f}, p={paired_test(c0_align,c1_align)['p']:.4f}")
        print(f"    C0 vs C8: d={test['d']:.2f}, p={test['p']:.4f} ({sig})")

        # C8 vs C1
        test2 = paired_test(c8_align, c1_align)
        laion_results[audio_name]['C8_vs_C1'] = test2
        sig2 = '***' if test2['p'] < 0.001 else ('**' if test2['p'] < 0.01 else ('*' if test2['p'] < 0.05 else 'n.s.'))
        print(f"    C8 vs C1: d={test2['d']:.2f}, p={test2['p']:.4f} ({sig2})")

# ============================================================
# [3] Microsoft CLAP alignment (with 500-char truncation)
# ============================================================
print("\n\n[3] Microsoft CLAP alignment (500-char truncation)...")

try:
    import msclap
    ms_clap = msclap.CLAP()
    print("  Microsoft CLAP loaded")

    TRUNC = 500

    def msclap_text_embed(texts, batch_size=32):
        all_emb = []
        for i in range(0, len(texts), batch_size):
            batch = [t[:TRUNC] for t in texts[i:i+batch_size]]
            try:
                embs = ms_clap.get_text_embeddings(batch)
                if isinstance(embs, torch.Tensor): embs = embs.cpu().numpy()
                all_emb.append(embs)
            except Exception as e:
                print(f"  MS-CLAP text error: {e}, filling zeros")
                all_emb.append(np.zeros((len(batch), 1024)))
        return np.vstack(all_emb)

    def msclap_audio_embed_files(paths, batch_size=16):
        all_emb = []
        for i in range(0, len(paths), batch_size):
            batch = paths[i:i+batch_size]
            try:
                embs = ms_clap.get_audio_embeddings(batch, resample=True)
                if isinstance(embs, torch.Tensor): embs = embs.cpu().numpy()
                all_emb.append(embs)
            except Exception as e:
                print(f"  MS-CLAP audio error: {e}, filling zeros")
                all_emb.append(np.zeros((len(batch), 1024)))
        return np.vstack(all_emb)

    print("  Computing MS-CLAP text embeddings (truncated to 500 chars)...")
    ms_text_embs = {}
    for c in CONDS:
        texts = [conditions[n][c][:TRUNC] for n in valid_full]
        ms_text_embs[c] = msclap_text_embed(texts)
        print(f"    {c}: done")

    if valid_c8:
        c8_texts = [conditions[n]['C8_paraphrase'][:TRUNC] for n in valid_c8]
        ms_text_embs['C8_paraphrase'] = msclap_text_embed(c8_texts)
        print(f"    C8_paraphrase: done")

    print("  Computing MS-CLAP audio embeddings...")
    hach_ms = msclap_audio_embed_files([hach_paths[n] for n in valid_full])
    orig_ms = msclap_audio_embed_files([orig_paths[n] for n in valid_full])
    print(f"    hach_ms: {hach_ms.shape}, orig_ms: {orig_ms.shape}")

    ms_results = {}
    print("\n  === Microsoft CLAP Results (cleaned data, 500-char trunc) ===")

    for audio_name, audio_embs in [("hach_audio", hach_ms), ("orig_audio", orig_ms)]:
        ms_results[audio_name] = {}
        print(f"\n  vs {audio_name}:")
        for c in CONDS:
            align = cos_sim(ms_text_embs[c], audio_embs)
            m, lo, hi = bootstrap_ci(align)
            ms_results[audio_name][c] = {
                'mean': float(align.mean()),
                'std': float(align.std()),
                'ci_lo': lo,
                'ci_hi': hi,
            }
            print(f"    {c}: mean={align.mean():.4f} (CI: [{lo:.4f}, {hi:.4f}])")

        # C0 vs C1
        c0_a = cos_sim(ms_text_embs['C0_orig_lyrics'], audio_embs)
        c1_a = cos_sim(ms_text_embs['C1_original'], audio_embs)
        test = paired_test(c0_a, c1_a)
        ms_results[audio_name]['C0_vs_C1'] = test
        sig = '***' if test['p'] < 0.001 else ('**' if test['p'] < 0.01 else ('*' if test['p'] < 0.05 else 'n.s.'))
        print(f"    C0 vs C1: d={test['d']:.2f}, p={test['p']:.4f} ({sig})")

    # C8 paraphrase for msCLAP
    if valid_c8:
        print("\n  === C8 Paraphrase MS-CLAP Results ===")
        hach_c8_ms = hach_ms[[valid_full.index(n) for n in valid_c8]]
        orig_c8_ms = orig_ms[[valid_full.index(n) for n in valid_c8]]

        for audio_name, audio_embs_sub in [("hach_audio", hach_c8_ms)]:
            c0_a = cos_sim(ms_text_embs['C0_orig_lyrics'][[valid_full.index(n) for n in valid_c8]], audio_embs_sub)
            c1_a = cos_sim(ms_text_embs['C1_original'][[valid_full.index(n) for n in valid_c8]], audio_embs_sub)
            c8_a = cos_sim(ms_text_embs['C8_paraphrase'], audio_embs_sub)

            m, lo, hi = bootstrap_ci(c8_a)
            ms_results[audio_name]['C8'] = {
                'mean': float(c8_a.mean()),
                'std': float(c8_a.std()),
                'ci_lo': lo,
                'ci_hi': hi,
            }

            test08 = paired_test(c0_a, c8_a)
            test01 = paired_test(c0_a, c1_a)
            test81 = paired_test(c8_a, c1_a)
            ms_results[audio_name]['C0_vs_C8'] = test08
            ms_results[audio_name]['C8_vs_C1'] = test81

            print(f"  vs {audio_name}:")
            print(f"    C0: {c0_a.mean():.4f}, C1: {c1_a.mean():.4f}, C8: {c8_a.mean():.4f}")
            print(f"    C0 vs C1: d={test01['d']:.2f}, p={test01['p']:.4f}")
            print(f"    C0 vs C8: d={test08['d']:.2f}, p={test08['p']:.4f}")
            print(f"    C8 vs C1: d={test81['d']:.2f}, p={test81['p']:.4f}")

except Exception as e:
    print(f"  Microsoft CLAP failed: {e}")
    import traceback
    traceback.print_exc()
    ms_results = {}

# ============================================================
# [4] Save all results
# ============================================================
print("\n[4] Saving results...")

RESULTS_FILE = RESULTS_DIR / "cleaned_results.json"
save_data = {
    'description': 'CLAP and msCLAP alignment results with cleaned conditions.json (metadata removed from C0 and C8)',
    'n_songs': len(valid_full),
    'n_songs_c8': len(valid_c8),
    'laion_clap': laion_results,
    'msclap': ms_results,
    'conditions_used': CONDS,
    'song_list': valid_full,
    'song_list_c8': valid_c8,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
}

with open(RESULTS_FILE, "w") as f:
    json.dump(save_data, f, indent=2, default=str)

print(f"  Saved to {RESULTS_FILE}")
print("\nDONE!")
