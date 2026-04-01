#!/usr/bin/env python3
"""
Step 5: Run CLAP/msCLAP alignment on temporally matched audio segments.

Instead of full songs, uses the aligned hachimi-original pairs from
segment_match_aligned.json to test whether temporal alignment improves
audio-text alignment scores.  Saves matched_segment_results.json.
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
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"
CONDITIONS_FILE = RESULTS_DIR / "conditions.json"
ALIGNED_FILE = RESULTS_DIR / "segment_match_aligned.json"

with open(CONDITIONS_FILE, "r", encoding="utf-8") as f:
    conditions = json.load(f)

with open(ALIGNED_FILE, "r", encoding="utf-8") as f:
    aligned = json.load(f)

print(f"Total songs in conditions: {len(conditions)}")
print(f"Aligned segment pairs: {len(aligned)}")

# ============================================================
# Filter to songs that exist in both aligned segments AND conditions
# ============================================================
valid_songs = [s for s in aligned if s['song'] in conditions]
print(f"Songs with both alignment data and text conditions: {len(valid_songs)}")

song_names = [s['song'] for s in valid_songs]
hach_seg_paths = [os.path.join(DATA_DIR, "matched_segments", s['files']['hachimi']) for s in valid_songs]
orig_seg_paths = [os.path.join(DATA_DIR, "matched_segments", s['files']['original_segment']) for s in valid_songs]

# Verify audio files exist
missing = [p for p in hach_seg_paths + orig_seg_paths if not os.path.exists(p)]
if missing:
    print(f"WARNING: {len(missing)} audio files missing!")
    for p in missing[:5]:
        print(f"  {p}")
    keep = [i for i in range(len(song_names))
            if os.path.exists(hach_seg_paths[i]) and os.path.exists(orig_seg_paths[i])]
    song_names = [song_names[i] for i in keep]
    hach_seg_paths = [hach_seg_paths[i] for i in keep]
    orig_seg_paths = [orig_seg_paths[i] for i in keep]
    print(f"After filtering: {len(song_names)} songs")

# Check C8 availability
valid_c8 = [n for n in song_names if 'C8_paraphrase' in conditions[n]]
print(f"Songs with C8 paraphrase: {len(valid_c8)}")

# ============================================================
# Utility functions
# ============================================================
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
    delta = a - b
    t, p = stats.ttest_rel(a, b)
    d = delta.mean() / (delta.std() + 1e-9)
    m, ci_lo, ci_hi = bootstrap_ci(delta)
    return {
        'mean_diff': float(delta.mean()),
        'd': float(d),
        'p': float(p),
        'ci_lo': float(ci_lo),
        'ci_hi': float(ci_hi),
    }

CONDS = ['C0_orig_lyrics', 'C1_original', 'C2_char_shuffle', 'C3_reversed',
         'C4_english_nonsense', 'C5_random_phonemes', 'C6_semantic_inversion',
         'C6b_semantic_negation', 'C7_random_chinese']

# ============================================================
# [1] LAION CLAP
# ============================================================
print("\n[1] LAION CLAP alignment on matched segments...")
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
            print(f"  Text embed error: {e}")
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
            print(f"  Audio embed error: {e}")
            all_emb.append(np.zeros((len(batch), 512)))
    return np.vstack(all_emb)

# Text embeddings
print("  Computing text embeddings...")
text_embs = {}
for c in CONDS:
    texts = [conditions[n][c] for n in song_names]
    text_embs[c] = clap_text_embed(texts)
    print(f"    {c}: done")

if valid_c8:
    c8_texts = [conditions[n]['C8_paraphrase'] for n in valid_c8]
    text_embs['C8_paraphrase'] = clap_text_embed(c8_texts)
    print(f"    C8_paraphrase: done ({len(valid_c8)} songs)")

# Audio embeddings (matched segments)
print("  Computing audio embeddings for matched segments...")
hach_seg_audio = clap_audio_embed_files(hach_seg_paths)
orig_seg_audio = clap_audio_embed_files(orig_seg_paths)
print(f"    hach_seg: {hach_seg_audio.shape}, orig_seg: {orig_seg_audio.shape}")

# Results
print("\n  === LAION CLAP Results (matched segments, N={}) ===".format(len(song_names)))
laion_results = {}

for audio_name, audio_embs in [("hach_seg_audio", hach_seg_audio), ("orig_seg_audio", orig_seg_audio)]:
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
        print(f"    {c}: mean={align.mean():.4f} (CI: [{lo:.4f}, {hi:.4f}])")

    # Key comparisons
    c0_a = cos_sim(text_embs['C0_orig_lyrics'], audio_embs)
    c1_a = cos_sim(text_embs['C1_original'], audio_embs)
    test = paired_test(c0_a, c1_a)
    laion_results[audio_name]['C0_vs_C1'] = test
    sig = '***' if test['p'] < 0.001 else ('**' if test['p'] < 0.01 else ('*' if test['p'] < 0.05 else 'n.s.'))
    print(f"    C0 vs C1: d={test['d']:.2f}, p={test['p']:.4f} ({sig})")

    # Per-song
    laion_results[audio_name]['per_song'] = {
        'C0': c0_a.tolist(),
        'C1': c1_a.tolist(),
        'song_names': song_names,
    }

# C8 paraphrase
if valid_c8:
    print("\n  === C8 Paraphrase (matched segments) ===")
    c8_idx = [song_names.index(n) for n in valid_c8]
    for audio_name, audio_embs in [("hach_seg_audio", hach_seg_audio), ("orig_seg_audio", orig_seg_audio)]:
        sub = audio_embs[c8_idx]
        c0_a = cos_sim(text_embs['C0_orig_lyrics'][c8_idx], sub)
        c1_a = cos_sim(text_embs['C1_original'][c8_idx], sub)
        c8_a = cos_sim(text_embs['C8_paraphrase'], sub)

        m, lo, hi = bootstrap_ci(c8_a)
        laion_results[audio_name]['C8'] = {
            'mean': float(c8_a.mean()),
            'std': float(c8_a.std()),
            'ci_lo': lo, 'ci_hi': hi,
        }

        test08 = paired_test(c0_a, c8_a)
        test81 = paired_test(c8_a, c1_a)
        laion_results[audio_name]['C0_vs_C8'] = test08
        laion_results[audio_name]['C8_vs_C1'] = test81

        sig08 = '***' if test08['p'] < 0.001 else ('**' if test08['p'] < 0.01 else ('*' if test08['p'] < 0.05 else 'n.s.'))
        sig81 = '***' if test81['p'] < 0.001 else ('**' if test81['p'] < 0.01 else ('*' if test81['p'] < 0.05 else 'n.s.'))
        print(f"  vs {audio_name}:")
        print(f"    C0: {c0_a.mean():.4f}, C1: {c1_a.mean():.4f}, C8: {c8_a.mean():.4f}")
        print(f"    C0 vs C1: d={paired_test(c0_a,c1_a)['d']:.2f}, p={paired_test(c0_a,c1_a)['p']:.4f}")
        print(f"    C0 vs C8: d={test08['d']:.2f}, p={test08['p']:.4f} ({sig08})")
        print(f"    C8 vs C1: d={test81['d']:.2f}, p={test81['p']:.4f} ({sig81})")

# ============================================================
# [2] Microsoft CLAP
# ============================================================
print("\n\n[2] Microsoft CLAP alignment on matched segments...")

try:
    import msclap
    ms_clap = msclap.CLAP()
    print("  Microsoft CLAP loaded")

    TRUNC = 500

    def msclap_text_embed(texts, batch_size=1):
        """Embed texts one-by-one to avoid tensor size mismatch in msclap."""
        all_emb = []
        errors = 0
        for text in texts:
            text = text[:TRUNC]
            try:
                emb = ms_clap.get_text_embeddings([text])
                if isinstance(emb, torch.Tensor): emb = emb.cpu().numpy()
                if emb.ndim == 1: emb = emb.reshape(1, -1)
                all_emb.append(emb)
            except Exception as e:
                errors += 1
                all_emb.append(np.zeros((1, 1024)))
        if errors:
            print(f"  {errors}/{len(texts)} errors")
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
                print(f"  MS-CLAP audio error: {e}")
                all_emb.append(np.zeros((len(batch), 1024)))
        return np.vstack(all_emb)

    print("  Computing MS-CLAP text embeddings (500-char trunc)...")
    ms_text_embs = {}
    for c in CONDS:
        texts = [conditions[n][c][:TRUNC] for n in song_names]
        ms_text_embs[c] = msclap_text_embed(texts)
        print(f"    {c}: done")

    if valid_c8:
        c8_texts = [conditions[n]['C8_paraphrase'][:TRUNC] for n in valid_c8]
        ms_text_embs['C8_paraphrase'] = msclap_text_embed(c8_texts)
        print(f"    C8_paraphrase: done")

    print("  Computing MS-CLAP audio embeddings for matched segments...")
    hach_seg_ms = msclap_audio_embed_files(hach_seg_paths)
    orig_seg_ms = msclap_audio_embed_files(orig_seg_paths)
    print(f"    hach_seg: {hach_seg_ms.shape}, orig_seg: {orig_seg_ms.shape}")

    ms_results = {}
    print("\n  === Microsoft CLAP Results (matched segments, N={}) ===".format(len(song_names)))

    for audio_name, audio_embs in [("hach_seg_audio", hach_seg_ms), ("orig_seg_audio", orig_seg_ms)]:
        ms_results[audio_name] = {}
        print(f"\n  vs {audio_name}:")
        for c in CONDS:
            align = cos_sim(ms_text_embs[c], audio_embs)
            m, lo, hi = bootstrap_ci(align)
            ms_results[audio_name][c] = {
                'mean': float(align.mean()),
                'std': float(align.std()),
                'ci_lo': lo, 'ci_hi': hi,
            }
            print(f"    {c}: mean={align.mean():.4f} (CI: [{lo:.4f}, {hi:.4f}])")

        c0_a = cos_sim(ms_text_embs['C0_orig_lyrics'], audio_embs)
        c1_a = cos_sim(ms_text_embs['C1_original'], audio_embs)
        test = paired_test(c0_a, c1_a)
        ms_results[audio_name]['C0_vs_C1'] = test
        sig = '***' if test['p'] < 0.001 else ('**' if test['p'] < 0.01 else ('*' if test['p'] < 0.05 else 'n.s.'))
        print(f"    C0 vs C1: d={test['d']:.2f}, p={test['p']:.4f} ({sig})")

    # C8 paraphrase
    if valid_c8:
        print("\n  === C8 Paraphrase MS-CLAP (matched segments) ===")
        c8_idx = [song_names.index(n) for n in valid_c8]
        for audio_name, audio_embs in [("hach_seg_audio", hach_seg_ms), ("orig_seg_audio", orig_seg_ms)]:
            sub = audio_embs[c8_idx]
            c0_a = cos_sim(ms_text_embs['C0_orig_lyrics'][c8_idx], sub)
            c1_a = cos_sim(ms_text_embs['C1_original'][c8_idx], sub)
            c8_a = cos_sim(ms_text_embs['C8_paraphrase'], sub)

            m, lo, hi = bootstrap_ci(c8_a)
            ms_results[audio_name]['C8'] = {
                'mean': float(c8_a.mean()),
                'std': float(c8_a.std()),
                'ci_lo': lo, 'ci_hi': hi,
            }

            test08 = paired_test(c0_a, c8_a)
            test81 = paired_test(c8_a, c1_a)
            ms_results[audio_name]['C0_vs_C8'] = test08
            ms_results[audio_name]['C8_vs_C1'] = test81

            print(f"  vs {audio_name}:")
            print(f"    C0: {c0_a.mean():.4f}, C1: {c1_a.mean():.4f}, C8: {c8_a.mean():.4f}")
            print(f"    C0 vs C1: d={paired_test(c0_a,c1_a)['d']:.2f}, p={paired_test(c0_a,c1_a)['p']:.4f}")
            print(f"    C0 vs C8: d={test08['d']:.2f}, p={test08['p']:.4f}")
            print(f"    C8 vs C1: d={test81['d']:.2f}, p={test81['p']:.4f}")

except Exception as e:
    print(f"  Microsoft CLAP failed: {e}")
    import traceback
    traceback.print_exc()
    ms_results = {}

# ============================================================
# [3] Save results
# ============================================================
print("\n[3] Saving results...")

RESULTS_FILE = RESULTS_DIR / "matched_segment_results.json"
RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

save_data = {
    'description': 'CLAP and msCLAP alignment on temporally matched audio segments',
    'n_songs': len(song_names),
    'n_songs_c8': len(valid_c8),
    'source': 'results/segment_match_aligned.json',
    'quality_filters': 'z_score >= 2.0, fm_sim >= 0.55, cross_validate_agree=true',
    'laion_clap': laion_results,
    'msclap': ms_results,
    'conditions_used': CONDS,
    'song_list': song_names,
    'song_list_c8': valid_c8,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
}

with open(RESULTS_FILE, "w") as f:
    json.dump(save_data, f, indent=2, default=str)

print(f"  Saved to {RESULTS_FILE}")
print("\nDONE!")
