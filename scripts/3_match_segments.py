#!/usr/bin/env python3
"""
Step 3: Audio Segment Matching via Chroma Cross-Correlation.

Uses chroma feature cross-correlation to find where hachimi clips
correspond to in original songs.  Cross-validates with full-mix vs
vocal-only matching.
"""
import os, sys, warnings, json, time
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import numpy as np
import librosa
from pathlib import Path

# ─── Config ────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = PROJECT_ROOT / "data" / "哈基米音乐和原曲对照合集"
OUTPUT = PROJECT_ROOT / "results" / "segment_match_results.json"

HOP_SEC = 0.5         # chroma hop in seconds
SR = 22050
NUM_TEST = 0           # 0 = all songs

def extract_chroma(path, hop_sec=HOP_SEC, sr=SR):
    """Extract normalized chroma features. Returns (n_frames, 12)."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    hop_samples = int(hop_sec * sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_samples)
    # L2-normalize each frame for cosine similarity
    norms = np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-8
    chroma = chroma / norms
    return chroma.T  # (n_frames, 12)

def slide_cosine(query, ref):
    """Slide shorter across longer, compute mean cosine similarity at each offset."""
    if query.shape[0] > ref.shape[0]:
        query, ref = ref, query

    n_q = query.shape[0]
    n_r = ref.shape[0]
    n_offsets = n_r - n_q + 1

    if n_offsets <= 0:
        return np.array([0.0])

    sims = np.zeros(n_offsets)
    for d in range(query.shape[1]):
        corr = np.correlate(ref[:, d], query[:, d], mode='valid')
        sims += corr
    sims /= n_q

    return sims

def find_best_offset(query_chroma, ref_chroma):
    """Find best offset and return analysis."""
    sims = slide_cosine(query_chroma, ref_chroma)

    best_offset = np.argmax(sims)
    best_sim = sims[best_offset]

    mean_sim = np.mean(sims)
    std_sim = np.std(sims)
    z_score = (best_sim - mean_sim) / (std_sim + 1e-8)

    sorted_sims = np.sort(sims)[::-1]
    sharpness = sorted_sims[0] / (sorted_sims[1] + 1e-8) if len(sorted_sims) > 1 else 1.0

    top3_idx = np.argsort(sims)[-3:][::-1]

    return {
        "best_offset": int(best_offset),
        "best_sim": round(best_sim, 4),
        "z_score": round(z_score, 2),
        "sharpness": round(sharpness, 3),
        "mean_sim": round(mean_sim, 4),
        "std_sim": round(std_sim, 4),
        "n_offsets": len(sims),
        "top3_offsets": [int(x) for x in top3_idx],
        "top3_sims": [round(sims[x], 4) for x in top3_idx],
    }

def find_files(song_dir, prefix):
    exclude = ('_vocals', '_main', '_harmony', '_noreverb', '_instrumental')
    return [f for f in os.listdir(song_dir)
            if f.startswith(prefix) and f.endswith('.mp3')
            and not any(s in f for s in exclude)]

def find_vocal_file(song_dir, prefix):
    for f in sorted(os.listdir(song_dir)):
        if f.startswith(prefix) and f.endswith('.mp3') and '_vocals' in f and '_harmony' not in f:
            return os.path.join(song_dir, f)
    return None

# ─── Main ──────────────────────────────────────────────────────────
print("Chroma Cross-Correlation Segment Matching")
print(f"Hop: {HOP_SEC}s | SR: {SR}Hz | Test: {NUM_TEST} songs\n")

song_dirs = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
if NUM_TEST > 0:
    song_dirs = song_dirs[:NUM_TEST]

results = []
t_total = time.time()

for i, d in enumerate(song_dirs):
    sdir = os.path.join(DATA_DIR, d)
    hach_files = find_files(sdir, "hachimi-")
    raw_files = find_files(sdir, "raw-")

    if not hach_files or not raw_files:
        continue

    hach_path = os.path.join(sdir, hach_files[0])
    raw_path = os.path.join(sdir, raw_files[0])

    hach_dur = librosa.get_duration(path=hach_path)
    orig_dur = librosa.get_duration(path=raw_path)

    result = {"song": d, "hach_duration": round(hach_dur, 1), "orig_duration": round(orig_dur, 1)}

    # ─── Full-mix matching ───
    t0 = time.time()
    hach_chroma = extract_chroma(hach_path)
    raw_chroma = extract_chroma(raw_path)
    fm = find_best_offset(hach_chroma, raw_chroma)

    fm_start = round(fm["best_offset"] * HOP_SEC, 1)
    fm_end = min(round((fm["best_offset"] + hach_chroma.shape[0]) * HOP_SEC, 1), orig_dur)

    result["full_mix"] = {**fm, "matched_start": fm_start, "matched_end": fm_end,
                          "hach_frames": hach_chroma.shape[0], "orig_frames": raw_chroma.shape[0],
                          "time_sec": round(time.time() - t0, 1)}

    # ─── Vocal-only matching (cross-validation) ───
    hach_vocal = find_vocal_file(sdir, "hachimi-")
    raw_vocal = find_vocal_file(sdir, "raw-")

    vm_info = "Vocal: N/A"
    if hach_vocal and raw_vocal:
        t0 = time.time()
        hv_chroma = extract_chroma(hach_vocal)
        rv_chroma = extract_chroma(raw_vocal)
        vm = find_best_offset(hv_chroma, rv_chroma)

        vm_start = round(vm["best_offset"] * HOP_SEC, 1)
        vm_end = min(round((vm["best_offset"] + hv_chroma.shape[0]) * HOP_SEC, 1), orig_dur)

        result["vocals"] = {**vm, "matched_start": vm_start, "matched_end": vm_end,
                            "time_sec": round(time.time() - t0, 1)}

        diff = abs(fm_start - vm_start)
        result["cross_validate_diff"] = round(diff, 1)
        result["cross_validate_agree"] = diff <= 15

        vm_info = f"Vocal@{vm_start}s (z={vm['z_score']:.1f}, sim={vm['best_sim']:.3f})"

    agree_str = "Y" if result.get("cross_validate_agree") else ("N" if "cross_validate_agree" in result else "?")
    print(f"[{i+1:2d}/{len(song_dirs)}] {d[:35]:35s}  H={hach_dur:.0f}s  O={orig_dur:.0f}s  "
          f"FM@{fm_start:5.1f}s(z={fm['z_score']:+.1f},sim={fm['best_sim']:.3f})  "
          f"{vm_info}  {agree_str}")

    results.append(result)

# ─── Summary ────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"Total: {len(results)} songs in {time.time()-t_total:.0f}s\n")

if results:
    agrees = sum(1 for r in results if r.get("cross_validate_agree"))
    has_v = sum(1 for r in results if r.get("vocals"))
    print(f"Cross-validation (within 15s): {agrees}/{has_v} ({agrees/max(1,has_v)*100:.0f}%)")

    fm_z = [r["full_mix"]["z_score"] for r in results]
    vm_z = [r["vocals"]["z_score"] for r in results if r.get("vocals")]
    fm_sim = [r["full_mix"]["best_sim"] for r in results]
    vm_sim = [r["vocals"]["best_sim"] for r in results if r.get("vocals")]

    print(f"\nFull-mix  z: mean={np.mean(fm_z):+.2f}  sim: mean={np.mean(fm_sim):.3f} (>{0.8} = confident)")
    print(f"Vocal     z: mean={np.mean(vm_z):+.2f}  sim: mean={np.mean(vm_sim):.3f}")

    fm_sharp = [r["full_mix"]["sharpness"] for r in results]
    vm_sharp = [r["vocals"]["sharpness"] for r in results if r.get("vocals")]
    print(f"\nSharpness (>{1.5} = clear winner): FM mean={np.mean(fm_sharp):.2f}  Vocal mean={np.mean(vm_sharp):.2f}")

    diffs = [r["cross_validate_diff"] for r in results if "cross_validate_diff" in r]
    if diffs:
        print(f"\nFM-Vocal offset diff: mean={np.mean(diffs):.1f}s  median={np.median(diffs):.1f}s  "
              f"<15s: {sum(1 for d in diffs if d <= 15)}/{len(diffs)}")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
print(f"\nSaved to: {OUTPUT}")
