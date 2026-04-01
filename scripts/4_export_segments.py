#!/usr/bin/env python3
"""
Step 4: Export matched segments.

Extracts the matched portion of original songs and their corresponding
hachimi audio as WAV files.  Applies quality filters (z-score, similarity,
cross-validation agreement) and saves metadata to segment_match_aligned.json.
"""
import os, sys, warnings, json, time
warnings.filterwarnings("ignore")

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

# ─── Paths (relative to project root) ────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = PROJECT_ROOT / "data" / "哈基米音乐和原曲对照合集"
MATCH_FILE = PROJECT_ROOT / "results" / "segment_match_results.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "matched_segments"

SR = 22050
HOP_SEC = 0.5

# Quality filters
MIN_Z_SCORE = 2.0       # z-score threshold
MIN_FM_SIM = 0.55        # full-mix similarity threshold
AGREE_THRESHOLD = 15     # FM vs Vocal agreement in seconds

# ─── Load match results ──────────────────────────────────────────────
matches = json.load(open(MATCH_FILE))
print(f"Loaded {len(matches)} match results")

# ─── Filter ──────────────────────────────────────────────────────────
aligned = []
excluded = []

for m in matches:
    fm = m["full_mix"]
    fm_z = fm["z_score"]
    fm_sim = fm["best_sim"]
    fm_start = fm["matched_start"]
    fm_end = fm["matched_end"]
    hach_dur = m["hach_duration"]
    orig_dur = m["orig_duration"]

    reasons = []

    # Filter 1: confidence
    if fm_z < MIN_Z_SCORE:
        reasons.append(f"low z={fm_z:.1f}")
    if fm_sim < MIN_FM_SIM:
        reasons.append(f"low sim={fm_sim:.3f}")

    # Filter 2: cross-validation
    if m.get("cross_validate_agree") == False:
        reasons.append(f"FM/Vocal disagree (diff={m.get('cross_validate_diff', '?')}s)")

    # Filter 3: sanity check — matched segment should be roughly hachimi length
    matched_dur = fm_end - fm_start
    if hach_dur <= orig_dur and matched_dur > 0:
        ratio = hach_dur / matched_dur
        if ratio > 3.0 or ratio < 0.33:
            reasons.append(f"duration mismatch (hach={hach_dur:.0f}s vs match={matched_dur:.0f}s, ratio={ratio:.1f})")

    if reasons:
        excluded.append({"song": m["song"], "reasons": reasons})
    else:
        aligned.append(m)

print(f"\nAligned: {len(aligned)} songs")
print(f"Excluded: {len(excluded)} songs")
for e in excluded[:10]:
    print(f"  X {e['song'][:40]:40s}  {', '.join(e['reasons'])}")
if len(excluded) > 10:
    print(f"  ... and {len(excluded)-10} more")

# ─── Extract matched segments ────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR / "audio", exist_ok=True)

export_results = []
t0 = time.time()

for i, m in enumerate(aligned):
    fm = m["full_mix"]
    fm_start = fm["matched_start"]
    fm_end = fm["matched_end"]
    hach_dur = m["hach_duration"]
    orig_dur = m["orig_duration"]
    song = m["song"]
    sdir = os.path.join(DATA_DIR, song)

    # Find files
    hach_files = [f for f in os.listdir(sdir) if f.startswith("hachimi-") and f.endswith('.mp3')
                  and not any(s in f for s in ('_vocals', '_main', '_harmony', '_noreverb', '_instrumental'))]
    raw_files = [f for f in os.listdir(sdir) if f.startswith("raw-") and f.endswith('.mp3')
                 and not any(s in f for s in ('_vocals', '_main', '_harmony', '_noreverb', '_instrumental'))]

    if not hach_files or not raw_files:
        continue

    raw_path = os.path.join(sdir, raw_files[0])
    hach_path = os.path.join(sdir, hach_files[0])

    # Load original audio and extract matched segment
    start_sample = int(fm_start * SR)
    end_sample = int(fm_end * SR)

    try:
        y_orig, _ = librosa.load(raw_path, sr=SR, mono=True)
        start_sample = max(0, start_sample)
        end_sample = min(len(y_orig), end_sample)
        y_segment = y_orig[start_sample:end_sample]
    except Exception as e:
        print(f"  ERROR loading {song}: {e}")
        continue

    # Save matched original segment
    safe_name = song.replace("/", "_").replace(" ", "_")
    seg_path = OUTPUT_DIR / "audio" / f"orig_segment_{safe_name}.wav"
    sf.write(str(seg_path), y_segment, SR)

    # Also load hachimi audio for reference
    try:
        y_hach, _ = librosa.load(hach_path, sr=SR, mono=True)
        hach_path_out = OUTPUT_DIR / "audio" / f"hachimi_{safe_name}.wav"
        sf.write(str(hach_path_out), y_hach, SR)
    except:
        hach_path_out = None

    # Use relative paths for portability
    export_results.append({
        "song": song,
        "hach_duration": hach_dur,
        "orig_duration": orig_dur,
        "matched_start": fm_start,
        "matched_end": fm_end,
        "matched_duration": round(fm_end - fm_start, 1),
        "z_score": fm["z_score"],
        "fm_sim": fm["best_sim"],
        "cross_validate_agree": m.get("cross_validate_agree"),
        "files": {
            "original_segment": f"audio/orig_segment_{safe_name}.wav",
            "hachimi": f"audio/hachimi_{safe_name}.wav" if hach_path_out else None,
        }
    })

    if (i + 1) % 20 == 0:
        print(f"  Processed {i+1}/{len(aligned)}...")

print(f"\nExported {len(export_results)} aligned pairs in {time.time()-t0:.0f}s")

# ─── Statistics ──────────────────────────────────────────────────────
z_scores = [r["z_score"] for r in export_results]
fm_sims = [r["fm_sim"] for r in export_results]
durs = [r["matched_duration"] for r in export_results]

print(f"\nAligned dataset statistics:")
print(f"  Songs: {len(export_results)}")
print(f"  Z-scores: mean={np.mean(z_scores):.1f}, min={np.min(z_scores):.1f}, max={np.max(z_scores):.1f}")
print(f"  FM similarity: mean={np.mean(fm_sims):.3f}, min={np.min(fm_sims):.3f}")
print(f"  Matched segment duration: mean={np.mean(durs):.0f}s, median={np.median(durs):.0f}s")
print(f"  Hachimi duration: mean={np.mean([r['hach_duration'] for r in export_results]):.0f}s")

# ─── Save aligned metadata ──────────────────────────────────────────
aligned_file = PROJECT_ROOT / "results" / "segment_match_aligned.json"
aligned_file.write_text(json.dumps(export_results, indent=2, ensure_ascii=False))
print(f"\nMetadata saved to: {aligned_file}")

excl_file = PROJECT_ROOT / "results" / "segment_match_excluded.json"
excl_file.write_text(json.dumps(excluded, indent=2, ensure_ascii=False))
print(f"Exclusion list saved to: {excl_file}")
