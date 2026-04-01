#!/usr/bin/env python3
"""
Experiment 3: Fused LAION CLAP Model
Uses laion/clap-htsat-fused with enable_fusion=True.
"""
import os, sys, json, time
import numpy as np
import torch
from pathlib import Path
from scipy import stats

SEED = 42
np.random.seed(SEED)

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data" / "\u54c8\u57fa\u7c73\u97f3\u4e50\u548c\u539f\u66f2\u5bf9\u7167\u5408\u96c6"

TRUNC = 500

print("=" * 60)
print("EXPERIMENT 3: FUSED LAION CLAP (enable_fusion=True)")
print("=" * 60)

with open(RESULTS_DIR / "conditions.json", "r", encoding="utf-8") as f:
    conditions = json.load(f)

def get_audio_path(song_name, audio_type):
    d = DATA_DIR / song_name
    if not d.exists():
        return None
    if audio_type == "hach":
        files = [f for f in d.glob("hachimi-*.mp3") if all(x not in f.name for x in ["_vocals", "_instrumental", "_noreverb"])]
        return str(files[0]) if files else None
    return None

song_list = []
for name in sorted(conditions.keys()):
    if "C8_paraphrase" in conditions[name] and get_audio_path(name, "hach"):
        song_list.append(name)

print(f"Songs with C8 + audio: {len(song_list)}")

c0_texts = [conditions[n]["C0_orig_lyrics"] for n in song_list]
c1_texts = [conditions[n].get("C1_original", "") for n in song_list]
c8_texts = [conditions[n]["C8_paraphrase"] for n in song_list]
homo_texts = [conditions[n].get("C_homophone", "") for n in song_list]
audio_paths = [get_audio_path(n, "hach") for n in song_list]

print("\nLoading fused LAION CLAP (enable_fusion=True)...")
from laion_clap import CLAP_Module
clap_fused = CLAP_Module(enable_fusion=True)

print("Computing text embeddings (fused)...")
c0_embs = clap_fused.get_text_embedding(c0_texts)
c1_embs = clap_fused.get_text_embedding(c1_texts)
c8_embs = clap_fused.get_text_embedding(c8_texts)

has_homo = any(h.strip() for h in homo_texts)
if has_homo:
    print("Computing homophone text embeddings (fused)...")
    homo_embs = clap_fused.get_text_embedding(homo_texts)

if isinstance(c0_embs, torch.Tensor):
    c0_embs = c0_embs.cpu().numpy()
    c1_embs = c1_embs.cpu().numpy()
    c8_embs = c8_embs.cpu().numpy()
    if has_homo:
        homo_embs = homo_embs.cpu().numpy()

print("Computing audio embeddings (fused, batched)...")
# Process audio in small batches to avoid OOM on fused model
all_audio_embs = []
BATCH = 4
for i in range(0, len(audio_paths), BATCH):
    batch_paths = audio_paths[i:i+BATCH]
    emb = clap_fused.get_audio_embedding_from_filelist(batch_paths)
    if isinstance(emb, torch.Tensor):
        emb = emb.cpu().numpy()
    all_audio_embs.append(emb)
    if isinstance(emb, torch.Tensor):
        torch.cuda.empty_cache()
    print(f"  Audio {min(i+BATCH, len(audio_paths))}/{len(audio_paths)} done")
audio_embs = np.vstack(all_audio_embs)

print(f"Embedding shapes: text={c0_embs.shape}, audio={audio_embs.shape}")

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

def cohen_d(a, b):
    diff = a - b
    return float(diff.mean() / (diff.std() + 1e-9))

c0_align = cos_sim(c0_embs, audio_embs)
c1_align = cos_sim(c1_embs, audio_embs)
c8_align = cos_sim(c8_embs, audio_embs)
if has_homo:
    homo_align = cos_sim(homo_embs, audio_embs)

print("\n" + "=" * 60)
print("FUSED LAION CLAP RESULTS")
print("=" * 60)

m0, l0, h0 = bootstrap_ci(c0_align)
m1, l1, h1 = bootstrap_ci(c1_align)
m8, l8, h8 = bootstrap_ci(c8_align)

print(f"\nC0 (original):  {m0:.4f} [{l0:.4f}, {h0:.4f}]")
print(f"C1 (hachimi):   {m1:.4f} [{l1:.4f}, {h1:.4f}]")
print(f"C8 (paraphrase):{m8:.4f} [{l8:.4f}, {h8:.4f}]")

if has_homo:
    mh, lh, hh = bootstrap_ci(homo_align)
    print(f"Homophone:      {mh:.4f} [{lh:.4f}, {hh:.4f}]")

pairs = [
    ("C0 vs C1", c0_align, c1_align),
    ("C0 vs C8", c0_align, c8_align),
    ("C1 vs C8", c1_align, c8_align),
]
if has_homo:
    pairs.append(("C0 vs Homo", c0_align, homo_align))
    pairs.append(("C1 vs Homo", c1_align, homo_align))

print("\nPairwise comparisons:")
for name, a, b in pairs:
    t, p = stats.ttest_rel(a, b)
    d = cohen_d(a, b)
    w, wp = stats.wilcoxon(a, b)
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
    print(f"  {name}: d={d:.3f}, t={t:.2f}, p={p:.4f} ({sig}), Wilcoxon p={wp:.4f}")

output = {
    "description": "Fused LAION CLAP alignment experiment",
    "model": "laion/clap-htsat-fused (enable_fusion=True)",
    "n_songs": len(song_list),
    "results": {
        "C0": {"mean": float(m0), "ci": [float(l0), float(h0)]},
        "C1": {"mean": float(m1), "ci": [float(l1), float(h1)]},
        "C8": {"mean": float(m8), "ci": [float(l8), float(h8)]},
    },
    "comparisons": {},
    "per_song": {
        "c0": c0_align.tolist(),
        "c1": c1_align.tolist(),
        "c8": c8_align.tolist(),
    },
}

if has_homo:
    output["results"]["homophone"] = {"mean": float(mh), "ci": [float(lh), float(hh)]}
    output["per_song"]["homophone"] = homo_align.tolist()

for name, a, b in pairs:
    t, p = stats.ttest_rel(a, b)
    d = cohen_d(a, b)
    output["comparisons"][name] = {"d": d, "p": float(p), "t": float(t)}

out_file = RESULTS_DIR / "fused_clap_results.json"
with open(out_file, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved to {out_file}")
print("DONE!")
