# When Meaning Dissolves: What Audio-Text Alignment Models Actually Encode

This repository contains the code, results, and paper for our study investigating what audio-text alignment models (CLAP) actually encode, using Chinese "hachimi" (哈基米) parody songs as a natural probe.

## Overview

Hachimi songs replace original meaningful lyrics with nonsense syllables ("ha-ji-mi") while preserving the melody, rhythm, and vocal timbre of the original. This creates a natural experiment: if CLAP models are truly sensitive to **semantic** content, then original lyrics should align better with audio than hachimi nonsense. Instead, we find the opposite — suggesting that acoustic and surface-level features dominate alignment scores.

### Key Finding

| Condition | Description | LAION CLAP | MS-CLAP |
|-----------|-------------|:----------:|:-------:|
| C0 | Original lyrics | 0.062 | 0.228 |
| C8 | Paraphrased lyrics | 0.063 | 0.197 |
| C1 | Hachimi nonsense | **0.084** | **0.253** |

Original lyrics (C0) ≈ Paraphrases (C8) < Hachimi nonsense (C1) in both models.

## Dataset

Audio data is hosted on HuggingFace: [heihei/hachimi-alignment](https://huggingface.co/datasets/heihei/hachimi-alignment)

The dataset includes:
- 166 paired hachimi-original song clips
- 236 temporally matched audio segments (WAV, 22050 Hz)
- Text conditions (C0-C8) for each song
- Paraphrase texts for semantic control

## Setup

```bash
# Clone the repo
git clone https://github.com/ngyygm/hachimi-alignment.git
cd hachimi-alignment

# Install dependencies
pip install -r requirements.txt

# Download audio data from HuggingFace
pip install huggingface_hub
huggingface-cli download heihei/hachimi-alignment --local-dir data/matched_segments
```

### Directory Structure

```
hachimi-alignment/
├── data/                        # Audio data (download from HF)
│   └── 哈基米音乐和原曲对照合集/    # Original + hachimi MP3s
├── paper/                       # LaTeX paper
│   ├── main.tex
│   └── figures/
├── scripts/                     # Analysis pipeline (run in order)
│   ├── 1_generate_paraphrases.py
│   ├── 2_compute_alignment.py
│   ├── 3_match_segments.py
│   ├── 4_export_segments.py
│   ├── 5_matched_alignment.py
│   ├── 6_segment_analysis.py
│   └── 7_generate_figures.py
└── results/                     # Pre-computed results (JSON)
```

## Reproduction

### Step 1: Generate paraphrases (requires MiniMax API key)
```bash
export MINIMAX_API_KEY="your-key-here"
python scripts/1_generate_paraphrases.py
```

### Step 2: Compute CLAP/msCLAP alignment
```bash
python scripts/2_compute_alignment.py
# Output: results/cleaned_results.json
```

### Step 3: Match temporal segments
```bash
python scripts/3_match_segments.py
# Output: results/segment_match_results.json
```

### Step 4: Export matched audio segments
```bash
python scripts/4_export_segments.py
# Output: results/segment_match_aligned.json + data/matched_segments/audio/
```

### Step 5: Compute alignment on matched segments
```bash
python scripts/5_matched_alignment.py
# Output: results/matched_segment_results.json
```

### Step 6: Run comparison experiments
```bash
python scripts/6_segment_analysis.py
# Output: results/comparison_experiments_results.json
```

### Step 7: Generate figures
```bash
python scripts/7_generate_figures.py
# Output: paper/figures/fig*.pdf
```

### Compile paper
```bash
cd paper && xelatex main.tex
```

## Citation

```bibtex
@inproceedings{author2025hachimi,
  title={When Meaning Dissolves: What Audio-Text Alignment Models Actually Encode},
  author={Author Name},
  booktitle={Proceedings of ACL},
  year={2025}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.
