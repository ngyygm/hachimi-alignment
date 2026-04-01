# When Meaning Fades: Probing Acoustic Properties in Audio-Text Alignment

[Paper (PDF)](paper/main.pdf) | [Dataset (HuggingFace)](https://huggingface.co/datasets/heihei/hachimi-alignment) | [BibTeX](#citation)

ACL 2025

## Overview

We investigate what audio-text alignment models (CLAP) actually encode, using Chinese **hachimi lyrics** (哈基米歌词) as a natural probe. Hachimi songs replace original meaningful lyrics with nonsense syllables like "ha-ji-mi" while preserving melody, instrumentation, and vocal timbre.

<p align="center">
<img src="paper/figures/fig_overview.png" width="100%">
</p>

## Key Findings

**Core result:** Meaning-preserving paraphrases produce alignment **indistinguishable** from originals ($d{=}{-}0.02$, $p{=}0.82$), while nonsense hachimi lyrics achieve **higher** alignment ($d{=}{-}0.24$, $p{=}0.002$). This is consistent with semantic content contributing far less to CLAP alignment than phonological regularity.

| Condition | Description | LAION CLAP | MS-CLAP |
|:---:|:---|:---:|:---:|
| **C0** | Original lyrics | 0.062 | 0.228 |
| **C8** | Paraphrased lyrics (same meaning, different words) | 0.063 | 0.197 |
| **C1** | Hachimi nonsense syllables | **0.084** | **0.253** |

**Pattern:** C0 ≈ C8 < C1 in both models.

## Results

### Alignment Across Conditions (LAION CLAP)

<p align="center">
<img src="paper/figures/fig1_alignment.png" width="90%">
</p>

All 10 text conditions tested against hachimi audio (N=166 songs). Error bars: 95% bootstrap CIs. English nonsense achieves the highest alignment due to LAION CLAP's English-dominant training.

### Cross-Model Comparison

<p align="center">
<img src="paper/figures/fig7_cross_model.png" width="90%">
</p>

Both CLAP variants agree: hachimi nonsense (C1) > original lyrics (C0) > paraphrases (C8). LAION CLAP cannot distinguish originals from paraphrases ($d{=}{-}0.02$).

### Acoustic Analysis

<p align="center">
<img src="paper/figures/fig8_vocal_only.png" width="90%">
</p>

**Left:** Removing instruments amplifies the original < hachimi separation in LAION CLAP (full-mix $d{=}{-}0.24$ → vocal-only $d{=}{-}0.47$).
**Right:** msCLAP shows the same direction ($d{=}{-}1.32$).

**Additional findings:**
- MFCC coefficients are the strongest alignment predictors (three survive Bonferroni correction, largest $|r|{=}0.36$)
- ZCA whitening eliminates all alignment (max = 0.008), showing alignment depends on cross-modal correlation structure
- Lowpass filtering *increases* alignment for Chinese text ($d{=}{-}0.54$), consistent with spectral mismatch

## Method

### Degeneration Spectrum

We construct 10 text conditions (C0--C8) systematically varying semantic and structural integrity:

| Condition | Operation | Tests |
|:---|:---|:---|
| C0: Original Lyrics | Original meaningful lyrics | Semantic baseline |
| C1: Hachimi | Nonsense hachimi lyrics | Non-semantic baseline |
| C2: Char-Shuffle | Characters scrambled within words | Orthographic cues |
| C3: Reversed | Word segments reversed | Sequential structure |
| C4: English Nonsense | English syllable walk | Cross-lingual phonotactics |
| C5: Random Phonemes | Random Chinese syllables | Phonetic texture |
| C6: Sem. Inversion | Negations in hachimi | Sem. ops on non-semantic |
| C6b: Sem. Negation | Negations in original lyrics | Sem. ops on meaningful |
| C8: Paraphrased Lyrics | LLM rewording, same meaning | **Semantic control** |

**Critical test:** C0 and C8 share identical meaning (different words), while C1 shares surface form but no meaning.

## Dataset

166 paired hachimi-original Chinese songs, including:
- Original and hachimi MP3 recordings
- Separated vocal and instrumental tracks
- Temporally matched audio segments (236 WAV files, 22,050 Hz)
- Text conditions (C0--C8) and LLM-generated paraphrases

## Setup

```bash
# Clone
git clone https://github.com/ngyygm/hachimi-alignment.git
cd hachimi-alignment

# Install
pip install -r requirements.txt

# Download audio data from HuggingFace
pip install huggingface_hub
huggingface-cli download heihei/hachimi-alignment --local-dir data
```

### Reproduce

```bash
# Step 1: Generate paraphrases (requires MINIMAX_API_KEY)
export MINIMAX_API_KEY="your-key"
python scripts/1_generate_paraphrases.py

# Step 2: Compute CLAP/msCLAP alignment
python scripts/2_compute_alignment.py

# Step 3: Match temporal segments
python scripts/3_match_segments.py

# Step 4: Export matched audio
python scripts/4_export_segments.py

# Step 5: Alignment on matched segments
python scripts/5_matched_alignment.py

# Step 6: Comparison experiments
python scripts/6_segment_analysis.py

# Step 7: Generate figures
python scripts/7_generate_figures.py
```

### Compile Paper

```bash
cd paper && latexmk -xelatex main.tex
```

## Citation

```bibtex
@inproceedings{author2025hachimi,
  title={When Meaning Fades: Probing Acoustic Properties in Audio-Text Alignment},
  author={Anonymous},
  booktitle={Proceedings of the Association for Computational Linguistics (ACL)},
  year={2025}
}
```

## License

MIT
