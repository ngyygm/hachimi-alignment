# Auto Review Loop — llm-hachimi

Started: 2026-04-01

## External Review Focus
Experimental design issues from AAAI-style review (4.5/10 Weak Reject):
1. Only cosine similarity metric — no retrieval metrics (Recall@K)
2. Dataset small (166 songs, Chinese-only)
3. C0 vs C1 confounds semantics with phonology
4. Whitening interpretation is causally wrong
5. Paraphrase control is flawed (LLM changes rhythm/token distribution)
6. Effect sizes small (d=-0.24)
7. Models outdated
8. Conclusion overgeneralizes

## Round Log

### Round 1 (2026-04-01)

**Assessment:**
- Score: 3/10
- Verdict: NOT READY
- Reviewer: Codex MCP (GPT-5.4, xhigh reasoning)

**Key Criticisms (ranked by severity):**
1. **FATAL** Core causal claim not identified: C0 vs C1 changes semantics, phonology, syllable structure, rhyme, tokenization, and lyric naturalness simultaneously
2. **FATAL** Paraphrase control not valid: LLM paraphrases change rhythm, length, lexical rarity, token boundaries
3. **FATAL** Evaluation too weak: only cosine similarity, no retrieval metrics (R@1/5/10, MRR)
4. **FATAL** Claim scope too broad: 166 Chinese songs, 2 old CLAP variants ≠ "when meaning fades"
5. **Serious** msCLAP disagreement and truncation explanation is post hoc
6. **Serious** Whitening result misinterpreted (destroys ALL embedding structure, not just semantic)
7. **Serious** Effect sizes weak (d=-0.24)
8. **Fixable** Acoustic analysis correlational and shallow
9. **Fixable** Statistical rigor under-specified
10. **Fixable** Dataset validity needs stronger verification

**Actions Taken (Round 1 → Round 2 fixes):**
1. ✅ Wrote retrieval experiment script (`scripts/retrieval_experiment.py`)
2. ✅ Ran retrieval experiment — KEY RESULTS:
   - C0 Recall@1 = 1.8% (3/166 songs), median rank = 6/9
   - C1 beats C0 in 59% of paired comparisons (p=0.002)
   - C8 vs C0 NOT significant (p=0.82)
   - Three-way: C1 ranks highest 48.5%, C0 23.0%, C8 28.5%
3. 🔄 Fixing references (agrawal title, formoso year, cite keys)
4. 🔄 Reframing whitening interpretation (remove causal claim)
5. 🔄 Narrowing conclusion scope
6. 🔄 Adding retrieval results paragraph to paper

**ThreadId:** 019d4922-7479-72e1-a58e-0bb780139ac1

### Round 2 (2026-04-01)

**Assessment:**
- Score: 5/10 (main track) — significant progress from 3/10
- Verdict: "Almost, but still not ready" for main track

**Key Remaining Criticisms:**
1. Still FATAL: Semantics not cleanly isolated from phonology
2. Paraphrase control validity under-validated
3. Still FATAL for main track: Scope narrow, models dated
4. Retrieval metrics added but practical significance modest
5. Mechanism story thinner than headline

### Round 3 (2026-04-01)

**Assessment:**
- Score: 4/10 — more credible but still not main-track ready
- Verdict: "Credible niche empirical pathology/case-study paper"

**Key Advice:** Reframe as observational case study, target workshop/lower venue

### Round 4 FINAL (2026-04-01)

**Assessment:**
- **Workshop: 7/10** ✅ — Likely acceptable!
- **Findings/short paper: 5.5/10** — Borderline
- Verdict: "Credible observational case study"

**Changes Made:**
1. Title reframed: "Do CLAP Models Need Meaning? A Case Study of Chinese Lyric Perturbations"
2. Discussion reframed as observational (not causal)
3. Pre-specified contrasts (not pre-registered)
4. All previous fixes (retrieval metrics, whitening, conclusion, references)

**What Still Holds It Back:**
1. Semantics still not cleanly isolated from phonology (needs TTS)
2. Model/data scope narrow (2 CLAP variants, Chinese-only)
3. No human validation of paraphrases (cheap to add!)
4. msCLAP truncation confound unresolved

**Recommendation:** Submit to multimodal robustness/music-language workshop

## Final Summary

| Round | Score | Verdict |
|-------|-------|---------|
| 1 | 3/10 | Not ready |
| 2 | 5/10 | Significant progress |
| 3 | 4/10 | Credible niche study |
| 4 | **7/10** (workshop) | **Likely acceptable** |

Score progression: 3 → 5 → 4 → **7** (workshop tier)

### Round 5 FINAL+ (2026-04-01)

**New experiments added:**
1. ✅ Homophone phonology control — same sound, different chars → d=-0.27 (p=0.0007)
2. ✅ msCLAP truncation control — C0-C8 gap survives length matching (d=0.38, p<0.001)

**Assessment:**
- **Workshop: 8/10** ✅ — ACCEPT
- **Findings/short paper: 6.5/10** — WEAK ACCEPT / BORDERLINE

**Reviewer's key statement:**
> "The homophone experiment materially improves the paper. It changes the identification issue from fatal to manageable."

**Final recommended claim (reviewer-approved):**
> "CLAP models show nontrivial sensitivity to lyric surface form under Chinese perturbations; this sensitivity is not reducible to meaning-preserving paraphrase effects or pure phonology preservation."


### Round 6 (2026-04-01)

**New experiment added:**
1. Fused LAION CLAP (enable_fusion=True) — third model variant with cross-modal fusion layers

**Key results:**
- C0 vs C8: d=0.137, p=0.08 (n.s.) — CONFIRMS C0~C8 pattern in fused variant
- C0 vs C1: d=-0.368, p<0.001 — LARGEST hachimi advantage across all models
- C1 vs C8: d=0.496, p<0.001

**Assessment:**
- **Workshop: 8.5/10** — comfortably above the bar
- **Findings/short paper: 7/10** — leaning accept

**Reviewer advice:**
- Rewrite main claim as model-dependent surface-form sensitivity (NOT universal)
- Tone down fused interpretation: "fusion did not rescue paraphrase invariance in this LAION checkpoint"
- Add human annotation if possible (highest-ROI remaining improvement)

**Changes made:**
1. Added fused CLAP results to cross-model table (3 columns)
2. Abstract rewritten: explicitly model-dependent, notes msCLAP DOES distinguish paraphrases
3. Contributions rewritten: model-dependent paraphrase sensitivity
4. Conclusion rewritten: model-dependent framing
5. Added caveat that fused result does not establish general architectural theorem
6. Significance notation changed from asterisks to p-values

