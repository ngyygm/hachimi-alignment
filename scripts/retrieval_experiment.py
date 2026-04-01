#!/usr/bin/env python3
"""
Retrieval Experiment: Condition-ranking metrics from per-song alignment data.

For each song, rank conditions by alignment score and compute:
- Which condition ranks #1 most often?
- Paired ranking: C1 vs C0 win rate
- Recall@K for C0 (semantic baseline)
- MRR by condition
"""
import json
import numpy as np
from pathlib import Path
from scipy import stats

RESULTS_DIR = Path(__file__).parent.parent / "results"


def main():
    print("=" * 60)
    print("RETRIEVAL EXPERIMENT: Condition-Ranking Analysis")
    print("=" * 60)

    # Load per-song data
    with open(RESULTS_DIR / "per_song_all_conditions.json") as f:
        ps_data = json.load(f)

    per_song = ps_data['per_song']
    song_list = ps_data['song_list']
    song_list_c8 = ps_data.get('song_list_c8', song_list)
    n_songs = len(song_list)

    conditions = list(per_song.keys())
    print(f"Conditions: {len(conditions)}")
    print(f"Songs: {n_songs}")

    # Short labels
    labels = {
        'C0_orig_lyrics': 'C0',
        'C1_original': 'C1',
        'C2_char_shuffle': 'C2',
        'C3_reversed': 'C3',
        'C4_english_nonsense': 'C4',
        'C5_random_phonemes': 'C5',
        'C6_semantic_inversion': 'C6',
        'C6b_semantic_negation': 'C6b',
        'C7_random_chinese': 'C7',
        'C8_paraphrase': 'C8',
    }

    # Use conditions with 166 songs for main analysis
    full_conds = [c for c in conditions if c != 'C8_paraphrase']

    # Build alignment matrix: (n_songs x n_conditions)
    matrix = np.column_stack([per_song[c] for c in full_conds])
    n_cond = len(full_conds)
    print(f"Matrix shape: {matrix.shape}")

    # ================================================================
    # 1. Condition-Ranking Accuracy
    # ================================================================
    print("\n" + "=" * 60)
    print("1. WHICH CONDITION RANKS #1?")
    print("   (Ranking all conditions by alignment score per song)")
    print("=" * 60)

    top1_counts = {}
    for i in range(n_songs):
        best_idx = np.argmax(matrix[i])
        best_cond = labels.get(full_conds[best_idx], full_conds[best_idx])
        top1_counts[best_cond] = top1_counts.get(best_cond, 0) + 1

    for cond, count in sorted(top1_counts.items(), key=lambda x: -x[1]):
        pct = count / n_songs * 100
        print(f"  {cond}: {count}/{n_songs} ({pct:.1f}%)")

    # ================================================================
    # 2. C0 rank distribution
    # ================================================================
    print("\n" + "=" * 60)
    print("2. C0 (ORIGINAL LYRICS) RANK DISTRIBUTION")
    print("=" * 60)

    c0_idx = full_conds.index('C0_orig_lyrics')
    c0_ranks = []
    for i in range(n_songs):
        scores = matrix[i]
        sorted_idx = np.argsort(-scores)
        rank = np.where(sorted_idx == c0_idx)[0][0] + 1
        c0_ranks.append(rank)

    c0_ranks = np.array(c0_ranks)
    print(f"\nC0 rank distribution:")
    for r in range(1, n_cond + 1):
        count = np.sum(c0_ranks == r)
        pct = count / n_songs * 100
        bar = "#" * int(pct)
        print(f"  Rank #{r}: {count:3d} songs ({pct:5.1f}%) {bar}")

    print(f"\n  C0 median rank: {np.median(c0_ranks):.0f}")
    print(f"  C0 mean rank:   {np.mean(c0_ranks):.2f}")

    # ================================================================
    # 3. Paired Ranking: C1 vs C0
    # ================================================================
    print("\n" + "=" * 60)
    print("3. PAIRED RANKING: C1 (hachimi) vs C0 (original)")
    print("=" * 60)

    c0_scores = np.array(per_song['C0_orig_lyrics'])
    c1_scores = np.array(per_song['C1_original'])

    c1_wins = int(np.sum(c1_scores > c0_scores))
    c0_wins = int(np.sum(c0_scores > c1_scores))
    ties = n_songs - c1_wins - c0_wins

    print(f"  C1 (hachimi) > C0:  {c1_wins}/{n_songs} ({c1_wins/n_songs:.1%})")
    print(f"  C0 (original) > C1: {c0_wins}/{n_songs} ({c0_wins/n_songs:.1%})")
    print(f"  Tied:               {ties}/{n_songs}")

    t, p = stats.ttest_rel(c1_scores, c0_scores)
    d = (c1_scores - c0_scores).mean() / (c1_scores - c0_scores).std()
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
    print(f"  Paired t-test: t={t:.3f}, p={p:.4f}, d={d:.3f} ({sig})")

    # ================================================================
    # 4. Recall@K for C0
    # ================================================================
    print("\n" + "=" * 60)
    print("4. RECALL@K: Does C0 appear in top-K conditions?")
    print("  (Ranking all 9 conditions by alignment per song)")
    print("=" * 60)

    recall_results = {}
    for k in [1, 3, 5, 10]:
        recall = float(np.sum(c0_ranks <= k) / n_songs)
        recall_results[f'R@{k}'] = recall
        print(f"  Recall@{k}: {recall:.3f} ({int(np.sum(c0_ranks <= k))}/{n_songs})")

    # ================================================================
    # 5. MRR by condition
    # ================================================================
    print("\n" + "=" * 60)
    print("5. MEAN RECIPROCAL RANK by condition")
    print("=" * 60)

    mrr_results = {}
    for cond in full_conds:
        cond_idx = full_conds.index(cond)
        ranks = []
        for i in range(n_songs):
            scores = matrix[i]
            sorted_idx = np.argsort(-scores)
            rank = int(np.where(sorted_idx == cond_idx)[0][0] + 1)
            ranks.append(rank)
        mrr = float(np.mean([1.0 / r for r in ranks]))
        median_rank = float(np.median(ranks))
        label = labels.get(cond, cond)
        mrr_results[label] = {'mrr': mrr, 'median_rank': median_rank}
        print(f"  {label:4s}: MRR={mrr:.3f}, median_rank={median_rank:.0f}")

    # ================================================================
    # 6. Three-way comparison: C0 vs C8 vs C1
    # ================================================================
    print("\n" + "=" * 60)
    print("6. THREE-WAY COMPARISON: C0 vs C8 vs C1")
    print("  (Using 165 songs that have all three conditions)")
    print("=" * 60)

    common_songs = [s for s in song_list if s in set(song_list_c8)]
    common_idx = [song_list.index(s) for s in common_songs]
    n_3 = len(common_idx)

    c0_sub = np.array(per_song['C0_orig_lyrics'])[common_idx]
    c1_sub = np.array(per_song['C1_original'])[common_idx]
    c8_sub = np.array(per_song['C8_paraphrase'])

    # Build 3-condition matrix
    mat3 = np.column_stack([c0_sub, c1_sub, c8_sub])
    labels3 = ['C0', 'C1', 'C8']

    c0_wins_3 = int(np.sum(np.argmax(mat3, axis=1) == 0))
    c1_wins_3 = int(np.sum(np.argmax(mat3, axis=1) == 1))
    c8_wins_3 = int(np.sum(np.argmax(mat3, axis=1) == 2))

    print(f"  N songs: {n_3}")
    print(f"  C0 (original) ranks highest:  {c0_wins_3}/{n_3} ({c0_wins_3/n_3:.1%})")
    print(f"  C1 (hachimi) ranks highest:   {c1_wins_3}/{n_3} ({c1_wins_3/n_3:.1%})")
    print(f"  C8 (paraphrase) ranks highest: {c8_wins_3}/{n_3} ({c8_wins_3/n_3:.1%})")

    # Paired comparisons
    c1_gt_c0 = int(np.sum(c1_sub > c0_sub))
    c8_gt_c0 = int(np.sum(c8_sub > c0_sub))
    c1_gt_c8 = int(np.sum(c1_sub > c8_sub))

    print(f"\n  C1 > C0: {c1_gt_c0}/{n_3} ({c1_gt_c0/n_3:.1%})")
    print(f"  C8 > C0: {c8_gt_c0}/{n_3} ({c8_gt_c0/n_3:.1%})")
    print(f"  C1 > C8: {c1_gt_c8}/{n_3} ({c1_gt_c8/n_3:.1%})")

    for name, (a, b) in [("C1 vs C0", (c1_sub, c0_sub)),
                          ("C8 vs C0", (c8_sub, c0_sub)),
                          ("C1 vs C8", (c1_sub, c8_sub))]:
        t_val, p_val = stats.ttest_rel(a, b)
        d_val = (a - b).mean() / (a - b).std()
        s = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'n.s.'))
        print(f"  {name}: d={d_val:.3f}, p={p_val:.4f} ({s})")

    # ================================================================
    # Save results
    # ================================================================
    output = {
        'description': 'Condition-ranking retrieval metrics',
        'n_songs': n_songs,
        'top1_condition_counts': top1_counts,
        'c0_rank_distribution': {int(r): int(np.sum(c0_ranks == r)) for r in range(1, n_cond + 1)},
        'c0_median_rank': float(np.median(c0_ranks)),
        'c0_mean_rank': float(np.mean(c0_ranks)),
        'paired_ranking': {
            'c1_vs_c0': {'c1_wins': c1_wins, 'c0_wins': c0_wins, 'ties': ties},
            'c1_win_rate': c1_wins / n_songs,
        },
        'recall_at_k': recall_results,
        'mrr_by_condition': mrr_results,
        'three_way': {
            'n_songs': n_3,
            'c0_wins': c0_wins_3,
            'c1_wins': c1_wins_3,
            'c8_wins': c8_wins_3,
            'c1_gt_c0': c1_gt_c0,
            'c8_gt_c0': c8_gt_c0,
            'c1_gt_c8': c1_gt_c8,
        },
    }

    out_file = RESULTS_DIR / "retrieval_results.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
