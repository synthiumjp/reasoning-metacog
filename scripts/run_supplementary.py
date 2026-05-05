# reasoning-metacog/scripts/run_supplementary.py
"""
Supplementary diagnostics to support the main findings.

1. NLP compression: variance, range, IQR in each condition
2. Bin collapse: why M-ratio is nan in thinking mode
3. Answer length confound: AUROC2 after length-matching
4. D1 decomposition: context conditioning vs metacognitive interpretation
5. H4 domain permutation test

Usage:
    python scripts/run_supplementary.py
"""

import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr, mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from local_config import PROJECT_DIR
from scripts.sdt_analysis import (
    compute_auroc2,
    compute_bin_boundaries,
    assign_bins,
    check_bin_occupancy,
    compute_type2_contingency,
    compute_metad,
)

RAW_DIR = os.path.join(PROJECT_DIR, "results", "raw")
PROCESSED_DIR = os.path.join(PROJECT_DIR, "results", "processed")
MODELS = ["qwen3-8b", "r1-distill-qwen-7b", "r1-distill-llama-8b"]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def load_trials(model, mode):
    path = os.path.join(RAW_DIR, f"trials_{model}_{mode}.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f]


def load_abridged(model):
    path = os.path.join(RAW_DIR, f"abridged_nlp_{model}.jsonl")
    if not os.path.exists(path):
        return {}
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line.strip())
            data[r["item_index"]] = r
    return data


def filter_valid(trials):
    valid = []
    for t in trials:
        if t.get("malformed", False) or t.get("correct") is None:
            continue
        nlp = t.get("answer_nlp")
        if nlp is None or nlp == float("-inf") or np.isinf(nlp):
            continue
        valid.append(t)
    return valid


# ---------------------------------------------------------------------------
# 1. NLP compression diagnostic
# ---------------------------------------------------------------------------

def nlp_compression(model, valid_nt, valid_t):
    """Quantify NLP distribution compression between modes."""
    print(f"\n  1. NLP COMPRESSION DIAGNOSTIC")

    nlp_nt = np.array([t["answer_nlp"] for t in valid_nt])
    nlp_t = np.array([t["answer_nlp"] for t in valid_t])

    stats = {}
    for label, nlp in [("non_thinking", nlp_nt), ("thinking", nlp_t)]:
        stats[label] = {
            "mean": float(np.mean(nlp)),
            "median": float(np.median(nlp)),
            "std": float(np.std(nlp)),
            "variance": float(np.var(nlp)),
            "iqr": float(np.percentile(nlp, 75) - np.percentile(nlp, 25)),
            "range": float(np.max(nlp) - np.min(nlp)),
            "p5": float(np.percentile(nlp, 5)),
            "p95": float(np.percentile(nlp, 95)),
            "p5_p95_range": float(np.percentile(nlp, 95) - np.percentile(nlp, 5)),
        }

    # Compression ratios
    var_ratio = stats["thinking"]["variance"] / stats["non_thinking"]["variance"]
    iqr_ratio = stats["thinking"]["iqr"] / stats["non_thinking"]["iqr"]
    range_ratio = stats["thinking"]["p5_p95_range"] / stats["non_thinking"]["p5_p95_range"]

    print(f"    {'':20s} {'Non-thinking':>14s} {'Thinking':>14s} {'Ratio':>8s}")
    print(f"    {'Mean':20s} {stats['non_thinking']['mean']:14.4f} {stats['thinking']['mean']:14.4f}")
    print(f"    {'Std':20s} {stats['non_thinking']['std']:14.4f} {stats['thinking']['std']:14.4f} {stats['thinking']['std']/stats['non_thinking']['std']:8.2f}")
    print(f"    {'Variance':20s} {stats['non_thinking']['variance']:14.6f} {stats['thinking']['variance']:14.6f} {var_ratio:8.2f}")
    print(f"    {'IQR':20s} {stats['non_thinking']['iqr']:14.4f} {stats['thinking']['iqr']:14.4f} {iqr_ratio:8.2f}")
    print(f"    {'5-95 range':20s} {stats['non_thinking']['p5_p95_range']:14.4f} {stats['thinking']['p5_p95_range']:14.4f} {range_ratio:8.2f}")

    # Separate by correctness to see if compression is symmetric
    for correctness_label, c_val in [("correct", True), ("incorrect", False)]:
        nlp_nt_c = np.array([t["answer_nlp"] for t in valid_nt if t["correct"] == c_val])
        nlp_t_c = np.array([t["answer_nlp"] for t in valid_t if t["correct"] == c_val])
        if len(nlp_nt_c) > 10 and len(nlp_t_c) > 10:
            print(f"    {correctness_label:20s} NT mean={np.mean(nlp_nt_c):.4f} std={np.std(nlp_nt_c):.4f} | "
                  f"T mean={np.mean(nlp_t_c):.4f} std={np.std(nlp_t_c):.4f}")

    # Effect size: separation between correct and incorrect NLP distributions
    nlp_nt_correct = np.array([t["answer_nlp"] for t in valid_nt if t["correct"]])
    nlp_nt_incorrect = np.array([t["answer_nlp"] for t in valid_nt if not t["correct"]])
    nlp_t_correct = np.array([t["answer_nlp"] for t in valid_t if t["correct"]])
    nlp_t_incorrect = np.array([t["answer_nlp"] for t in valid_t if not t["correct"]])

    sep_nt = np.mean(nlp_nt_correct) - np.mean(nlp_nt_incorrect)
    sep_t = np.mean(nlp_t_correct) - np.mean(nlp_t_incorrect)
    pooled_sd_nt = np.sqrt((np.var(nlp_nt_correct) + np.var(nlp_nt_incorrect)) / 2)
    pooled_sd_t = np.sqrt((np.var(nlp_t_correct) + np.var(nlp_t_incorrect)) / 2)
    cohen_d_nt = sep_nt / pooled_sd_nt if pooled_sd_nt > 0 else np.nan
    cohen_d_t = sep_t / pooled_sd_t if pooled_sd_t > 0 else np.nan

    print(f"\n    Correct-Incorrect separation:")
    print(f"      NT: Δmean={sep_nt:.4f}, pooled_sd={pooled_sd_nt:.4f}, Cohen's d={cohen_d_nt:.3f}")
    print(f"      T:  Δmean={sep_t:.4f}, pooled_sd={pooled_sd_t:.4f}, Cohen's d={cohen_d_t:.3f}")

    stats["compression"] = {
        "variance_ratio": var_ratio,
        "iqr_ratio": iqr_ratio,
        "range_ratio": range_ratio,
        "separation_nt": sep_nt,
        "separation_t": sep_t,
        "cohen_d_nt": cohen_d_nt,
        "cohen_d_t": cohen_d_t,
    }

    return stats


# ---------------------------------------------------------------------------
# 2. Bin collapse diagnostic
# ---------------------------------------------------------------------------

def bin_collapse_diagnostic(model, valid_nt, valid_t):
    """Show exactly why M-ratio is nan in thinking mode."""
    print(f"\n  2. BIN COLLAPSE DIAGNOSTIC")

    nlp_nt = np.array([t["answer_nlp"] for t in valid_nt])
    correct_nt = np.array([1 if t["correct"] else 0 for t in valid_nt])
    nlp_t = np.array([t["answer_nlp"] for t in valid_t])
    correct_t = np.array([1 if t["correct"] else 0 for t in valid_t])

    bin_boundaries = compute_bin_boundaries(nlp_nt, k=4)

    results = {}
    for label, nlp, correct in [("non_thinking", nlp_nt, correct_nt),
                                  ("thinking", nlp_t, correct_t)]:
        bins = assign_bins(nlp, bin_boundaries)
        occupancy = check_bin_occupancy(bins, correct, k=4)

        # Show distribution across bins
        bin_counts = {}
        for b in range(1, 5):
            n_correct = ((bins == b) & (correct == 1)).sum()
            n_incorrect = ((bins == b) & (correct == 0)).sum()
            bin_counts[b] = {"correct": int(n_correct), "incorrect": int(n_incorrect),
                            "total": int(n_correct + n_incorrect)}

        # Percentage of trials in each bin
        total = len(bins)
        print(f"\n    {label}:")
        print(f"      Bin boundaries: {bin_boundaries}")
        for b in range(1, 5):
            bc = bin_counts[b]
            pct = bc["total"] / total * 100
            print(f"      Bin {b}: {bc['total']:5d} ({pct:5.1f}%) | "
                  f"correct={bc['correct']:4d} incorrect={bc['incorrect']:4d}")
        print(f"      Occupancy passes: {occupancy['pass']}")
        print(f"      N occupied bins: {occupancy['n_occupied_bins']}")
        print(f"      Bin collapse: {occupancy['bin_collapse']}")

        # Try computing metad to see the specific error
        contingency = compute_type2_contingency(bins, correct, k=4)
        sdt = compute_metad(contingency["nR_S1"], contingency["nR_S2"])
        print(f"      d'={sdt['d_prime']}, meta-d'={sdt['meta_d_prime']}, "
              f"converged={sdt['converged']}")
        if "error" in sdt:
            print(f"      Error: {sdt['error']}")

        results[label] = {
            "bin_counts": {str(k): v for k, v in bin_counts.items()},
            "occupancy": {k: v for k, v in occupancy.items() if k != "cells"},
            "sdt": sdt,
        }

    return results


# ---------------------------------------------------------------------------
# 3. Answer length confound
# ---------------------------------------------------------------------------

def answer_length_confound(model, valid_nt, valid_t):
    """Check if answer length drives AUROC2 differences."""
    print(f"\n  3. ANSWER LENGTH CONFOUND")

    nlp_nt = np.array([t["answer_nlp"] for t in valid_nt])
    correct_nt = np.array([1 if t["correct"] else 0 for t in valid_nt])
    len_nt = np.array([t.get("answer_length_tokens", 0) for t in valid_nt])
    nlp_t = np.array([t["answer_nlp"] for t in valid_t])
    correct_t = np.array([1 if t["correct"] else 0 for t in valid_t])
    len_t = np.array([t.get("answer_length_tokens", 0) for t in valid_t])

    # Correlation between answer length and NLP
    rho_len_nlp_nt, p_nt = spearmanr(len_nt, nlp_nt)
    rho_len_nlp_t, p_t = spearmanr(len_t, nlp_t)
    print(f"    Length-NLP correlation: NT rho={rho_len_nlp_nt:.3f} (p={p_nt:.2e}), "
          f"T rho={rho_len_nlp_t:.3f} (p={p_t:.2e})")

    # Length-matched analysis: restrict to trials with similar answer lengths
    # Find overlap range
    len_min = max(np.percentile(len_nt, 10), np.percentile(len_t, 10))
    len_max = min(np.percentile(len_nt, 90), np.percentile(len_t, 90))

    mask_nt = (len_nt >= len_min) & (len_nt <= len_max)
    mask_t = (len_t >= len_min) & (len_t <= len_max)

    if mask_nt.sum() > 100 and mask_t.sum() > 100:
        auroc2_nt_matched = compute_auroc2(nlp_nt[mask_nt], correct_nt[mask_nt])
        auroc2_t_matched = compute_auroc2(nlp_t[mask_t], correct_t[mask_t])
        auroc2_nt_full = compute_auroc2(nlp_nt, correct_nt)
        auroc2_t_full = compute_auroc2(nlp_t, correct_t)

        print(f"    Length range [{len_min:.0f}, {len_max:.0f}]: "
              f"NT n={mask_nt.sum()}, T n={mask_t.sum()}")
        print(f"    Full AUROC2:     NT={auroc2_nt_full:.4f}, T={auroc2_t_full:.4f}, "
              f"Δ={auroc2_t_full - auroc2_nt_full:+.4f}")
        print(f"    Matched AUROC2:  NT={auroc2_nt_matched:.4f}, T={auroc2_t_matched:.4f}, "
              f"Δ={auroc2_t_matched - auroc2_nt_matched:+.4f}")

        # Does the finding hold?
        delta_full = auroc2_t_full - auroc2_nt_full
        delta_matched = auroc2_t_matched - auroc2_nt_matched
        print(f"    Conclusion: {'Finding holds after length matching' if delta_matched < 0 else 'Finding attenuated/reversed after length matching'}")

        return {
            "rho_len_nlp_nt": rho_len_nlp_nt,
            "rho_len_nlp_t": rho_len_nlp_t,
            "auroc2_full_delta": delta_full,
            "auroc2_matched_delta": delta_matched,
            "n_matched_nt": int(mask_nt.sum()),
            "n_matched_t": int(mask_t.sum()),
        }
    else:
        print(f"    Insufficient overlap for length matching")
        return {"rho_len_nlp_nt": rho_len_nlp_nt, "rho_len_nlp_t": rho_len_nlp_t}


# ---------------------------------------------------------------------------
# 4. D1 decomposition
# ---------------------------------------------------------------------------

def d1_decomposition(model, valid_nt, valid_t):
    """Detailed D1 analysis: separate context conditioning from metacognition."""
    print(f"\n  4. D1 DECOMPOSITION")

    abridged = load_abridged(model)
    if not abridged:
        print(f"    No abridged data")
        return {}

    nlp_nt = np.array([t["answer_nlp"] for t in valid_nt])
    correct_nt = np.array([1 if t["correct"] else 0 for t in valid_nt])
    auroc2_nt = compute_auroc2(nlp_nt, correct_nt)

    # Build matched arrays: thinking NLP, abridged NLP, correctness
    t_by_idx = {t["item_index"]: t for t in valid_t}
    matched_thinking_nlp = []
    matched_abridged_nlp = []
    matched_correct = []

    for t in valid_t:
        idx = t["item_index"]
        if idx in abridged:
            ab_nlp = abridged[idx].get("abridged_nlp")
            if ab_nlp is not None and not np.isinf(ab_nlp):
                matched_thinking_nlp.append(t["answer_nlp"])
                matched_abridged_nlp.append(ab_nlp)
                matched_correct.append(1 if t["correct"] else 0)

    if len(matched_correct) < 100:
        print(f"    Insufficient matched trials ({len(matched_correct)})")
        return {}

    thinking_nlp = np.array(matched_thinking_nlp)
    abridged_nlp = np.array(matched_abridged_nlp)
    correct = np.array(matched_correct)

    auroc2_thinking = compute_auroc2(thinking_nlp, correct)
    auroc2_abridged = compute_auroc2(abridged_nlp, correct)

    # Correlation between thinking NLP and abridged NLP
    rho, p = spearmanr(thinking_nlp, abridged_nlp)

    # NLP shift: how much does the trace change NLP?
    nlp_shift = thinking_nlp - abridged_nlp
    mean_shift = np.mean(nlp_shift)
    mean_shift_correct = np.mean(nlp_shift[correct == 1])
    mean_shift_incorrect = np.mean(nlp_shift[correct == 0])

    print(f"    N matched trials: {len(correct)}")
    print(f"    AUROC2 non-thinking:  {auroc2_nt:.4f} (reference)")
    print(f"    AUROC2 abridged:      {auroc2_abridged:.4f} (same answers, no trace context)")
    print(f"    AUROC2 thinking:      {auroc2_thinking:.4f} (same answers, with trace context)")
    print(f"    NLP correlation (thinking vs abridged): rho={rho:.4f}, p={p:.2e}")
    print(f"\n    NLP shift (thinking - abridged):")
    print(f"      Overall:   {mean_shift:+.4f}")
    print(f"      Correct:   {mean_shift_correct:+.4f}")
    print(f"      Incorrect: {mean_shift_incorrect:+.4f}")
    print(f"      Differential (correct - incorrect shift): "
          f"{mean_shift_correct - mean_shift_incorrect:+.4f}")

    # Interpretation
    if auroc2_abridged < auroc2_nt:
        print(f"\n    Interpretation: Abridged AUROC2 < non-thinking AUROC2.")
        print(f"      The thinking-mode *answers* (not just the trace context) carry less")
        print(f"      discriminative NLP signal than non-thinking answers. This suggests")
        print(f"      the trace changes the answer distribution itself, not just the context.")
    if auroc2_thinking > auroc2_abridged:
        print(f"      However, thinking AUROC2 > abridged AUROC2, meaning the trace")
        print(f"      context *partially recovers* the signal. The trace helps, but not enough.")

    return {
        "auroc2_nt": auroc2_nt,
        "auroc2_abridged": auroc2_abridged,
        "auroc2_thinking": auroc2_thinking,
        "nlp_correlation": rho,
        "mean_shift_overall": mean_shift,
        "mean_shift_correct": mean_shift_correct,
        "mean_shift_incorrect": mean_shift_incorrect,
        "differential_shift": mean_shift_correct - mean_shift_incorrect,
        "n_matched": len(correct),
    }


# ---------------------------------------------------------------------------
# 5. Domain permutation test
# ---------------------------------------------------------------------------

def domain_permutation_test(model, valid_nt, valid_t, n_permutations=5000):
    """Permutation test on domain labels for H4 domain interaction."""
    print(f"\n  5. DOMAIN PERMUTATION TEST")

    # Get per-domain AUROC2 deltas
    domains = sorted(set(t.get("domain", "Unknown") for t in valid_nt))
    domain_deltas = []

    for domain in domains:
        dom_nt = [t for t in valid_nt if t.get("domain") == domain]
        dom_t = [t for t in valid_t if t.get("domain") == domain]
        if len(dom_nt) < 50 or len(dom_t) < 50:
            continue

        nlp_nt_d = np.array([t["answer_nlp"] for t in dom_nt])
        cor_nt_d = np.array([1 if t["correct"] else 0 for t in dom_nt])
        nlp_t_d = np.array([t["answer_nlp"] for t in dom_t])
        cor_t_d = np.array([1 if t["correct"] else 0 for t in dom_t])

        auroc2_nt = compute_auroc2(nlp_nt_d, cor_nt_d)
        auroc2_t = compute_auroc2(nlp_t_d, cor_t_d)

        if not (np.isnan(auroc2_nt) or np.isnan(auroc2_t)):
            domain_deltas.append(auroc2_t - auroc2_nt)

    if len(domain_deltas) < 3:
        print(f"    Insufficient domains ({len(domain_deltas)})")
        return {}

    # Observed: variance of per-domain deltas (measures domain interaction)
    observed_var = np.var(domain_deltas)

    # Also compute Spearman rho between baseline AUROC2 and delta
    baseline_aurocs = []
    deltas = []
    for domain in domains:
        dom_nt = [t for t in valid_nt if t.get("domain") == domain]
        dom_t = [t for t in valid_t if t.get("domain") == domain]
        if len(dom_nt) < 50 or len(dom_t) < 50:
            continue
        nlp_nt_d = np.array([t["answer_nlp"] for t in dom_nt])
        cor_nt_d = np.array([1 if t["correct"] else 0 for t in dom_nt])
        nlp_t_d = np.array([t["answer_nlp"] for t in dom_t])
        cor_t_d = np.array([1 if t["correct"] else 0 for t in dom_t])
        a_nt = compute_auroc2(nlp_nt_d, cor_nt_d)
        a_t = compute_auroc2(nlp_t_d, cor_t_d)
        if not (np.isnan(a_nt) or np.isnan(a_t)):
            baseline_aurocs.append(a_nt)
            deltas.append(a_t - a_nt)

    rho_baseline_delta, p_rho = spearmanr(baseline_aurocs, deltas) if len(baseline_aurocs) >= 3 else (np.nan, np.nan)
    pattern = "compensatory" if rho_baseline_delta < -0.3 else ("amplificatory" if rho_baseline_delta > 0.3 else "null")

    # Permutation: shuffle domain labels across items, recompute variance
    rng = np.random.default_rng(42)
    all_nt_domains = [t.get("domain", "Unknown") for t in valid_nt]
    all_t_domains = [t.get("domain", "Unknown") for t in valid_t]

    n_exceed = 0
    for _ in range(n_permutations):
        # Shuffle domains within each mode
        perm_nt_domains = rng.permutation(all_nt_domains)
        perm_t_domains = rng.permutation(all_t_domains)

        perm_deltas = []
        for domain in domains:
            perm_dom_nt_idx = [i for i, d in enumerate(perm_nt_domains) if d == domain]
            perm_dom_t_idx = [i for i, d in enumerate(perm_t_domains) if d == domain]
            if len(perm_dom_nt_idx) < 50 or len(perm_dom_t_idx) < 50:
                continue

            nlp_nt_arr = np.array([valid_nt[i]["answer_nlp"] for i in perm_dom_nt_idx])
            cor_nt_arr = np.array([1 if valid_nt[i]["correct"] else 0 for i in perm_dom_nt_idx])
            nlp_t_arr = np.array([valid_t[i]["answer_nlp"] for i in perm_dom_t_idx])
            cor_t_arr = np.array([1 if valid_t[i]["correct"] else 0 for i in perm_dom_t_idx])

            a_nt = compute_auroc2(nlp_nt_arr, cor_nt_arr)
            a_t = compute_auroc2(nlp_t_arr, cor_t_arr)
            if not (np.isnan(a_nt) or np.isnan(a_t)):
                perm_deltas.append(a_t - a_nt)

        if len(perm_deltas) >= 3:
            perm_var = np.var(perm_deltas)
            if perm_var >= observed_var:
                n_exceed += 1

    p_perm = (n_exceed + 1) / (n_permutations + 1)

    print(f"    N domains: {len(domain_deltas)}")
    print(f"    Observed delta variance: {observed_var:.6f}")
    print(f"    Permutation p-value: {p_perm:.4f} ({n_permutations} permutations)")
    print(f"    Baseline-delta rho: {rho_baseline_delta:.3f} (p={p_rho:.3f})")
    print(f"    Pattern: {pattern}")

    return {
        "observed_var": observed_var,
        "p_permutation": p_perm,
        "rho_baseline_delta": rho_baseline_delta,
        "p_rho": p_rho,
        "pattern": pattern,
        "n_domains": len(domain_deltas),
        "domain_deltas": domain_deltas,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print(f"{'='*70}")
    print(f"  SUPPLEMENTARY DIAGNOSTICS")
    print(f"{'='*70}")

    all_results = {}

    for model in MODELS:
        print(f"\n{'='*70}")
        print(f"  MODEL: {model}")
        print(f"{'='*70}")

        valid_nt = filter_valid(load_trials(model, "non_thinking"))
        valid_t = filter_valid(load_trials(model, "thinking"))

        if not valid_nt or not valid_t:
            print(f"  SKIPPING: missing data")
            continue

        model_results = {
            "nlp_compression": nlp_compression(model, valid_nt, valid_t),
            "bin_collapse": bin_collapse_diagnostic(model, valid_nt, valid_t),
            "answer_length": answer_length_confound(model, valid_nt, valid_t),
            "d1_decomposition": d1_decomposition(model, valid_nt, valid_t),
            "domain_permutation": domain_permutation_test(model, valid_nt, valid_t),
        }

        all_results[model] = model_results

    # Save
    def sanitize_keys(obj):
        if isinstance(obj, dict):
            return {str(k): sanitize_keys(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_keys(v) for v in obj]
        return obj

    output_path = os.path.join(PROCESSED_DIR, "supplementary_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sanitize_keys(all_results), f, indent=2, cls=NumpyEncoder)

    print(f"\n  Results saved to {output_path}")
    print(f"\n{'='*70}")
    print(f"  SUPPLEMENTARY DIAGNOSTICS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
