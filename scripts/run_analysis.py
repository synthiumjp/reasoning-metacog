# reasoning-metacog/scripts/run_analysis.py
"""
Full analysis pipeline per pre-registration.
Loads all inference results, computes all pre-registered metrics,
runs bootstrap inference, and outputs structured results.

Usage:
    python scripts/run_analysis.py
    python scripts/run_analysis.py --skip-bootstrap   # Fast run without CIs
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from local_config import PROJECT_DIR
from scripts.sdt_analysis import (
    compute_auroc2,
    compute_bin_boundaries,
    assign_bins,
    full_sdt_pipeline,
    check_nlp_monotonicity,
    compute_validity_indices,
    compute_d_prime_accuracy,
    compute_ece,
)
from scripts.bootstrap import bootstrap_paired_delta
from scripts.scoring import score_answer, is_malformed

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


def sanitize_keys(obj):
    """Recursively convert non-string dict keys to strings for JSON."""
    if isinstance(obj, dict):
        return {str(k): sanitize_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_keys(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trials(model: str, mode: str) -> list[dict]:
    """Load trial-level data from JSONL."""
    path = os.path.join(RAW_DIR, f"trials_{model}_{mode}.jsonl")
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found")
        return []
    trials = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            trials.append(json.loads(line.strip()))
    return trials


def load_abridged_nlp(model: str) -> dict:
    """Load D1 abridged-NLP data, keyed by item_index."""
    path = os.path.join(RAW_DIR, f"abridged_nlp_{model}.jsonl")
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found")
        return {}
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            data[record["item_index"]] = record
    return data


def filter_valid_trials(trials: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Filter to valid (non-malformed, non-None correctness) trials.
    Returns (nlp_values, correctness, filtered_trials).
    """
    valid = []
    for t in trials:
        if t.get("malformed", False):
            continue
        if t.get("correct") is None:
            continue
        nlp = t.get("answer_nlp")
        if nlp is None or nlp == float("-inf") or np.isinf(nlp):
            continue
        valid.append(t)

    nlp = np.array([t["answer_nlp"] for t in valid])
    correct = np.array([1 if t["correct"] else 0 for t in valid])
    return nlp, correct, valid


# ---------------------------------------------------------------------------
# Per-model analysis
# ---------------------------------------------------------------------------

def analyse_model(model: str, skip_bootstrap: bool = False) -> dict:
    """Run the full pre-registered analysis pipeline for one model."""
    print(f"\n{'='*70}")
    print(f"  MODEL: {model}")
    print(f"{'='*70}")

    # Load data
    trials_nt = load_trials(model, "non_thinking")
    trials_t = load_trials(model, "thinking")
    abridged = load_abridged_nlp(model)

    if not trials_nt or not trials_t:
        print(f"  SKIPPING {model}: missing data")
        return {}

    # Filter valid trials
    nlp_nt, correct_nt, valid_nt = filter_valid_trials(trials_nt)
    nlp_t, correct_t, valid_t = filter_valid_trials(trials_t)

    print(f"\n  Non-thinking: {len(valid_nt)} valid trials, "
          f"accuracy={correct_nt.mean():.1%}")
    print(f"  Thinking:     {len(valid_t)} valid trials, "
          f"accuracy={correct_t.mean():.1%}")

    # --- Exclusion check: non-thinking accuracy < 20% ---
    if correct_nt.mean() < 0.20:
        print(f"  EXCLUDED: non-thinking accuracy {correct_nt.mean():.1%} < 20%")
        return {"model": model, "excluded": True, "reason": "accuracy < 20%"}

    # --- Compute bin boundaries from non-thinking NLP (reference distribution) ---
    bin_boundaries = compute_bin_boundaries(nlp_nt, k=4)
    print(f"  Bin boundaries (from non-thinking): {bin_boundaries}")

    # --- D3: NLP monotonicity ---
    d3_nt = check_nlp_monotonicity(nlp_nt, correct_nt)
    d3_t = check_nlp_monotonicity(nlp_t, correct_t)
    print(f"\n  D3 NLP monotonicity:")
    print(f"    Non-thinking: rho={d3_nt['spearman_rho']:.3f}, "
          f"monotonic={d3_nt['monotonic']}, passes={d3_nt['passes']}")
    print(f"    Thinking:     rho={d3_t['spearman_rho']:.3f}, "
          f"monotonic={d3_t['monotonic']}, passes={d3_t['passes']}")

    d3_passes = d3_nt["passes"] and d3_t["passes"]
    if not d3_passes:
        print(f"  WARNING: D3 fails — model excluded from H1a, H1b, H2")

    # --- Full SDT pipeline per condition ---
    print(f"\n  Running SDT pipeline...")
    sdt_nt = full_sdt_pipeline(nlp_nt, correct_nt, bin_boundaries)
    sdt_t = full_sdt_pipeline(nlp_t, correct_t, bin_boundaries)

    print(f"\n  Non-thinking SDT:")
    print(f"    AUROC2:   {sdt_nt['auroc2']:.4f}")
    print(f"    d':       {sdt_nt['d_prime_nlp']:.4f}")
    print(f"    meta-d':  {sdt_nt['meta_d_prime']:.4f}")
    print(f"    M-ratio:  {sdt_nt['m_ratio']:.4f}")
    print(f"    ECE:      {sdt_nt['ece']:.4f}")

    print(f"\n  Thinking SDT:")
    print(f"    AUROC2:   {sdt_t['auroc2']:.4f}")
    print(f"    d':       {sdt_t['d_prime_nlp']:.4f}")
    print(f"    meta-d':  {sdt_t['meta_d_prime']:.4f}")
    print(f"    M-ratio:  {sdt_t['m_ratio']:.4f}")
    print(f"    ECE:      {sdt_t['ece']:.4f}")

    # --- Deltas ---
    delta_auroc2 = sdt_t["auroc2"] - sdt_nt["auroc2"]
    delta_mratio = sdt_t["m_ratio"] - sdt_nt["m_ratio"]
    delta_ece = sdt_t["ece"] - sdt_nt["ece"]

    print(f"\n  Deltas (thinking - non-thinking):")
    print(f"    ΔAUROC2:  {delta_auroc2:+.4f}")
    print(f"    ΔM-ratio: {delta_mratio:+.4f}")
    print(f"    ΔECE:     {delta_ece:+.4f}")

    # --- D4: Validity screening ---
    d4_nt = compute_validity_indices(nlp_nt, correct_nt)
    d4_t = compute_validity_indices(nlp_t, correct_t)
    print(f"\n  D4 Validity screening:")
    print(f"    Non-thinking: {d4_nt['classification']} "
          f"(L={d4_nt['L']:.3f}, Fp={d4_nt['Fp']:.3f}, "
          f"RBS={d4_nt['RBS']:.3f}, r={d4_nt['r_confidence_correct']:.3f})")
    print(f"    Thinking:     {d4_t['classification']} "
          f"(L={d4_t['L']:.3f}, Fp={d4_t['Fp']:.3f}, "
          f"RBS={d4_t['RBS']:.3f}, r={d4_t['r_confidence_correct']:.3f})")

    # --- D1: Abridged-NLP ---
    d1_result = None
    if abridged:
        # Match abridged NLP to thinking-mode trials
        abridged_nlps = []
        abridged_correct = []
        for t in valid_t:
            idx = t["item_index"]
            if idx in abridged:
                ab = abridged[idx]
                ab_nlp = ab.get("abridged_nlp")
                if ab_nlp is not None and not np.isinf(ab_nlp):
                    abridged_nlps.append(ab_nlp)
                    abridged_correct.append(1 if t["correct"] else 0)

        if len(abridged_nlps) > 100:
            ab_nlp_arr = np.array(abridged_nlps)
            ab_correct_arr = np.array(abridged_correct)

            d1_auroc2 = compute_auroc2(ab_nlp_arr, ab_correct_arr)
            d1_sdt = full_sdt_pipeline(ab_nlp_arr, ab_correct_arr, bin_boundaries)

            d1_result = {
                "n_trials": len(abridged_nlps),
                "auroc2": d1_auroc2,
                "m_ratio": d1_sdt["m_ratio"],
                "d_prime_nlp": d1_sdt["d_prime_nlp"],
                "meta_d_prime": d1_sdt["meta_d_prime"],
                "ece": d1_sdt["ece"],
            }
            print(f"\n  D1 Abridged-NLP ({d1_result['n_trials']} trials):")
            print(f"    AUROC2:   {d1_result['auroc2']:.4f} "
                  f"(thinking: {sdt_t['auroc2']:.4f})")
            print(f"    M-ratio:  {d1_result['m_ratio']:.4f} "
                  f"(thinking: {sdt_t['m_ratio']:.4f})")
        else:
            print(f"\n  D1: insufficient matched trials ({len(abridged_nlps)})")

    # --- D2: Accuracy-based d' stress test ---
    d2_nt_dprime_acc = compute_d_prime_accuracy(correct_nt.mean())
    d2_t_dprime_acc = compute_d_prime_accuracy(correct_t.mean())
    d2_nt_mratio = (sdt_nt["meta_d_prime"] / d2_nt_dprime_acc
                    if abs(d2_nt_dprime_acc) > 1e-10 else np.nan)
    d2_t_mratio = (sdt_t["meta_d_prime"] / d2_t_dprime_acc
                   if abs(d2_t_dprime_acc) > 1e-10 else np.nan)
    print(f"\n  D2 Accuracy-based d' stress test:")
    print(f"    Non-thinking: d'_acc={d2_nt_dprime_acc:.4f}, "
          f"M-ratio_acc={d2_nt_mratio:.4f}")
    print(f"    Thinking:     d'_acc={d2_t_dprime_acc:.4f}, "
          f"M-ratio_acc={d2_t_mratio:.4f}")

    # --- Bootstrap inference ---
    bootstrap_result = None
    if not skip_bootstrap:
        print(f"\n  Running bootstrap (10,000 resamples)...")
        t0 = time.time()

        # Need paired data: same items in both conditions
        # Build paired arrays by item_index
        nt_by_idx = {t["item_index"]: t for t in valid_nt}
        t_by_idx = {t["item_index"]: t for t in valid_t}
        shared_idx = sorted(set(nt_by_idx.keys()) & set(t_by_idx.keys()))

        paired_nlp_nt = np.array([nt_by_idx[i]["answer_nlp"] for i in shared_idx])
        paired_correct_nt = np.array([1 if nt_by_idx[i]["correct"] else 0
                                       for i in shared_idx])
        paired_nlp_t = np.array([t_by_idx[i]["answer_nlp"] for i in shared_idx])
        paired_correct_t = np.array([1 if t_by_idx[i]["correct"] else 0
                                      for i in shared_idx])

        print(f"    Paired items: {len(shared_idx)}")

        bootstrap_result = bootstrap_paired_delta(
            nlp_think=paired_nlp_t,
            correct_think=paired_correct_t,
            nlp_nothink=paired_nlp_nt,
            correct_nothink=paired_correct_nt,
            bin_boundaries=bin_boundaries,
            n_resamples=10000,
            seed=42,
        )

        elapsed = time.time() - t0
        print(f"    Bootstrap complete in {elapsed/60:.1f} min")
        print(f"\n  H1a ΔAUROC2: {bootstrap_result['delta_auroc2']['point']:+.4f} "
              f"[{bootstrap_result['delta_auroc2']['ci_lower']:+.4f}, "
              f"{bootstrap_result['delta_auroc2']['ci_upper']:+.4f}] "
              f"{'*' if bootstrap_result['delta_auroc2']['excludes_zero'] else 'ns'}")
        print(f"  H1b ΔM-ratio: {bootstrap_result['delta_m_ratio']['point']:+.4f} "
              f"[{bootstrap_result['delta_m_ratio']['ci_lower']:+.4f}, "
              f"{bootstrap_result['delta_m_ratio']['ci_upper']:+.4f}] "
              f"{'*' if bootstrap_result['delta_m_ratio']['excludes_zero'] else 'ns'}")
        print(f"  H3  ΔECE: {bootstrap_result['delta_ece']['point']:+.4f} "
              f"[{bootstrap_result['delta_ece']['ci_lower']:+.4f}, "
              f"{bootstrap_result['delta_ece']['ci_upper']:+.4f}] "
              f"{'*' if bootstrap_result['delta_ece']['excludes_zero'] else 'ns'}")
        print(f"    M-ratio exclusion rate: "
              f"{bootstrap_result['mratio_exclusion_rate']:.1%}")

    # --- Thinking-mode trace statistics ---
    trace_lengths = [t.get("trace_length_tokens", 0) for t in valid_t
                     if not t.get("is_empty_trace", True)]
    trace_stats = {}
    if trace_lengths:
        trace_stats = {
            "n_with_trace": len(trace_lengths),
            "mean": float(np.mean(trace_lengths)),
            "median": float(np.median(trace_lengths)),
            "min": int(np.min(trace_lengths)),
            "max": int(np.max(trace_lengths)),
            "n_truncated": sum(1 for t in valid_t
                               if t.get("is_truncated", False)),
        }

    # --- Answer length comparison ---
    ans_len_nt = np.array([t.get("answer_length_tokens", 0) for t in valid_nt])
    ans_len_t = np.array([t.get("answer_length_tokens", 0) for t in valid_t])
    length_ratio = ans_len_t.mean() / ans_len_nt.mean() if ans_len_nt.mean() > 0 else np.nan
    print(f"\n  Answer length: NT mean={ans_len_nt.mean():.1f}, "
          f"T mean={ans_len_t.mean():.1f}, ratio={length_ratio:.2f}")
    if abs(length_ratio - 1.0) > 0.50:
        print(f"  WARNING: answer length differs by >{50}% between modes")

    # --- Domain analysis (H4) ---
    domain_results = {}
    domains = sorted(set(t.get("domain", "Unknown") for t in valid_nt))
    for domain in domains:
        dom_nt = [t for t in valid_nt if t.get("domain") == domain]
        dom_t = [t for t in valid_t if t.get("domain") == domain]
        if len(dom_nt) < 50 or len(dom_t) < 50:
            continue
        dom_nlp_nt = np.array([t["answer_nlp"] for t in dom_nt])
        dom_correct_nt = np.array([1 if t["correct"] else 0 for t in dom_nt])
        dom_nlp_t = np.array([t["answer_nlp"] for t in dom_t])
        dom_correct_t = np.array([1 if t["correct"] else 0 for t in dom_t])

        dom_auroc2_nt = compute_auroc2(dom_nlp_nt, dom_correct_nt)
        dom_auroc2_t = compute_auroc2(dom_nlp_t, dom_correct_t)

        domain_results[domain] = {
            "n_nt": len(dom_nt),
            "n_t": len(dom_t),
            "acc_nt": float(dom_correct_nt.mean()),
            "acc_t": float(dom_correct_t.mean()),
            "auroc2_nt": dom_auroc2_nt,
            "auroc2_t": dom_auroc2_t,
            "delta_auroc2": dom_auroc2_t - dom_auroc2_nt,
        }

    if domain_results:
        print(f"\n  H4 Domain results:")
        for dom, dr in sorted(domain_results.items()):
            print(f"    {dom:25s}: acc NT={dr['acc_nt']:.1%} T={dr['acc_t']:.1%} | "
                  f"AUROC2 NT={dr['auroc2_nt']:.3f} T={dr['auroc2_t']:.3f} "
                  f"Δ={dr['delta_auroc2']:+.3f}")

    # --- Assemble results ---
    result = {
        "model": model,
        "excluded": False,
        "non_thinking": {
            "n_total": len(trials_nt),
            "n_valid": len(valid_nt),
            "n_malformed": sum(1 for t in trials_nt if t.get("malformed", False)),
            "accuracy": float(correct_nt.mean()),
            "sdt": sdt_nt,
            "d3": d3_nt,
            "d4": d4_nt,
        },
        "thinking": {
            "n_total": len(trials_t),
            "n_valid": len(valid_t),
            "n_malformed": sum(1 for t in trials_t if t.get("malformed", False)),
            "accuracy": float(correct_t.mean()),
            "sdt": sdt_t,
            "d3": d3_t,
            "d4": d4_t,
            "trace_stats": trace_stats,
        },
        "deltas": {
            "auroc2": delta_auroc2,
            "m_ratio": delta_mratio,
            "ece": delta_ece,
        },
        "d1_abridged": d1_result,
        "d2_stress_test": {
            "non_thinking": {
                "d_prime_acc": d2_nt_dprime_acc,
                "m_ratio_acc": d2_nt_mratio,
            },
            "thinking": {
                "d_prime_acc": d2_t_dprime_acc,
                "m_ratio_acc": d2_t_mratio,
            },
        },
        "d3_passes": d3_passes,
        "bin_boundaries": bin_boundaries.tolist(),
        "bootstrap": bootstrap_result,
        "answer_length": {
            "non_thinking_mean": float(ans_len_nt.mean()),
            "thinking_mean": float(ans_len_t.mean()),
            "ratio": length_ratio,
        },
        "domains": domain_results,
    }

    return result


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]):
    """Print a summary table across all models."""
    print(f"\n{'='*70}")
    print(f"  SUMMARY TABLE")
    print(f"{'='*70}\n")

    header = (f"  {'Model':30s} {'AccNT':>6s} {'AccT':>6s} "
              f"{'AUROC2_NT':>9s} {'AUROC2_T':>9s} {'ΔAUROC2':>8s} "
              f"{'Mrat_NT':>8s} {'Mrat_T':>8s} {'ΔMrat':>8s} "
              f"{'D3':>3s} {'D4_NT':>6s} {'D4_T':>6s}")
    print(header)
    print(f"  {'-'*len(header)}")

    for r in results:
        if r.get("excluded"):
            print(f"  {r['model']:30s} EXCLUDED ({r.get('reason', '')})")
            continue

        nt = r["non_thinking"]
        t = r["thinking"]
        d = r["deltas"]

        d4_nt_class = nt["d4"]["classification"][:3]
        d4_t_class = t["d4"]["classification"][:3]

        print(f"  {r['model']:30s} "
              f"{nt['accuracy']:6.1%} {t['accuracy']:6.1%} "
              f"{nt['sdt']['auroc2']:9.4f} {t['sdt']['auroc2']:9.4f} "
              f"{d['auroc2']:+8.4f} "
              f"{nt['sdt']['m_ratio']:8.4f} {t['sdt']['m_ratio']:8.4f} "
              f"{d['m_ratio']:+8.4f} "
              f"{'OK' if r['d3_passes'] else 'FAIL':>3s} "
              f"{d4_nt_class:>6s} {d4_t_class:>6s}")

    # Bootstrap CIs if available
    has_bootstrap = any(r.get("bootstrap") for r in results if not r.get("excluded"))
    if has_bootstrap:
        print(f"\n  Bootstrap 95% CIs (10,000 resamples, seed 42):")
        print(f"  {'Model':30s} {'H1a ΔAUROC2':>30s} {'H1b ΔM-ratio':>30s} "
              f"{'H3 ΔECE':>30s}")
        for r in results:
            if r.get("excluded") or not r.get("bootstrap"):
                continue
            b = r["bootstrap"]
            for key, label in [("delta_auroc2", "H1a"), ("delta_m_ratio", "H1b"),
                                ("delta_ece", "H3")]:
                pass
            da = b["delta_auroc2"]
            dm = b["delta_m_ratio"]
            de = b["delta_ece"]
            sig_a = "*" if da["excludes_zero"] else " "
            sig_m = "*" if dm["excludes_zero"] else " "
            sig_e = "*" if de["excludes_zero"] else " "
            print(f"  {r['model']:30s} "
                  f"{da['point']:+.4f} [{da['ci_lower']:+.4f}, {da['ci_upper']:+.4f}]{sig_a} "
                  f"{dm['point']:+.4f} [{dm['ci_lower']:+.4f}, {dm['ci_upper']:+.4f}]{sig_m} "
                  f"{de['point']:+.4f} [{de['ci_lower']:+.4f}, {de['ci_upper']:+.4f}]{sig_e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Reasoning-Metacog Analysis Pipeline")
    parser.add_argument("--skip-bootstrap", action="store_true",
                        help="Skip bootstrap (faster, no CIs)")
    parser.add_argument("--models", nargs="+", default=MODELS,
                        help="Models to analyse")
    args = parser.parse_args()

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print(f"{'='*70}")
    print(f"  REASONING-METACOG ANALYSIS PIPELINE")
    print(f"  Pre-registered: OSF [30 April 2026]")
    print(f"{'='*70}")

    results = []
    for model in args.models:
        result = analyse_model(model, skip_bootstrap=args.skip_bootstrap)
        if result:
            results.append(result)

    # Save full results
    output_path = os.path.join(PROCESSED_DIR, "analysis_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sanitize_keys(results), f, indent=2, cls=NumpyEncoder)
    print(f"\n  Full results saved to {output_path}")

    # Print summary table
    print_summary(results)

    # Decision table per pre-registration §4.1
    print(f"\n{'='*70}")
    print(f"  DECISION TABLE")
    print(f"{'='*70}")
    for r in results:
        if r.get("excluded"):
            continue
        model = r["model"]
        d3_ok = r["d3_passes"]
        d4_nt = r["non_thinking"]["d4"]["classification"]
        d4_t = r["thinking"]["d4"]["classification"]
        d1 = r.get("d1_abridged")
        b = r.get("bootstrap")

        print(f"\n  {model}:")
        print(f"    D3 passes: {d3_ok}")
        print(f"    D4: NT={d4_nt}, T={d4_t}")

        if b:
            h1a_sig = b["delta_auroc2"]["excludes_zero"]
            h1b_sig = b["delta_m_ratio"]["excludes_zero"]
            h3_sig = b["delta_ece"]["excludes_zero"]

            if d3_ok:
                print(f"    H1a (ΔAUROC2): {'SIGNIFICANT' if h1a_sig else 'NOT SIGNIFICANT'}")
                print(f"    H1b (ΔM-ratio): {'SIGNIFICANT' if h1b_sig else 'NOT SIGNIFICANT'}")
            else:
                print(f"    H1a, H1b: EXCLUDED (D3 failure)")

            print(f"    H3 (ΔECE): {'SIGNIFICANT' if h3_sig else 'NOT SIGNIFICANT'}")

            if d1:
                d1_delta_auroc2 = d1["auroc2"] - r["non_thinking"]["sdt"]["auroc2"]
                thinking_delta = r["deltas"]["auroc2"]
                if h1a_sig and abs(d1_delta_auroc2) < abs(thinking_delta) * 0.5:
                    print(f"    D1 interpretation: Effect likely metacognitive "
                          f"(D1 ΔAUROC2={d1_delta_auroc2:+.4f} << "
                          f"thinking ΔAUROC2={thinking_delta:+.4f})")
                elif h1a_sig:
                    print(f"    D1 interpretation: Effect may be context conditioning "
                          f"(D1 ΔAUROC2={d1_delta_auroc2:+.4f} ~ "
                          f"thinking ΔAUROC2={thinking_delta:+.4f})")

    print(f"\n{'='*70}")
    print(f"  ANALYSIS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
