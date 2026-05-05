# reasoning-metacog/scripts/run_exploratory.py
"""
Exploratory analyses E1-E7, Q8 quantisation check, and K-robustness.
Run after run_analysis.py completes.

Usage:
    python scripts/run_exploratory.py
"""

import json
import os
import sys
import re

import numpy as np
from scipy.stats import spearmanr, friedmanchisquare, rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from local_config import PROJECT_DIR
from scripts.sdt_analysis import (
    compute_auroc2,
    compute_bin_boundaries,
    full_sdt_pipeline,
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
# E1: Item difficulty interaction
# ---------------------------------------------------------------------------

def run_e1(model, valid_nt, valid_t):
    """M-ratio and AUROC2 delta by difficulty tercile."""
    print(f"\n  E1: Item difficulty interaction")

    # Difficulty = proportion incorrect in non-thinking mode per item
    # Since each item is seen once, use non-thinking correctness as difficulty proxy
    # Split items into terciles by non-thinking NLP (proxy for difficulty)
    nlp_nt = np.array([t["answer_nlp"] for t in valid_nt])
    correct_nt = np.array([1 if t["correct"] else 0 for t in valid_nt])

    # Tercile boundaries on non-thinking accuracy
    # Group by item difficulty: items where NT got it wrong are "hard"
    nt_by_idx = {t["item_index"]: t for t in valid_nt}
    t_by_idx = {t["item_index"]: t for t in valid_t}
    shared = sorted(set(nt_by_idx.keys()) & set(t_by_idx.keys()))

    if len(shared) < 100:
        print(f"    Insufficient paired items ({len(shared)})")
        return {}

    # Sort by non-thinking NLP as difficulty proxy (lower NLP = harder)
    items_sorted = sorted(shared, key=lambda i: nt_by_idx[i]["answer_nlp"])
    tercile_size = len(items_sorted) // 3
    terciles = {
        "hard": items_sorted[:tercile_size],
        "medium": items_sorted[tercile_size:2*tercile_size],
        "easy": items_sorted[2*tercile_size:],
    }

    bin_boundaries = compute_bin_boundaries(nlp_nt, k=4)
    results = {}

    for label, indices in terciles.items():
        nlp_t = np.array([t_by_idx[i]["answer_nlp"] for i in indices])
        cor_t = np.array([1 if t_by_idx[i]["correct"] else 0 for i in indices])
        nlp_n = np.array([nt_by_idx[i]["answer_nlp"] for i in indices])
        cor_n = np.array([1 if nt_by_idx[i]["correct"] else 0 for i in indices])

        auroc2_nt = compute_auroc2(nlp_n, cor_n) if len(np.unique(cor_n)) == 2 else np.nan
        auroc2_t = compute_auroc2(nlp_t, cor_t) if len(np.unique(cor_t)) == 2 else np.nan

        results[label] = {
            "n": len(indices),
            "acc_nt": float(cor_n.mean()),
            "acc_t": float(cor_t.mean()),
            "auroc2_nt": auroc2_nt,
            "auroc2_t": auroc2_t,
            "delta_auroc2": auroc2_t - auroc2_nt if not (np.isnan(auroc2_t) or np.isnan(auroc2_nt)) else np.nan,
        }
        print(f"    {label:8s}: n={len(indices)}, acc NT={cor_n.mean():.1%} T={cor_t.mean():.1%}, "
              f"AUROC2 NT={auroc2_nt:.3f} T={auroc2_t:.3f} Δ={results[label]['delta_auroc2']:+.3f}")

    return results


# ---------------------------------------------------------------------------
# E2: Trace length–accuracy
# ---------------------------------------------------------------------------

def run_e2(model, valid_t):
    """Spearman rho between thinking trace length and correctness."""
    print(f"\n  E2: Trace length–accuracy")

    traces = [(t["trace_length_tokens"], 1 if t["correct"] else 0)
              for t in valid_t if t.get("trace_length_tokens", 0) > 0]

    if len(traces) < 50:
        print(f"    Insufficient traces ({len(traces)})")
        return {}

    lengths = np.array([x[0] for x in traces])
    correct = np.array([x[1] for x in traces])

    rho, p = spearmanr(lengths, correct)
    print(f"    rho={rho:.4f}, p={p:.4e}, n={len(traces)}")

    return {"spearman_rho": rho, "p_value": p, "n": len(traces)}


# ---------------------------------------------------------------------------
# E3: Trace length–confidence (directional prediction from saturation paper)
# ---------------------------------------------------------------------------

def run_e3(model, valid_t):
    """Spearman rho between trace length and NLP, separately for correct/incorrect."""
    print(f"\n  E3: Trace length–confidence")

    traces = [(t["trace_length_tokens"], t["answer_nlp"], 1 if t["correct"] else 0)
              for t in valid_t if t.get("trace_length_tokens", 0) > 0]

    if len(traces) < 50:
        print(f"    Insufficient traces ({len(traces)})")
        return {}

    lengths = np.array([x[0] for x in traces])
    nlps = np.array([x[1] for x in traces])
    correct = np.array([x[2] for x in traces])

    # Overall
    rho_all, p_all = spearmanr(lengths, nlps)

    # Correct only
    mask_c = correct == 1
    rho_correct, p_correct = spearmanr(lengths[mask_c], nlps[mask_c]) if mask_c.sum() > 20 else (np.nan, np.nan)

    # Incorrect only
    mask_i = correct == 0
    rho_incorrect, p_incorrect = spearmanr(lengths[mask_i], nlps[mask_i]) if mask_i.sum() > 20 else (np.nan, np.nan)

    # Directional prediction: rho < 0 for R1-Distill (replicating saturation paper)
    is_r1 = model.startswith("r1-distill")
    directional = "predicted_negative" if is_r1 else "no_prediction"
    if is_r1:
        replicates = rho_all < 0 and p_all < 0.05
    else:
        replicates = None

    print(f"    Overall:   rho={rho_all:.4f}, p={p_all:.4e}")
    print(f"    Correct:   rho={rho_correct:.4f}, p={p_correct:.4e}")
    print(f"    Incorrect: rho={rho_incorrect:.4f}, p={p_incorrect:.4e}")
    if is_r1:
        print(f"    Directional (R1-Distill, predicted rho<0): {'REPLICATES' if replicates else 'DOES NOT REPLICATE'}")

    return {
        "rho_all": rho_all, "p_all": p_all,
        "rho_correct": rho_correct, "p_correct": p_correct,
        "rho_incorrect": rho_incorrect, "p_incorrect": p_incorrect,
        "directional": directional,
        "replicates": replicates,
        "n": len(traces),
    }


# ---------------------------------------------------------------------------
# E4: Correct vs error trace length asymmetry
# ---------------------------------------------------------------------------

def run_e4(model, valid_nt, valid_t):
    """Mean trace length for correct vs incorrect, stratified by difficulty tercile."""
    print(f"\n  E4: Trace length asymmetry (correct vs error)")

    nt_by_idx = {t["item_index"]: t for t in valid_nt}
    traces = [(t, nt_by_idx.get(t["item_index"]))
              for t in valid_t if t.get("trace_length_tokens", 0) > 0]
    traces = [(t, nt) for t, nt in traces if nt is not None]

    if len(traces) < 100:
        print(f"    Insufficient traces ({len(traces)})")
        return {}

    # Sort by NT NLP for difficulty terciles
    traces_sorted = sorted(traces, key=lambda x: x[1]["answer_nlp"])
    tercile_size = len(traces_sorted) // 3

    results = {}
    for label, start, end in [("hard", 0, tercile_size),
                               ("medium", tercile_size, 2*tercile_size),
                               ("easy", 2*tercile_size, len(traces_sorted))]:
        subset = traces_sorted[start:end]
        correct_lens = [t["trace_length_tokens"] for t, _ in subset if t["correct"]]
        error_lens = [t["trace_length_tokens"] for t, _ in subset if not t["correct"]]

        mean_c = np.mean(correct_lens) if correct_lens else np.nan
        mean_e = np.mean(error_lens) if error_lens else np.nan

        results[label] = {
            "n_correct": len(correct_lens),
            "n_error": len(error_lens),
            "mean_correct": float(mean_c),
            "mean_error": float(mean_e),
            "ratio": float(mean_e / mean_c) if mean_c > 0 and not np.isnan(mean_e) else np.nan,
        }
        print(f"    {label:8s}: correct={mean_c:.0f} ({len(correct_lens)}), "
              f"error={mean_e:.0f} ({len(error_lens)}), "
              f"ratio={results[label]['ratio']:.2f}")

    return results


# ---------------------------------------------------------------------------
# E5: Thinking trace structural features
# ---------------------------------------------------------------------------

def run_e5(model, valid_t):
    """Proportion of traces with backtracking, verification, alternative markers."""
    print(f"\n  E5: Trace structural features")

    backtrack_markers = ["wait", "actually", "no,", "let me reconsider",
                         "i was wrong", "on second thought", "correction"]
    verify_markers = ["let me check", "let me verify", "double check",
                      "to confirm", "verifying", "checking"]
    alternative_markers = ["alternatively", "another possibility", "or maybe",
                           "it could also be", "on the other hand", "but also"]

    traces = [t for t in valid_t if t.get("trace_text", "").strip()]
    if not traces:
        print(f"    No traces available")
        return {}

    n = len(traces)
    n_backtrack = 0
    n_verify = 0
    n_alternative = 0
    backtrack_correct = []

    for t in traces:
        trace_lower = t.get("trace_text", "").lower()
        has_bt = any(m in trace_lower for m in backtrack_markers)
        has_vf = any(m in trace_lower for m in verify_markers)
        has_alt = any(m in trace_lower for m in alternative_markers)

        if has_bt:
            n_backtrack += 1
            backtrack_correct.append(1 if t["correct"] else 0)
        if has_vf:
            n_verify += 1
        if has_alt:
            n_alternative += 1

    # Does backtracking predict correctness above baseline?
    baseline_acc = np.mean([1 if t["correct"] else 0 for t in traces])
    backtrack_acc = np.mean(backtrack_correct) if backtrack_correct else np.nan

    results = {
        "n_traces": n,
        "backtracking_rate": n_backtrack / n,
        "verification_rate": n_verify / n,
        "alternative_rate": n_alternative / n,
        "baseline_accuracy": float(baseline_acc),
        "backtrack_accuracy": float(backtrack_acc),
        "backtrack_above_baseline": float(backtrack_acc - baseline_acc) if not np.isnan(backtrack_acc) else np.nan,
    }

    print(f"    Traces analysed: {n}")
    print(f"    Backtracking: {n_backtrack}/{n} ({n_backtrack/n:.1%})")
    print(f"    Verification: {n_verify}/{n} ({n_verify/n:.1%})")
    print(f"    Alternatives: {n_alternative}/{n} ({n_alternative/n:.1%})")
    print(f"    Backtrack accuracy: {backtrack_acc:.1%} vs baseline {baseline_acc:.1%} "
          f"(Δ={results['backtrack_above_baseline']:+.1%})")

    return results


# ---------------------------------------------------------------------------
# E6: Cross-model consistency
# ---------------------------------------------------------------------------

def run_e6(all_results):
    """Whether all three models show the same direction of H1a effect."""
    print(f"\n{'='*70}")
    print(f"  E6: Cross-model consistency")
    print(f"{'='*70}")

    directions = []
    for model, res in all_results.items():
        delta = res.get("delta_auroc2")
        if delta is not None and not np.isnan(delta):
            direction = "negative" if delta < 0 else "positive"
            directions.append(direction)
            print(f"    {model}: ΔAUROC2 = {delta:+.4f} ({direction})")

    all_same = len(set(directions)) == 1
    print(f"    All same direction: {all_same} ({'YES — consistent' if all_same else 'NO — inconsistent'})")

    return {
        "all_same_direction": all_same,
        "direction": directions[0] if all_same else "mixed",
        "n_models": len(directions),
    }


# ---------------------------------------------------------------------------
# E7: Rank-normalised M-ratio
# ---------------------------------------------------------------------------

def run_e7(model, valid_nt, valid_t):
    """Recompute M-ratio after rank-normalising NLP within each mode."""
    print(f"\n  E7: Rank-normalised M-ratio")

    from scipy.stats import norm as sp_norm

    nlp_nt = np.array([t["answer_nlp"] for t in valid_nt])
    correct_nt = np.array([1 if t["correct"] else 0 for t in valid_nt])
    nlp_t = np.array([t["answer_nlp"] for t in valid_t])
    correct_t = np.array([1 if t["correct"] else 0 for t in valid_t])

    # Rank-normalise: rank -> quantile -> probit
    def rank_normalise(x):
        ranks = rankdata(x)
        quantiles = (ranks - 0.5) / len(ranks)
        return sp_norm.ppf(quantiles)

    nlp_nt_rn = rank_normalise(nlp_nt)
    nlp_t_rn = rank_normalise(nlp_t)

    bin_boundaries_rn = compute_bin_boundaries(nlp_nt_rn, k=4)

    sdt_nt = full_sdt_pipeline(nlp_nt_rn, correct_nt, bin_boundaries_rn)
    sdt_t = full_sdt_pipeline(nlp_t_rn, correct_t, bin_boundaries_rn)

    delta_auroc2 = sdt_t["auroc2"] - sdt_nt["auroc2"]
    delta_mratio = sdt_t["m_ratio"] - sdt_nt["m_ratio"] if not (np.isnan(sdt_t["m_ratio"]) or np.isnan(sdt_nt["m_ratio"])) else np.nan

    print(f"    Rank-normalised NT: AUROC2={sdt_nt['auroc2']:.4f}, M-ratio={sdt_nt['m_ratio']:.4f}")
    print(f"    Rank-normalised T:  AUROC2={sdt_t['auroc2']:.4f}, M-ratio={sdt_t['m_ratio']:.4f}")
    print(f"    ΔAUROC2={delta_auroc2:+.4f}, ΔM-ratio={delta_mratio:+.4f}")

    # AUROC2 should be identical (rank-preserving transform)
    # If M-ratio delta reverses, the original finding is distributional
    return {
        "nt": {"auroc2": sdt_nt["auroc2"], "m_ratio": sdt_nt["m_ratio"]},
        "t": {"auroc2": sdt_t["auroc2"], "m_ratio": sdt_t["m_ratio"]},
        "delta_auroc2": delta_auroc2,
        "delta_m_ratio": delta_mratio,
    }


# ---------------------------------------------------------------------------
# Q8 quantisation check
# ---------------------------------------------------------------------------

def run_q8_check():
    """Q8 vs Q5 sign/magnitude replication for Qwen3-8B."""
    print(f"\n{'='*70}")
    print(f"  Q8 QUANTISATION CHECK")
    print(f"{'='*70}")

    q5_path = os.path.join(RAW_DIR, "trials_qwen3-8b_non_thinking.jsonl")
    q8_path = os.path.join(RAW_DIR, "trials_qwen3-8b-q8_non_thinking.jsonl")

    if not os.path.exists(q8_path):
        print(f"  Q8 data not found: {q8_path}")
        return {}

    q5_all = [json.loads(l) for l in open(q5_path)]
    q8_all = [json.loads(l) for l in open(q8_path)]

    q5_valid = filter_valid(q5_all)
    q8_valid = filter_valid(q8_all)

    # Match to same items (Q8 only has 200)
    q8_indices = set(t["item_index"] for t in q8_valid)
    q5_matched = [t for t in q5_valid if t["item_index"] in q8_indices]

    nlp_q5 = np.array([t["answer_nlp"] for t in q5_matched])
    cor_q5 = np.array([1 if t["correct"] else 0 for t in q5_matched])
    nlp_q8 = np.array([t["answer_nlp"] for t in q8_valid])
    cor_q8 = np.array([1 if t["correct"] else 0 for t in q8_valid])

    auroc2_q5 = compute_auroc2(nlp_q5, cor_q5) if len(np.unique(cor_q5)) == 2 else np.nan
    auroc2_q8 = compute_auroc2(nlp_q8, cor_q8) if len(np.unique(cor_q8)) == 2 else np.nan

    acc_q5 = cor_q5.mean()
    acc_q8 = cor_q8.mean()

    # NLP correlation between Q5 and Q8 on matched items
    q5_by_idx = {t["item_index"]: t for t in q5_matched}
    q8_by_idx = {t["item_index"]: t for t in q8_valid}
    shared = sorted(set(q5_by_idx.keys()) & set(q8_by_idx.keys()))
    nlp_q5_paired = np.array([q5_by_idx[i]["answer_nlp"] for i in shared])
    nlp_q8_paired = np.array([q8_by_idx[i]["answer_nlp"] for i in shared])
    rho_nlp, p_nlp = spearmanr(nlp_q5_paired, nlp_q8_paired)

    print(f"  Q5 (matched 200): acc={acc_q5:.1%}, AUROC2={auroc2_q5:.4f}")
    print(f"  Q8 (200 items):   acc={acc_q8:.1%}, AUROC2={auroc2_q8:.4f}")
    print(f"  NLP correlation:  rho={rho_nlp:.4f}, p={p_nlp:.4e}")
    print(f"  AUROC2 difference: {auroc2_q8 - auroc2_q5:+.4f}")

    return {
        "q5_acc": float(acc_q5), "q8_acc": float(acc_q8),
        "q5_auroc2": auroc2_q5, "q8_auroc2": auroc2_q8,
        "nlp_rho": rho_nlp, "nlp_p": p_nlp,
        "n_matched": len(shared),
    }


# ---------------------------------------------------------------------------
# K-robustness (K=3 and K=6)
# ---------------------------------------------------------------------------

def run_k_robustness(model, valid_nt, valid_t):
    """Recompute SDT with K=3 and K=6 bins."""
    print(f"\n  K-robustness check")

    nlp_nt = np.array([t["answer_nlp"] for t in valid_nt])
    correct_nt = np.array([1 if t["correct"] else 0 for t in valid_nt])
    nlp_t = np.array([t["answer_nlp"] for t in valid_t])
    correct_t = np.array([1 if t["correct"] else 0 for t in valid_t])

    results = {}
    for k in [3, 6]:
        boundaries = compute_bin_boundaries(nlp_nt, k=k)
        sdt_nt = full_sdt_pipeline(nlp_nt, correct_nt, boundaries, k=k)
        sdt_t = full_sdt_pipeline(nlp_t, correct_t, boundaries, k=k)

        delta_mratio = sdt_t["m_ratio"] - sdt_nt["m_ratio"] if not (
            np.isnan(sdt_t["m_ratio"]) or np.isnan(sdt_nt["m_ratio"])) else np.nan

        results[f"K{k}"] = {
            "nt_auroc2": sdt_nt["auroc2"],
            "t_auroc2": sdt_t["auroc2"],
            "delta_auroc2": sdt_t["auroc2"] - sdt_nt["auroc2"],
            "nt_m_ratio": sdt_nt["m_ratio"],
            "t_m_ratio": sdt_t["m_ratio"],
            "delta_m_ratio": delta_mratio,
        }
        print(f"    K={k}: ΔAUROC2={results[f'K{k}']['delta_auroc2']:+.4f}, "
              f"ΔM-ratio={delta_mratio:+.4f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print(f"{'='*70}")
    print(f"  EXPLORATORY ANALYSES (E1–E7, Q8, K-robustness)")
    print(f"{'='*70}")

    all_results = {}

    for model in MODELS:
        print(f"\n{'='*70}")
        print(f"  MODEL: {model}")
        print(f"{'='*70}")

        trials_nt = filter_valid(load_trials(model, "non_thinking"))
        trials_t = filter_valid(load_trials(model, "thinking"))

        if not trials_nt or not trials_t:
            print(f"  SKIPPING: missing data")
            continue

        nlp_nt = np.array([t["answer_nlp"] for t in trials_nt])
        nlp_t = np.array([t["answer_nlp"] for t in trials_t])
        correct_nt = np.array([1 if t["correct"] else 0 for t in trials_nt])
        correct_t = np.array([1 if t["correct"] else 0 for t in trials_t])

        delta_auroc2 = compute_auroc2(nlp_t, correct_t) - compute_auroc2(nlp_nt, correct_nt)

        model_results = {
            "delta_auroc2": delta_auroc2,
            "E1": run_e1(model, trials_nt, trials_t),
            "E2": run_e2(model, trials_t),
            "E3": run_e3(model, trials_t),
            "E4": run_e4(model, trials_nt, trials_t),
            "E5": run_e5(model, trials_t),
            "E7": run_e7(model, trials_nt, trials_t),
            "K_robustness": run_k_robustness(model, trials_nt, trials_t),
        }

        all_results[model] = model_results

    # E6: cross-model
    e6 = run_e6(all_results)
    all_results["E6_cross_model"] = e6

    # Q8 check
    q8 = run_q8_check()
    all_results["Q8"] = q8

    # Save
    def sanitize_keys(obj):
        if isinstance(obj, dict):
            return {str(k): sanitize_keys(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_keys(v) for v in obj]
        return obj

    output_path = os.path.join(PROCESSED_DIR, "exploratory_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sanitize_keys(all_results), f, indent=2, cls=NumpyEncoder)

    print(f"\n  Results saved to {output_path}")
    print(f"\n{'='*70}")
    print(f"  EXPLORATORY ANALYSES COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
