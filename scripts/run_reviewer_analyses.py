# reasoning-metacog/scripts/run_reviewer_analyses.py
"""
Analyses requested by adversarial reviewers:

1. First-token softmax AUROC₂ — a second confidence channel already in the data
2. Exact-match-only AUROC₂ — stability check without containment fallback
3. Within-trial vs between-trial compression decomposition
4. D1 competing mechanisms decomposition

Usage:
    python scripts/run_reviewer_analyses.py
"""

import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from local_config import PROJECT_DIR
from scripts.sdt_analysis import compute_auroc2
from scripts.scoring import score_answer, normalise_answer

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


def filter_valid(trials, require_first_token=False):
    """Filter to valid trials. If require_first_token, also require non-zero first_token_softmax."""
    valid = []
    for t in trials:
        if t.get("malformed", False) or t.get("correct") is None:
            continue
        nlp = t.get("answer_nlp")
        if nlp is None or nlp == float("-inf") or np.isinf(nlp):
            continue
        if require_first_token:
            fts = t.get("first_token_softmax", 0)
            if fts is None or fts <= 0:
                continue
        valid.append(t)
    return valid


# ---------------------------------------------------------------------------
# 1. First-token softmax AUROC₂
# ---------------------------------------------------------------------------

def first_token_auroc2(model):
    """Compute AUROC₂ using first-token softmax as the confidence signal."""
    print(f"\n  1. FIRST-TOKEN SOFTMAX AUROC₂")

    results = {}
    for mode in ["non_thinking", "thinking"]:
        trials = filter_valid(load_trials(model, mode), require_first_token=True)
        if len(trials) < 100:
            print(f"    {mode}: insufficient trials with first_token_softmax ({len(trials)})")
            results[mode] = {"auroc2": np.nan, "n": len(trials)}
            continue

        fts = np.array([t["first_token_softmax"] for t in trials])
        nlp = np.array([t["answer_nlp"] for t in trials])
        correct = np.array([1 if t["correct"] else 0 for t in trials])

        if len(np.unique(correct)) < 2:
            results[mode] = {"auroc2": np.nan, "n": len(trials)}
            continue

        auroc2_fts = compute_auroc2(fts, correct)
        auroc2_nlp = compute_auroc2(nlp, correct)

        # Correlation between the two signals
        rho, p = spearmanr(fts, nlp)

        # Variance
        var_fts = float(np.var(fts))
        mean_fts = float(np.mean(fts))

        # Cohen's d for first-token softmax
        fts_correct = fts[correct == 1]
        fts_incorrect = fts[correct == 0]
        if len(fts_correct) > 5 and len(fts_incorrect) > 5:
            pooled_sd = np.sqrt((np.var(fts_correct) + np.var(fts_incorrect)) / 2)
            cohen_d = (np.mean(fts_correct) - np.mean(fts_incorrect)) / pooled_sd if pooled_sd > 0 else np.nan
        else:
            cohen_d = np.nan

        results[mode] = {
            "auroc2_fts": auroc2_fts,
            "auroc2_nlp": auroc2_nlp,
            "fts_nlp_rho": rho,
            "mean_fts": mean_fts,
            "var_fts": var_fts,
            "cohen_d_fts": float(cohen_d),
            "n": len(trials),
        }

        print(f"    {mode:15s}: AUROC₂(FTS)={auroc2_fts:.4f}  AUROC₂(NLP)={auroc2_nlp:.4f}  "
              f"rho(FTS,NLP)={rho:.3f}  d(FTS)={cohen_d:.3f}  "
              f"mean={mean_fts:.4f}  var={var_fts:.6f}  n={len(trials)}")

    # Delta
    if "non_thinking" in results and "thinking" in results:
        fts_nt = results["non_thinking"].get("auroc2_fts", np.nan)
        fts_t = results["thinking"].get("auroc2_fts", np.nan)
        nlp_nt = results["non_thinking"].get("auroc2_nlp", np.nan)
        nlp_t = results["thinking"].get("auroc2_nlp", np.nan)

        if not (np.isnan(fts_nt) or np.isnan(fts_t)):
            print(f"    ΔAUROC₂(FTS): {fts_t - fts_nt:+.4f}")
            print(f"    ΔAUROC₂(NLP): {nlp_t - nlp_nt:+.4f}")
            results["delta_fts"] = fts_t - fts_nt
            results["delta_nlp"] = nlp_t - nlp_nt

            # Key question: do both channels degrade?
            both_degrade = (fts_t - fts_nt < 0) and (nlp_t - nlp_nt < 0)
            print(f"    Both channels degrade: {both_degrade}")
            results["both_degrade"] = both_degrade

    return results


# ---------------------------------------------------------------------------
# 2. Exact-match-only AUROC₂ (no containment fallback)
# ---------------------------------------------------------------------------

def exact_match_only_auroc2(model):
    """Recompute AUROC₂ using strict exact-match + Levenshtein only (no containment)."""
    print(f"\n  2. EXACT-MATCH-ONLY AUROC₂ (no containment)")

    from Levenshtein import ratio as levenshtein_ratio

    results = {}
    for mode in ["non_thinking", "thinking"]:
        trials = filter_valid(load_trials(model, mode))
        if not trials:
            continue

        # Re-score without containment
        strict_correct = []
        containment_rescued = 0
        for t in trials:
            answer = t.get("answer_text", "")
            gold = t.get("gold_answers", [])
            pred_norm = normalise_answer(answer)

            # Strict scoring: exact match + Levenshtein only
            is_strict_correct = False
            if pred_norm:
                for g in gold:
                    g_norm = normalise_answer(g)
                    if not g_norm:
                        continue
                    if pred_norm == g_norm:
                        is_strict_correct = True
                        break
                    if levenshtein_ratio(pred_norm, g_norm) >= 0.85:
                        is_strict_correct = True
                        break

            strict_correct.append(is_strict_correct)

            # Track how many were rescued by containment
            if t["correct"] and not is_strict_correct:
                containment_rescued += 1

        nlp = np.array([t["answer_nlp"] for t in trials])
        correct_orig = np.array([1 if t["correct"] else 0 for t in trials])
        correct_strict = np.array([1 if c else 0 for c in strict_correct])

        auroc2_orig = compute_auroc2(nlp, correct_orig) if len(np.unique(correct_orig)) == 2 else np.nan
        auroc2_strict = compute_auroc2(nlp, correct_strict) if len(np.unique(correct_strict)) == 2 else np.nan

        acc_orig = correct_orig.mean()
        acc_strict = correct_strict.mean()

        results[mode] = {
            "auroc2_containment": auroc2_orig,
            "auroc2_strict": auroc2_strict,
            "acc_containment": float(acc_orig),
            "acc_strict": float(acc_strict),
            "n_rescued_by_containment": containment_rescued,
            "n": len(trials),
        }

        print(f"    {mode:15s}: AUROC₂(strict)={auroc2_strict:.4f}  "
              f"AUROC₂(containment)={auroc2_orig:.4f}  "
              f"acc strict={acc_strict:.1%} vs {acc_orig:.1%}  "
              f"rescued={containment_rescued}")

    # Delta comparison
    if "non_thinking" in results and "thinking" in results:
        delta_strict = (results["thinking"]["auroc2_strict"] -
                       results["non_thinking"]["auroc2_strict"])
        delta_contain = (results["thinking"]["auroc2_containment"] -
                        results["non_thinking"]["auroc2_containment"])

        print(f"    ΔAUROC₂(strict):      {delta_strict:+.4f}")
        print(f"    ΔAUROC₂(containment): {delta_contain:+.4f}")
        print(f"    Finding stable: {(delta_strict < 0) == (delta_contain < 0)}")

        results["delta_strict"] = delta_strict
        results["delta_containment"] = delta_contain
        results["finding_stable"] = (delta_strict < 0) == (delta_contain < 0)

    return results


# ---------------------------------------------------------------------------
# 3. Within-trial vs between-trial compression
# ---------------------------------------------------------------------------

def compression_decomposition(model):
    """
    Decompose NLP compression into:
    - Within-trial smoothing: individual token logprobs are higher (less variable)
    - Between-trial homogenisation: trial-level NLP values cluster together

    We measure:
    - Between-trial: variance of trial-level NLP (already computed)
    - Answer-length effect: does shorter answer → higher NLP mechanically?
    - NLP by answer length bins: is compression uniform or length-dependent?
    """
    print(f"\n  3. COMPRESSION DECOMPOSITION")

    results = {}
    for mode in ["non_thinking", "thinking"]:
        trials = filter_valid(load_trials(model, mode))
        if not trials:
            continue

        nlp = np.array([t["answer_nlp"] for t in trials])
        correct = np.array([1 if t["correct"] else 0 for t in trials])
        lengths = np.array([t.get("answer_length_tokens", 1) for t in trials])

        # Between-trial variance (the main compression metric)
        between_var = float(np.var(nlp))

        # NLP by answer-length quartiles
        len_quartiles = np.percentile(lengths, [25, 50, 75])
        len_bins = np.digitize(lengths, len_quartiles)

        nlp_by_length = {}
        for b in range(4):
            mask = len_bins == b
            if mask.sum() > 10:
                nlp_by_length[f"q{b+1}"] = {
                    "mean_nlp": float(np.mean(nlp[mask])),
                    "var_nlp": float(np.var(nlp[mask])),
                    "mean_len": float(np.mean(lengths[mask])),
                    "n": int(mask.sum()),
                }

        # Partial correlation: NLP-correctness controlling for length
        # Residualise NLP on length, then correlate with correctness
        if len(nlp) > 50:
            from numpy.polynomial.polynomial import polyfit
            # Linear residualisation
            coeffs = polyfit(lengths, nlp, 1)
            nlp_predicted = coeffs[0] + coeffs[1] * lengths
            nlp_resid = nlp - nlp_predicted
            rho_raw, _ = spearmanr(nlp, correct)
            rho_resid, _ = spearmanr(nlp_resid, correct)
            auroc2_resid = compute_auroc2(nlp_resid, correct) if len(np.unique(correct)) == 2 else np.nan
        else:
            rho_raw = rho_resid = auroc2_resid = np.nan

        results[mode] = {
            "between_trial_var": between_var,
            "nlp_by_length": nlp_by_length,
            "rho_nlp_correct_raw": float(rho_raw),
            "rho_nlp_correct_length_controlled": float(rho_resid),
            "auroc2_length_controlled": auroc2_resid,
            "n": len(trials),
        }

        print(f"    {mode:15s}: var={between_var:.6f}  "
              f"rho(raw)={rho_raw:.3f}  rho(len-ctrl)={rho_resid:.3f}  "
              f"AUROC₂(len-ctrl)={auroc2_resid:.4f}")

    # Compare
    if "non_thinking" in results and "thinking" in results:
        var_ratio = results["thinking"]["between_trial_var"] / results["non_thinking"]["between_trial_var"]
        print(f"    Variance ratio (T/NT): {var_ratio:.3f}")

        auroc2_nt = results["non_thinking"]["auroc2_length_controlled"]
        auroc2_t = results["thinking"]["auroc2_length_controlled"]
        if not (np.isnan(auroc2_nt) or np.isnan(auroc2_t)):
            print(f"    ΔAUROC₂(length-controlled): {auroc2_t - auroc2_nt:+.4f}")
            print(f"    Length-controlling does {'NOT ' if (auroc2_t - auroc2_nt) < 0 else ''}eliminate the effect")

    return results


# ---------------------------------------------------------------------------
# 4. D1 competing mechanisms
# ---------------------------------------------------------------------------

def d1_competing_mechanisms(model):
    """
    Decompose the D1 result into competing mechanisms:
    - Answer-distribution effect: how much does changing the answer hurt?
    - Context-conditioning effect: how much does the trace help?

    Three AUROC₂ values tell the story:
    - NT answers in NT context (baseline)
    - T answers in NT context (abridged = answer-distribution effect isolated)
    - T answers in T context (thinking = answer-distribution + context recovery)
    """
    print(f"\n  4. D1 COMPETING MECHANISMS")

    trials_nt = filter_valid(load_trials(model, "non_thinking"))
    trials_t = filter_valid(load_trials(model, "thinking"))

    abridged_path = os.path.join(RAW_DIR, f"abridged_nlp_{model}.jsonl")
    if not os.path.exists(abridged_path):
        print(f"    No abridged data")
        return {}

    abridged = {}
    with open(abridged_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line.strip())
            abridged[r["item_index"]] = r

    # Non-thinking baseline
    nlp_nt = np.array([t["answer_nlp"] for t in trials_nt])
    correct_nt = np.array([1 if t["correct"] else 0 for t in trials_nt])
    auroc2_baseline = compute_auroc2(nlp_nt, correct_nt)

    # Matched thinking trials with abridged
    thinking_nlps = []
    abridged_nlps = []
    correct_t_matched = []
    for t in trials_t:
        idx = t["item_index"]
        if idx in abridged:
            ab_nlp = abridged[idx].get("abridged_nlp")
            if ab_nlp is not None and not np.isinf(ab_nlp):
                thinking_nlps.append(t["answer_nlp"])
                abridged_nlps.append(ab_nlp)
                correct_t_matched.append(1 if t["correct"] else 0)

    if len(correct_t_matched) < 100:
        print(f"    Insufficient matched trials")
        return {}

    thinking_nlp = np.array(thinking_nlps)
    abridged_nlp = np.array(abridged_nlps)
    correct_matched = np.array(correct_t_matched)

    auroc2_thinking = compute_auroc2(thinking_nlp, correct_matched)
    auroc2_abridged = compute_auroc2(abridged_nlp, correct_matched)

    # Decomposition
    answer_dist_effect = auroc2_abridged - auroc2_baseline  # Negative = answers are worse
    context_recovery = auroc2_thinking - auroc2_abridged    # Positive = trace helps
    total_effect = auroc2_thinking - auroc2_baseline         # Net

    print(f"    AUROC₂ baseline (NT answers, NT context):    {auroc2_baseline:.4f}")
    print(f"    AUROC₂ abridged (T answers, NT context):     {auroc2_abridged:.4f}")
    print(f"    AUROC₂ thinking (T answers, T context):      {auroc2_thinking:.4f}")
    print(f"")
    print(f"    Answer-distribution effect (abridged - baseline): {answer_dist_effect:+.4f}")
    print(f"    Context recovery (thinking - abridged):           {context_recovery:+.4f}")
    print(f"    Total effect (thinking - baseline):               {total_effect:+.4f}")
    print(f"")

    # Proportional decomposition
    if abs(total_effect) > 0.001:
        pct_answer = answer_dist_effect / total_effect * 100
        pct_context = context_recovery / total_effect * 100
        print(f"    Decomposition: {pct_answer:.0f}% answer-distribution, "
              f"{pct_context:.0f}% context recovery")
    else:
        pct_answer = pct_context = np.nan

    return {
        "auroc2_baseline": auroc2_baseline,
        "auroc2_abridged": auroc2_abridged,
        "auroc2_thinking": auroc2_thinking,
        "answer_distribution_effect": answer_dist_effect,
        "context_recovery": context_recovery,
        "total_effect": total_effect,
        "pct_answer_distribution": pct_answer,
        "pct_context_recovery": pct_context,
        "n_matched": len(correct_matched),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print(f"{'='*70}")
    print(f"  REVIEWER-REQUESTED ANALYSES")
    print(f"{'='*70}")

    all_results = {}

    for model in MODELS:
        print(f"\n{'='*70}")
        print(f"  MODEL: {model}")
        print(f"{'='*70}")

        model_results = {
            "first_token": first_token_auroc2(model),
            "exact_match_only": exact_match_only_auroc2(model),
            "compression_decomposition": compression_decomposition(model),
            "d1_mechanisms": d1_competing_mechanisms(model),
        }

        all_results[model] = model_results

    # Cross-model summary for first-token
    print(f"\n{'='*70}")
    print(f"  CROSS-MODEL SUMMARY: First-Token Softmax")
    print(f"{'='*70}")
    print(f"  {'Model':30s} {'ΔAUROC₂(FTS)':>14s} {'ΔAUROC₂(NLP)':>14s} {'Both degrade':>14s}")
    for model in MODELS:
        r = all_results[model]["first_token"]
        delta_fts = r.get("delta_fts", np.nan)
        delta_nlp = r.get("delta_nlp", np.nan)
        both = r.get("both_degrade", None)
        print(f"  {model:30s} {delta_fts:+14.4f} {delta_nlp:+14.4f} {'YES' if both else 'NO':>14s}")

    print(f"\n{'='*70}")
    print(f"  CROSS-MODEL SUMMARY: D1 Decomposition")
    print(f"{'='*70}")
    print(f"  {'Model':30s} {'Ans-dist':>10s} {'Context':>10s} {'Total':>10s} {'% Ans-dist':>12s}")
    for model in MODELS:
        r = all_results[model]["d1_mechanisms"]
        if r:
            print(f"  {model:30s} {r['answer_distribution_effect']:+10.4f} "
                  f"{r['context_recovery']:+10.4f} {r['total_effect']:+10.4f} "
                  f"{r.get('pct_answer_distribution', np.nan):10.0f}%")

    # Save
    def sanitize_keys(obj):
        if isinstance(obj, dict):
            return {str(k): sanitize_keys(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_keys(v) for v in obj]
        return obj

    output_path = os.path.join(PROCESSED_DIR, "reviewer_analyses.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sanitize_keys(all_results), f, indent=2, cls=NumpyEncoder)

    print(f"\n  Results saved to {output_path}")
    print(f"\n{'='*70}")
    print(f"  REVIEWER ANALYSES COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
