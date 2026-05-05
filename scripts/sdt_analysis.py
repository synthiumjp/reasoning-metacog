# reasoning-metacog/scripts/sdt_analysis.py
"""
Type-2 SDT analysis pipeline per pre-registration.
Computes d', meta-d', M-ratio, AUROC2, ECE, and validity indices.
All parameters from study_config.py.
"""

import numpy as np
from scipy.stats import norm, spearmanr
from sklearn.metrics import roc_auc_score
from metadpy.mle import metad


def compute_nlp(token_logprobs: list[float]) -> float:
    """Normalised log-probability: mean of token log-probs."""
    if not token_logprobs or len(token_logprobs) == 0:
        return float("-inf")
    return np.mean(token_logprobs)


def compute_auroc2(nlp_values: np.ndarray, correctness: np.ndarray) -> float:
    """
    AUROC2: non-parametric, unbinned Type-2 AUROC.
    P(NLP_correct > NLP_incorrect).
    """
    if len(np.unique(correctness)) < 2:
        return np.nan
    return roc_auc_score(correctness, nlp_values)


def compute_bin_boundaries(nlp_values: np.ndarray, k: int = 4) -> np.ndarray:
    """
    Compute equal-quantile bin boundaries from reference distribution.
    Returns k-1 boundary values.
    """
    quantiles = np.linspace(0, 1, k + 1)[1:-1]
    boundaries = np.quantile(nlp_values, quantiles)
    return boundaries


def assign_bins(nlp_values: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """
    Assign NLP values to bins using fixed boundaries.
    Returns bin indices (1 to k).
    """
    return np.digitize(nlp_values, boundaries) + 1


def check_bin_occupancy(
    bins: np.ndarray,
    correctness: np.ndarray,
    k: int = 4,
    min_trials: int = 5,
    min_pct: float = 0.01,
) -> dict:
    """
    Check bin occupancy per pre-registration.
    Returns dict with pass/fail status and cell counts.
    """
    n_total = len(bins)
    cells = {}
    any_violation = False
    
    for b in range(1, k + 1):
        for c in [0, 1]:
            mask = (bins == b) & (correctness == c)
            count = mask.sum()
            cells[(b, c)] = count
            if count < min_trials or count < min_pct * n_total:
                any_violation = True
    
    n_occupied_bins = len(set(bins))
    bin_collapse = n_occupied_bins < 3
    
    return {
        "pass": not any_violation,
        "cells": cells,
        "any_violation": any_violation,
        "bin_collapse": bin_collapse,
        "n_occupied_bins": n_occupied_bins,
    }


def compute_type2_contingency(
    bins: np.ndarray, correctness: np.ndarray, k: int = 4, hautus: float = 0.5
) -> dict:
    """
    Build Type-2 contingency table with Hautus correction.
    Returns nR_S1 and nR_S2 arrays for metadpy.
    """
    nR_S1 = np.zeros(k)  # Incorrect trials: count per confidence bin
    nR_S2 = np.zeros(k)  # Correct trials: count per confidence bin
    
    for b in range(1, k + 1):
        nR_S1[b - 1] = ((bins == b) & (correctness == 0)).sum() + hautus
        nR_S2[b - 1] = ((bins == b) & (correctness == 1)).sum() + hautus
    
    return {"nR_S1": nR_S1, "nR_S2": nR_S2}


def compute_metad(nR_S1: np.ndarray, nR_S2: np.ndarray, n_ratings: int = 2) -> dict:
    """
    Compute meta-d' and d' via MLE using metadpy.
    nR_S1/nR_S2 should be length 2*nRatings arrays.
    For K=4 bins, use nRatings=2 (4 elements each).
    Returns dict with d_prime, meta_d_prime, m_ratio.
    """
    try:
        result = metad(
            nR_S1=nR_S1,
            nR_S2=nR_S2,
            nRatings=n_ratings,
            padding=False,
        )
        d_prime = result["dprime"].values[0]
        meta_d_prime = result["meta_d"].values[0]
        
        if d_prime is None or meta_d_prime is None:
            return {
                "d_prime": np.nan,
                "meta_d_prime": np.nan,
                "m_ratio": np.nan,
                "converged": False,
            }
        
        d_prime = float(d_prime)
        meta_d_prime = float(meta_d_prime)
        
        if abs(d_prime) < 1e-10:
            m_ratio = np.nan
        else:
            m_ratio = meta_d_prime / d_prime
        
        return {
            "d_prime": d_prime,
            "meta_d_prime": meta_d_prime,
            "m_ratio": m_ratio,
            "converged": True,
        }
    except Exception as e:
        return {
            "d_prime": np.nan,
            "meta_d_prime": np.nan,
            "m_ratio": np.nan,
            "converged": False,
            "error": str(e),
        }


def compute_d_prime_accuracy(accuracy: float) -> float:
    """
    D2 stress test: d'_acc = probit(accuracy).
    Clipped to [0.001, 0.999].
    """
    acc_clipped = np.clip(accuracy, 0.001, 0.999)
    return norm.ppf(acc_clipped) * np.sqrt(2)  # 2AFC equivalent


def compute_ece(
    nlp_values: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error with min-max rescaled NLP.
    10 equal-width bins per pre-registration.
    """
    nlp_min, nlp_max = nlp_values.min(), nlp_values.max()
    if nlp_max - nlp_min < 1e-10:
        return np.nan
    
    confidence = (nlp_values - nlp_min) / (nlp_max - nlp_min)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    
    ece = 0.0
    n_total = len(correctness)
    
    for i in range(n_bins):
        mask = (confidence >= bin_edges[i]) & (confidence < bin_edges[i + 1])
        if i == n_bins - 1:  # Include right edge in last bin
            mask = (confidence >= bin_edges[i]) & (confidence <= bin_edges[i + 1])
        
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        
        acc_bin = correctness[mask].mean()
        conf_bin = confidence[mask].mean()
        ece += (n_bin / n_total) * abs(acc_bin - conf_bin)
    
    return ece


def compute_validity_indices(
    nlp_values: np.ndarray, correctness: np.ndarray
) -> dict:
    """
    Validity screening protocol (D4) per arXiv:2604.17714.
    Median-split on NLP for KEEP/WITHDRAW.
    """
    median_nlp = np.median(nlp_values)
    keep = nlp_values >= median_nlp  # High confidence
    withdraw = ~keep  # Low confidence
    
    correct = correctness.astype(bool)
    incorrect = ~correct
    
    a = (keep & correct).sum()      # KEEP + correct
    b = (keep & incorrect).sum()    # KEEP + incorrect
    c = (withdraw & correct).sum()  # WITHDRAW + correct
    d = (withdraw & incorrect).sum() # WITHDRAW + incorrect
    
    n = len(correctness)
    
    L = b / (b + d) if (b + d) > 0 else 0.0
    Fp = c / (a + c) if (a + c) > 0 else 0.0
    RBS = Fp - (1 - L)
    TRIN = max(keep.sum(), withdraw.sum()) / n
    
    r_conf_correct, p_val = spearmanr(nlp_values, correctness)
    
    # Classification
    invalid = (Fp >= 0.50) or (L >= 0.95) or (RBS > 0)  # Simplified; full version uses CI on RBS
    
    return {
        "L": L,
        "Fp": Fp,
        "RBS": RBS,
        "TRIN": TRIN,
        "r_confidence_correct": r_conf_correct,
        "r_p_value": p_val,
        "classification": "Invalid" if invalid else "Valid",
        "a": a, "b": b, "c": c, "d": d,
    }


def check_nlp_monotonicity(
    nlp_values: np.ndarray, correctness: np.ndarray, n_quartiles: int = 4
) -> dict:
    """
    D3: NLP monotonicity check.
    Accuracy must increase across NLP quartiles.
    Spearman rho > 0.10.
    """
    quartile_boundaries = np.quantile(nlp_values, np.linspace(0, 1, n_quartiles + 1))
    quartile_accs = []
    
    for i in range(n_quartiles):
        if i < n_quartiles - 1:
            mask = (nlp_values >= quartile_boundaries[i]) & (nlp_values < quartile_boundaries[i + 1])
        else:
            mask = (nlp_values >= quartile_boundaries[i]) & (nlp_values <= quartile_boundaries[i + 1])
        
        if mask.sum() == 0:
            quartile_accs.append(np.nan)
        else:
            quartile_accs.append(correctness[mask].mean())
    
    # Check monotonicity
    monotonic = all(
        quartile_accs[i] <= quartile_accs[i + 1]
        for i in range(len(quartile_accs) - 1)
        if not (np.isnan(quartile_accs[i]) or np.isnan(quartile_accs[i + 1]))
    )
    
    rho, p = spearmanr(nlp_values, correctness)
    
    passes = monotonic and (rho > 0.10)
    
    return {
        "monotonic": monotonic,
        "spearman_rho": rho,
        "spearman_p": p,
        "quartile_accuracies": quartile_accs,
        "passes": passes,
    }


def run_h2_simulation(
    d_prime_nonthinking: float,
    type2_criteria_z: np.ndarray,
    d_prime_thinking: float,
    n_correct_thinking: int,
    n_incorrect_thinking: int,
    n_simulations: int = 10000,
    sigma_noise: float = 1.0,
    seed: int = 42,
) -> dict:
    """
    H2: Simulation-based constancy test.
    Simulate meta-d' under preserved confidence mapping.
    """
    rng = np.random.default_rng(seed)
    simulated_metad = []
    
    for _ in range(n_simulations):
        # Draw evidence from signal and noise distributions
        signal = rng.normal(d_prime_thinking / 2, 1.0, size=n_correct_thinking)
        noise = rng.normal(-d_prime_thinking / 2, sigma_noise, size=n_incorrect_thinking)
        
        # Assign confidence bins using non-thinking criteria
        all_evidence = np.concatenate([noise, signal])
        all_correct = np.concatenate([
            np.zeros(n_incorrect_thinking),
            np.ones(n_correct_thinking)
        ])
        
        bins = np.digitize(all_evidence, type2_criteria_z) + 1
        
        # Build contingency table
        k = len(type2_criteria_z) + 1
        nR_S1 = np.array([(bins[all_correct == 0] == b).sum() + 0.5 for b in range(1, k + 1)])
        nR_S2 = np.array([(bins[all_correct == 1] == b).sum() + 0.5 for b in range(1, k + 1)])
        
        try:
            result = metad(nR_S1=nR_S1.tolist(), nR_S2=nR_S2.tolist())
            md = result["meta_d1"]
            if abs(md) < 100:  # Exclude extreme values
                simulated_metad.append(md)
        except Exception:
            continue
    
    simulated_metad = np.array(simulated_metad)
    
    return {
        "simulated_metad": simulated_metad,
        "ci_lower": np.percentile(simulated_metad, 2.5),
        "ci_upper": np.percentile(simulated_metad, 97.5),
        "mean": np.mean(simulated_metad),
        "n_valid": len(simulated_metad),
    }


def full_sdt_pipeline(
    nlp_values: np.ndarray,
    correctness: np.ndarray,
    bin_boundaries: np.ndarray,
    k: int = 4,
) -> dict:
    """
    Run the complete SDT pipeline on one condition.
    Returns all pre-registered aggregate metrics.
    """
    accuracy = correctness.mean()
    auroc2 = compute_auroc2(nlp_values, correctness)
    
    # Bin assignment with fixed boundaries
    bins = assign_bins(nlp_values, bin_boundaries)
    
    # Bin occupancy check
    occupancy = check_bin_occupancy(bins, correctness, k=k)
    
    # Type-2 contingency table
    contingency = compute_type2_contingency(bins, correctness, k=k)
    
    # meta-d' and d'
    sdt_result = compute_metad(contingency["nR_S1"], contingency["nR_S2"])
    
    # d'_acc (D2)
    d_prime_acc = compute_d_prime_accuracy(accuracy)
    
    # M-ratio with d'_acc (D2)
    if abs(d_prime_acc) < 1e-10 or np.isnan(sdt_result["meta_d_prime"]):
        m_ratio_acc = np.nan
    else:
        m_ratio_acc = sdt_result["meta_d_prime"] / d_prime_acc
    
    # ECE
    ece = compute_ece(nlp_values, correctness)
    
    # NLP monotonicity (D3)
    monotonicity = check_nlp_monotonicity(nlp_values, correctness)
    
    # Validity indices (D4)
    validity = compute_validity_indices(nlp_values, correctness)
    
    return {
        "accuracy": accuracy,
        "n_trials": len(correctness),
        "n_correct": int(correctness.sum()),
        "n_incorrect": int((~correctness.astype(bool)).sum()),
        "auroc2": auroc2,
        "d_prime_nlp": sdt_result["d_prime"],
        "meta_d_prime": sdt_result["meta_d_prime"],
        "m_ratio": sdt_result["m_ratio"],
        "d_prime_acc": d_prime_acc,
        "m_ratio_acc": m_ratio_acc,
        "ece": ece,
        "sdt_converged": sdt_result["converged"],
        "bin_occupancy": occupancy,
        "monotonicity": monotonicity,
        "validity": validity,
        "contingency_nR_S1": contingency["nR_S1"].tolist(),
        "contingency_nR_S2": contingency["nR_S2"].tolist(),
    }
