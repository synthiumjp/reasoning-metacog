# reasoning-metacog/scripts/bootstrap.py
"""
Bootstrap inference per pre-registration.
10,000 resamples, seed 42, item-level, full pipeline recomputation.
"""

import numpy as np
from scripts.sdt_analysis import full_sdt_pipeline


def bootstrap_paired_delta(
    nlp_think: np.ndarray,
    correct_think: np.ndarray,
    nlp_nothink: np.ndarray,
    correct_nothink: np.ndarray,
    bin_boundaries: np.ndarray,
    n_resamples: int = 10000,
    seed: int = 42,
    mratio_threshold: float = 10.0,
) -> dict:
    """
    Bootstrap paired difference for all pre-registered metrics.
    Each resample recomputes the full pipeline.
    
    Returns CIs and point estimates for:
    - delta_auroc2 (H1a)
    - delta_m_ratio (H1b)
    - delta_ece (H3)
    """
    rng = np.random.default_rng(seed)
    n_items = len(nlp_think)
    
    deltas_auroc2 = []
    deltas_m_ratio = []
    deltas_ece = []
    n_mratio_excluded = 0
    
    for i in range(n_resamples):
        idx = rng.integers(0, n_items, size=n_items)
        
        # Resample both conditions with same indices (paired)
        nlp_t = nlp_think[idx]
        cor_t = correct_think[idx]
        nlp_nt = nlp_nothink[idx]
        cor_nt = correct_nothink[idx]
        
        # Full pipeline on each condition
        result_t = full_sdt_pipeline(nlp_t, cor_t, bin_boundaries)
        result_nt = full_sdt_pipeline(nlp_nt, cor_nt, bin_boundaries)
        
        # AUROC2 delta
        if not (np.isnan(result_t["auroc2"]) or np.isnan(result_nt["auroc2"])):
            deltas_auroc2.append(result_t["auroc2"] - result_nt["auroc2"])
        
        # M-ratio delta (with extreme exclusion)
        mr_t = result_t["m_ratio"]
        mr_nt = result_nt["m_ratio"]
        if (
            not (np.isnan(mr_t) or np.isnan(mr_nt))
            and abs(mr_t) <= mratio_threshold
            and abs(mr_nt) <= mratio_threshold
        ):
            deltas_m_ratio.append(mr_t - mr_nt)
        else:
            n_mratio_excluded += 1
        
        # ECE delta
        if not (np.isnan(result_t["ece"]) or np.isnan(result_nt["ece"])):
            deltas_ece.append(result_t["ece"] - result_nt["ece"])
    
    deltas_auroc2 = np.array(deltas_auroc2)
    deltas_m_ratio = np.array(deltas_m_ratio)
    deltas_ece = np.array(deltas_ece)
    
    def ci(arr):
        if len(arr) == 0:
            return {"point": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "n": 0}
        return {
            "point": np.median(arr),
            "mean": np.mean(arr),
            "ci_lower": np.percentile(arr, 2.5),
            "ci_upper": np.percentile(arr, 97.5),
            "excludes_zero": (np.percentile(arr, 2.5) > 0) or (np.percentile(arr, 97.5) < 0),
            "n": len(arr),
        }
    
    return {
        "delta_auroc2": ci(deltas_auroc2),
        "delta_m_ratio": ci(deltas_m_ratio),
        "delta_ece": ci(deltas_ece),
        "n_mratio_excluded": n_mratio_excluded,
        "mratio_exclusion_rate": n_mratio_excluded / n_resamples,
        "n_resamples": n_resamples,
    }
