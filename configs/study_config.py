# reasoning-metacog/configs/study_config.py
"""
Pre-registered study configuration.
All values locked at OSF registration. Changes require disclosure.
"""

# === MODELS ===
MODELS = {
    "qwen3-8b": {
        "repo": "Qwen/Qwen3-8B-GGUF",
        "filename": "qwen3-8b-q5_k_m.gguf",  # Adjust to actual filename
        "quant": "Q5_K_M",
        "type": "native",
        "role": "primary",
        "thinking_disable_method": "enable_thinking_false",
    },
    "r1-distill-qwen-7b": {
        "repo": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B-GGUF",
        "filename": "deepseek-r1-distill-qwen-7b-q5_k_m.gguf",
        "quant": "Q5_K_M",
        "type": "distilled",
        "role": "robustness",
        "thinking_disable_method": "prefix_suppression",
    },
    "r1-distill-llama-8b": {
        "repo": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B-GGUF",
        "filename": "deepseek-r1-distill-llama-8b-q5_k_m.gguf",
        "quant": "Q5_K_M",
        "type": "distilled",
        "role": "robustness",
        "thinking_disable_method": "prefix_suppression",
    },
}

# Q8 check model (pre-committed, not optional)
Q8_MODEL = {
    "repo": "Qwen/Qwen3-8B-GGUF",
    "filename": "qwen3-8b-q8_0.gguf",
    "quant": "Q8_0",
    "n_items": 200,
}

# === SAMPLING PARAMETERS (matched across modes) ===
SAMPLING = {
    "temperature": 0.6,
    "top_k": 20,
    "top_p": 0.95,
    "min_p": 0.0,
    "max_thinking_tokens": 4096,
    "max_answer_tokens": 256,
    "seed": 42,  # Generation seed where supported
}

# === DATA ===
DATA = {
    "dataset": "trivia_qa",
    "split": "rc.nocontext/validation",
    "n_items": 2000,
    "selection_seed": 42,
    "domains": [
        "History & Politics",
        "Arts & Literature",
        "Geography",
        "Science & Technology",
        "Sports & Games",
    ],
}

# === SCORING ===
SCORING = {
    "levenshtein_threshold": 0.85,
    "normalise_lowercase": True,
    "strip_articles": True,
    "strip_whitespace": True,
}

# === SDT PARAMETERS ===
SDT = {
    "n_bins": 4,  # K=4 equal-quantile
    "bin_method": "equal_quantile",
    "bin_reference_condition": "non_thinking",  # Fixed boundaries from non-thinking
    "evsdt_restarts": 50,
    "hautus_correction": 0.5,
    "metadpy_version": "0.1.2",
    "extreme_mratio_threshold": 10.0,  # |M-ratio| > 10 excluded
}

# === BOOTSTRAP ===
BOOTSTRAP = {
    "n_resamples": 10000,
    "seed": 42,
    "ci_level": 0.95,
    "alpha": 0.05,
}

# === H2 SIMULATION ===
H2_SIMULATION = {
    "n_simulations": 10000,
    "uvsdt_aic_threshold": 4.0,
    "uvsdt_sigma_bounds": (0.1, 10.0),
}

# === VALIDITY SCREENING (D4) ===
VALIDITY = {
    "invalid_fp_threshold": 0.50,
    "invalid_l_threshold": 0.95,
    "keep_withdraw_method": "median_split_nlp",
}

# === DIAGNOSTICS ===
DIAGNOSTICS = {
    "d3_min_spearman_rho": 0.10,
    "bin_occupancy_min_trials": 5,
    "bin_occupancy_min_pct": 0.01,
    "bin_collapse_min_occupied": 3,
    "min_accuracy_threshold": 0.20,
    "max_malformed_rate": 0.10,
    "max_empty_trace_rate": 0.20,
    "truncation_caveat_threshold": 0.30,
    "answer_length_diff_threshold": 0.50,
}

# === PILOT ===
PILOT = {
    "n_items": 50,
    "min_trace_generation_rate": 0.80,
    "min_coherence_rate": 0.90,
    "max_truncation_rate": 0.20,
    "coherence_last_n_tokens": 50,
    "contradiction_markers": [
        "actually", "on second thought", "that is incorrect", "I was wrong"
    ],
    "max_adjustment_attempts": 2,
    "max_coherence_failure_rate": 0.10,
}

# === Q8 REPLICATION ===
Q8_REPLICATION = {
    "magnitude_tolerance": 0.50,  # ±50% of Q5 point estimate
}

# === ECE ===
ECE = {
    "n_bins": 10,
    "bin_method": "equal_width",
    "confidence_rescale": "minmax",  # min-max normalisation within condition
}

# === EXPLORATORY ===
EXPLORATORY = {
    "e3_directional_prediction_model": "r1-distill-qwen-7b",
    "e3_predicted_direction": "negative",  # rho < 0
    "e5_backtracking_markers": ["wait", "let me reconsider", "actually", "no,"],
    "e5_verification_markers": ["let me check", "to verify", "let me confirm"],
    "e5_alternative_markers": ["another approach", "alternatively", "or maybe"],
    "h4_permutation_shuffles": 1000,
}

# === PATHS ===
PATHS = {
    "raw_outputs": "results/raw/",
    "processed": "results/processed/",
    "figures": "results/figures/",
    "pilot": "pilot/",
    "models_dir": "models/",  # Local GGUF storage
}
