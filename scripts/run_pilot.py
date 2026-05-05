# reasoning-metacog/scripts/run_pilot.py
"""
Pilot check: 50 items per model in thinking mode.
Verifies trace generation, coherence, and trace length.
Log results on OSF before proceeding to full data collection.

Usage:
    python scripts/run_pilot.py --model qwen3-8b
    python scripts/run_pilot.py --model r1-distill-qwen-7b
    python scripts/run_pilot.py --model r1-distill-llama-8b
    python scripts/run_pilot.py --model all
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from local_config import MODEL_PATHS, TRIVIAQA_PATH, PROJECT_DIR
from scripts.reasoning_engine import ReasoningInferenceEngine, check_pilot_coherence, NumpyEncoder
from scripts.scoring import extract_answer_from_response, is_malformed

PILOT_N = 50
MODELS = ["qwen3-8b", "r1-distill-qwen-7b", "r1-distill-llama-8b"]


def load_pilot_items(triviaqa_path: str, n: int = 50) -> list[dict]:
    """Load first N items from the sampled TriviaQA file."""
    items = []
    with open(triviaqa_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            items.append(json.loads(line.strip()))
    print(f"Loaded {len(items)} pilot items from {triviaqa_path}")
    return items


def run_pilot(model_key: str, items: list[dict]):
    """Run pilot for one model. Report pass/fail on all three checks."""
    if model_key not in MODEL_PATHS:
        print(f"ERROR: '{model_key}' not in local_config.py MODEL_PATHS")
        return

    model_path = MODEL_PATHS[model_key]
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        return

    output_dir = os.path.join(PROJECT_DIR, "pilot")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"pilot_{model_key}.jsonl")

    print(f"\n{'='*60}")
    print(f"PILOT: {model_key}")
    print(f"Path:  {model_path}")
    print(f"Items: {len(items)}")
    print(f"{'='*60}\n")

    engine = ReasoningInferenceEngine(model_key, model_path)

    records = []
    empty_traces = 0
    truncated_traces = 0
    coherent_count = 0
    t_start = time.time()

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, item in enumerate(items):
            result = engine.generate(
                question=item["question"],
                mode="thinking",
                max_thinking_tokens=4096,
                max_answer_tokens=256,
                temperature=0.6,
                top_k=20,
                top_p=0.95,
                seed=idx,
            )

            # Coherence check
            coherence = check_pilot_coherence(
                result["trace_text"], result["answer_text"]
            )

            record = {
                "item_index": idx,
                "item_id": item.get("item_id", idx),
                "question": item["question"][:100],
                "answer_text": result["answer_text"][:200],
                "trace_length_tokens": result["trace_length_tokens"],
                "is_empty_trace": result["is_empty_trace"],
                "is_truncated": result.get("is_truncated", False),
                "coherent": coherence["coherent"],
                "answer_in_trace": coherence["answer_in_trace"],
                "has_contradiction": coherence["has_contradiction"],
                "answer_nlp": result["answer_nlp"],
                "elapsed_s": result["elapsed_s"],
            }
            records.append(record)
            f.write(json.dumps(record, cls=NumpyEncoder) + "\n")

            if result["is_empty_trace"]:
                empty_traces += 1
            if result.get("is_truncated", False):
                truncated_traces += 1
            if coherence["coherent"]:
                coherent_count += 1

            if (idx + 1) % 10 == 0:
                print(f"  [{idx+1}/{len(items)}] trace_len={result['trace_length_tokens']}, "
                      f"coherent={coherence['coherent']}, empty={result['is_empty_trace']}")

    elapsed = time.time() - t_start
    n = len(records)

    # Compute pilot metrics
    trace_generated_rate = (n - empty_traces) / n
    coherence_rate = coherent_count / n
    truncation_rate = truncated_traces / n

    # Pre-registered thresholds
    check_a = trace_generated_rate >= 0.80
    check_b = coherence_rate >= 0.90
    check_c = truncation_rate < 0.20

    print(f"\n{'='*60}")
    print(f"PILOT RESULTS: {model_key}")
    print(f"{'='*60}")
    print(f"  (a) Trace generation rate: {trace_generated_rate:.1%} "
          f"{'PASS' if check_a else 'FAIL'} (threshold: >=80%)")
    print(f"  (b) Coherence rate:        {coherence_rate:.1%} "
          f"{'PASS' if check_b else 'FAIL'} (threshold: >=90%)")
    print(f"  (c) Truncation rate:       {truncation_rate:.1%} "
          f"{'PASS' if check_c else 'FAIL'} (threshold: <20%)")
    print(f"  Overall: {'ALL CHECKS PASS — proceed to data collection' if (check_a and check_b and check_c) else 'CHECKS FAILED — adjust parameters'}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/n:.1f}s/item)")
    print(f"  Output: {output_path}")

    # Trace length distribution
    trace_lengths = [r["trace_length_tokens"] for r in records if not r["is_empty_trace"]]
    if trace_lengths:
        import numpy as np
        print(f"\n  Trace length distribution (non-empty):")
        print(f"    N:      {len(trace_lengths)}")
        print(f"    Mean:   {np.mean(trace_lengths):.0f} tokens")
        print(f"    Median: {np.median(trace_lengths):.0f} tokens")
        print(f"    Min:    {np.min(trace_lengths)} tokens")
        print(f"    Max:    {np.max(trace_lengths)} tokens")
        print(f"    >1000:  {sum(1 for t in trace_lengths if t > 1000)}/{len(trace_lengths)}")

    print(f"\n  >>> Log these results on OSF before proceeding to full data collection <<<")
    print(f"{'='*60}\n")

    engine.unload()


def main():
    parser = argparse.ArgumentParser(description="Reasoning-Metacog Pilot (50 items, thinking mode)")
    parser.add_argument("--model", choices=MODELS + ["all"], required=True)
    parser.add_argument("--items", default=TRIVIAQA_PATH)
    args = parser.parse_args()

    items = load_pilot_items(args.items, PILOT_N)
    models = MODELS if args.model == "all" else [args.model]

    for model_key in models:
        run_pilot(model_key, items)


if __name__ == "__main__":
    main()
