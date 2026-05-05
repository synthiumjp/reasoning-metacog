# reasoning-metacog/scripts/run_abridged_nlp.py
"""
D1: Abridged-NLP diagnostic.
Teacher-forces thinking-mode answers in fresh context WITHOUT thinking trace.

For each thinking-mode trial, recomputes NLP of the answer text
in a context containing only the original prompt (no <think> trace).

Usage:
    python scripts/run_abridged_nlp.py --model qwen3-8b
    python scripts/run_abridged_nlp.py --model all
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from local_config import MODEL_PATHS, PROJECT_DIR
from scripts.reasoning_engine import ReasoningInferenceEngine, NumpyEncoder

MODELS = ["qwen3-8b", "r1-distill-qwen-7b", "r1-distill-llama-8b"]


def run_abridged_nlp(model_key: str, raw_dir: str, output_dir: str):
    """
    Load thinking-mode trials, teacher-force each answer in fresh context,
    compute abridged NLP.
    """
    if model_key not in MODEL_PATHS:
        print(f"ERROR: '{model_key}' not in local_config.py MODEL_PATHS")
        return

    thinking_path = os.path.join(raw_dir, f"trials_{model_key}_thinking.jsonl")
    if not os.path.exists(thinking_path):
        print(f"ERROR: Thinking-mode data not found: {thinking_path}")
        print(f"  Run thinking mode inference first.")
        return

    model_path = MODEL_PATHS[model_key]
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"abridged_nlp_{model_key}.jsonl")

    # Load thinking-mode trials
    trials = []
    with open(thinking_path, "r", encoding="utf-8") as f:
        for line in f:
            trials.append(json.loads(line.strip()))
    print(f"Loaded {len(trials)} thinking-mode trials for {model_key}")

    # Filter out malformed answers (no point teacher-forcing empty answers)
    valid_trials = [t for t in trials if not t.get("malformed", False) and t["answer_text"].strip()]
    print(f"  {len(valid_trials)} valid trials (non-malformed, non-empty answer)")

    print(f"\n{'='*60}")
    print(f"D1 ABRIDGED-NLP: {model_key}")
    print(f"Model: {model_path}")
    print(f"Trials: {len(valid_trials)}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")

    engine = ReasoningInferenceEngine(model_key, model_path, n_ctx=2048)

    n_done = 0
    n_failed = 0
    t_start = time.time()

    with open(output_path, "w", encoding="utf-8") as f:
        for trial in valid_trials:
            try:
                abridged_nlp = engine.force_decode_nlp(
                    question=trial["question"],
                    answer_text=trial["answer_text"],
                    mode="non_thinking",  # Fresh context without trace
                )
            except Exception as e:
                print(f"  WARNING: force_decode failed for item {trial['item_index']}: {e}")
                abridged_nlp = float("-inf")
                n_failed += 1

            record = {
                "item_index": trial["item_index"],
                "item_id": trial["item_id"],
                "model": model_key,
                "answer_text": trial["answer_text"],
                "thinking_nlp": trial["answer_nlp"],   # Original NLP (with trace in context)
                "abridged_nlp": abridged_nlp,           # D1: NLP without trace
                "correct": trial["correct"],
                "domain": trial.get("domain", "Unknown"),
            }
            f.write(json.dumps(record, cls=NumpyEncoder) + "\n")

            n_done += 1
            if n_done % 200 == 0:
                elapsed = time.time() - t_start
                rate = n_done / elapsed if elapsed > 0 else 0
                remaining = (len(valid_trials) - n_done) / rate if rate > 0 else 0
                print(f"  [{n_done}/{len(valid_trials)}] "
                      f"failed={n_failed} | "
                      f"{rate:.1f} items/s | ETA {remaining/60:.0f}min")
                f.flush()

    elapsed_total = time.time() - t_start

    print(f"\n{'='*60}")
    print(f"D1 COMPLETE: {model_key}")
    print(f"  Processed: {n_done}")
    print(f"  Failed:    {n_failed}")
    print(f"  Time:      {elapsed_total/60:.1f} min ({elapsed_total/n_done:.2f}s/item)")
    print(f"  Output:    {output_path}")
    print(f"{'='*60}\n")

    engine.unload()


def main():
    parser = argparse.ArgumentParser(description="D1: Abridged-NLP Diagnostic")
    parser.add_argument("--model", choices=MODELS + ["all"], required=True)
    parser.add_argument("--raw-dir", default=os.path.join(PROJECT_DIR, "results", "raw"))
    parser.add_argument("--output", default=os.path.join(PROJECT_DIR, "results", "raw"))
    args = parser.parse_args()

    models = MODELS if args.model == "all" else [args.model]
    for model_key in models:
        run_abridged_nlp(model_key, args.raw_dir, args.output)


if __name__ == "__main__":
    main()
