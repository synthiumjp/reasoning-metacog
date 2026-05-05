# reasoning-metacog/scripts/run_inference.py
"""
Main inference script for reasoning-metacog study.
Runs thinking or non-thinking mode on 2,000 TriviaQA items.

Usage:
    python scripts/run_inference.py --model qwen3-8b --mode non_thinking
    python scripts/run_inference.py --model qwen3-8b --mode thinking
    python scripts/run_inference.py --model qwen3-8b --mode thinking --resume 500
    python scripts/run_inference.py --model qwen3-8b-q8 --mode non_thinking --n-items 200
    python scripts/run_inference.py --model all --mode non_thinking
    python scripts/run_inference.py --model all --mode thinking
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from local_config import MODEL_PATHS, TRIVIAQA_PATH, PROJECT_DIR
from scripts.reasoning_engine import ReasoningInferenceEngine, NumpyEncoder
from scripts.scoring import score_answer, is_malformed, extract_answer_from_response

MODELS = ["qwen3-8b", "r1-distill-qwen-7b", "r1-distill-llama-8b"]


def load_items(path: str, n_items: int = None) -> list[dict]:
    """Load items from JSONL."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line.strip()))
            if n_items and len(items) >= n_items:
                break
    print(f"Loaded {len(items)} items from {path}")
    return items


def run_model_mode(
    model_key: str,
    mode: str,
    items: list[dict],
    output_dir: str,
    resume_from: int = 0,
):
    """Run one model in one mode on all items."""
    if model_key not in MODEL_PATHS:
        print(f"ERROR: '{model_key}' not in local_config.py MODEL_PATHS")
        return

    model_path = MODEL_PATHS[model_key]
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"trials_{model_key}_{mode}.jsonl")

    print(f"\n{'='*60}")
    print(f"Model: {model_key}")
    print(f"Mode:  {mode}")
    print(f"Path:  {model_path}")
    print(f"Items: {len(items)}")
    print(f"Output: {output_path}")
    if resume_from > 0:
        print(f"Resuming from item {resume_from}")
    print(f"{'='*60}\n")

    engine = ReasoningInferenceEngine(model_key, model_path)

    file_mode = "a" if resume_from > 0 else "w"
    total = 0
    n_correct = 0
    n_malformed = 0
    n_empty_trace = 0
    n_truncated = 0
    t_start = time.time()

    with open(output_path, file_mode, encoding="utf-8") as f:
        for idx in range(resume_from, len(items)):
            item = items[idx]

            result = engine.generate(
                question=item["question"],
                mode=mode,
                max_thinking_tokens=4096,
                max_answer_tokens=256,
                temperature=0.6,
                top_k=20,
                top_p=0.95,
                seed=idx,
            )

            # Score answer
            answer = result["answer_text"]
            malformed = is_malformed(answer)
            correct = None
            if not malformed:
                correct = score_answer(answer, item["answers"])

            record = {
                "item_index": idx,
                "item_id": item.get("item_id", idx),
                "domain": item.get("domain", "Unknown"),
                "model": model_key,
                "mode": mode,
                "question": item["question"],
                "gold_answers": item["answers"][:5],  # First 5 aliases
                "answer_text": answer,
                "correct": correct,
                "malformed": malformed,
                "answer_nlp": result["answer_nlp"],
                "answer_length_tokens": result["answer_length_tokens"],
                "first_token_softmax": result["first_token_softmax"],
                "elapsed_s": result["elapsed_s"],
            }

            # Thinking-mode specific fields
            if mode == "thinking":
                record["trace_text"] = result["trace_text"]
                record["trace_length_tokens"] = result["trace_length_tokens"]
                record["is_empty_trace"] = result["is_empty_trace"]
                record["is_truncated"] = result.get("is_truncated", False)

                if result["is_empty_trace"]:
                    n_empty_trace += 1
                if result.get("is_truncated", False):
                    n_truncated += 1

            f.write(json.dumps(record, cls=NumpyEncoder) + "\n")

            total += 1
            if malformed:
                n_malformed += 1
            elif correct:
                n_correct += 1

            # Progress every 100 items
            if (idx + 1) % 100 == 0:
                elapsed = time.time() - t_start
                items_done = idx + 1 - resume_from
                rate = items_done / elapsed if elapsed > 0 else 0
                remaining = (len(items) - idx - 1) / rate if rate > 0 else 0
                n_valid = total - n_malformed
                acc = n_correct / n_valid if n_valid > 0 else 0

                status = (f"  [{idx+1}/{len(items)}] "
                         f"acc={acc:.1%} | "
                         f"malformed={n_malformed} | ")
                if mode == "thinking":
                    status += f"empty={n_empty_trace} trunc={n_truncated} | "
                status += f"{rate:.1f} items/s | ETA {remaining/60:.0f}min"
                print(status)
                f.flush()

    elapsed_total = time.time() - t_start
    n_valid = total - n_malformed
    acc = n_correct / n_valid if n_valid > 0 else 0

    print(f"\n{'='*60}")
    print(f"COMPLETE: {model_key} / {mode}")
    print(f"  Trials:    {total}")
    print(f"  Accuracy:  {acc:.1%} ({n_correct}/{n_valid} valid)")
    print(f"  Malformed: {n_malformed} ({n_malformed/total*100:.1f}%)")
    if mode == "thinking":
        print(f"  Empty traces:     {n_empty_trace} ({n_empty_trace/total*100:.1f}%)")
        print(f"  Truncated traces: {n_truncated} ({n_truncated/total*100:.1f}%)")
    print(f"  Time: {elapsed_total/60:.1f} min ({elapsed_total/total:.1f}s/item)")
    print(f"  Output: {output_path}")
    print(f"{'='*60}\n")

    engine.unload()


def main():
    parser = argparse.ArgumentParser(description="Reasoning-Metacog Inference")
    parser.add_argument("--model", choices=MODELS + ["qwen3-8b-q8", "all"], required=True)
    parser.add_argument("--mode", choices=["thinking", "non_thinking"], required=True)
    parser.add_argument("--items", default=TRIVIAQA_PATH)
    parser.add_argument("--output", default=os.path.join(PROJECT_DIR, "results", "raw"))
    parser.add_argument("--resume", type=int, default=0)
    parser.add_argument("--n-items", type=int, default=None, help="Override item count (for Q8 check)")
    args = parser.parse_args()

    items = load_items(args.items, args.n_items)

    if args.model == "all":
        for model_key in MODELS:
            run_model_mode(model_key, args.mode, items, args.output, args.resume)
    else:
        run_model_mode(args.model, args.mode, items, args.output, args.resume)


if __name__ == "__main__":
    main()
