# reasoning-metacog/scripts/download_triviaqa.py
"""
Download TriviaQA and sample 2,000 items with domain labels.
Deterministic selection: numpy.random.default_rng(seed=42).

Usage:
    python scripts/download_triviaqa.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from local_config import PROJECT_DIR

OUTPUT_PATH = os.path.join(PROJECT_DIR, "data", "triviaqa_sampled_2000.jsonl")
N_ITEMS = 2000
SEED = 42

# Domain taxonomy from M1 (arXiv:2603.25112)
DOMAIN_KEYWORDS = {
    "History & Politics": ["history", "politic", "war", "royal", "president", "king", "queen",
                           "empire", "revolution", "civil", "colonial", "dynasty", "treaty"],
    "Arts & Literature": ["literature", "art", "music", "film", "movie", "novel", "author",
                          "paint", "song", "album", "poet", "theatre", "theater", "actor",
                          "direct", "book", "compos", "symphon"],
    "Geography": ["geograph", "country", "capital", "river", "mountain", "ocean", "island",
                  "continent", "city", "border", "lake", "desert", "peninsula"],
    "Science & Technology": ["science", "tech", "math", "physic", "chemi", "biolog", "medic",
                             "element", "planet", "space", "comput", "invent", "discover",
                             "atom", "cell", "species", "disease", "gene"],
    "Sports & Games": ["sport", "game", "olymp", "football", "soccer", "cricket", "tennis",
                       "basebal", "basket", "golf", "rugby", "champion", "world cup",
                       "medal", "athlet", "race", "swim"],
}


def assign_domain(question: str, answer_aliases: list[str]) -> str:
    """Assign domain from question text and answer aliases."""
    text = (question + " " + " ".join(answer_aliases)).lower()

    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        scores[domain] = score

    best_domain = max(scores, key=scores.get)
    if scores[best_domain] == 0:
        return "History & Politics"  # Default = majority category
    return best_domain


def main():
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: datasets library not installed.")
        print("Run: pip install datasets")
        return

    print("Downloading TriviaQA rc.nocontext validation split...")
    dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    print(f"  Total items in validation: {len(dataset)}")

    # Deterministic sampling
    rng = np.random.default_rng(SEED)
    indices = rng.choice(len(dataset), size=min(N_ITEMS, len(dataset)), replace=False)
    indices.sort()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    items = []
    domain_counts = {}

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for i, idx in enumerate(indices):
            item = dataset[int(idx)]

            # Extract answers
            answer_obj = item.get("answer", {})
            aliases = answer_obj.get("aliases", [])
            normalised = answer_obj.get("normalized_aliases", [])
            value = answer_obj.get("value", "")
            all_answers = list(set(aliases + normalised + ([value] if value else [])))

            question = item.get("question", "")
            domain = assign_domain(question, all_answers)

            domain_counts[domain] = domain_counts.get(domain, 0) + 1

            record = {
                "item_id": int(idx),
                "item_index": i,
                "question": question,
                "answers": all_answers,
                "domain": domain,
            }
            f.write(json.dumps(record) + "\n")
            items.append(record)

    print(f"\n  Sampled {len(items)} items -> {OUTPUT_PATH}")
    print(f"  Domain distribution:")
    for domain, count in sorted(domain_counts.items()):
        print(f"    {domain}: {count} ({count/len(items)*100:.1f}%)")
    print(f"\n  Seed: {SEED}")
    print(f"  Ready for pilot and inference.")


if __name__ == "__main__":
    main()
