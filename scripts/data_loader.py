# reasoning-metacog/scripts/data_loader.py
"""
TriviaQA data loading per pre-registration.
2,000 items from rc.nocontext validation, seed 42, with domain labels.
"""

import json
import numpy as np
from pathlib import Path


# Domain taxonomy from M1 (arXiv:2603.25112)
# Maps TriviaQA EntityPages categories to study domains
DOMAIN_MAP = {
    # History & Politics
    "History": "History & Politics",
    "Politics": "History & Politics",
    "War": "History & Politics",
    "Royalty": "History & Politics",
    # Arts & Literature
    "Literature": "Arts & Literature",
    "Art": "Arts & Literature",
    "Music": "Arts & Literature",
    "Film": "Arts & Literature",
    "Television": "Arts & Literature",
    "Theatre": "Arts & Literature",
    # Geography
    "Geography": "Geography",
    "Countries": "Geography",
    "Cities": "Geography",
    # Science & Technology
    "Science": "Science & Technology",
    "Technology": "Science & Technology",
    "Mathematics": "Science & Technology",
    "Medicine": "Science & Technology",
    "Nature": "Science & Technology",
    # Sports & Games
    "Sport": "Sports & Games",
    "Sports": "Sports & Games",
    "Games": "Sports & Games",
    "Olympics": "Sports & Games",
}

DEFAULT_DOMAIN = "History & Politics"  # Majority category for unmatched


def load_triviaqa_items(
    data_path: str,
    n_items: int = 2000,
    seed: int = 42,
) -> list[dict]:
    """
    Load and sample TriviaQA items.
    
    Args:
        data_path: Path to TriviaQA rc.nocontext validation JSON
        n_items: Number of items to sample
        seed: Random seed for deterministic selection
    
    Returns:
        List of dicts with keys: question, answers, domain, item_id
    """
    with open(data_path, "r") as f:
        data = json.load(f)
    
    items = data["Data"] if "Data" in data else data
    
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(items), size=min(n_items, len(items)), replace=False)
    indices.sort()
    
    selected = []
    for idx in indices:
        item = items[idx]
        
        # Extract answer aliases
        answer_obj = item.get("Answer", {})
        aliases = answer_obj.get("Aliases", [])
        normalised = answer_obj.get("NormalizedAliases", [])
        value = answer_obj.get("Value", "")
        
        all_answers = list(set(aliases + normalised + ([value] if value else [])))
        
        # Assign domain
        domain = assign_domain(item)
        
        selected.append({
            "item_id": int(idx),
            "question": item.get("Question", ""),
            "answers": all_answers,
            "domain": domain,
        })
    
    return selected


def assign_domain(item: dict) -> str:
    """Assign domain from TriviaQA metadata using M1 taxonomy."""
    # Try EntityPages categories first
    entity_pages = item.get("EntityPages", [])
    for page in entity_pages:
        filename = page.get("Filename", "")
        for key, domain in DOMAIN_MAP.items():
            if key.lower() in filename.lower():
                return domain
    
    # Try SearchResults
    search_results = item.get("SearchResults", [])
    for result in search_results:
        filename = result.get("Filename", "")
        for key, domain in DOMAIN_MAP.items():
            if key.lower() in filename.lower():
                return domain
    
    return DEFAULT_DOMAIN


def format_prompt(question: str, system_prompt: str = None) -> list[dict]:
    """
    Format question as chat messages for llama-cpp-python.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    return messages


# Default system prompt (minimal, matches M1 style)
SYSTEM_PROMPT = "Answer the following question concisely. Give only the answer, no explanation."
