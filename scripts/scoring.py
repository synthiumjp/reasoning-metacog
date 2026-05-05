# reasoning-metacog/scripts/scoring.py
"""
TriviaQA answer scoring per pre-registration.
Exact match against aliases + normalised Levenshtein similarity >= 0.85,
with containment fallback for verbose model outputs.

The containment check is standard TriviaQA evaluation practice and handles
models (e.g., R1-Distill) that embed correct answers in longer responses.
Applied uniformly across all models and conditions.
"""

import re
import string
from Levenshtein import ratio as levenshtein_ratio


def normalise_answer(text: str) -> str:
    """Normalise answer: lowercase, strip articles, collapse whitespace."""
    text = text.lower().strip()
    # Strip leading articles
    text = re.sub(r"^(the|a|an)\s+", "", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def score_answer(predicted: str, gold_answers: list[str], threshold: float = 0.85) -> bool:
    """
    Score a predicted answer against TriviaQA gold answers.
    
    Returns True if any of the following hold for any gold alias:
    1. Exact match (after normalisation), OR
    2. Normalised Levenshtein similarity >= threshold, OR
    3. Containment: a normalised gold alias (>= 4 chars) appears as a
       substring within the normalised predicted text.
    
    The containment fallback (step 3) follows standard TriviaQA evaluation
    practice ("contains" baseline in Thakur et al. 2024; flexible-extract
    in lm-evaluation-harness). It handles verbose model outputs where the
    correct entity is embedded in a longer response. The 3-character minimum
    prevents spurious matches on short gold aliases (e.g., "II", "TV", "art").
    
    Applied uniformly across all models and conditions.
    """
    pred_norm = normalise_answer(predicted)
    if not pred_norm:
        return False
    
    for gold in gold_answers:
        gold_norm = normalise_answer(gold)
        if not gold_norm:
            continue
        # 1. Exact match
        if pred_norm == gold_norm:
            return True
        # 2. Levenshtein similarity
        if levenshtein_ratio(pred_norm, gold_norm) >= threshold:
            return True
    
    # 3. Containment fallback: check if any gold alias appears within
    #    the predicted text. Only for aliases >= 3 characters to avoid
    #    spurious substring matches on very short aliases.
    #    Also strips parenthetical disambiguators from gold aliases
    #    (e.g., "Pink Floyd (band)" -> "Pink Floyd") since TriviaQA
    #    aliases often include these but model responses do not.
    for gold in gold_answers:
        gold_clean = re.sub(r"\s*\(.*?\)\s*", " ", gold)  # Strip (band), (river), etc.
        gold_norm = normalise_answer(gold_clean)
        if len(gold_norm) >= 4 and gold_norm in pred_norm:
            return True
    
    return False


def is_malformed(answer_text: str) -> bool:
    """
    Check if answer is malformed per pre-registration:
    no alphabetic characters or empty after whitespace stripping.
    """
    stripped = answer_text.strip() if answer_text else ""
    if not stripped:
        return True
    if not any(c.isalpha() for c in stripped):
        return True
    return False


def extract_answer_from_response(response: str, mode: str = "non_thinking") -> str:
    """
    Extract the final answer from model response.
    In thinking mode, answer is after </think> tag.
    In non-thinking mode, entire response is the answer.
    """
    if mode == "thinking":
        # Find content after </think> tag
        if "</think>" in response:
            answer = response.split("</think>")[-1].strip()
        else:
            # No closing tag — treat entire response as answer
            answer = response.strip()
    else:
        answer = response.strip()
    
    # Clean common prefixes
    for prefix in ["The answer is ", "Answer: ", "A: "]:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()
    
    # Take first line if multi-line
    answer = answer.split("\n")[0].strip()
    
    return answer


def extract_thinking_trace(response: str) -> tuple[str, bool]:
    """
    Extract thinking trace from response.
    Returns (trace_text, is_empty).
    """
    if "<think>" not in response:
        return "", True
    
    if "</think>" not in response:
        # Truncated — extract everything after <think>
        trace = response.split("<think>", 1)[1].strip()
        return trace, (len(trace.strip()) == 0)
    
    trace = response.split("<think>", 1)[1].split("</think>", 1)[0].strip()
    is_empty = len(trace) == 0
    return trace, is_empty
