# reasoning-metacog/scripts/reasoning_engine.py
"""
Inference engine for reasoning-metacog study.
Adapted from SDT Calibration inference_engine.py.

Handles:
- Thinking mode generation with <think> trace extraction
- Non-thinking mode generation
- Abridged-NLP (D1): teacher-forced NLP without thinking trace
- NLP extraction from token log-probs
- First-token softmax extraction

Hardware: AMD RX 7900 GRE 16GB, Vulkan backend, llama-cpp-python

IMPORTANT: Uses raw llm() completion API with manual chat templates.
create_chat_completion() returns logprobs: None on the Vulkan build.
"""

import json
import math
import re
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# JSON helper (from inference_engine.py)
# ---------------------------------------------------------------------------

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Answer the following question with a short factual answer. "
    "Respond with only the answer, nothing else."
)

# Qwen3 thinking mode: do NOT include /no_think
# Qwen3 non-thinking mode: include /no_think in system prompt
SYSTEM_PROMPT_QWEN3_NOTHINK = (
    "/no_think Answer the following question with a short factual answer. "
    "Respond with only the answer, nothing else."
)

# DeepSeek-R1-Distill models use <think> tags natively
# Non-thinking: suppressed via empty-think prefill (<think>\n</think>\n)
# No need for "Do not use <think> tags" instruction — the prefill handles it
SYSTEM_PROMPT_R1_NOTHINK = (
    "Answer the following question with a short factual answer. "
    "Respond with only the answer, nothing else."
)


def get_system_prompt(model_key: str, mode: str) -> str:
    """Return appropriate system prompt for model and mode."""
    if mode == "thinking":
        return SYSTEM_PROMPT  # All models get the same prompt in thinking mode
    else:  # non_thinking
        if model_key == "qwen3-8b" or model_key == "qwen3-8b-q8":
            return SYSTEM_PROMPT_QWEN3_NOTHINK
        elif model_key.startswith("r1-distill"):
            return SYSTEM_PROMPT_R1_NOTHINK
        else:
            return SYSTEM_PROMPT


# Model-specific stop tokens
STOP_TOKENS = {
    "qwen3-8b": ["<|endoftext|>", "<|im_end|>"],
    "qwen3-8b-q8": ["<|endoftext|>", "<|im_end|>"],
    "r1-distill-qwen-7b": ["<|endoftext|>", "<|im_end|>"],
    "r1-distill-llama-8b": ["<|eot_id|>", "<|end_of_text|>"],
}


# ---------------------------------------------------------------------------
# Thinking trace extraction
# ---------------------------------------------------------------------------

def extract_thinking_trace(response: str) -> dict:
    """
    Extract thinking trace and final answer from response.
    Returns dict with trace_text, answer_text, is_empty_trace, is_truncated.
    """
    if "<think>" not in response:
        return {
            "trace_text": "",
            "answer_text": response.strip(),
            "is_empty_trace": True,
            "has_think_tag": False,
        }

    has_closing = "</think>" in response

    if has_closing:
        parts = response.split("<think>", 1)
        think_and_rest = parts[1]
        trace_parts = think_and_rest.split("</think>", 1)
        trace_text = trace_parts[0].strip()
        answer_text = trace_parts[1].strip() if len(trace_parts) > 1 else ""
    else:
        # No closing tag — trace was truncated
        trace_text = response.split("<think>", 1)[1].strip()
        answer_text = ""

    is_empty = len(trace_text) == 0

    return {
        "trace_text": trace_text,
        "answer_text": answer_text,
        "is_empty_trace": is_empty,
        "has_think_tag": True,
        "is_truncated": not has_closing,
    }


def check_pilot_coherence(trace_text: str, answer_text: str, last_n_tokens: int = 50) -> dict:
    """
    Pilot coherence check per pre-registration.
    Coherent if:
      (i) final answer appears verbatim in last 50 tokens of trace, OR
      (ii) no explicit contradiction markers referring to final answer.
    """
    contradiction_markers = [
        "actually", "on second thought", "that is incorrect", "I was wrong"
    ]

    # Check (i): answer in last portion of trace
    # Use character-level approximation (50 tokens ≈ 200 chars)
    trace_tail = trace_text[-800:] if len(trace_text) > 800 else trace_text
    answer_in_trace = answer_text.lower().strip() in trace_tail.lower() if answer_text.strip() else False

    # Check (ii): no contradiction markers
    has_contradiction = False
    for marker in contradiction_markers:
        if marker.lower() in trace_text.lower():
            # Only flag if it appears near the end (last 30% of trace)
            marker_pos = trace_text.lower().rfind(marker.lower())
            if marker_pos > len(trace_text) * 0.7:
                has_contradiction = True
                break

    coherent = answer_in_trace or (not has_contradiction)

    return {
        "coherent": coherent,
        "answer_in_trace": answer_in_trace,
        "has_contradiction": has_contradiction,
    }


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------

class ReasoningInferenceEngine:
    """
    Inference engine for reasoning-metacog study.
    Adapted from SDTInferenceEngine.

    Uses raw llm() completion API with manual chat templates.
    create_chat_completion() does NOT return logprobs on the Vulkan build.

    Supports:
    - Thinking mode: generate with <think> traces, extract answer NLP
    - Non-thinking mode: generate directly, extract NLP
    - Abridged-NLP (D1): teacher-force answer in fresh context
    """

    def __init__(self, model_key: str, model_path: str, n_ctx: int = 5120):
        from llama_cpp import Llama

        self.model_key = model_key
        self.model_path = model_path
        self.stop_tokens = STOP_TOKENS.get(model_key, [])

        print(f"Loading {model_key} from {model_path}...")
        t0 = time.time()
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=n_ctx,  # 5120: prompt (~100) + trace (4096) + answer (256) + margin
            logits_all=True,  # Required for logprobs=True in llm() API AND for force_decode_nlp
            verbose=False,
        )
        print(f"  Loaded in {time.time() - t0:.1f}s. Vocab: {self.llm.n_vocab()}")

    def _build_raw_prompt(self, question: str, system_prompt: str, mode: str = "non_thinking") -> str:
        """
        Build raw prompt string for the raw llm() API.
        Model-specific chat templates applied manually.

        For R1-Distill models in thinking mode, appends '<think>\n' as an
        assistant prefill to force the model into reasoning. Per DeepSeek's
        official recommendation: "To ensure that the model engages in thorough
        reasoning, we recommend enforcing the model to initiate its response
        with '<think>\n' at the beginning of every output."

        Without this prefill, R1-Distill models skip thinking on easy factual
        questions and produce empty <think></think> tags. Documented as a
        permissible non-deviation (pre-registration §10.1) before full data
        collection.
        """
        # Determine whether to add prefill for R1-Distill models
        # Qwen3 handles thinking/non-thinking natively via /no_think
        #
        # R1-Distill models need explicit prefill in BOTH modes:
        #   - Thinking:     "<think>\n"           forces model to reason
        #   - Non-thinking: "<think>\n</think>\n"  tells model reasoning is done
        #
        # Without non-thinking prefill, R1-Distill generates chain-of-thought
        # as plain text (no tags), producing verbose wrong answers (~1% accuracy).
        # The empty-think prefill is the community-standard approach documented
        # on HuggingFace (deepseek-ai/DeepSeek-R1-Distill-Qwen-14B/discussions/11).
        think_prefill = ""
        if self.model_key.startswith("r1-distill"):
            if mode == "thinking":
                think_prefill = "<think>\n"
            else:
                think_prefill = "<think>\n</think>\n"

        if self.model_key.startswith("qwen3") or self.model_key.startswith("r1-distill-qwen"):
            # Qwen / ChatML template
            return (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{question}<|im_end|>\n"
                f"<|im_start|>assistant\n{think_prefill}"
            )
        elif self.model_key.startswith("r1-distill-llama"):
            # Llama-3 instruct template
            return (
                f"<|start_header_id|>system<|end_header_id|>\n\n"
                f"{system_prompt}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{question}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n{think_prefill}"
            )
        else:
            # Fallback: ChatML
            return (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{question}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

    def generate(
        self,
        question: str,
        mode: str,  # "thinking" or "non_thinking"
        max_thinking_tokens: int = 4096,
        max_answer_tokens: int = 256,
        temperature: float = 0.6,
        top_k: int = 20,
        top_p: float = 0.95,
        seed: int = 42,
    ) -> dict:
        """
        Generate response using raw llm() API and extract NLP signals.

        Returns dict with:
        - raw_response: full model output
        - answer_text: extracted answer (after </think> if thinking mode)
        - trace_text: thinking trace (empty string if non-thinking)
        - trace_length_tokens: number of tokens in thinking trace
        - answer_nlp: NLP of answer tokens only
        - answer_length_tokens: number of tokens in answer
        - first_token_softmax: softmax prob of first answer token
        - is_empty_trace: whether thinking trace is empty
        - is_truncated: whether thinking trace was cut off
        - elapsed_s: generation time
        """
        system_prompt = get_system_prompt(self.model_key, mode)
        prompt = self._build_raw_prompt(question, system_prompt, mode=mode)

        max_tokens = (
            max_answer_tokens
            if mode == "non_thinking"
            else (max_thinking_tokens + max_answer_tokens)
        )

        t0 = time.perf_counter()
        result = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            logprobs=1,  # Returns top-1 logprob per token; True triggers O(V) sort per token
            stop=self.stop_tokens,
        )
        elapsed = time.perf_counter() - t0

        # --- Extract text and logprobs from raw API result ---
        raw_response = result["choices"][0]["text"]
        logprobs_data = result["choices"][0].get("logprobs", {})

        # When we used <think>\n prefill for R1-Distill models, the generated
        # text starts AFTER <think>\n — it contains the trace content, then
        # </think>, then the answer. We need to prepend <think> so that
        # extract_thinking_trace() can find the boundary correctly.
        used_think_prefill = (
            mode == "thinking" and self.model_key.startswith("r1-distill")
        )
        if used_think_prefill:
            raw_response = "<think>" + raw_response

        # Token strings and their log-probabilities
        tokens = logprobs_data.get("tokens", []) if logprobs_data else []
        token_logprobs = logprobs_data.get("token_logprobs", []) if logprobs_data else []

        # --- Extract thinking trace and answer from text ---
        if mode == "thinking":
            trace_info = extract_thinking_trace(raw_response)
            trace_text = trace_info["trace_text"]
            answer_text = trace_info["answer_text"]
            is_empty_trace = trace_info["is_empty_trace"]
            is_truncated = trace_info.get("is_truncated", False)
        else:
            trace_text = ""
            answer_text = raw_response.strip()
            # Remove empty think tags that Qwen3 sometimes produces in /no_think
            answer_text = re.sub(r"<think>\s*</think>\s*", "", answer_text).strip()
            is_empty_trace = True
            is_truncated = False

        # --- Compute NLP from token logprobs ---
        answer_nlp, answer_length, first_token_softmax, trace_length = (
            self._compute_answer_nlp(tokens, token_logprobs, mode)
        )

        return {
            "raw_response": raw_response,
            "answer_text": answer_text,
            "trace_text": trace_text,
            "trace_length_tokens": trace_length,
            "answer_nlp": answer_nlp,
            "answer_length_tokens": answer_length,
            "first_token_softmax": first_token_softmax,
            "is_empty_trace": is_empty_trace,
            "is_truncated": is_truncated,
            "elapsed_s": round(elapsed, 3),
        }

    def _compute_answer_nlp(
        self,
        tokens: list[str],
        token_logprobs: list[float],
        mode: str,
    ) -> tuple[float, int, float, int]:
        """
        Compute NLP for answer tokens only using token strings and logprobs
        from the raw llm() API.

        Returns (answer_nlp, answer_length, first_token_softmax, trace_length).
        """
        if not tokens or not token_logprobs:
            return float("-inf"), 0, 0.0, 0

        if mode == "non_thinking":
            # All tokens are answer tokens
            valid_lps = [lp for lp in token_logprobs if lp is not None]
            if not valid_lps:
                return float("-inf"), 0, 0.0, 0
            first_softmax = math.exp(valid_lps[0]) if valid_lps else 0.0
            return sum(valid_lps) / len(valid_lps), len(valid_lps), first_softmax, 0

        # --- Thinking mode: find </think> boundary in token stream ---
        # Walk the token list, accumulating text to find </think>
        cumulative = ""
        think_end_idx = None
        for i, tok in enumerate(tokens):
            cumulative += tok
            if "</think>" in cumulative and think_end_idx is None:
                think_end_idx = i + 1  # First token AFTER </think>

        if think_end_idx is None:
            # No </think> found
            if "<think>" not in cumulative:
                # No thinking tags at all — treat all as answer
                valid_lps = [lp for lp in token_logprobs if lp is not None]
                if not valid_lps:
                    return float("-inf"), 0, 0.0, 0
                return sum(valid_lps) / len(valid_lps), len(valid_lps), math.exp(valid_lps[0]), 0
            else:
                # Truncated thinking trace — no answer tokens
                return float("-inf"), 0, 0.0, len(tokens)

        trace_length = think_end_idx
        answer_lps = token_logprobs[think_end_idx:]

        if not answer_lps:
            return float("-inf"), 0, 0.0, trace_length

        valid_lps = [lp for lp in answer_lps if lp is not None]
        if not valid_lps:
            return float("-inf"), 0, 0.0, trace_length

        first_softmax = math.exp(valid_lps[0])
        answer_nlp = sum(valid_lps) / len(valid_lps)

        return answer_nlp, len(valid_lps), first_softmax, trace_length

    def force_decode_nlp(self, question: str, answer_text: str, mode: str = "non_thinking") -> float:
        """
        D1: Abridged-NLP — teacher-force answer in fresh context WITHOUT thinking trace.

        Always uses non-thinking system prompt and context.
        This is the binding diagnostic that isolates prefix conditioning from metacognition.

        Adapted from SDTInferenceEngine.force_decode_nlp().
        NLP = (1/L) * sum(log p(t_i | t_{<i}))
        """
        # Always use non-thinking prompt for D1 (that's the point)
        system_prompt = get_system_prompt(self.model_key, "non_thinking")

        # Build the prompt using manual chat template
        prompt = self._build_raw_prompt(question, system_prompt)

        # Tokenize prompt + answer
        prompt_tokens = self.llm.tokenize(prompt.encode("utf-8"))
        answer_tokens = self.llm.tokenize(answer_text.encode("utf-8"), add_bos=False)

        if len(answer_tokens) == 0:
            return float("-inf")

        # Evaluate full sequence
        full_tokens = prompt_tokens + answer_tokens
        self.llm.reset()
        self.llm.eval(full_tokens)

        scores = self.llm.scores
        if scores is None or len(scores) == 0:
            return float("-inf")

        prompt_len = len(prompt_tokens)
        answer_len = len(answer_tokens)

        log_probs_sum = 0.0
        valid_count = 0
        for i in range(answer_len):
            logit_pos = prompt_len - 1 + i
            if logit_pos >= len(scores):
                break
            logits = np.array(scores[logit_pos], dtype=np.float64)
            logits_shifted = logits - np.max(logits)
            log_sum_exp = np.log(np.sum(np.exp(logits_shifted)))
            log_prob = logits_shifted[answer_tokens[i]] - log_sum_exp
            log_probs_sum += float(log_prob)
            valid_count += 1

        if valid_count == 0:
            return float("-inf")

        return log_probs_sum / valid_count

    def unload(self):
        """Free model from GPU memory."""
        if hasattr(self, "llm") and self.llm is not None:
            del self.llm
            self.llm = None
            import gc
            gc.collect()
