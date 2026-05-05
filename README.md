# Does Thinking Mode Improve Metacognitive Efficiency in Reasoning Language Models?

**Pre-registered on OSF: [link TBD]**

## Setup (AMD RX 7900 GRE, Windows 11)

### 1. Install llama-cpp-python with Vulkan backend

```bash
# Option A: Vulkan (recommended for AMD)
CMAKE_ARGS="-DGGML_VULKAN=ON" pip install llama-cpp-python --force-reinstall --no-cache-dir

# Option B: If Vulkan fails, try ROCm
# CMAKE_ARGS="-DGGML_HIPBLAS=ON" pip install llama-cpp-python --force-reinstall --no-cache-dir

# Option C: CPU fallback (slow but guaranteed)
# pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Download models

```bash
# Create models directory
mkdir -p models

# Qwen3-8B Q5_K_M (primary)
huggingface-cli download Qwen/Qwen3-8B-GGUF qwen3-8b-q5_k_m.gguf --local-dir models/

# DeepSeek-R1-Distill-Qwen-7B Q5_K_M
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B-GGUF deepseek-r1-distill-qwen-7b-q5_k_m.gguf --local-dir models/

# DeepSeek-R1-Distill-Llama-8B Q5_K_M
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Llama-8B-GGUF deepseek-r1-distill-llama-8b-q5_k_m.gguf --local-dir models/

# Qwen3-8B Q8_0 (pre-committed quantisation check)
huggingface-cli download Qwen/Qwen3-8B-GGUF qwen3-8b-q8_0.gguf --local-dir models/
```

**Note:** Check exact GGUF filenames on HuggingFace — they may differ slightly from the above. The quantisation level (Q5_K_M, Q8_0) is what matters.

### 4. Download TriviaQA

```bash
python scripts/download_triviaqa.py
```

### 5. Run pilot (50 items per model, thinking mode)

```bash
python scripts/run_pilot.py --model qwen3-8b
python scripts/run_pilot.py --model r1-distill-qwen-7b
python scripts/run_pilot.py --model r1-distill-llama-8b
```

Review pilot results in `pilot/`. Log on OSF before proceeding.

### 6. Run full data collection

```bash
# Non-thinking mode (all models)
python scripts/run_inference.py --model qwen3-8b --mode non_thinking
python scripts/run_inference.py --model r1-distill-qwen-7b --mode non_thinking
python scripts/run_inference.py --model r1-distill-llama-8b --mode non_thinking

# Thinking mode (all models)
python scripts/run_inference.py --model qwen3-8b --mode thinking
python scripts/run_inference.py --model r1-distill-qwen-7b --mode thinking
python scripts/run_inference.py --model r1-distill-llama-8b --mode thinking

# Abridged-NLP (D1) — teacher-forcing
python scripts/run_abridged_nlp.py --model qwen3-8b
python scripts/run_abridged_nlp.py --model r1-distill-qwen-7b
python scripts/run_abridged_nlp.py --model r1-distill-llama-8b

# Q8 check (Qwen3-8B only, 200 items)
python scripts/run_inference.py --model qwen3-8b --mode non_thinking --quant q8 --n-items 200
python scripts/run_inference.py --model qwen3-8b --mode thinking --quant q8 --n-items 200
```

### 7. Run analysis

```bash
python scripts/run_analysis.py
```

Results written to `results/`.

## Project structure

```
reasoning-metacog/
├── configs/
│   └── study_config.py        # All pre-registered parameters
├── scripts/
│   ├── data_loader.py         # TriviaQA loading and domain assignment
│   ├── scoring.py             # Answer scoring (exact match + Levenshtein)
│   ├── sdt_analysis.py        # d', meta-d', M-ratio, AUROC2, ECE, validity
│   ├── bootstrap.py           # Bootstrap inference
│   ├── run_pilot.py           # Pilot execution (TODO)
│   ├── run_inference.py       # Main inference loop (TODO)
│   ├── run_abridged_nlp.py    # D1 teacher-forcing (TODO)
│   ├── run_analysis.py        # Full analysis pipeline (TODO)
│   └── download_triviaqa.py   # Data download helper (TODO)
├── models/                    # Local GGUF files (not tracked)
├── data/                      # TriviaQA data
├── pilot/                     # Pilot results
├── results/
│   ├── raw/                   # Per-trial outputs
│   ├── processed/             # Aggregate metrics
│   └── figures/               # Publication figures
├── requirements.txt
└── README.md
```

## Pre-registration

Filed on OSF: [link TBD]
Pre-registration document: v3, 30 April 2026
12 independent adversarial reviews across 3 rounds before filing.
```
