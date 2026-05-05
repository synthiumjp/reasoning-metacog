# reasoning-metacog/local_config.py
"""
Local machine configuration. Not tracked in git.
Edit paths to match your system.
"""

MODEL_PATHS = {
    "qwen3-8b": r"D:\beyond_the_mean\models\Qwen3-8B-Q5_K_M.gguf",
    "r1-distill-llama-8b": r"C:\sdt_calibration\models\DeepSeek-R1-Distill-Llama-8B-Q5_K_M.gguf",
    # Download this one: huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B-GGUF
    "r1-distill-qwen-7b": r"D:\reasoning-metacog\models\DeepSeek-R1-Distill-Qwen-7B-Q5_K_M.gguf",
    # Q8 check (pre-committed, Qwen3-8B only)
    # Download: huggingface-cli download Qwen/Qwen3-8B-GGUF
    "qwen3-8b-q8": r"D:\reasoning-metacog\models\Qwen3-8B-Q8_0.gguf",
}

# Project root
PROJECT_DIR = r"D:\reasoning-metacog"

# TriviaQA data path (downloaded by scripts/download_triviaqa.py)
TRIVIAQA_PATH = r"D:\reasoning-metacog\data\triviaqa_sampled_2000.jsonl"
