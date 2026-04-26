"""
Configuration for the Agentic Planning Study.
Supports: OpenAI, Gemini (Google AI), Groq, or any OpenAI-compatible API.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM Provider Configuration ---
# Supported providers: "gemini", "openai", "groq"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

# --- API Keys ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# --- Provider-specific settings ---
PROVIDER_CONFIG = {
    "gemini": {
        "api_key": GEMINI_API_KEY,
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    },
    "openai": {
        "api_key": OPENAI_API_KEY,
        "base_url": None,  # Uses default OpenAI endpoint
        "model": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
    },
    "groq": {
        "api_key": GROQ_API_KEY,
        "base_url": "https://api.groq.com/openai/v1",
        "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    },
}

def get_provider_config():
    """Get the active provider configuration."""
    provider = LLM_PROVIDER.lower()
    if provider not in PROVIDER_CONFIG:
        raise ValueError(f"Unknown provider: {provider}. Use: gemini, openai, or groq")
    cfg = PROVIDER_CONFIG[provider]
    if not cfg["api_key"]:
        raise ValueError(
            f"API key not set for provider '{provider}'. "
            f"Set {provider.upper()}_API_KEY in your .env file."
        )
    return cfg

# --- General LLM Settings ---
TEMPERATURE = 0.0  # Deterministic for reproducibility
MAX_TOKENS = 1024

# --- Experiment Configuration ---
NUM_RUNS_PER_TASK = 3          # Runs per task for stability measurement
MAX_STEPS_PER_TASK = 15        # Safety limit to prevent infinite loops
TIMEOUT_SECONDS = 120          # Per-task timeout

# --- Retry Configuration (handles rate limits) ---
MAX_RETRIES = 15
RETRY_BASE_DELAY = 15           # seconds — initial wait before retry
RETRY_BACKOFF_FACTOR = 1.5       # exponential backoff multiplier

# --- Tree of Thoughts Configuration ---
TOT_BRANCHES = 3
TOT_MAX_DEPTH = 3
TOT_EVALUATION_STRATEGY = "vote"

# --- Output Paths ---
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
