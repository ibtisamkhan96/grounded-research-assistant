"""Configuration and logging.

Everything tunable lives here so the rest of the code reads cleanly. Keys come
from the environment (or a .env you load), never hard-coded.
"""
import os
import logging

# --- Model: Claude via Anthropic. Override with the MODEL env var. ---
MODEL = os.environ.get("MODEL", "claude-sonnet-5")

# --- Retrieval settings ---
OPENALEX_K = 8                  # peer-reviewed papers to pull per question
ARXIV_K = 4                     # preprint top-up when peer-reviewed results are thin
MIN_PEER_REVIEWED = 4           # below this many peer-reviewed hits, top up with arXiv
ABSTRACT_CHARS = 2500           # max abstract length stored per paper
CONTEXT_ABSTRACT_CHARS = 1500   # abstract length handed to the model per paper
HTTP_TIMEOUT = 30               # seconds
CACHE_TTL = 600                 # seconds to cache identical searches

USER_AGENT = "grounded-research/1.0 (mailto:khanibtisam38@gmail.com)"

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def require_anthropic_key() -> str:
    """Return the Anthropic key or fail with a clear message (used before any LLM call)."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it, or copy .env.example to .env and fill it in. "
            "You only need it for answering; --search-only works without a key."
        )
    return key
