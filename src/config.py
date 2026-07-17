"""
config.py — Centralized configuration for Multi-Source RAG
Loads environment variables and initializes the Groq LLM + Gemini embeddings.
"""
import os
from dotenv import load_dotenv

# Load .env from project root FIRST (before any langchain imports that check env)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# --- API Keys ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "multi-source-rag")

# Disable LangSmith if the key looks like a placeholder (do this BEFORE importing langchain)
_langsmith_key = os.getenv("LANGCHAIN_API_KEY", "")
if not _langsmith_key or "dummy" in _langsmith_key or "replace" in _langsmith_key.lower():
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    TRACING_ENABLED = False
else:
    TRACING_ENABLED = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set. Please add it to your .env file.")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not set. Please add it to your .env file.")

from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- Model: Groq Llama 3.3 70B (fast, reliable, and free of Gemini's tight quota) ---
# temperature=0 for deterministic routing
LLM = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=GROQ_API_KEY,
    max_tokens=2048,
)

# --- Synthesis LLM (slightly more creative for answers) ---
SYNTHESIS_LLM = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    groq_api_key=GROQ_API_KEY,
    max_tokens=2048,
)

# --- Embeddings for FAISS ---
EMBEDDINGS = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY,
)

# --- Cost estimation (Groq Llama 3.3 70B pricing) ---
# Input: $0.59/1M tokens, Output: $0.79/1M tokens
COST_PER_INPUT_TOKEN = 0.59 / 1_000_000
COST_PER_OUTPUT_TOKEN = 0.79 / 1_000_000


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD for a given token count."""
    return (input_tokens * COST_PER_INPUT_TOKEN) + (output_tokens * COST_PER_OUTPUT_TOKEN)


def get_text(content) -> str:
    """
    Extract plain text from response content.
    Handles string, lists, and dict format content safely.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


if __name__ == "__main__":
    print(f"✅ Config loaded. LangSmith tracing: {TRACING_ENABLED}")
    print(f"   Project: {LANGCHAIN_PROJECT}")
    print(f"   Model: llama-3.3-70b-versatile (via Groq)")
    # Quick LLM smoke test
    resp = LLM.invoke("Say 'config OK' and nothing else.")
    print(f"   LLM test: {get_text(resp.content)}")
