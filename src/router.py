"""
router.py — LLM-based router with few-shot examples.

The router is the most critical node in the graph. It reads the user query
and decides which ONE tool to use (or whether to answer directly).

Design decisions:
  - Few-shot examples reduce ambiguity between Wikipedia ("tell me about X")
    and Arxiv ("recent research on X").
  - A "direct_answer" path avoids unnecessary tool calls for simple queries,
    reducing cost + latency.
  - Zero temperature makes routing deterministic and auditable.
"""
import re
import time
import random
from langchain_core.messages import HumanMessage, SystemMessage

# Import LLM from config
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import LLM, get_text

# ─────────────────────────────────────────────
# Router System Prompt (with few-shot examples)
# ─────────────────────────────────────────────
ROUTER_SYSTEM_PROMPT = """\
You are a query router for a multi-source RAG system. Your job is to read a user query
and decide which ONE of these 4 options should handle it:

  - wikipedia       → General knowledge, encyclopedia facts, people, places, concepts, history
  - arxiv           → Scientific papers, recent research, machine learning, physics, mathematics, academic
  - langsmith_search → LangSmith docs, LangChain tracing, LLM observability, evaluation frameworks
  - direct_answer   → Simple math, greetings, conversational follow-ups, or anything that doesn't need external lookup

RULES:
1. Output ONLY the option name. Nothing else. No explanation, no punctuation.
2. If the query could be Wikipedia OR Arxiv, choose arxiv if it mentions "research", "paper", "study", "model", "recent", or a paper ID. Otherwise choose wikipedia.
3. Choose direct_answer when no retrieval adds value.

FEW-SHOT EXAMPLES:
Query: "Who is Alan Turing?"
Answer: wikipedia

Query: "What is the capital of France?"
Answer: wikipedia

Query: "Tell me about quantum computing"
Answer: wikipedia

Query: "What are the latest advances in transformer architectures?"
Answer: arxiv

Query: "Summarize the paper 2305.10601"
Answer: arxiv

Query: "Recent research on reinforcement learning from human feedback"
Answer: arxiv

Query: "What is attention mechanism in transformers?"
Answer: arxiv

Query: "How does LangSmith tracing work?"
Answer: langsmith_search

Query: "What is LangSmith used for?"
Answer: langsmith_search

Query: "How do I set up LangSmith evaluation?"
Answer: langsmith_search

Query: "What is 2 + 2?"
Answer: direct_answer

Query: "Thanks, that's helpful"
Answer: direct_answer

Query: "Tell me more"
Answer: direct_answer

Query: "What year is it?"
Answer: direct_answer

Now classify the following query:
"""

VALID_ROUTES = {"wikipedia", "arxiv", "langsmith_search", "direct_answer"}


def route_query(query: str) -> str:
    """
    Route a user query to one of: wikipedia, arxiv, langsmith_search, direct_answer.
    Returns the route name as a string.
    """
    messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=f'Query: "{query}"\nAnswer:'),
    ]

    # Retry with exponential backoff on rate limits
    # NOTE: gemini-flash-latest has ~2 RPM on free tier — waits can be 30-60s
    backoff_times = [30, 60, 120]  # seconds
    for attempt in range(3):
        try:
            response = LLM.invoke(messages)
            break
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait = backoff_times[attempt] + random.uniform(0, 5)
                print(f"   ⏳ Rate limit hit (attempt {attempt+1}/3). Waiting {wait:.0f}s...")
                time.sleep(wait)
            else:
                raise
    else:
        print("⚠️  Router max retries exceeded. Defaulting to 'wikipedia'.")
        return "wikipedia"

    raw = get_text(response.content).strip().lower()

    # Clean up any extra punctuation or whitespace
    route = re.sub(r"[^a-z_]", "", raw)

    if route not in VALID_ROUTES:
        # Fallback: try to find a valid route substring
        for valid in VALID_ROUTES:
            if valid in raw:
                route = valid
                break
        else:
            # Default fallback to wikipedia if completely unparseable
            print(f"⚠️  Router returned unexpected value: '{raw}'. Defaulting to 'wikipedia'.")
            route = "wikipedia"

    return route


if __name__ == "__main__":
    test_queries = [
        ("Who invented the telephone?", "wikipedia"),
        ("Recent papers on large language models", "arxiv"),
        ("How do I use LangSmith for debugging?", "langsmith_search"),
        ("What is 5 times 7?", "direct_answer"),
        ("Tell me about machine learning", "wikipedia"),
        ("Paper about GPT-4 capabilities", "arxiv"),
    ]

    print("Router test:\n" + "─" * 50)
    correct = 0
    for query, expected in test_queries:
        predicted = route_query(query)
        status = "RIGHT" if predicted == expected else "WRONG"
        print(f"  {status} [{predicted}] ← '{query}'")
        if predicted == expected:
            correct += 1

    print(f"\nAccuracy: {correct}/{len(test_queries)}")
