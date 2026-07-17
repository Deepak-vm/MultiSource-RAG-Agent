"""
test_queries.py — 20 labeled evaluation queries for the Multi-Source RAG system.

Each query has:
  - query: the user's input
  - expected_tool: which tool SHOULD handle it
  - keywords: expected keywords in the retrieved content (for retrieval quality eval)
  - notes: why this query belongs to this category

Categories:
  - wikipedia (5): general knowledge, factual, encyclopedic
  - arxiv (5): academic papers, research, ML/science
  - langsmith_search (5): LangSmith docs, LangChain observability
  - direct_answer (5): no retrieval needed
"""

TEST_QUERIES = [
    # ─── Wikipedia (5) ───────────────────────────────────────
    {
        "query": "Who is Alan Turing?",
        "expected_tool": "wikipedia",
        "keywords": ["mathematician", "computer", "Turing", "computing"],
        "notes": "Famous person — pure encyclopedia entry",
    },
    {
        "query": "What is the theory of relativity?",
        "expected_tool": "wikipedia",
        "keywords": ["Einstein", "relativity", "mass", "energy", "spacetime"],
        "notes": "Scientific concept with encyclopedic explanation",
    },
    {
        "query": "Tell me about the French Revolution",
        "expected_tool": "wikipedia",
        "keywords": ["France", "revolution", "1789", "monarchy", "republic"],
        "notes": "Historical event — Wikipedia territory",
    },
    {
        "query": "What is quantum computing?",
        "expected_tool": "wikipedia",
        "keywords": ["qubit", "quantum", "superposition", "computing"],
        "notes": "General concept explanation, not recent research",
    },
    {
        "query": "Who founded Google?",
        "expected_tool": "wikipedia",
        "keywords": ["Larry Page", "Sergey Brin", "Stanford", "Google"],
        "notes": "Company founding — factual Wikipedia entry",
    },

    # ─── Arxiv (5) ───────────────────────────────────────────
    {
        "query": "Recent research on transformer architectures",
        "expected_tool": "arxiv",
        "keywords": ["transformer", "attention", "architecture", "model"],
        "notes": "Contains 'recent research' — Arxiv signal",
    },
    {
        "query": "Papers about reinforcement learning from human feedback",
        "expected_tool": "arxiv",
        "keywords": ["RLHF", "reinforcement", "human", "feedback", "reward"],
        "notes": "'Papers about' is explicit Arxiv signal",
    },
    {
        "query": "What is the paper 1706.03762 about?",
        "expected_tool": "arxiv",
        "keywords": ["Attention Is All You Need", "transformer", "attention"],
        "notes": "Paper ID lookup — only Arxiv handles these",
    },
    {
        "query": "Latest advances in diffusion models for image generation",
        "expected_tool": "arxiv",
        "keywords": ["diffusion", "image", "generation", "model", "DDPM"],
        "notes": "'Latest advances' = research signal",
    },
    {
        "query": "Research on large language model hallucinations",
        "expected_tool": "arxiv",
        "keywords": ["hallucination", "language model", "factual", "accuracy"],
        "notes": "'Research on' + technical ML topic = Arxiv",
    },

    # ─── LangSmith (5) ───────────────────────────────────────
    {
        "query": "How does LangSmith tracing work?",
        "expected_tool": "langsmith_search",
        "keywords": ["trace", "LangSmith", "run", "span"],
        "notes": "Direct LangSmith product question",
    },
    {
        "query": "What is LangSmith used for?",
        "expected_tool": "langsmith_search",
        "keywords": ["LangSmith", "observability", "trace", "evaluation"],
        "notes": "LangSmith overview question",
    },
    {
        "query": "How do I set up LangSmith for my LangChain project?",
        "expected_tool": "langsmith_search",
        "keywords": ["LANGCHAIN_TRACING_V2", "API_KEY", "setup", "project"],
        "notes": "Setup/integration question for LangSmith",
    },
    {
        "query": "How do I evaluate my RAG pipeline with LangSmith?",
        "expected_tool": "langsmith_search",
        "keywords": ["evaluation", "dataset", "LangSmith", "benchmark"],
        "notes": "Evaluation feature of LangSmith",
    },
    {
        "query": "What are LangSmith datasets and how do I create one?",
        "expected_tool": "langsmith_search",
        "keywords": ["dataset", "LangSmith", "examples", "create"],
        "notes": "LangSmith-specific feature",
    },

    # ─── Direct Answer (5) ───────────────────────────────────
    {
        "query": "What is 15 multiplied by 7?",
        "expected_tool": "direct_answer",
        "keywords": ["105"],
        "notes": "Simple arithmetic — no retrieval needed",
    },
    {
        "query": "Hello, how are you?",
        "expected_tool": "direct_answer",
        "keywords": ["hello", "fine", "well", "help"],
        "notes": "Greeting — direct conversational response",
    },
    {
        "query": "Thank you for the information",
        "expected_tool": "direct_answer",
        "keywords": ["welcome", "glad", "help"],
        "notes": "Conversational closing — no retrieval needed",
    },
    {
        "query": "What day comes after Monday?",
        "expected_tool": "direct_answer",
        "keywords": ["Tuesday"],
        "notes": "Trivial factual — LLM knows this directly",
    },
    {
        "query": "Can you help me?",
        "expected_tool": "direct_answer",
        "keywords": ["yes", "certainly", "help", "assist"],
        "notes": "Open-ended conversational — no retrieval value",
    },
]

if __name__ == "__main__":
    from collections import Counter
    counts = Counter(q["expected_tool"] for q in TEST_QUERIES)
    print(f"Total queries: {len(TEST_QUERIES)}")
    for tool, count in sorted(counts.items()):
        print(f"  {tool}: {count}")
