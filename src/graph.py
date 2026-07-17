"""
graph.py — LangGraph StateGraph: the core architecture of the RAG agent.

Architecture:
  START
    └─→ router_node         (decides which tool to call)
         ├─→ wikipedia_node  (calls Wikipedia)
         ├─→ arxiv_node      (calls Arxiv)
         ├─→ langsmith_node  (calls FAISS retriever)
         └─→ direct_answer_node (no tool, LLM answers directly)
              │ (all tool nodes)
              └─→ answer_node   (synthesizes final response)
                  └─→ END

State schema carries: query, route, tool_output, answer, tracking data.

WHY LangGraph over AgentExecutor:
  1. Explicit, inspectable state — you can see exactly what's in the graph at each step
  2. Node-level tracing in LangSmith — each node shows up as a separate span
  3. Conditional routing with clear if/else logic, not hidden chain logic
  4. Easy to extend: add memory, add nodes, add loops without refactoring
  5. Checkpointing support for multi-turn conversations
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import time
import random
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage

from config import LLM, SYNTHESIS_LLM, get_text, estimate_cost
from router import route_query
from tools import wikipedia_tool, arxiv_tool, get_langsmith_retriever_tool
from tracking import (
    QueryTrace, NodeTrace, timed_node, build_node_trace,
    extract_token_usage, print_trace_summary,
)

# ─────────────────────────────────────────────
# State Schema
# ─────────────────────────────────────────────
class AgentState(TypedDict):
    """The state that flows between nodes in the graph."""
    query: str                          # Original user query
    route: str                          # Chosen tool (set by router_node)
    tool_output: str                    # Raw output from tool (if any)
    answer: str                         # Final synthesized answer
    # Tracking fields
    router_latency_ms: float
    router_input_tokens: int
    router_output_tokens: int
    tool_latency_ms: float
    answer_latency_ms: float
    answer_input_tokens: int
    answer_output_tokens: int
    total_cost_usd: float
    error: Optional[str]


# ─────────────────────────────────────────────
# Initialize tools (do this once at module level)
# ─────────────────────────────────────────────
print("🔧 Initializing tools...")
_langsmith_tool = get_langsmith_retriever_tool()
print("Tools ready")

TOOL_MAP = {
    "wikipedia": wikipedia_tool,
    "arxiv": arxiv_tool,
    "langsmith_search": _langsmith_tool,
}


# ─────────────────────────────────────────────
# Node Definitions
# ─────────────────────────────────────────────

def router_node(state: AgentState) -> AgentState:
    """
    Router node: given the query, decide which tool to use.
    This is the key architectural insight — routing is explicit, not hidden in an LLM loop.
    """
    start = time.perf_counter()
    
    # Use the router (includes its own LLM call)
    route = route_query(state["query"])
    
    latency_ms = (time.perf_counter() - start) * 1000
    print(f"   🧭 Router → [{route}] ({latency_ms:.0f}ms)")

    return {
        **state,
        "route": route,
        "router_latency_ms": latency_ms,
        # Token tracking for router: approximate since router LLM call is inside route_query()
        "router_input_tokens": 0,  # populated in enhanced version
        "router_output_tokens": 0,
    }


def wikipedia_node(state: AgentState) -> AgentState:
    """Call Wikipedia tool and store raw output."""
    start = time.perf_counter()
    try:
        result = wikipedia_tool.invoke(state["query"])
        latency_ms = (time.perf_counter() - start) * 1000
        print(f"   📖 Wikipedia ({latency_ms:.0f}ms) → {len(result)} chars")
        return {**state, "tool_output": result, "tool_latency_ms": latency_ms}
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        return {**state, "tool_output": f"Wikipedia error: {e}", "tool_latency_ms": latency_ms}


def arxiv_node(state: AgentState) -> AgentState:
    """Call Arxiv tool and store raw output."""
    start = time.perf_counter()
    try:
        result = arxiv_tool.invoke(state["query"])
        latency_ms = (time.perf_counter() - start) * 1000
        print(f"   📄 Arxiv ({latency_ms:.0f}ms) → {len(result)} chars")
        return {**state, "tool_output": result, "tool_latency_ms": latency_ms}
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        return {**state, "tool_output": f"Arxiv error: {e}", "tool_latency_ms": latency_ms}


def langsmith_node(state: AgentState) -> AgentState:
    """Call FAISS retriever for LangSmith docs and store raw output."""
    start = time.perf_counter()
    try:
        result = _langsmith_tool.invoke(state["query"])
        latency_ms = (time.perf_counter() - start) * 1000
        print(f"   🔍 LangSmith FAISS ({latency_ms:.0f}ms) → {len(result)} chars")
        return {**state, "tool_output": result, "tool_latency_ms": latency_ms}
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        return {**state, "tool_output": f"LangSmith retriever error: {e}", "tool_latency_ms": latency_ms}


def direct_answer_node(state: AgentState) -> AgentState:
    """
    No tool needed — answer directly from LLM.
    This path avoids unnecessary retrieval for simple queries,
    reducing both cost and latency.
    """
    return {**state, "tool_output": "", "tool_latency_ms": 0.0}


SYNTHESIS_PROMPT = """\
You are a helpful research assistant. Based on the retrieved information below, 
provide a clear, accurate, and well-structured answer to the user's question.

If retrieved information is provided, base your answer primarily on it.
If no retrieved information is provided (empty), answer from your own knowledge.
Keep your answer concise but complete — 2-4 paragraphs maximum.

Retrieved Information:
{context}

User Question: {query}

Answer:"""


def _invoke_with_retry(llm, messages, max_retries=3):
    """Call LLM with exponential backoff on rate limit errors."""
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait = (2 ** attempt) + random.uniform(0.5, 2.0)
                print(f"   ⏳ Rate limit hit, waiting {wait:.1f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise  # Re-raise non-rate-limit errors
    raise RuntimeError(f"Max retries ({max_retries}) exceeded due to rate limiting")


def answer_node(state: AgentState) -> AgentState:
    """
    Synthesis node: combine tool output with user query to produce final answer.
    This is where the LLM reasons over retrieved content.
    """
    start = time.perf_counter()

    context = state.get("tool_output", "")
    query = state["query"]

    if context:
        prompt_content = SYNTHESIS_PROMPT.format(context=context[:3000], query=query)
    else:
        prompt_content = f"Answer this question directly and helpfully: {query}"

    messages = [HumanMessage(content=prompt_content)]
    response = _invoke_with_retry(SYNTHESIS_LLM, messages)

    latency_ms = (time.perf_counter() - start) * 1000
    inp, out = extract_token_usage(response)
    cost = estimate_cost(inp, out)

    # Also add router cost (estimate: ~500 input + 5 output tokens for router)
    router_cost = estimate_cost(500, 5)
    total_cost = cost + router_cost

    print(f"   ✍️  Answer ({latency_ms:.0f}ms) | {inp}/{out} tokens | ${cost:.6f}")

    return {
        **state,
        "answer": get_text(response.content),
        "answer_latency_ms": latency_ms,
        "answer_input_tokens": inp,
        "answer_output_tokens": out,
        "total_cost_usd": total_cost,
    }


# ─────────────────────────────────────────────
# Routing Function (conditional edges)
# ─────────────────────────────────────────────

def get_next_node(state: AgentState) -> str:
    """
    Conditional edge function: maps the route string to the next node name.
    This is how LangGraph implements branching — it's explicit, not hidden.
    """
    route = state.get("route", "wikipedia")
    node_map = {
        "wikipedia": "wikipedia_node",
        "arxiv": "arxiv_node",
        "langsmith_search": "langsmith_node",
        "direct_answer": "direct_answer_node",
    }
    return node_map.get(route, "wikipedia_node")


# ─────────────────────────────────────────────
# Build the Graph
# ─────────────────────────────────────────────

def build_graph():
    """
    Construct and compile the LangGraph StateGraph.
    
    Graph topology:
      START → router_node → [tool_node] → answer_node → END
    """
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("router_node", router_node)
    builder.add_node("wikipedia_node", wikipedia_node)
    builder.add_node("arxiv_node", arxiv_node)
    builder.add_node("langsmith_node", langsmith_node)
    builder.add_node("direct_answer_node", direct_answer_node)
    builder.add_node("answer_node", answer_node)

    # Entry point
    builder.add_edge(START, "router_node")

    # Conditional routing from router to tool nodes
    builder.add_conditional_edges(
        "router_node",
        get_next_node,
        {
            "wikipedia_node": "wikipedia_node",
            "arxiv_node": "arxiv_node",
            "langsmith_node": "langsmith_node",
            "direct_answer_node": "direct_answer_node",
        }
    )

    # All tool nodes → answer_node
    builder.add_edge("wikipedia_node", "answer_node")
    builder.add_edge("arxiv_node", "answer_node")
    builder.add_edge("langsmith_node", "answer_node")
    builder.add_edge("direct_answer_node", "answer_node")

    # answer_node → END
    builder.add_edge("answer_node", END)

    return builder.compile()


# Module-level compiled graph
graph = build_graph()


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def run_query(query: str, verbose: bool = True) -> dict:
    """
    Run a single query through the graph.
    
    Returns a dict with:
      - query, route, answer
      - tool_output (raw retrieval)
      - latency and cost metrics
    """
    if verbose:
        print(f"\n{'─'*60}")
        print(f"🔍 Query: {query}")

    initial_state: AgentState = {
        "query": query,
        "route": "",
        "tool_output": "",
        "answer": "",
        "router_latency_ms": 0.0,
        "router_input_tokens": 0,
        "router_output_tokens": 0,
        "tool_latency_ms": 0.0,
        "answer_latency_ms": 0.0,
        "answer_input_tokens": 0,
        "answer_output_tokens": 0,
        "total_cost_usd": 0.0,
        "error": None,
    }

    try:
        final_state = graph.invoke(initial_state)
    except Exception as e:
        if verbose:
            print(f"❌ Error: {e}")
        return {
            "query": query,
            "route": "error",
            "answer": f"Error: {e}",
            "tool_output": "",
            "total_latency_ms": 0,
            "total_cost_usd": 0,
            "error": str(e),
        }

    total_latency = (
        final_state["router_latency_ms"]
        + final_state["tool_latency_ms"]
        + final_state["answer_latency_ms"]
    )

    result = {
        "query": query,
        "route": final_state["route"],
        "answer": final_state["answer"],
        "tool_output": final_state.get("tool_output", ""),
        "router_latency_ms": final_state["router_latency_ms"],
        "tool_latency_ms": final_state["tool_latency_ms"],
        "answer_latency_ms": final_state["answer_latency_ms"],
        "total_latency_ms": total_latency,
        "answer_input_tokens": final_state["answer_input_tokens"],
        "answer_output_tokens": final_state["answer_output_tokens"],
        "total_cost_usd": final_state["total_cost_usd"],
        "error": final_state.get("error"),
    }

    if verbose:
        print(f"\n📝 Answer: {final_state['answer'][:300]}..." if len(final_state["answer"]) > 300 else f"\n📝 Answer: {final_state['answer']}")
        print(f"\n   Total: {total_latency:.0f}ms | ${final_state['total_cost_usd']:.6f}")
        print(f"{'─'*60}")

    return result


# ─────────────────────────────────────────────
# CLI Smoke Test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("🚀 Multi-Source RAG — LangGraph Demo")
    print("=" * 60)

    test_queries = [
        "Who is Marie Curie?",
        "Recent research on attention mechanisms in large language models",
        "What is LangSmith used for?",
        "What is 15 * 7?",
    ]

    results = []
    for q in test_queries:
        r = run_query(q)
        results.append(r)

    print("\n\n📊 Summary:")
    print(f"{'Query':<45} {'Route':<18} {'Latency':>8} {'Cost':>10}")
    print("─" * 85)
    for r in results:
        q_short = r["query"][:44]
        print(f"{q_short:<45} {r['route']:<18} {r['total_latency_ms']:>7.0f}ms {r['total_cost_usd']:>9.6f}$")
