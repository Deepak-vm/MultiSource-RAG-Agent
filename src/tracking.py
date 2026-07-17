"""
tracking.py — Cost and latency tracking for every node in the graph.

This module provides:
  - A timer context manager for measuring node latency
  - Token/cost extraction from Gemini LLM responses
  - A per-query log accumulator
  - Summary statistics printer

Design goal: every graph run produces a TrackingRecord with full cost/latency breakdown.
This is what "production engineer" thinking looks like on a resume.
"""
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
import os

from config import estimate_cost, COST_PER_INPUT_TOKEN, COST_PER_OUTPUT_TOKEN


@dataclass
class NodeTrace:
    """Timing and token data for a single graph node."""
    node_name: str
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class QueryTrace:
    """Full trace for a single user query through the graph."""
    query: str
    route: str = ""
    tool_output_chars: int = 0
    answer_chars: int = 0
    total_latency_ms: float = 0.0
    node_traces: List[NodeTrace] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    error: Optional[str] = None

    def add_node(self, trace: NodeTrace):
        self.node_traces.append(trace)
        self.total_latency_ms += trace.latency_ms
        self.total_input_tokens += trace.input_tokens
        self.total_output_tokens += trace.output_tokens
        self.total_cost_usd += trace.cost_usd

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "route": self.route,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "tool_output_chars": self.tool_output_chars,
            "answer_chars": self.answer_chars,
            "node_traces": [
                {
                    "node": t.node_name,
                    "latency_ms": round(t.latency_ms, 2),
                    "input_tokens": t.input_tokens,
                    "output_tokens": t.output_tokens,
                    "cost_usd": round(t.cost_usd, 6),
                }
                for t in self.node_traces
            ],
            "error": self.error,
        }


@contextmanager
def timed_node(node_name: str):
    """Context manager that returns a NodeTrace populated with timing."""
    trace = NodeTrace(node_name=node_name)
    start = time.perf_counter()
    try:
        yield trace
    finally:
        trace.latency_ms = (time.perf_counter() - start) * 1000


def extract_token_usage(llm_response) -> tuple[int, int]:
    """
    Extract (input_tokens, output_tokens) from an LLM response.
    Handles both Groq (dict) and Gemini (object) usage_metadata formats.
    Returns (0, 0) if metadata is unavailable.
    """
    try:
        usage = llm_response.usage_metadata
        if usage:
            # Groq returns a dict: {'input_tokens': N, 'output_tokens': N}
            if isinstance(usage, dict):
                inp = usage.get("input_tokens", 0)
                out = usage.get("output_tokens", 0)
            else:
                # Gemini returns an object with attributes
                inp = getattr(usage, "input_tokens", 0) or getattr(usage, "prompt_token_count", 0)
                out = getattr(usage, "output_tokens", 0) or getattr(usage, "candidates_token_count", 0)
            return int(inp), int(out)
    except Exception:
        pass
    return 0, 0


def build_node_trace(node_name: str, latency_ms: float, llm_response=None) -> NodeTrace:
    """Build a NodeTrace from a completed LLM call."""
    inp, out = extract_token_usage(llm_response) if llm_response else (0, 0)
    cost = estimate_cost(inp, out)
    return NodeTrace(
        node_name=node_name,
        latency_ms=latency_ms,
        input_tokens=inp,
        output_tokens=out,
        cost_usd=cost,
    )


def print_trace_summary(trace: QueryTrace):
    """Print a formatted summary of a single query trace."""
    print(f"\n{'─'*60}")
    print(f"📊 Query Trace: '{trace.query[:60]}...' " if len(trace.query) > 60 else f"📊 Query Trace: '{trace.query}'")
    print(f"   Route: {trace.route}")
    print(f"   Total latency: {trace.total_latency_ms:.0f}ms")
    print(f"   Tokens: {trace.total_input_tokens} in / {trace.total_output_tokens} out")
    print(f"   Estimated cost: ${trace.total_cost_usd:.6f}")
    print(f"\n   Node breakdown:")
    for nt in trace.node_traces:
        print(f"     [{nt.node_name}] {nt.latency_ms:.0f}ms | {nt.input_tokens}/{nt.output_tokens} tokens | ${nt.cost_usd:.6f}")
    if trace.error:
        print(f"   ⚠️  Error: {trace.error}")
    print(f"{'─'*60}")


def print_aggregate_stats(traces: List[QueryTrace]):
    """Print aggregate statistics across multiple query traces."""
    if not traces:
        print("No traces to summarize.")
        return

    valid = [t for t in traces if not t.error]
    n = len(valid)

    if n == 0:
        print("All traces had errors.")
        return

    avg_latency = sum(t.total_latency_ms for t in valid) / n
    avg_cost = sum(t.total_cost_usd for t in valid) / n
    total_cost = sum(t.total_cost_usd for t in valid)
    avg_tokens_in = sum(t.total_input_tokens for t in valid) / n
    avg_tokens_out = sum(t.total_output_tokens for t in valid) / n

    # Per-route breakdown
    routes: Dict[str, list] = {}
    for t in valid:
        routes.setdefault(t.route, []).append(t)

    print(f"\n{'═'*60}")
    print(f"📈 Aggregate Stats ({n} queries)")
    print(f"{'═'*60}")
    print(f"  Avg latency:      {avg_latency:.0f}ms")
    print(f"  Avg cost:         ${avg_cost:.6f}")
    print(f"  Total cost:       ${total_cost:.5f}")
    print(f"  Avg tokens in:    {avg_tokens_in:.0f}")
    print(f"  Avg tokens out:   {avg_tokens_out:.0f}")
    print(f"\n  Per-route breakdown:")
    for route, route_traces in sorted(routes.items()):
        rl = sum(t.total_latency_ms for t in route_traces) / len(route_traces)
        rc = sum(t.total_cost_usd for t in route_traces) / len(route_traces)
        print(f"    [{route}] {len(route_traces)} queries | avg {rl:.0f}ms | avg ${rc:.6f}")
    print(f"{'═'*60}")


def save_traces(traces: List[QueryTrace], path: str):
    """Save all traces to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump([t.to_dict() for t in traces], f, indent=2)
    print(f"💾 Saved {len(traces)} traces to {path}")
