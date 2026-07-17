"""
run_eval.py — Evaluation harness for Multi-Source RAG.

Measures:
  1. Tool-selection accuracy (routing): did the router pick the right tool?
  2. Per-tool precision and recall
  3. Retrieval quality: does the tool_output contain expected keywords?
  4. Latency per tool (avg, min, max)
  5. Cost per query and total run cost

Produces:
  - Console report with before → after comparison
  - results/eval_results.json with full trace data

Usage:
  # Full eval (20 queries) — will take ~10 min with free-tier rate limits
  python eval/run_eval.py

  # Quick eval (first 5 queries only, useful for smoke test)
  python eval/run_eval.py --quick

  # Custom inter-query delay (seconds)
  python eval/run_eval.py --delay 20
"""
import sys, os
import json
import time
import random
from collections import defaultdict
from datetime import datetime

# Path setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from graph import run_query
from test_queries import TEST_QUERIES


# ─────────────────────────────────────────────
# Retrieval Quality Check
# ─────────────────────────────────────────────

def check_retrieval_quality(tool_output: str, keywords: list, route: str) -> dict:
    """
    Check if the retrieved content contains expected keywords.
    Only applies to wikipedia, arxiv, and langsmith_search routes.
    Returns hit_rate (0–1).
    """
    if route == "direct_answer" or not tool_output:
        return {"applicable": False, "hit_rate": None, "hits": [], "misses": []}

    tool_output_lower = tool_output.lower()
    hits = [kw for kw in keywords if kw.lower() in tool_output_lower]
    misses = [kw for kw in keywords if kw.lower() not in tool_output_lower]
    hit_rate = len(hits) / len(keywords) if keywords else 0.0

    return {
        "applicable": True,
        "hit_rate": hit_rate,
        "hits": hits,
        "misses": misses,
    }


# ─────────────────────────────────────────────
# Rate-Limit-Aware Query Runner
# ─────────────────────────────────────────────

def run_query_safe(query: str, delay_seconds: int = 20, max_total_wait: int = 120) -> dict:
    """
    Run a single query with adaptive retry on rate limits.
    Waits `delay_seconds` between attempts.
    """
    attempt = 0
    total_waited = 0

    while total_waited < max_total_wait:
        result = run_query(query, verbose=False)

        if result.get("route") == "error":
            err = result.get("error", "")
            if "429" in str(err) or "RESOURCE_EXHAUSTED" in str(err):
                wait = delay_seconds + random.uniform(0, 5)
                print(f"   ⏳ API rate limit on query. Waiting {wait:.0f}s before retry...")
                time.sleep(wait)
                total_waited += wait
                attempt += 1
                continue
            # Non-rate-limit error — return as-is
            return result

        return result  # Success

    # Exhausted all retries
    print(f"   ⚠️  Max wait time exceeded for query. Recording as error.")
    return {
        "query": query, "route": "error", "answer": "Rate limit exhausted",
        "tool_output": "", "total_latency_ms": 0, "total_cost_usd": 0,
        "error": "Rate limit exhausted after max retries"
    }


# ─────────────────────────────────────────────
# Main Evaluation Runner
# ─────────────────────────────────────────────

def run_evaluation(queries=None, run_label="eval", save_results=True, delay_seconds=20):
    """
    Run the evaluation harness over all test queries.

    Args:
        queries: list of query dicts (defaults to TEST_QUERIES)
        run_label: label for this run (e.g., "before_fewshot", "after_fewshot")
        save_results: whether to save JSON results
        delay_seconds: pause between queries (helps with rate limits)

    Returns:
        dict with full metrics
    """
    if queries is None:
        queries = TEST_QUERIES

    total_time_est = len(queries) * (delay_seconds + 15) / 60
    print(f"\n{'='*70}")
    print(f"🧪 Multi-Source RAG Evaluation — {run_label}")
    print(f"   Queries: {len(queries)} | Delay: {delay_seconds}s | Est. time: ~{total_time_est:.0f}min")
    print(f"   Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*70}\n")

    results = []
    per_tool_data = defaultdict(lambda: {"correct": 0, "total": 0, "predicted_for": defaultdict(int)})
    tool_latencies = defaultdict(list)
    tool_costs = defaultdict(list)
    retrieval_quality_scores = []

    for i, test_case in enumerate(queries):
        # Rate-limit-aware pause (skip before first query)
        if i > 0:
            print(f"   ⌛ Waiting {delay_seconds}s (rate limit courtesy pause)...")
            time.sleep(delay_seconds)

        query = test_case["query"]
        expected = test_case["expected_tool"]
        keywords = test_case.get("keywords", [])

        q_display = f"'{query[:52]}...'" if len(query) > 55 else f"'{query}'"
        print(f"[{i+1:02d}/{len(queries)}] {q_display}")
        print(f"        Expected: {expected}")

        result = run_query_safe(query, delay_seconds=delay_seconds)
        predicted = result["route"]

        # Routing accuracy
        is_correct = (predicted == expected)
        per_tool_data[expected]["total"] += 1
        per_tool_data[expected]["predicted_for"][predicted] += 1
        if is_correct:
            per_tool_data[expected]["correct"] += 1

        # Retrieval quality
        rq = check_retrieval_quality(result.get("tool_output", ""), keywords, predicted)

        # Track latency / cost
        tool_latencies[predicted].append(result["total_latency_ms"])
        tool_costs[predicted].append(result["total_cost_usd"])
        if rq["applicable"]:
            retrieval_quality_scores.append(rq["hit_rate"])

        status = "✅ RIGHT" if is_correct else "❌ WRONG"
        rq_str = f"| retrieval: {rq['hit_rate']:.0%}" if rq["applicable"] else ""
        print(f"        Predicted: {predicted} {status} | {result['total_latency_ms']:.0f}ms | ${result['total_cost_usd']:.6f} {rq_str}")

        results.append({
            "query": query,
            "expected_tool": expected,
            "predicted_tool": predicted,
            "correct": is_correct,
            "total_latency_ms": result["total_latency_ms"],
            "total_cost_usd": result["total_cost_usd"],
            "answer": result["answer"][:500] if result.get("answer") else "",
            "tool_output_chars": len(result.get("tool_output", "")),
            "retrieval_quality": rq,
            "notes": test_case.get("notes", ""),
        })

    # ─── Compute Metrics ───────────────────────────────────────
    n = len(results)
    correct_count = sum(1 for r in results if r["correct"])
    overall_accuracy = correct_count / n

    print(f"\n{'='*70}")
    print(f"📊 EVALUATION RESULTS — {run_label}")
    print(f"{'='*70}")
    print(f"\n  Overall Routing Accuracy: {correct_count}/{n} = {overall_accuracy:.1%}")

    print(f"\n  Per-Tool Breakdown:")
    print(f"  {'Tool':<22} {'Correct':>7} {'Total':>6} {'Precision':>10} {'Recall':>7} {'Avg Lat':>9} {'Avg Cost':>10}")
    print(f"  {'─'*72}")

    per_tool_metrics = {}
    for tool in ["wikipedia", "arxiv", "langsmith_search", "direct_answer"]:
        data = per_tool_data[tool]
        total_expected = data["total"]
        total_predicted = sum(per_tool_data[t]["predicted_for"][tool] for t in per_tool_data)
        tool_correct = data["correct"]

        recall = tool_correct / total_expected if total_expected > 0 else 0.0
        precision = tool_correct / total_predicted if total_predicted > 0 else 0.0
        avg_lat = (sum(tool_latencies[tool]) / len(tool_latencies[tool])) if tool_latencies[tool] else 0
        avg_cost = (sum(tool_costs[tool]) / len(tool_costs[tool])) if tool_costs[tool] else 0

        print(f"  {tool:<22} {tool_correct:>7} {total_expected:>6} {precision:>10.1%} {recall:>7.1%} {avg_lat:>8.0f}ms {avg_cost:>9.6f}$")
        per_tool_metrics[tool] = {
            "correct": tool_correct, "total": total_expected,
            "precision": precision, "recall": recall,
            "avg_latency_ms": avg_lat, "avg_cost_usd": avg_cost,
        }

    # Retrieval quality
    avg_hit_rate = None
    if retrieval_quality_scores:
        avg_hit_rate = sum(retrieval_quality_scores) / len(retrieval_quality_scores)
        print(f"\n  Retrieval Quality (keyword hit rate): {avg_hit_rate:.1%}")
        print(f"  Evaluated on {len(retrieval_quality_scores)} retrieval-based queries")

    # Overall cost/latency
    all_latencies = [r["total_latency_ms"] for r in results]
    all_costs = [r["total_cost_usd"] for r in results]
    print(f"\n  Latency:  avg={sum(all_latencies)/n:.0f}ms | min={min(all_latencies):.0f}ms | max={max(all_latencies):.0f}ms")
    print(f"  Cost:     avg=${sum(all_costs)/n:.6f} | total=${sum(all_costs):.5f}")
    print(f"\n  💡 Insight: direct_answer is fastest (no external API) | arxiv is slowest (network + content)")
    print(f"  💡 Cost driver: synthesis LLM (answer_node) >> router node")
    print(f"{'='*70}")

    # ─── Misses Analysis ─────────────────────────────────────────
    misses = [r for r in results if not r["correct"]]
    if misses:
        print(f"\n❌ Routing Misses ({len(misses)} total) — Use these to improve the router prompt:")
        print(f"  {'Query':<50} {'Expected':<20} {'Got'}")
        print(f"  {'─'*85}")
        for r in misses:
            print(f"  {r['query'][:49]:<50} {r['expected_tool']:<20} {r['predicted_tool']}")
        print()

    # ─── Save Results ────────────────────────────────────────────
    metrics = {
        "run_label": run_label,
        "timestamp": datetime.now().isoformat(),
        "overall_accuracy": overall_accuracy,
        "correct": correct_count,
        "total": n,
        "per_tool_metrics": per_tool_metrics,
        "avg_retrieval_hit_rate": avg_hit_rate,
        "avg_latency_ms": sum(all_latencies) / n,
        "avg_cost_usd": sum(all_costs) / n,
        "total_cost_usd": sum(all_costs),
        "results": results,
    }

    if save_results:
        results_dir = os.path.join(project_root, "results")
        os.makedirs(results_dir, exist_ok=True)
        fname = f"eval_{run_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out_path = os.path.join(results_dir, fname)
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"💾 Results saved to {out_path}")

    return metrics


# ─────────────────────────────────────────────
# Before → After Comparison
# ─────────────────────────────────────────────

def compare_runs(before: dict, after: dict):
    """Print a before → after comparison table."""
    print(f"\n{'='*70}")
    print("📈 BEFORE → AFTER IMPROVEMENT REPORT")
    print(f"{'='*70}")

    b_acc = before["overall_accuracy"]
    a_acc = after["overall_accuracy"]
    print(f"\n  Overall Accuracy: {b_acc:.1%} → {a_acc:.1%}  (Δ {a_acc - b_acc:+.1%})")

    print(f"\n  Per-Tool Recall:")
    print(f"  {'Tool':<22} {'Before':>10} {'After':>10} {'Delta':>8}")
    print(f"  {'─'*52}")
    for tool in ["wikipedia", "arxiv", "langsmith_search", "direct_answer"]:
        b = before["per_tool_metrics"].get(tool, {})
        a = after["per_tool_metrics"].get(tool, {})
        br = b.get("recall", 0)
        ar = a.get("recall", 0)
        print(f"  {tool:<22} {br:>10.1%} {ar:>10.1%} {ar-br:>+8.1%}")

    print(f"\n  Latency: {before.get('avg_latency_ms',0):.0f}ms → {after.get('avg_latency_ms',0):.0f}ms")
    print(f"  Cost:    ${before.get('avg_cost_usd',0):.6f} → ${after.get('avg_cost_usd',0):.6f}")
    print(f"{'='*70}")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Multi-Source RAG evaluation")
    parser.add_argument("--label", default="run", help="Label for this eval run")
    parser.add_argument("--no-save", action="store_true", help="Don't save JSON results")
    parser.add_argument("--quick", action="store_true", help="Run first 5 queries only (smoke test)")
    parser.add_argument("--delay", type=int, default=20,
                        help="Seconds to wait between queries (default: 20, for rate limit compliance)")
    args = parser.parse_args()

    queries = TEST_QUERIES[:5] if args.quick else TEST_QUERIES

    metrics = run_evaluation(
        queries=queries,
        run_label=args.label,
        save_results=not args.no_save,
        delay_seconds=args.delay,
    )

    print(f"\n✅ Evaluation complete!")
    print(f"   Overall accuracy: {metrics['overall_accuracy']:.1%}")
    print(f"   Avg cost/query:   ${metrics['avg_cost_usd']:.6f}")
    print(f"   Total eval cost:  ${metrics['total_cost_usd']:.5f}")
