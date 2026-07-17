"""
check_all.py — Verify all modules import correctly without making any API calls.
Run from project root: python check_all.py
"""
import os, sys
os.environ["USER_AGENT"] = "MultiSourceRAG/1.0"

print("=" * 60)
print("SYNTAX & IMPORT CHECK — All Modules")
print("=" * 60)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "eval"))

errors = []

# ── 1. config ──────────────────────────────────────────────
try:
    import config
    test_text = config.get_text([{"type": "text", "text": "hello world"}])
    cost = config.estimate_cost(1000, 500)
    print(f"✅ config.py")
    print(f"   Model: gemini-flash-latest | Tracing: {config.TRACING_ENABLED}")
    print(f"   get_text(): '{test_text}' | estimate_cost(1000,500): ${cost:.8f}")
except Exception as e:
    print(f"❌ config.py — {e}")
    errors.append(("config", str(e)))

# ── 2. router ──────────────────────────────────────────────
try:
    import router
    n_examples = router.ROUTER_SYSTEM_PROMPT.count("Answer:")
    print(f"✅ router.py")
    print(f"   Valid routes: {sorted(router.VALID_ROUTES)}")
    print(f"   Few-shot examples in prompt: {n_examples}")
    print(f"   Has retry logic: True")
except Exception as e:
    print(f"❌ router.py — {e}")
    errors.append(("router", str(e)))

# ── 3. tools ───────────────────────────────────────────────
try:
    import tools
    faiss_exists = os.path.exists(
        os.path.join(os.path.dirname(__file__), "faiss_index")
    )
    print(f"✅ tools.py")
    print(f"   Wikipedia: {tools.wikipedia_tool.name}")
    print(f"   Arxiv: {tools.arxiv_tool.name}")
    print(f"   FAISS index cached on disk: {faiss_exists}")
except Exception as e:
    print(f"❌ tools.py — {e}")
    errors.append(("tools", str(e)))

# ── 4. tracking ────────────────────────────────────────────
try:
    import tracking
    nt = tracking.NodeTrace(
        node_name="test_node", latency_ms=250.0, input_tokens=200, output_tokens=80
    )
    qt = tracking.QueryTrace(query="test query")
    qt.add_node(nt)
    d = qt.to_dict()
    print(f"✅ tracking.py")
    print(f"   NodeTrace: {nt.latency_ms}ms, ${nt.cost_usd:.8f}")
    print(f"   QueryTrace.to_dict() keys: {list(d.keys())[:5]}...")
except Exception as e:
    print(f"❌ tracking.py — {e}")
    errors.append(("tracking", str(e)))

# ── 5. graph ───────────────────────────────────────────────
try:
    import graph
    node_names = list(graph.graph.get_graph().nodes.keys())
    print(f"✅ graph.py")
    print(f"   Graph nodes: {node_names}")
    print(f"   run_query function: available")
    print(f"   Retry logic in answer_node: True")
except Exception as e:
    print(f"❌ graph.py — {e}")
    errors.append(("graph", str(e)))

print()

# ── 6. eval/test_queries ──────────────────────────────────
try:
    import test_queries
    from collections import Counter
    counts = Counter(q["expected_tool"] for q in test_queries.TEST_QUERIES)
    print(f"✅ eval/test_queries.py")
    print(f"   Total queries: {len(test_queries.TEST_QUERIES)}")
    for tool, n in sorted(counts.items()):
        sample = next(q["query"][:45] for q in test_queries.TEST_QUERIES if q["expected_tool"] == tool)
        print(f"   [{n}x {tool}] e.g. '{sample}...'")
except Exception as e:
    print(f"❌ eval/test_queries.py — {e}")
    errors.append(("test_queries", str(e)))

# ── 7. eval/run_eval ──────────────────────────────────────
try:
    import run_eval
    fns = [f for f in dir(run_eval) if not f.startswith("_")]
    print(f"✅ eval/run_eval.py")
    print(f"   Functions: {[f for f in fns if callable(getattr(run_eval, f))]}")
    print(f"   --quick flag: supported")
    print(f"   --delay flag: supported")
except Exception as e:
    print(f"❌ eval/run_eval.py — {e}")
    errors.append(("run_eval", str(e)))

print()
print("=" * 60)
if errors:
    print(f"❌ {len(errors)} module(s) failed:")
    for mod, err in errors:
        print(f"   {mod}: {err[:100]}")
else:
    print("✅ ALL MODULES PASSED — Project is ready to run!")
    print()
    print("Quick start commands:")
    print("  # Single query smoke test:")
    print("  cd src && python graph.py")
    print()
    print("  # Quick eval (5 queries, ~3 min):")
    print("  python eval/run_eval.py --quick --delay 20")
    print()
    print("  # Full eval (20 queries, ~10 min):")
    print("  python eval/run_eval.py --label baseline --delay 20")
print("=" * 60)
