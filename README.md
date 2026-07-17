# Multi-Source RAG Agent — LangGraph + Groq + Gemini Embeddings

> **Production-Ready Multi-Source RAG Agent** — Rebuilt from a basic `AgentExecutor` prototype into a modular, inspectable, and production-grade LangGraph `StateGraph` architecture. Powered by **Groq Llama 3.3 70B** for ultra-fast, high-accuracy routing and answer synthesis, and **Gemini Embeddings** with a local FAISS vector store.

---

## 📊 Performance Results (Groq Llama 3.3 70B Baseline)

The agent was evaluated using a structured evaluation suite of **20 labeled test cases** covering Wikipedia, Arxiv, LangSmith documentation, and direct conversation.

| Metric | Target | Result (Gemini 1.5) | Result (Groq Llama 3.3 70B) |
|--------|--------|---------------------|-----------------------------|
| **Overall Routing Accuracy** | >90.0% | ~85.0% (Rate limited) | **100.0% (20/20)** ✅ |
| **Average Latency** | <5000ms | ~15000ms | **4483ms** ✅ |
| **Average Cost per Query** | <$0.01 | ~$0.0001 | **$0.000299** ✅ |
| **Total Evaluation Cost (20 Queries)**| <$0.10 | ~$0.002 | **$0.005979** ✅ |
| **Retrieval Quality (Keyword Hit Rate)**| >20.0%| N/A | **22.7%** ✅ |

### Per-Tool Routing Breakdown

* **Wikipedia**: Recall **100%** | Precision **100%** (Avg Latency: ~9.6s)
* **Arxiv**: Recall **100%** | Precision **100%** (Avg Latency: ~2.8s)
* **LangSmith Docs**: Recall **100%** | Precision **100%** (Avg Latency: ~3.8s)
* **Direct Answer**: Recall **100%** | Precision **100%** (Avg Latency: ~1.6s)

---

## 🏗️ Architecture

```
                      User Query
                          │
                          ▼
             ┌─────────────────────────┐
             │       router_node       │
             │ (Llama 3.3 70B, temp=0) │
             │  Few-Shot Query Router  │
             └────────────┬────────────┘
                          │
         ┌────────────────┼───────────────┬────────────────┐
         ▼                ▼               ▼                ▼
   wikipedia_node     arxiv_node     langsmith_node  direct_answer_node
  (Wikipedia API)    (Arxiv API)    (FAISS Retriever) (Bypasses Tools)
         │                │               │                │
         └────────────────┴───────────────┴────────────────┘
                          │
                          ▼
                  ┌───────────────┐
                  │  answer_node  │
                  │ (Llama 3.3)   │
                  │  Synthesis    │
                  └───────┬───────┘
                          │
                          ▼
                     Final Answer
```

### Why LangGraph over LangChain's AgentExecutor?
1. **Explicit Inspectable State**: The entire state of the agent (`query`, `route`, `tool_output`, `answer`) flows through nodes as an immutable dictionary. You can view the graph's contents at every single step.
2. **Node-Level Tracing**: LangSmith shows each node as a separate span in the trace tree, giving deep visibility into execution latency per component.
3. **Deterministic Branching**: Branching is handled by explicit conditional edges in Python code, rather than letting the LLM guess the next tool inside an unstructured loop.
4. **Resiliency**: If a tool fails, LangGraph easily recovers by capturing the exception and routing to synthesis or using fallback parameters.

---

## 🛠️ Features

* **3 Retrieval Tools**: Wikipedia API, Arxiv API, and a local FAISS vector database built from scraped LangSmith documentation.
* **Direct Answer Path**: Bypasses retrieval entirely for simple math, greetings, and conversational follow-ups. Reduces latency by **~65%** and saves input tokens.
* **Rate-Limit-Resilient Execution**: Includes built-in exponential backoff retry logic (`30s / 60s / 120s`) for LLM invocations and custom inter-query spacing in the evaluation runner.
* **Cost & Latency Instrumentation**: Every query logs per-node latencies, input/output tokens, and estimated USD cost using real-world API pricing tiers.

---

## 🚀 Quick Start

### 1. Clone & Setup Environment
```bash
cd Multi-SourceRAG
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Credentials (`.env`)
Create or edit your `.env` file at the root:
```env
GOOGLE_API_KEY=your_gemini_api_key      # Kept for GoogleEmbeddings
GROQ_API_KEY=your_groq_api_key          # Used for LLM and Synthesis LLM
LANGCHAIN_TRACING_V2=true               # Set to true for LangSmith observability
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=multi-source-rag
```

### 3. Verify Code Setup
Run the import and syntax check tool:
```bash
python check_all.py
```

### 4. Run the Interactive Notebook
Start the demo notebook to visualize the LangGraph topology and test interactively:
```bash
jupyter notebook notebooks/demo.ipynb
```

### 5. Execute the Evaluation Harness
Run the evaluation runner:
```bash
# Run first 5 queries (Quick Smoke Test)
python eval/run_eval.py --quick --delay 5

# Run full 20-query evaluation
python eval/run_eval.py --label groq_baseline --delay 5
```

---

## 📖 Directory Structure

```
Multi-SourceRAG/
├── .env                    # API keys (Groq, Gemini, LangSmith)
├── requirements.txt        # All dependencies
├── check_all.py            # Diagnostic script to verify imports & structure
├── src/
│   ├── config.py           # Model initializations (Groq Llama 3.3 + Gemini Embeddings)
│   ├── tools.py            # Tool wrapper configurations (Wiki, Arxiv, FAISS)
│   ├── router.py           # LLM query classifier using few-shot prompt
│   ├── graph.py            # LangGraph StateGraph pipeline
│   └── tracking.py         # Node-level timing and cost logging utilities
├── eval/
│   ├── test_queries.py     # 20 labeled test cases
│   └── run_eval.py         # Evaluation harness with rate-limit compliance
├── results/                # Evaluation output JSON files
├── notebooks/
│   └── demo.ipynb          # Interactive visualization & testing notebook
└── faiss_index/            # Local vector database index directory
```

---

## 💡 Interview Talking Points

1. **"Why LangGraph?"**
   * *Talking Point*: Transitioning from `AgentExecutor` to `StateGraph` gives us control over the agent loop. It decouples LLM classification from tool execution, makes token/cost tracking straightforward, and opens the door to adding checkpoints for conversation memory.
2. **"Why Llama 3.3 70B via Groq?"**
   * *Talking Point*: Replaced Gemini API for LLM operations. Using Groq reduced query routing classification times to **<1s** (down from ~3s) and synthesis times to **~1.5s**, while achieving **100% routing accuracy** due to its superior system prompt following capabilities.
3. **"How did you improve routing accuracy?"**
   * *Talking Point*: Zero-shot models initially misclassified border-case queries (e.g. general science vs paper reviews). Engineered a few-shot prompt system with 14 examples to establish clear decision boundaries, resulting in zero routing errors during evaluation.
4. **"Cost Optimization Strategy"**
   * *Talking Point*: Implemented a `direct_answer` route to identify queries that do not require external search (like basic math or chit-chat). Bypassing retrieval and document-based synthesis reduced cost by **70%** and latency by **60%** for those queries.
