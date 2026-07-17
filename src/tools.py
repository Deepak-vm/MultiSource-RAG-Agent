"""
tools.py — Three retrieval tools: Wikipedia, Arxiv, FAISS (LangSmith docs)
Each tool is a standard LangChain Tool object, ready to call by name.
"""
import os
import arxiv as arxiv_lib
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.tools import Tool

# Local config (only imports embeddings, avoids re-initializing LLM)
from config import EMBEDDINGS

# ─────────────────────────────────────────────
# 1. Wikipedia Tool
# ─────────────────────────────────────────────
_wiki_api = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=500)
wikipedia_tool = WikipediaQueryRun(api_wrapper=_wiki_api)
wikipedia_tool.name = "wikipedia"
wikipedia_tool.description = (
    "Search Wikipedia for factual, encyclopedic information about people, places, "
    "companies, historical events, concepts, or general knowledge. "
    "Use this for 'what is', 'who is', 'tell me about' type questions."
)

# ─────────────────────────────────────────────
# 2. Arxiv Tool (using new arxiv.Client() API)
# ─────────────────────────────────────────────
def _arxiv_search(query: str) -> str:
    """Search Arxiv using the new Client-based API (arxiv>=2.1)."""
    try:
        client = arxiv_lib.Client()
        search = arxiv_lib.Search(query=query, max_results=3)
        results = list(client.results(search))
        if not results:
            return "No Arxiv papers found for this query."
        parts = []
        for r in results:
            parts.append(
                f"Title: {r.title}\n"
                f"Authors: {', '.join(a.name for a in r.authors[:3])}\n"
                f"Published: {r.published.strftime('%Y-%m-%d')}\n"
                f"Summary: {r.summary[:400]}\n"
                f"URL: {r.entry_id}"
            )
        return "\n\n---\n\n".join(parts)
    except Exception as e:
        return f"Arxiv search error: {e}"

arxiv_tool = Tool(
    name="arxiv",
    func=_arxiv_search,
    description=(
        "Search Arxiv for scientific papers and recent research. "
        "Use this for questions about recent research, machine learning papers, "
        "physics, mathematics, computer science, or any academic/scientific topic. "
        "Good for 'latest research on', 'paper about', 'study on' type questions."
    ),
)

# ─────────────────────────────────────────────
# 3. FAISS Retriever Tool (LangSmith docs)
# ─────────────────────────────────────────────
_FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "faiss_index")
_SCRAPE_URLS = [
    "https://docs.smith.langchain.com/",
    "https://docs.smith.langchain.com/concepts",
    "https://docs.smith.langchain.com/how_to_guides",
]

def _build_faiss_retriever():
    """Build or load FAISS index from LangSmith documentation."""
    if os.path.exists(_FAISS_INDEX_PATH):
        print("Loading cached FAISS index...")
        vectordb = FAISS.load_local(
            _FAISS_INDEX_PATH,
            EMBEDDINGS,
            allow_dangerous_deserialization=True,
        )
    else:
        print(" Building FAISS index from LangSmith docs (first run only)...")
        all_docs = []
        for url in _SCRAPE_URLS:
            try:
                loader = WebBaseLoader(url)
                docs = loader.load()
                all_docs.extend(docs)
                print(f"   ✓ Scraped: {url}")
            except Exception as e:
                print(f"   ✗ Failed: {url} — {e}")

        if not all_docs:
            raise RuntimeError("Could not scrape any LangSmith docs. Check your internet connection.")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(all_docs)
        print(f"   ✓ {len(chunks)} chunks created")

        vectordb = FAISS.from_documents(chunks, EMBEDDINGS)
        vectordb.save_local(_FAISS_INDEX_PATH)
        print(f"   ✓ FAISS index saved to {_FAISS_INDEX_PATH}")

    return vectordb.as_retriever(search_kwargs={"k": 3})


def get_langsmith_retriever_tool():
    """Build FAISS retriever and wrap it as a LangChain tool."""
    retriever = _build_faiss_retriever()
    return create_retriever_tool(
        retriever,
        name="langsmith_search",
        description=(
            "Search the LangSmith documentation for information about LangSmith features, "
            "tracing, evaluation, monitoring, and LangChain observability tools. "
            "Use this for questions specifically about LangSmith, LangChain tracing, "
            "or LLM observability."
        ),
    )


def get_all_tools():
    """Return all three tools as a list."""
    langsmith_tool = get_langsmith_retriever_tool()
    return [wikipedia_tool, arxiv_tool, langsmith_tool]


if __name__ == "__main__":
    print("Testing tools...")
    print(f"  Wikipedia: {wikipedia_tool.name}")
    print(f"  Arxiv: {arxiv_tool.name}")
    ls_tool = get_langsmith_retriever_tool()
    print(f"  LangSmith: {ls_tool.name}")
    print("All tools initialized")
