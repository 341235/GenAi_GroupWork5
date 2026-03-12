import os
import pickle
import hashlib
import numpy as np
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from embeddings import NomicEmbeddings

load_dotenv()

CHROMA_PATH = "./chroma_python_docs"
BM25_CACHE_DIR = ".bm25_cache"
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_GUARD_MODEL = "llama-3.2-3b-preview"


def _load_or_build_bm25(docs: list, k: int) -> BM25Retriever:
    """
    Load a cached BM25Retriever from disk if the corpus hasn't changed,
    otherwise build from scratch and cache for the next startup.
    The cache key is an MD5 fingerprint of the document contents.
    """
    os.makedirs(BM25_CACHE_DIR, exist_ok=True)
    fingerprint = hashlib.md5(
        "".join(d.page_content[:120] for d in docs).encode()
    ).hexdigest()
    cache_path = os.path.join(BM25_CACHE_DIR, f"{fingerprint}_k{k}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    retriever = BM25Retriever.from_documents(docs, k=k)
    with open(cache_path, "wb") as f:
        pickle.dump(retriever, f)
    return retriever

# --- Topic routing keyword sets ---
_DATETIME_KEYWORDS   = {"datetime", "timedelta", "strftime", "strptime", "timezone"}
_ASYNCIO_KEYWORDS    = {"asyncio", "coroutine", "event loop", "async def", "await "}
_THREADING_KEYWORDS  = {"threading", "thread", "lock", "semaphore", "barrier"}
_NUMPY_KEYWORDS      = {"numpy", "ndarray", "np.", "linspace", "reshape", "broadcasting", "np.array", "np.zeros", "np.ones"}
_PANDAS_KEYWORDS     = {"pandas", "dataframe", "series", "groupby", "merge", "iloc", "loc", "read_csv", "to_csv", "pd."}
_MATPLOTLIB_KEYWORDS = {"matplotlib", "pyplot", "plt.", "scatter", "histogram", "subplot", "plt.plot", "plt.show"}
_SKLEARN_KEYWORDS    = {"sklearn", "scikit", "fit(", "predict(", "train_test_split", "pipeline", "classifier", "regressor", "cross_val"}
_REQUESTS_KEYWORDS   = {"requests", "requests.get", "requests.post", "response.json", "status_code", "http request"}
_ITERTOOLS_KEYWORDS  = {"itertools", "itertools.chain", "itertools.islice", "itertools.groupby",
                        "itertools.product", "itertools.combinations", "itertools.permutations",
                        "itertools.cycle", "itertools.repeat", "itertools.starmap",
                        "itertools.takewhile", "itertools.dropwhile", "itertools.count",
                        "itertools.zip_longest", "chain.from_iterable"}
_BUILTINS_KEYWORDS   = {"sorted(", "sorted()", "map(", "map()", "filter(", "filter()",
                        "enumerate()", "zip()", "range()", "len()", "type()", "isinstance(",
                        "hasattr(", "getattr(", "callable(", "iter(", "next(", "abs(",
                        "min(", "max(", "sum(", "round(", "repr(", "hash(", "id(", "vars(",
                        "dir(", "builtin function", "built-in function"}
_DATASTRUCTURES_KEYWORDS = {"list.append", "list.extend", "list.insert", "list.remove",
                             "list.pop", "list.sort", "list.reverse", "list.index",
                             "list.count", "list.clear", "list.copy",
                             "dict.keys", "dict.values", "dict.items", "dict.get",
                             "dict.update", "dict.pop", "dict.setdefault",
                             "set.add", "set.remove", "set.union", "set.intersection",
                             "set.difference", "set.discard"}

# --- Semantic routing: natural-language phrases per topic used to build centroids ---
_TOPIC_PHRASES = {
    "datetime":       ["parse a date string", "format datetime object",
                       "convert timezone python", "calculate time difference"],
    "asyncio":        ["write async await code", "run coroutines concurrently",
                       "create event loop", "asyncio task gather"],
    "threading":      ["run code in background thread", "synchronize threads with lock",
                       "thread pool executor"],
    "numpy":          ["create numpy array", "reshape array operations",
                       "matrix multiplication broadcasting"],
    "pandas":         ["read csv into dataframe", "filter rows in dataframe",
                       "groupby aggregate pandas"],
    "matplotlib":     ["plot a line chart", "create scatter plot",
                       "customize figure axes matplotlib"],
    "sklearn":        ["train a classifier", "split data train test",
                       "evaluate model accuracy cross validation"],
    "requests":       ["make http get request", "send post request with json",
                       "handle api response status code"],
    "itertools":      ["chain multiple iterables", "generate combinations permutations",
                       "lazy iteration tools"],
    # builtins and datastructures are semantically indistinguishable to the
    # embedding model — merged into one topic to eliminate mis-routing
    "python_basics":  ["append items to list", "access dict keys values",
                       "set union intersection difference",
                       "use sorted with key function", "map filter over list",
                       "enumerate zip builtin functions",
                       "remove duplicates from a list", "merge two dictionaries",
                       "apply a function to every element"],
}

_ROUTING_THRESHOLD = 0.65  # calibrated from calibrate_routing.py output


def _cosine_sim(a, b) -> float:
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _build_ensemble_reranker(vectorstore, all_docs, cross_encoder,
                              chroma_filter=None, doc_filter_fn=None):
    """
    Build a hybrid (BM25 + semantic MMR) + cross-encoder retriever.

    chroma_filter  — optional ChromaDB `where` dict applied to the semantic search.
    doc_filter_fn  — optional predicate applied to all_docs before building BM25.
    """
    semantic_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6, "fetch_k": 12,
            **({"filter": chroma_filter} if chroma_filter else {}),
        },
    )

    bm25_docs = [d for d in all_docs if doc_filter_fn(d)] if doc_filter_fn else all_docs
    bm25_retriever = _load_or_build_bm25(bm25_docs, k=6)

    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[0.2, 0.8],
    )
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=5)
    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=ensemble,
    )


def load_retriever():
    embeddings = NomicEmbeddings()
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
    )

    raw = vectorstore.get()
    all_docs = [
        Document(page_content=pc, metadata=m)
        for pc, m in zip(raw["documents"], raw["metadatas"])
    ]

    cross_encoder = HuggingFaceCrossEncoder(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    # Full-corpus retriever — used for most queries
    full_retriever = _build_ensemble_reranker(vectorstore, all_docs, cross_encoder)

    def _page_filter(fragment):
        return lambda d: fragment in d.metadata.get("source", "")

    def _domain_filter(domain):
        return lambda d: domain in d.metadata.get("source", "")

    # Python stdlib topic retrievers
    datetime_retriever  = _build_ensemble_reranker(
        vectorstore, all_docs, cross_encoder,
        chroma_filter={"source": {"$contains": "datetime.html"}},
        doc_filter_fn=_page_filter("datetime.html"),
    )
    threading_retriever = _build_ensemble_reranker(
        vectorstore, all_docs, cross_encoder,
        chroma_filter={"source": {"$contains": "threading.html"}},
        doc_filter_fn=_page_filter("threading.html"),
    )
    asyncio_retriever   = _build_ensemble_reranker(
        vectorstore, all_docs, cross_encoder,
        chroma_filter={"source": {"$contains": "asyncio.html"}},
        doc_filter_fn=_page_filter("asyncio.html"),
    )

    # Third-party library retrievers
    numpy_retriever = _build_ensemble_reranker(
        vectorstore, all_docs, cross_encoder,
        chroma_filter={"source": {"$contains": "numpy.org"}},
        doc_filter_fn=_domain_filter("numpy.org"),
    )
    pandas_retriever = _build_ensemble_reranker(
        vectorstore, all_docs, cross_encoder,
        chroma_filter={"source": {"$contains": "pandas.pydata.org"}},
        doc_filter_fn=_domain_filter("pandas.pydata.org"),
    )
    matplotlib_retriever = _build_ensemble_reranker(
        vectorstore, all_docs, cross_encoder,
        chroma_filter={"source": {"$contains": "matplotlib.org"}},
        doc_filter_fn=_domain_filter("matplotlib.org"),
    )
    sklearn_retriever = _build_ensemble_reranker(
        vectorstore, all_docs, cross_encoder,
        chroma_filter={"source": {"$contains": "scikit-learn.org"}},
        doc_filter_fn=_domain_filter("scikit-learn.org"),
    )
    requests_retriever = _build_ensemble_reranker(
        vectorstore, all_docs, cross_encoder,
        chroma_filter={"source": {"$contains": "requests.readthedocs.io"}},
        doc_filter_fn=_domain_filter("requests.readthedocs.io"),
    )

    # Merged: builtins (functions.html) + data structures (datastructures.html)
    # These topics are semantically adjacent — one retriever covers both.
    python_basics_retriever = _build_ensemble_reranker(
        vectorstore, all_docs, cross_encoder,
        chroma_filter={"source": {"$in": ["functions.html", "datastructures.html"]}},
        doc_filter_fn=lambda d: any(
            f in d.metadata.get("source", "")
            for f in ("functions.html", "datastructures.html")
        ),
    )
    # itertools module — itertools.html
    itertools_retriever = _build_ensemble_reranker(
        vectorstore, all_docs, cross_encoder,
        chroma_filter={"source": {"$contains": "itertools.html"}},
        doc_filter_fn=_page_filter("itertools.html"),
    )

    # Map topic names to their retrievers (used by semantic routing below)
    _topic_retriever_map = {
        "datetime":       datetime_retriever,
        "asyncio":        asyncio_retriever,
        "threading":      threading_retriever,
        "numpy":          numpy_retriever,
        "pandas":         pandas_retriever,
        "matplotlib":     matplotlib_retriever,
        "sklearn":        sklearn_retriever,
        "requests":       requests_retriever,
        "itertools":      itertools_retriever,
        "python_basics":  python_basics_retriever,
    }

    # Pre-compute per-topic centroids by averaging embeddings of representative phrases.
    # Uses embed_query (search_query: prefix) because the phrases are query-like.
    _topic_centroids = {
        topic: np.mean([embeddings.embed_query(p) for p in phrases], axis=0)
        for topic, phrases in _TOPIC_PHRASES.items()
    }

    def smart_retrieve(query: str) -> list:
        """Route the query to a topic retriever via keyword match, then semantic fallback."""
        q = query.lower()
        # Fast path: exact keyword match
        if any(kw in q for kw in _DATETIME_KEYWORDS):
            return datetime_retriever.invoke(query)
        if any(kw in q for kw in _THREADING_KEYWORDS):
            return threading_retriever.invoke(query)
        if any(kw in q for kw in _ASYNCIO_KEYWORDS):
            return asyncio_retriever.invoke(query)
        if any(kw in q for kw in _NUMPY_KEYWORDS):
            return numpy_retriever.invoke(query)
        if any(kw in q for kw in _PANDAS_KEYWORDS):
            return pandas_retriever.invoke(query)
        if any(kw in q for kw in _MATPLOTLIB_KEYWORDS):
            return matplotlib_retriever.invoke(query)
        if any(kw in q for kw in _SKLEARN_KEYWORDS):
            return sklearn_retriever.invoke(query)
        if any(kw in q for kw in _REQUESTS_KEYWORDS):
            return requests_retriever.invoke(query)
        if any(kw in q for kw in _ITERTOOLS_KEYWORDS):
            return itertools_retriever.invoke(query)
        if any(kw in q for kw in _DATASTRUCTURES_KEYWORDS):
            return python_basics_retriever.invoke(query)
        if any(kw in q for kw in _BUILTINS_KEYWORDS):
            return python_basics_retriever.invoke(query)
        # Semantic routing fallback: embed the query and score against all topic centroids
        q_vec = embeddings.embed_query(query)
        scores = {t: _cosine_sim(q_vec, c) for t, c in _topic_centroids.items()}
        above = [(t, s) for t, s in scores.items() if s >= _ROUTING_THRESHOLD]
        # Single confident topic → use its filtered retriever
        if len(above) == 1:
            return _topic_retriever_map[above[0][0]].invoke(query)
        # Multiple topics above threshold (e.g. "numpy array to pandas DataFrame")
        # or no confident match → search full corpus to avoid missing any topic
        return full_retriever.invoke(query)

    return smart_retrieve


def _multi_query_retrieve(query: str, smart_fn, llm) -> list:
    """
    Generate 2 alternative phrasings of the query with the LLM, then merge
    and deduplicate results from all variants for broader retrieval coverage.
    """
    variant_prompt = (
        f"Generate 2 alternative phrasings of this Python programming question "
        f"to improve document retrieval. Return only the 2 questions, one per line, no numbering.\n"
        f"Question: {query}"
    )
    try:
        variants_text = llm.invoke(variant_prompt).content.strip()
        variants = [v.strip() for v in variants_text.split("\n") if v.strip()][:2]
    except Exception:
        variants = []

    # Run queries sequentially — NomicBERT's rotary embedding cache is not thread-safe
    seen: set = set()
    merged: list = []
    for q in [query] + variants:
        for doc in smart_fn(q):
            key = hash(doc.page_content)
            if key not in seen:
                seen.add(key)
                merged.append(doc)
    return merged


_OFF_TOPIC_REPLY = (
    "I'm a Python documentation assistant — I can only help with questions about "
    "Python, NumPy, Pandas, Matplotlib, scikit-learn, or Requests. "
    "Could you ask me something related to one of those topics?"
)

_GUARD_PROMPT = (
    "You are a topic classifier. Answer only YES or NO.\n"
    "Is the following question related to Python programming, "
    "Python libraries (NumPy, Pandas, Matplotlib, scikit-learn, Requests), "
    "or general software development concepts?\n"
    "Question: {question}\n"
    "Answer (YES or NO):"
)


def _is_on_topic(question: str, guard_llm) -> bool:
    """Return True if the question is Python/programming related."""
    try:
        response = guard_llm.invoke(_GUARD_PROMPT.format(question=question)).content.strip().upper()
        return response.startswith("YES")
    except Exception:
        return True  # fail open — better to answer than to block


_RETRIEVAL_NEEDED_PROMPT = (
    "You are a classifier. Answer only YES or NO.\n"
    "Does answering this message require looking up Python documentation?\n"
    "Answer NO only for purely conversational messages that need no Python knowledge "
    "(e.g. 'thanks', 'ok', 'can you repeat that?', 'what did you just say?').\n"
    "Message: {question}\n"
    "Answer (YES or NO):"
)

_COMPRESS_PROMPT = (
    "You are a documentation summarizer. From the following Python documentation excerpts, "
    "extract and keep only the information directly relevant to answering this question: "
    "\"{question}\"\n\n"
    "Rules:\n"
    "- Preserve all code examples exactly as written\n"
    "- Remove repeated or off-topic content\n"
    "- Be concise but complete\n\n"
    "Documentation:\n{context}\n\nSummary:"
)


def _needs_retrieval(question: str, guard_llm) -> bool:
    """Return False for purely conversational messages that need no doc lookup."""
    try:
        resp = guard_llm.invoke(_RETRIEVAL_NEEDED_PROMPT.format(question=question)).content.strip().upper()
        return not resp.startswith("NO")
    except Exception:
        return True  # fail open


_REWRITE_PROMPT = (
    "Rewrite the last question as a fully self-contained, standalone question "
    "using context from the conversation. If it is already standalone, return it unchanged. "
    "Return only the rewritten question, nothing else.\n\n"
    "Conversation:\n{history}\n\n"
    "Last question: {question}\n\n"
    "Standalone question:"
)


def _rewrite_standalone(question: str, chat_history: list, llm) -> str:
    """Resolve pronouns / references using recent history so the retrieval query is self-contained."""
    history_text = "\n".join(f"Human: {h}\nAssistant: {a}" for h, a in chat_history[-3:])
    prompt = _REWRITE_PROMPT.format(history=history_text, question=question)
    try:
        return llm.invoke(prompt).content.strip()
    except Exception:
        return question


def _compress_context(raw_context: str, question: str, summarizer_llm) -> str:
    """Summarize retrieved chunks down to what is relevant for the question."""
    prompt = _COMPRESS_PROMPT.format(question=question, context=raw_context[:5000])
    try:
        return summarizer_llm.invoke(prompt).content.strip()
    except Exception:
        return raw_context  # fall back to full context on error


def build_qa_chain():
    llm = ChatGroq(model=GROQ_MODEL, temperature=0)
    guard_llm = ChatGroq(model=GROQ_GUARD_MODEL, temperature=0)

    smart_fn = load_retriever()

    prompt = ChatPromptTemplate.from_template("""
You are a Python documentation assistant. Answer ONLY using the provided context.

STRICT RULES:
- Every statement in your answer MUST be directly supported by the context below.
- Do NOT add explanations, comparisons, or details that are not explicitly present in the context.
- If the context does not contain enough information to answer, say exactly: "I couldn't find this in the Python docs."
- Include code examples ONLY if they appear in the context.

Before answering, verify: can I point to a specific part of the context for each claim I am about to make? If not, omit that claim.

CONVERSATION RULES:
- If there is a previous conversation, do NOT repeat information already stated. Only add NEW information that was not yet covered.
- Build on what was already explained — assume the user remembers the previous answers.

ADDITIONAL BEHAVIORS:
- Rubberducking: ONLY apply if the user explicitly shares an error message, a traceback, or says something is broken/not working. Do not apply for general how-to questions.
- Proactive Suggestions: At the end of your response, add a "See also:" section ONLY if you can suggest something directly relevant to THIS specific question (not just the general topic). Skip the section entirely if nothing fits precisely. You MUST be able to explain in one sentence WHY it is relevant to the user's exact question — if you cannot, omit it.

FORMATTING & TONE:
- Language: ALWAYS answer in the exact same language the user used for their question.
- Structure: Use Markdown formatting. Always wrap code examples in proper ```python code blocks.
- Tone: Be encouraging and pedagogical.
- Ambiguity: If the user's question is too vague to answer even with the context, politely ask them to provide their specific error message or code snippet.
{chat_history_text}
Context:
{context}

Question: {question}

Answer:""")

    direct_prompt = ChatPromptTemplate.from_template(
        "{chat_history_text}You are a helpful Python assistant. "
        "Answer the following conversational message briefly and naturally.\n\n"
        "Message: {question}\n\nAnswer:"
    )

    def format_docs(docs):
        return "\n\n".join(
            f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in docs
        )

    def format_history(history: list) -> str:
        if not history:
            return ""
        lines = ["\nPrevious conversation:"]
        for human_msg, ai_msg in history:
            lines.append(f"Human: {human_msg}")
            lines.append(f"Assistant: {ai_msg}")
        return "\n".join(lines) + "\n"

    def _prepare(question: str, chat_history: list):
        """Retrieve docs and build the prompt value — shared by invoke and stream."""
        history_text = format_history(chat_history)

        # Adaptive retrieval: skip docs lookup for purely conversational messages
        if not _needs_retrieval(question, guard_llm):
            prompt_value = direct_prompt.invoke({
                "question": question,
                "chat_history_text": history_text,
            })
            return [], prompt_value

        # Full RAG path: retrieve → compress → prompt
        # Rewrite follow-up questions into standalone queries so the retriever
        # receives a coherent, self-contained question rather than a fragment.
        retrieval_query = _rewrite_standalone(question, chat_history, llm) if chat_history else question
        docs = _multi_query_retrieve(retrieval_query, smart_fn, llm)
        raw_context = format_docs(docs)
        # Use the main LLM (8B) for compression — the 3B guard model lacks
        # the capacity to reliably summarize technical documentation.
        compressed_context = _compress_context(raw_context, question, llm)
        prompt_value = prompt.invoke({
            "context": compressed_context,
            "question": question,
            "chat_history_text": history_text,
        })
        return docs, prompt_value

    class RAGPipeline:
        def invoke(self, input_dict: dict) -> dict:
            """Blocking call — used by evaluate.py."""
            question = input_dict["question"]
            chat_history = input_dict.get("chat_history", [])
            if not _is_on_topic(question, guard_llm):
                return {"answer": _OFF_TOPIC_REPLY, "source_docs": []}
            docs, prompt_value = _prepare(question, chat_history)
            answer = StrOutputParser().invoke(llm.invoke(prompt_value))
            return {"answer": answer, "source_docs": docs}

        def stream(self, question: str, chat_history: list = None):
            """Returns (source_docs, token_generator) for streaming in app.py."""
            chat_history = chat_history or []
            if not _is_on_topic(question, guard_llm):
                return [], iter([_OFF_TOPIC_REPLY])
            docs, prompt_value = _prepare(question, chat_history)
            token_stream = (chunk.content for chunk in llm.stream(prompt_value))
            return docs, token_stream

    return RAGPipeline()


def ask(pipeline, question: str, chat_history: list = None) -> dict:
    result = pipeline.invoke({"question": question, "chat_history": chat_history or []})
    return {
        "answer": result["answer"],
        "sources": [doc.metadata.get("source", "unknown") for doc in result["source_docs"]],
    }