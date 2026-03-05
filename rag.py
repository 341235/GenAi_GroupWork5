import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # === NEU ===
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory # === NEU ===
from langchain_community.chat_message_histories import ChatMessageHistory # === NEU ===
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from embeddings import NomicEmbeddings

load_dotenv()

CHROMA_PATH = "./chroma_python_docs"
OLLAMA_MODEL = "mistral"

# Keywords that indicate a query belongs to an isolated embedding cluster.
_DATETIME_KEYWORDS  = {"datetime", "timedelta", "strftime", "strptime", "timezone"}
_ASYNCIO_KEYWORDS   = {"asyncio", "coroutine", "event loop", "async def", "await "}
_THREADING_KEYWORDS = {"threading", "thread", "lock", "semaphore", "barrier"}


def _build_ensemble_reranker(vectorstore, all_docs, cross_encoder,
                             chroma_filter=None, doc_filter_fn=None):
    """
    Build a hybrid (BM25 + semantic MMR) + cross-encoder retriever.
    """
    semantic_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8, "fetch_k": 20,
            **({"filter": chroma_filter} if chroma_filter else {}),
        },
    )

    bm25_docs = [d for d in all_docs if doc_filter_fn(d)] if doc_filter_fn else all_docs
    bm25_retriever = BM25Retriever.from_documents(bm25_docs, k=8)

    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[0.2, 0.8],
    )
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=6)
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

    full_retriever = _build_ensemble_reranker(vectorstore, all_docs, cross_encoder)

    def _page_filter(page_filename):
        return lambda d: page_filename in d.metadata.get("source", "")

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

    def smart_retrieve(query: str):
        """Route the query to a filtered retriever when topic keywords match."""
        q = query.lower()
        if any(kw in q for kw in _DATETIME_KEYWORDS):
            return datetime_retriever.invoke(query)
        if any(kw in q for kw in _THREADING_KEYWORDS):
            return threading_retriever.invoke(query)
        if any(kw in q for kw in _ASYNCIO_KEYWORDS):
            return asyncio_retriever.invoke(query)
        return full_retriever.invoke(query)

    return RunnableLambda(smart_retrieve)


# === NEW: a global memory storage ===
# With this new message history, it should be possible to have a chat and return to your former chats.
session_store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]


def build_qa_chain():
    retriever = load_retriever()
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
    )

    # === NEW: Prompt adapted to from_messages for history support ===
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Python documentation assistant. Answer ONLY using the provided context.

STRICT RULES:
- Every statement in your answer MUST be directly supported by the context below.
- Do NOT add explanations, comparisons, or details that are not explicitly present in the context.
- If the context does not contain enough information to answer, say exactly: "I couldn't find this in the Python docs."
- Include code examples ONLY if they appear in the context.

Before answering, verify: can I point to a specific part of the context for each claim I am about to make? If not, omit that claim.

ADDITIONAL BEHAVIORS:
- Rubberducking for Errors: If the user's question involves debugging or an error, do not just give the direct answer.
Instead, use the context to ask 1-2 guiding questions that nudge the user to spot the mistake themselves.
- Proactive Suggestions: At the end of your response, add a "See also:" section. 
Suggest 1-2 related functions or modules, but ONLY if they are mentioned in the provided context and are relevant to the user's intent.

FORMATTING & TONE:
- Language: ALWAYS answer in the exact same language the user used for their question.
- Structure: Use Markdown formatting. Always wrap code examples in proper ```python code blocks.
- Tone: Be encouraging and pedagogical. 
- Ambiguity: If the user's question is too vague to answer even with the context, politely ask them to provide their specific error message or code snippet.

Context:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"), # Adds the old messages
        ("human", "{question}")
    ])
    # =========================================================================

    def format_docs(docs):
        return "\n\n".join(
            f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in docs
        )

    # === New: Adapted Chain-Logic so it is able to use the input-format of the history ===
    chain = (
        RunnablePassthrough.assign(
            context=lambda x: retriever.invoke(x["question"])
        )
        | RunnableParallel({
            "answer": (
                {
                    "context": lambda x: format_docs(x["context"]), 
                    "question": lambda x: x["question"],
                    "chat_history": lambda x: x["chat_history"]
                }
                | prompt
                | llm
                | StrOutputParser()
            ),
            "source_docs": lambda x: x["context"],
        })
    )

    # Memory-Wrapper around the finished chain
    conversational_chain = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
        output_messages_key="answer", # Important: Tells the history which key has the LLM answer
    )
    # =============================================================================================

    return conversational_chain, retriever

# === New: The ask function now needs a session id ===
def ask(chain_and_retriever, question: str, session_id: str = "default_user_session") -> dict:
    chain, _ = chain_and_retriever
    
    # if used, we need to give the config, so langchain knows which history to use
    result = chain.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}}
    )
    
    return {
        "answer": result["answer"],
        "sources": [doc.metadata.get("source", "unknown") for doc in result["source_docs"]]
    }
