"""
rag_baseline.py — Page-Level (ParentDocumentRetriever-style) RAG pipeline

Architecture
------------
This is an intentionally simple baseline for academic comparison with rag.py.

Concept (mirrors LangChain's ParentDocumentRetriever):
  1. Small child chunks are stored in ChromaDB for precise similarity search.
  2. When a query arrives, top-k child chunks are retrieved by cosine similarity.
  3. Each matched chunk's source URL is the "parent page".
  4. ALL chunks from those parent pages are returned as context (full-page view).

What is deliberately left out (unlike rag.py):
  - No BM25 hybrid retrieval
  - No cross-encoder reranking
  - No keyword-based topic routing
  - No multi-query expansion
  - No off-topic guard
  - No adaptive retrieval
  - No post-retrieval compression
  - No conversation history handling

This lets us measure exactly how much the advanced features in rag.py contribute.
"""

import os
from collections import defaultdict
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from embeddings import NomicEmbeddings

load_dotenv()

CHROMA_PATH = "./chroma_python_docs"
GROQ_MODEL = "llama-3.1-8b-instant"

# How many child chunks to use as "seeds" for page expansion
CHILD_K = 5
# Max total chunks returned after page expansion
MAX_EXPANDED = 12


def load_page_retriever():
    """
    Returns a callable that, given a query string, performs:
      1. Semantic child-chunk retrieval (top CHILD_K by cosine similarity)
      2. Page expansion — returns all chunks from matched source pages
    """
    embeddings = NomicEmbeddings()
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
    )

    # Build an in-memory page → chunks index (the "parent store")
    raw = vectorstore.get()
    page_to_chunks: dict[str, list[Document]] = defaultdict(list)
    for pc, m in zip(raw["documents"], raw["metadatas"]):
        source = m.get("source", "unknown")
        page_to_chunks[source].append(Document(page_content=pc, metadata=m))

    # Plain semantic child retriever — no MMR, no filters
    child_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": CHILD_K},
    )

    def retrieve(query: str) -> list[Document]:
        # Step 1: find most relevant child chunks
        child_docs = child_retriever.invoke(query)

        # Step 2: collect unique parent pages
        seen_sources: set[str] = set()
        expanded: list[Document] = []
        for doc in child_docs:
            src = doc.metadata.get("source", "unknown")
            if src not in seen_sources:
                seen_sources.add(src)
                expanded.extend(page_to_chunks[src])

        # Deduplicate by content prefix and cap size
        seen_content: set[str] = set()
        result: list[Document] = []
        for d in expanded:
            key = d.page_content[:150]
            if key not in seen_content:
                seen_content.add(key)
                result.append(d)
        return result[:MAX_EXPANDED]

    return retrieve


def build_qa_chain():
    llm = ChatGroq(model=GROQ_MODEL, temperature=0)
    retrieve = load_page_retriever()

    prompt = ChatPromptTemplate.from_template(
        "You are a Python documentation assistant. "
        "Answer the question using only the information in the context below. "
        "If the answer is not in the context, say: "
        "\"I couldn't find this in the Python docs.\"\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    def format_docs(docs: list[Document]) -> str:
        return "\n\n".join(
            f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in docs
        )

    class BaselineRAGPipeline:
        def invoke(self, input_dict: dict) -> dict:
            question = input_dict["question"]
            docs = retrieve(question)
            prompt_value = prompt.invoke({
                "context": format_docs(docs),
                "question": question,
            })
            answer = StrOutputParser().invoke(llm.invoke(prompt_value))
            return {"answer": answer, "source_docs": docs}

        def stream(self, question: str, chat_history: list = None):
            docs = retrieve(question)
            prompt_value = prompt.invoke({
                "context": format_docs(docs),
                "question": question,
            })
            token_stream = (chunk.content for chunk in llm.stream(prompt_value))
            return docs, token_stream

    return BaselineRAGPipeline()


def ask(pipeline, question: str, chat_history: list = None) -> dict:
    result = pipeline.invoke({"question": question, "chat_history": chat_history or []})
    return {
        "answer": result["answer"],
        "sources": [doc.metadata.get("source", "unknown") for doc in result["source_docs"]],
    }