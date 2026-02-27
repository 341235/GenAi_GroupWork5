import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

load_dotenv()

CHROMA_PATH = "./chroma_python_docs"
OLLAMA_MODEL = "mistral"

def load_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    # Semantic retriever — higher k to give re-ranker a large candidate pool
    semantic_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 20}
    )

    # BM25 retriever — keyword-based, built from docs stored in Chroma
    # Ensures exact function names like "zip" or "for" always get a match
    raw = vectorstore.get()
    all_docs = [
        Document(page_content=pc, metadata=m)
        for pc, m in zip(raw["documents"], raw["metadatas"])
    ]
    bm25_retriever = BM25Retriever.from_documents(all_docs, k=8)

    # Hybrid ensemble: BM25 (keyword, 20%) + semantic (80%)
    # Low BM25 weight avoids false positives on common keywords like
    # "open", "import", "for" which appear in nearly every code chunk.
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[0.2, 0.8]
    )

    # Cross-encoder re-ranker: re-scores every candidate against the query
    # and keeps the top 6 — much more accurate than embedding similarity alone
    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=6)

    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=ensemble_retriever
    )

def build_qa_chain():
    retriever = load_retriever()
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_template("""
You are a Python documentation assistant. Answer ONLY using the provided context.

STRICT RULES:
- Every statement in your answer MUST be directly supported by the context below.
- Do NOT add explanations, comparisons, or details that are not explicitly present in the context.
- If the context does not contain enough information to answer, say exactly: "I couldn't find this in the Python docs."
- Include code examples ONLY if they appear in the context.

Before answering, verify: can I point to a specific part of the context for each claim I am about to make? If not, omit that claim.

Context:
{context}

Question: {question}

Answer:""")

    def format_docs(docs):
        return "\n\n".join(
            f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in docs
        )

    chain = (
        RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
        | RunnableParallel({
            "answer": (
                {"context": lambda x: format_docs(x["context"]), "question": lambda x: x["question"]}
                | prompt
                | llm
                | StrOutputParser()
            ),
            "source_docs": lambda x: x["context"],
        })
    )

    return chain, retriever

def ask(chain_and_retriever, question: str) -> dict:
    chain, _ = chain_and_retriever
    result = chain.invoke(question)
    return {
        "answer": result["answer"],
        "sources": [doc.metadata.get("source", "unknown") for doc in result["source_docs"]]
    }