import os
import streamlit as st
from rag import build_qa_chain

# Expose Groq API key to langchain_groq (reads from Streamlit secrets or .env)
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

st.set_page_config(
    page_title="Python Docs RAG",
    page_icon="🐍",
    layout="centered"
)

st.title("🐍 Python Documentation Assistant")
st.caption("Ask questions about Python, NumPy, Pandas, Matplotlib, scikit-learn and more — powered by Groq and ChromaDB.")

@st.cache_resource
def get_chain():
    return build_qa_chain()

chain = get_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def get_chat_history() -> list:
    """Build (human, assistant) pairs from session messages for RAG context."""
    history = []
    msgs = st.session_state.messages
    for i in range(0, len(msgs) - 1, 2):
        if i + 1 < len(msgs):
            if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
                history.append((msgs[i]["content"], msgs[i + 1]["content"]))
    return history

if question := st.chat_input("Ask something about Python..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching docs..."):
            source_docs, token_stream = chain.stream(question, chat_history=get_chat_history())

        answer = st.write_stream(token_stream)

        with st.expander("📚 Sources"):
            for src in set(doc.metadata.get("source", "unknown") for doc in source_docs):
                st.markdown(f"- {src}")

    st.session_state.messages.append({"role": "assistant", "content": answer})