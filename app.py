import streamlit as st
from rag import build_qa_chain, ask

st.set_page_config(
    page_title="Python Docs RAG",
    page_icon="🐍",
    layout="centered"
)

st.title("🐍 Python Documentation Assistant")
st.caption("Ask questions about the Python standard library and tutorial — powered by Ollama and ChromaDB.")

@st.cache_resource
def get_chain():
    return build_qa_chain()

chain_and_retriever = get_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask something about Python..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching docs..."):
            response = ask(chain_and_retriever, question)

        st.markdown(response["answer"])

        with st.expander("📚 Sources"):
            for src in set(response["sources"]):
                st.markdown(f"- {src}")

    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})