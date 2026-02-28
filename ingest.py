import os
os.environ["USER_AGENT"] = "MyRAGApp/1.0"
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from embeddings import NomicEmbeddings

load_dotenv()

CHROMA_PATH = "./chroma_python_docs"

def load_docs():
    print("Loading Python-Documentation...")
    urls = [
        "https://docs.python.org/3/tutorial/index.html",
        "https://docs.python.org/3/tutorial/introduction.html",
        "https://docs.python.org/3/tutorial/datastructures.html",
        "https://docs.python.org/3/library/functions.html",
        "https://docs.python.org/3/library/stdtypes.html",
        "https://docs.python.org/3/library/exceptions.html",
        "https://docs.python.org/3/library/collections.html",
        "https://docs.python.org/3/library/itertools.html",
        "https://docs.python.org/3/tutorial/controlflow.html",
        "https://docs.python.org/3/tutorial/classes.html",
        "https://docs.python.org/3/tutorial/errors.html",
        "https://docs.python.org/3/tutorial/modules.html",
        "https://docs.python.org/3/tutorial/inputoutput.html",
        "https://docs.python.org/3/library/os.html",
        "https://docs.python.org/3/library/os.path.html",
        "https://docs.python.org/3/library/sys.html",
        "https://docs.python.org/3/library/json.html",
        "https://docs.python.org/3/library/datetime.html",
        "https://docs.python.org/3/library/re.html",
        "https://docs.python.org/3/library/math.html",
        "https://docs.python.org/3/library/random.html",
        "https://docs.python.org/3/library/string.html",
        "https://docs.python.org/3/library/pathlib.html",
        "https://docs.python.org/3/library/functools.html",
        "https://docs.python.org/3/library/typing.html",
        "https://docs.python.org/3/library/asyncio.html",
        "https://docs.python.org/3/library/threading.html",
        "https://docs.python.org/3/library/csv.html",
        "https://docs.python.org/3/library/io.html",
    ]
    loader = WebBaseLoader(urls)
    docs = loader.load()
    print(f"{len(docs)} Sites loaded")
    return docs

_BOILERPLATE_MARKERS = [
    "© Copyright 2001 Python Software Foundation",
    "Theme\n    \nAuto\nLight\nDark",
    "Found a bug?\n",
    "Created using Sphinx",
    "Navigation\n\n\nindex\n\nmodules",
]


def _is_boilerplate(text: str) -> bool:
    return any(marker in text for marker in _BOILERPLATE_MARKERS)


def split_docs(docs):
    print("Creating Chunks...")
    # Tutorial pages are narrative prose — larger chunks preserve context
    tutorial_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
    )
    # Reference pages (functions.html, re.html, stdtypes.html, etc.) list
    # every API entry alphabetically. Smaller chunks keep each function
    # description in its own chunk so retrieval can target them precisely.
    reference_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=180,
    )
    chunks = []
    for doc in docs:
        url = doc.metadata.get("source", "")
        if "/library/" in url:
            chunks.extend(reference_splitter.split_documents([doc]))
        else:
            chunks.extend(tutorial_splitter.split_documents([doc]))
    print(f"{len(chunks)} Chunks created")

    # Remove HTML navigation/footer boilerplate that WebBaseLoader captures.
    # These chunks (theme toggles, copyright notices, nav links) form a spurious
    # outlier cluster in the embedding space and hurt retrieval precision.
    before = len(chunks)
    chunks = [c for c in chunks if not _is_boilerplate(c.page_content)]
    print(f"Removed {before - len(chunks)} boilerplate chunks ({len(chunks)} remaining)")
    return chunks

def build_vectorstore(chunks):
    print("Creating Embeddings & VektorDB (may take a few minutes)...")
    embeddings = NomicEmbeddings()
    # Alte DB löschen und neu aufbauen
    import shutil
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("🗑️  Old vectorstore deleted")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print(f"✅ Saved under: {CHROMA_PATH}")

if __name__ == "__main__":
    docs = load_docs()
    chunks = split_docs(docs)
    build_vectorstore(chunks)
    print("\n🎉 Done! You can now run the app with: streamlit run app.py")