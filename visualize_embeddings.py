"""
visualize_embeddings.py
-----------------------
Generates an interactive 2D scatter plot of every chunk stored in the
vector database, coloured by documentation category.

Each dot is one text chunk. Chunks that discuss similar topics cluster
together. Hover over any dot to see which page it came from and a preview
of its content.

Usage:
    python visualize_embeddings.py

Output:
    embedding_visualization.html  — open this in any browser
"""

import numpy as np
import plotly.graph_objects as go
import umap
from langchain_chroma import Chroma
from embeddings import NomicEmbeddings

CHROMA_PATH = "./chroma_python_docs"

# Group the 29 indexed pages into readable categories for the legend
CATEGORY_MAP = {
    "index":          "Tutorial",
    "introduction":   "Tutorial",
    "datastructures": "Tutorial",
    "controlflow":    "Tutorial",
    "classes":        "Tutorial",
    "errors":         "Tutorial",
    "modules":        "Tutorial",
    "inputoutput":    "Tutorial",
    "functions":      "Builtins",
    "stdtypes":       "Builtins",
    "exceptions":     "Exceptions",
    "collections":    "Data Structures",
    "itertools":      "Data Structures",
    "functools":      "Functional",
    "os":             "OS / Files",
    "os.path":        "OS / Files",
    "pathlib":        "OS / Files",
    "sys":            "OS / Files",
    "io":             "OS / Files",
    "csv":            "OS / Files",
    "json":           "Serialization",
    "datetime":       "Date & Time",
    "re":             "Text",
    "string":         "Text",
    "math":           "Math",
    "random":         "Math",
    "typing":         "Types",
    "asyncio":        "Concurrency",
    "threading":      "Concurrency",
}

CATEGORY_COLORS = {
    "Tutorial":        "#4C72B0",
    "Builtins":        "#DD8452",
    "Exceptions":      "#C44E52",
    "Data Structures": "#55A868",
    "Functional":      "#8172B3",
    "OS / Files":      "#937860",
    "Serialization":   "#DA8BC3",
    "Date & Time":     "#CCB974",
    "Text":            "#64B5CD",
    "Math":            "#8BC34A",
    "Types":           "#FF7043",
    "Concurrency":     "#26C6DA",
}


def page_name(url: str) -> str:
    return url.split("/")[-1].replace(".html", "")


def category(url: str) -> str:
    return CATEGORY_MAP.get(page_name(url), "Other")


def main():
    print("Loading embeddings from vectorstore...")
    embeddings_model = NomicEmbeddings()
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings_model,
    )

    raw = vectorstore.get(include=["embeddings", "metadatas", "documents"])
    vectors   = np.array(raw["embeddings"])
    metadatas = raw["metadatas"]
    documents = raw["documents"]
    n = len(documents)
    print(f"  {n} chunks loaded.")

    print("Running UMAP — this takes about 30 seconds for large databases...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    coords = reducer.fit_transform(vectors)

    # Build per-chunk labels
    sources  = [m.get("source", "unknown") for m in metadatas]
    pages    = [page_name(s) for s in sources]
    cats     = np.array([category(s) for s in sources])
    # Hover text: bold page name + first 300 chars of the chunk
    previews = [
        f"<b>{pages[i]}</b><br>"
        + documents[i][:300].replace("<", "&lt;").replace(">", "&gt;")
        + "..."
        for i in range(n)
    ]

    print("Building interactive plot...")
    fig = go.Figure()

    for cat in sorted(set(cats)):
        mask = cats == cat
        fig.add_trace(go.Scatter(
            x=coords[mask, 0],
            y=coords[mask, 1],
            mode="markers",
            name=cat,
            text=np.array(previews)[mask],
            hovertemplate="%{text}<extra></extra>",
            marker=dict(
                size=6,
                color=CATEGORY_COLORS.get(cat, "#888888"),
                opacity=0.75,
                line=dict(width=0),
            ),
        ))

    fig.update_layout(
        title=dict(
            text="Python Docs — Embedding Space (UMAP)",
            font=dict(size=20),
        ),
        xaxis=dict(title="UMAP dimension 1", showgrid=False, zeroline=False),
        yaxis=dict(title="UMAP dimension 2", showgrid=False, zeroline=False),
        legend=dict(title="Category", itemsizing="constant"),
        hovermode="closest",
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="white",
        width=1100,
        height=750,
    )

    output = "embedding_visualization.html"
    fig.write_html(output)
    print(f"\nDone! Saved to {output}")
    print("Open it in your browser to explore the embedding space.")
    fig.show()


if __name__ == "__main__":
    main()
