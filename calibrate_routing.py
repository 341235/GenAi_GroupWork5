"""
calibrate_routing.py
--------------------
Finds the optimal semantic routing threshold for smart_retrieve().

Run once after (re-)ingesting the vector DB:
    python calibrate_routing.py

Prints per-query similarity scores and recommends a threshold value.
Update _ROUTING_THRESHOLD in rag.py with the printed suggestion.
"""

import numpy as np
from embeddings import NomicEmbeddings
from rag import _TOPIC_PHRASES, _cosine_sim, _ROUTING_THRESHOLD

# ---------------------------------------------------------------------------
# Labeled test queries — add more to improve calibration confidence.
# Use None for cross-topic queries that should fall through to full_retriever.
# ---------------------------------------------------------------------------
LABELED: list[tuple[str, str | None]] = [
    # datetime
    ("how do I work with time zones",               "datetime"),
    ("convert a string to a date object",           "datetime"),
    ("get the current date and time",               "datetime"),
    # asyncio
    ("run two coroutines at the same time",         "asyncio"),
    ("what is an event loop in Python",             "asyncio"),
    ("how do I cancel an async task",               "asyncio"),
    # threading
    ("protect shared data between threads",         "threading"),
    ("how do I start a background thread",          "threading"),
    # numpy
    ("create a 2D array of zeros",                  "numpy"),
    ("how do I reshape an array",                   "numpy"),
    ("compute a dot product of two vectors",        "numpy"),
    # pandas
    ("load a CSV file into memory",                 "pandas"),
    ("filter rows where a column is greater than",  "pandas"),
    ("how do I group and aggregate data",           "pandas"),
    # matplotlib
    ("draw a scatter plot",                         "matplotlib"),
    ("how do I add a legend to my chart",           "matplotlib"),
    ("save a figure to a PNG file",                 "matplotlib"),
    # sklearn
    ("how do I train a random forest classifier",   "sklearn"),
    ("split data into training and test sets",      "sklearn"),
    ("evaluate model performance",                  "sklearn"),
    # requests
    ("send a POST request with a JSON body",        "requests"),
    ("how do I handle a 404 response",              "requests"),
    # itertools
    ("get all combinations of a list",              "itertools"),
    ("how do I chain two iterables together",       "itertools"),
    # python_basics (merged: builtins + datastructures)
    ("remove duplicates from a list",               "python_basics"),
    ("how do I merge two dictionaries",             "python_basics"),
    ("apply a function to every element",           "python_basics"),
    ("sort a list of tuples by the second value",   "python_basics"),
    # cross-topic — should NOT be routed to any single topic
    ("convert a numpy array to a pandas DataFrame", None),
    ("plot a pandas DataFrame with matplotlib",     None),
    ("what is Python",                              None),
    ("explain object oriented programming",         None),
]

# ---------------------------------------------------------------------------

def main():
    embeddings = NomicEmbeddings()

    print("Building topic centroids...")
    centroids = {
        topic: np.mean([embeddings.embed_query(p) for p in phrases], axis=0)
        for topic, phrases in _TOPIC_PHRASES.items()
    }

    correct_scores: list[float] = []
    cross_scores:   list[float] = []
    errors = 0

    print(f"\n{'Query':<50} {'Expected':<16} {'Got':<16} {'Score':>6}  {'Expected Score':>14}")
    print("-" * 110)

    for query, expected in LABELED:
        q_vec = embeddings.embed_query(query)
        scores = {t: _cosine_sim(q_vec, c) for t, c in centroids.items()}
        best_topic = max(scores, key=scores.get)
        best_score = scores[best_topic]

        if expected is not None:
            correct_score = scores[expected]
            correct = best_topic == expected
            marker = "✅" if correct else "❌"
            if not correct:
                errors += 1
            print(f"{marker} {query:<48} {expected:<16} {best_topic:<16} {best_score:>6.3f}  {correct_score:>14.3f}")
            correct_scores.append(correct_score)
            cross_scores.append(best_score if not correct else 0.0)
        else:
            print(f"⚠️  {query:<48} {'(cross-topic)':<16} {best_topic:<16} {best_score:>6.3f}  {'—':>14}")
            cross_scores.append(best_score)

    # Remove zeros added as placeholders for correct matches
    cross_scores = [s for s in cross_scores if s > 0]

    print("\n" + "=" * 110)
    print(f"Correct-topic scores  — min: {min(correct_scores):.3f}  mean: {sum(correct_scores)/len(correct_scores):.3f}  max: {max(correct_scores):.3f}")
    if cross_scores:
        print(f"Cross/wrong scores    — min: {min(cross_scores):.3f}  mean: {sum(cross_scores)/len(cross_scores):.3f}  max: {max(cross_scores):.3f}")
        midpoint = (min(correct_scores) + max(cross_scores)) / 2
        print(f"\nSuggested threshold (midpoint): {midpoint:.3f}")
        print(f"Current  threshold (_ROUTING_THRESHOLD): {_ROUTING_THRESHOLD}")
        if abs(midpoint - _ROUTING_THRESHOLD) > 0.02:
            print(f"  → Consider updating _ROUTING_THRESHOLD to {midpoint:.2f} in rag.py")
        else:
            print("  → Current threshold looks good.")

    single_topic_errors = errors
    print(f"\nRouting errors (wrong single topic): {single_topic_errors} / {sum(1 for _, e in LABELED if e is not None)}")


if __name__ == "__main__":
    main()