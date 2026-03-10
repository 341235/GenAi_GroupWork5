import os
import re
import sys
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from bert_score import score as bert_score_fn

load_dotenv()

TEST_CASES = [
    # --- Basic data structures ---
    {
        "question": "How do I create a list in Python?",
        "ground_truth": "You can create a list using square brackets like [1, 2, 3] or the list() constructor."
    },
    {
        "question": "What is the difference between a list and a tuple?",
        "ground_truth": "Lists are mutable and use square brackets, tuples are immutable and use parentheses."
    },
    {
        "question": "What is a dictionary in Python?",
        "ground_truth": "A dictionary is a mutable mapping of key-value pairs, created with curly braces like {'key': 'value'}."
    },
    {
        "question": "What is the difference between list.append() and list.extend()?",
        "ground_truth": "append() adds a single element to the end of the list, while extend() adds all elements from an iterable."
    },
    {
        "question": "How do I write a list comprehension in Python?",
        "ground_truth": "A list comprehension uses the syntax [expr for item in iterable], optionally with an if condition at the end."
    },
    # --- Control flow ---
    {
        "question": "How do I use a for loop in Python?",
        "ground_truth": "Use 'for item in iterable:' syntax to iterate over sequences like lists, tuples, or ranges."
    },
    {
        "question": "How does the range() function work?",
        "ground_truth": "range(stop) or range(start, stop, step) generates a sequence of integers, commonly used in for loops."
    },
    {
        "question": "What does the enumerate() function do?",
        "ground_truth": "enumerate() adds a counter to an iterable and returns (index, value) pairs, useful in for loops."
    },
    {
        "question": "What does the zip() function do?",
        "ground_truth": "zip() takes multiple iterables and aggregates them into tuples, stopping at the shortest iterable."
    },
    # --- Exception handling ---
    {
        "question": "How do I handle exceptions in Python?",
        "ground_truth": "Use try/except blocks to catch and handle exceptions in Python."
    },
    {
        "question": "What is the finally clause in a try/except block?",
        "ground_truth": "The finally clause runs regardless of whether an exception was raised, typically used for cleanup actions."
    },
    # --- Functions & builtins ---
    {
        "question": "What does the sorted() built-in function do?",
        "ground_truth": "sorted() returns a new sorted list from the elements of an iterable, accepting optional key and reverse arguments."
    },
    {
        "question": "What does the map() function do in Python?",
        "ground_truth": "map() applies a function to every item of an iterable and returns an iterator of the results."
    },
    {
        "question": "What is a lambda function in Python?",
        "ground_truth": "A lambda function is a small anonymous function defined with the lambda keyword, e.g. lambda x: x + 1."
    },
    # --- Strings & formatting ---
    {
        "question": "How do I format a string using f-strings in Python?",
        "ground_truth": "Use f-strings with the syntax f'text {variable}' to embed expressions directly in string literals."
    },
    # --- Modules & stdlib ---
    {
        "question": "How do I parse JSON in Python?",
        "ground_truth": "Use json.loads() to parse a JSON string into a Python object, and json.dumps() to serialize a Python object to a JSON string."
    },
    {
        "question": "How do I use regular expressions in Python?",
        "ground_truth": "Import the re module and use re.search(), re.match(), or re.findall() to find patterns in strings."
    },
    {
        "question": "What does itertools.chain() do?",
        "ground_truth": "itertools.chain() takes multiple iterables and chains them together, yielding elements from each sequentially."
    },
    {
        "question": "How do I use collections.Counter?",
        "ground_truth": "Counter is a dict subclass for counting hashable objects. Pass an iterable to Counter() to get a mapping of elements to their counts."
    },
    # --- File I/O & OOP ---
    {
        "question": "How do I open and read a file in Python?",
        "ground_truth": "Use open() with a file path and mode. The 'with open(...) as f:' pattern ensures the file is automatically closed after use."
    },
    {
        "question": "How do I use pathlib to work with file paths?",
        "ground_truth": "Use pathlib.Path to create path objects. Paths can be joined with the / operator, and methods like .read_text() read file contents."
    },
    {
        "question": "What is a class in Python and how do I define one?",
        "ground_truth": "A class is defined with the class keyword and serves as a blueprint for creating objects with shared methods and attributes."
    },
    {
        "question": "What is the purpose of the __init__ method in a class?",
        "ground_truth": "__init__ is the initializer method called when a new instance is created, used to set up instance attributes."
    },
    # --- Advanced ---
    {
        "question": "How do I import a module in Python?",
        "ground_truth": "Use 'import module_name' or 'from module_name import name' to bring module functionality into scope."
    },
    {
        "question": "What does functools.reduce() do?",
        "ground_truth": "functools.reduce() applies a two-argument function cumulatively to items of an iterable to reduce it to a single value."
    },
]

GROQ_GUARD_MODEL = "llama-3.2-3b-preview"
judge = ChatGroq(model=GROQ_GUARD_MODEL, temperature=0)

def score(prompt: str) -> float:
    """Ask the judge LLM to score something 1-5, return normalized 0-1."""
    try:
        response = judge.invoke(prompt).content.strip()
        match = re.search(r'\b([1-5])\b', response)
        if match:
            return (int(match.group(1)) - 1) / 4.0
        return 0.5
    except Exception as e:
        print(f"    ⚠️  Judge error: {e}")
        return None

def eval_faithfulness(answer: str, context: str) -> float:
    prompt = f"""Rate from 1-5 how faithful this answer is to the context (1=contradicts context, 5=fully supported by context).
Context: {context[:3000]}
Answer: {answer}
Reply with just a single number 1-5."""
    return score(prompt)

def eval_answer_relevancy(question: str, answer: str) -> float:
    prompt = f"""Rate from 1-5 how well this answer addresses the question (1=completely irrelevant, 5=perfectly answers the question).
Question: {question}
Answer: {answer}
Reply with just a single number 1-5."""
    return score(prompt)

def eval_context_precision(question: str, context: str) -> float:
    prompt = f"""Rate from 1-5 how relevant the retrieved context is for answering the question (1=completely irrelevant, 5=perfectly relevant).
Question: {question}
Context: {context[:3000]}
Reply with just a single number 1-5."""
    return score(prompt)

def eval_bertscore(answer: str, ground_truth: str) -> float:
    """Semantic similarity between answer and ground truth via BERTScore F1.
    Downloads a BERT model on first run (~500 MB, cached afterwards)."""
    try:
        _, _, F1 = bert_score_fn([answer], [ground_truth], lang="en", verbose=False)
        return F1[0].item()
    except Exception as e:
        print(f"    ⚠️  BERTScore error: {e}")
        return None

def eval_context_recall(ground_truth: str, context: str) -> float:
    prompt = f"""Rate from 1-5 how much of the information needed for the reference answer is present in the context (1=nothing useful, 5=everything needed is there).
Reference Answer: {ground_truth}
Context: {context[:3000]}
Reply with just a single number 1-5."""
    return score(prompt)

def run_evaluation(pipeline_name: str = "advanced"):
    """
    pipeline_name: "advanced" → rag.py (full pipeline)
                   "baseline"  → rag_baseline.py (page-level retriever)
    """
    if pipeline_name == "baseline":
        from rag_baseline import build_qa_chain
    else:
        from rag import build_qa_chain
        pipeline_name = "advanced"

    print(f"🔧 Loading RAG chain ({pipeline_name})...")
    chain = build_qa_chain()

    results = []
    print(f"📝 Running {len(TEST_CASES)} test questions...\n")

    for i, tc in enumerate(TEST_CASES):
        question = tc["question"]
        ground_truth = tc["ground_truth"]
        print(f"  [{i+1}/{len(TEST_CASES)}] {question}")

        result = chain.invoke({"question": question, "chat_history": []})
        answer = result["answer"]
        docs = result["source_docs"]
        context = "\n\n".join(doc.page_content for doc in docs)

        print(f"    📊 Scoring...")
        faith  = eval_faithfulness(answer, context)
        relev  = eval_answer_relevancy(question, answer)
        prec   = eval_context_precision(question, context)
        recall = eval_context_recall(ground_truth, context)
        bert   = eval_bertscore(answer, ground_truth)

        results.append({
            "question": question,
            "answer": answer[:100] + "...",
            "faithfulness": faith,
            "answer_relevancy": relev,
            "context_precision": prec,
            "context_recall": recall,
            "bertscore_f1": bert,
        })
        print(f"    ✅ faith={faith:.2f}  relevancy={relev:.2f}  precision={prec:.2f}  recall={recall:.2f}  bert={bert:.2f}")

    df = pd.DataFrame(results)

    print("\n" + "="*60)
    print(f"📈 EVALUATION RESULTS — pipeline: {pipeline_name.upper()}")
    print("="*60)
    print(df[["question", "faithfulness", "answer_relevancy", "context_precision", "context_recall", "bertscore_f1"]].to_string(index=False))

    print("\n--- Averages ---")
    metrics = {
        "faithfulness":      "Erfindet das Modell nichts?  1.0 = perfekt",
        "answer_relevancy":  "Beantwortet es die Frage?    1.0 = perfekt",
        "context_precision": "Richtige Chunks gefunden?    1.0 = perfekt",
        "context_recall":    "Kein wichtiger Info fehlt?   1.0 = perfekt",
        "bertscore_f1":      "Semantische Nähe zur Referenz 1.0 = perfekt",
    }
    for metric, desc in metrics.items():
        val = df[metric].mean()
        print(f"  {metric:<22} {val:.3f}  ({desc})")
    print("="*60)

    out_file = f"evaluation_results_{pipeline_name}.csv"
    df.to_csv(out_file, index=False)
    print(f"\n💾 Results saved to {out_file}")

if __name__ == "__main__":
    pipeline = "baseline" if "--baseline" in sys.argv else "advanced"
    run_evaluation(pipeline)