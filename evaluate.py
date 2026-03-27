import re
import time
import pandas as pd
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from bert_score import score as bert_score_fn
from rouge_score import rouge_scorer as rouge_scorer_lib

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

judge = ChatOllama(model="llama3.2", temperature=0, timeout=120)
_rouge = rouge_scorer_lib.RougeScorer(["rougeL"], use_stemmer=True)


def _wait_from_error(msg: str, default: float = 65.0) -> float:
    """Parse 'try again in X.XXs' from a Groq rate-limit error message."""
    m = re.search(r'try again in (\d+\.?\d*)s', str(msg))
    return float(m.group(1)) + 2 if m else default


def _groq_invoke_with_retry(fn, *args, max_retries: int = 8, **kwargs):
    """Call fn(*args, **kwargs), retrying automatically on Groq 429 errors."""
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            msg = str(e)
            if "rate_limit_exceeded" in msg or "429" in msg:
                wait = _wait_from_error(msg)
                print(f"    ⏳ Rate limit — waiting {wait:.0f}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Groq rate limit: max retries exceeded")


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def _llm_score(prompt: str) -> float:
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
    prompt = (
        f"Rate from 1-5 how faithful this answer is to the context "
        f"(1=contradicts context, 5=fully supported by context).\n"
        f"Context: {context[:3000]}\nAnswer: {answer}\n"
        f"Reply with just a single number 1-5."
    )
    return _llm_score(prompt)


def eval_answer_relevancy(question: str, answer: str) -> float:
    prompt = (
        f"Rate from 1-5 how well this answer addresses the question "
        f"(1=completely irrelevant, 5=perfectly answers the question).\n"
        f"Question: {question}\nAnswer: {answer}\n"
        f"Reply with just a single number 1-5."
    )
    return _llm_score(prompt)


def eval_context_precision(question: str, context: str) -> float:
    prompt = (
        f"Rate from 1-5 how relevant the retrieved context is for answering the question "
        f"(1=completely irrelevant, 5=perfectly relevant).\n"
        f"Question: {question}\nContext: {context[:3000]}\n"
        f"Reply with just a single number 1-5."
    )
    return _llm_score(prompt)


def eval_context_recall(ground_truth: str, context: str) -> float:
    prompt = (
        f"Rate from 1-5 how much of the information needed for the reference answer "
        f"is present in the context (1=nothing useful, 5=everything needed is there).\n"
        f"Reference Answer: {ground_truth}\nContext: {context[:3000]}\n"
        f"Reply with just a single number 1-5."
    )
    return _llm_score(prompt)


def eval_bertscore(answer: str, ground_truth: str) -> float:
    """Semantic similarity between answer and ground truth via BERTScore F1."""
    try:
        _, _, F1 = bert_score_fn([answer], [ground_truth], lang="en", verbose=False)
        return F1[0].item()
    except Exception as e:
        print(f"    ⚠️  BERTScore error: {e}")
        return None


def eval_rouge_l(answer: str, ground_truth: str) -> float:
    """ROUGE-L F1 between answer and ground truth (longest common subsequence)."""
    try:
        scores = _rouge.score(ground_truth, answer)
        return scores["rougeL"].fmeasure
    except Exception as e:
        print(f"    ⚠️  ROUGE error: {e}")
        return None


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(pipeline_name: str = "pageindex") -> pd.DataFrame:
    """Run the full test suite against the PageIndex pipeline."""
    from rag_pageindex import build_qa_chain

    print(f"\n{'='*60}")
    print(f"🔧 Loading pipeline: {pipeline_name.upper()}")
    print(f"{'='*60}")
    chain = build_qa_chain()

    results = []
    print(f"📝 Running {len(TEST_CASES)} test questions...\n")

    for i, tc in enumerate(TEST_CASES):
        question    = tc["question"]
        ground_truth = tc["ground_truth"]
        print(f"  [{i+1}/{len(TEST_CASES)}] {question}")

        result  = _groq_invoke_with_retry(chain.invoke, {"question": question, "chat_history": []})
        answer  = result["answer"]
        docs    = result["source_docs"]
        context = "\n\n".join(doc.page_content for doc in docs)

        print(f"    📊 Scoring...")
        faith  = eval_faithfulness(answer, context)
        relev  = eval_answer_relevancy(question, answer)
        prec   = eval_context_precision(question, context)
        recall = eval_context_recall(ground_truth, context)
        bert   = eval_bertscore(answer, ground_truth)
        rouge  = eval_rouge_l(answer, ground_truth)

        results.append({
            "pipeline":          pipeline_name,
            "question":          question,
            "answer":            answer[:120] + "...",
            "faithfulness":      faith,
            "answer_relevancy":  relev,
            "context_precision": prec,
            "context_recall":    recall,
            "bertscore_f1":      bert,
            "rouge_l":           rouge,
        })

        def _fmt(v): return f"{v:.2f}" if v is not None else "N/A"
        print(
            f"    ✅ faith={_fmt(faith)}  relevancy={_fmt(relev)}  "
            f"prec={_fmt(prec)}  recall={_fmt(recall)}  "
            f"bert={_fmt(bert)}  rouge={_fmt(rouge)}"
        )

    df = pd.DataFrame(results)

    # Per-pipeline summary
    print(f"\n{'='*60}")
    print(f"📈 RESULTS — {pipeline_name.upper()}")
    print(f"{'='*60}")
    metrics = ["faithfulness", "answer_relevancy", "context_precision",
               "context_recall", "bertscore_f1", "rouge_l"]
    descriptions = {
        "faithfulness":      "No hallucination?          1.0 = perfect",
        "answer_relevancy":  "Answers the question?      1.0 = perfect",
        "context_precision": "Right chunks retrieved?    1.0 = perfect",
        "context_recall":    "No key info missing?       1.0 = perfect",
        "bertscore_f1":      "Semantic match to ref?     1.0 = perfect",
        "rouge_l":           "Lexical overlap with ref?  1.0 = perfect",
    }
    for m in metrics:
        val = df[m].mean()
        print(f"  {m:<22} {val:.3f}  ({descriptions[m]})")

    out_file = f"evaluation_results_{pipeline_name}.csv"
    df.to_csv(out_file, index=False)
    print(f"\n💾 Saved → {out_file}")

    return df


def print_comparison(dfs: list[pd.DataFrame]) -> None:
    """Print a side-by-side comparison table and save combined CSV."""
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv("evaluation_results_comparison.csv", index=False)

    metrics = ["faithfulness", "answer_relevancy", "context_precision",
               "context_recall", "bertscore_f1", "rouge_l"]
    pipelines = combined["pipeline"].unique()

    print(f"\n{'='*70}")
    print("📊 COMPARISON SUMMARY")
    print(f"{'='*70}")
    header = f"{'Metric':<22}" + "".join(f"  {p.upper():<12}" for p in pipelines)
    print(header)
    print("-" * len(header))

    for m in metrics:
        row = f"{m:<22}"
        for p in pipelines:
            val = combined[combined["pipeline"] == p][m].mean()
            row += f"  {val:.3f}{'':9}"
        print(row)

    print(f"{'='*70}")
    print("💾 Combined results saved → evaluation_results_comparison.csv")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_evaluation("pageindex")
