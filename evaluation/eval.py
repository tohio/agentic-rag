"""
eval.py
-------
Evaluates the agentic RAG pipeline against the ground truth Q&A dataset.

Extends rag-pipeline/eval.py with agent-specific metrics:

    1. Retrieval Quality (inherited)
       - Hit Rate    : was the correct chunk retrieved?

    2. Answer Quality (inherited)
       - Faithfulness: is the answer grounded in retrieved context?
       - Relevance   : does the answer address the question?
       - Correctness : does the answer match ground truth?

    3. Agent-Specific Metrics (new)
       - Route Accuracy    : did the agent choose the correct route?
       - Tool Precision    : did the agent use the right tools?
       - Iteration Count   : how many reasoning steps were needed?
       - Escalation Rate   : how often did the agent escalate?

    4. Performance
       - Latency     : end-to-end response time per query

Usage:
    # Run full evaluation
    python evaluation/eval.py

    # Limit to first N questions
    python evaluation/eval.py --limit 10

    # Save results
    python evaluation/eval.py --output evaluation/results.json
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Optional
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_QA_PATH = "data/raw/qa_pairs.json"
DEFAULT_DOCS_PATH = "data/raw"

# -------------------------------------------------------------------
# Expected route mappings for evaluation Q&A pairs
# Used to score route accuracy
# -------------------------------------------------------------------
EXPECTED_ROUTES = {
    "Easy": "hr_retrieval",
    "Medium": "hr_retrieval",
    "Hard": "multi_step",
}

# -------------------------------------------------------------------
# LLM-as-judge prompts
# -------------------------------------------------------------------

FAITHFULNESS_PROMPT = """\
You are evaluating whether an AI-generated answer is faithful to the provided context.
Faithful means the answer only contains information present in the context.

Context:
{context}

Answer:
{answer}

Score faithfulness 1-5 (5=completely faithful). Respond with ONLY a single integer.
"""

RELEVANCE_PROMPT = """\
You are evaluating whether an AI-generated answer is relevant to the question.

Question:
{question}

Answer:
{answer}

Score relevance 1-5 (5=directly answers the question). Respond with ONLY a single integer.
"""

CORRECTNESS_PROMPT = """\
You are evaluating whether an AI-generated answer matches the expected ground truth.

Question:
{question}

Expected Answer:
{expected_answer}

Generated Answer:
{generated_answer}

Score correctness 1-5 (5=completely correct). Respond with ONLY a single integer.
"""


def load_qa_pairs(qa_path: str) -> list[dict]:
    """Load evaluation Q&A pairs from a JSON file."""
    path = Path(qa_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Q&A pairs file not found: {qa_path}\n"
            f"Generate it by converting meridian-evaluation-qa.md to JSON."
        )
    with open(path, "r") as f:
        data = json.load(f)
        pairs = data["pairs"]

    logger.info(f"Loaded {len(pairs)} Q&A pairs from {qa_path}")
    return pairs


def _score_with_llm(llm, prompt: str) -> int:
    """
    Use the LLM to score a response on a 1-5 scale.
    Uses LangChain's .invoke() which returns an AIMessage with .content.
    """
    try:
        response = llm.invoke(prompt)
        score = int(response.content.strip())
        return max(1, min(5, score))
    except (ValueError, AttributeError) as e:
        logger.warning(f"Failed to parse LLM score: {e}")
        return 0


def evaluate_faithfulness(llm, answer: str, context: str) -> int:
    prompt = FAITHFULNESS_PROMPT.format(context=context, answer=answer)
    return _score_with_llm(llm, prompt)


def evaluate_relevance(llm, question: str, answer: str) -> int:
    prompt = RELEVANCE_PROMPT.format(question=question, answer=answer)
    return _score_with_llm(llm, prompt)


def evaluate_correctness(
    llm, question: str, expected_answer: str, generated_answer: str
) -> int:
    prompt = CORRECTNESS_PROMPT.format(
        question=question,
        expected_answer=expected_answer,
        generated_answer=generated_answer,
    )
    return _score_with_llm(llm, prompt)


def evaluate_route_accuracy(actual_route: str, difficulty: str) -> bool:
    """
    Check whether the agent chose the expected route for this question.

    Args:
        actual_route (str): Route chosen by the agent.
        difficulty (str): Question difficulty (Easy/Medium/Hard).

    Returns:
        bool: True if the route matches the expected route for this difficulty.
    """
    expected = EXPECTED_ROUTES.get(difficulty, "hr_retrieval")
    return actual_route == expected


def evaluate_tool_precision(tool_calls: list[dict], route: str) -> bool:
    """
    Check whether the agent used appropriate tools for the given route.

    Args:
        tool_calls (list[dict]): Tool calls made by the agent.
        route (str): The classified route.

    Returns:
        bool: True if tool usage was appropriate for the route.
    """
    if not tool_calls:
        return route in ("out_of_scope", "escalation")

    tool_names = {tc.get("tool") for tc in tool_calls}

    route_tool_map = {
        "hr_retrieval": {"hr_policy_retriever"},
        "calculation": {"calculator"},
        "date_lookup": {"date_calculator"},
        "multi_step": {"hr_policy_retriever", "calculator", "date_calculator"},
        "escalation": {"escalation_router"},
    }

    expected_tools = route_tool_map.get(route, set())
    return bool(tool_names & expected_tools)


def evaluate_retrieval_hit(sources: list, expected_answer: str) -> bool:
    """
    Check if any retrieved source contains content relevant to the expected answer.
    Works with the agentic pipeline's sources list format (list of dicts).

    Args:
        sources (list): Source dicts from pipeline response.
        expected_answer (str): Ground truth answer.

    Returns:
        bool: True if a relevant source was retrieved.
    """
    if not sources:
        return False

    expected_lower = expected_answer.lower()
    key_terms = [w for w in expected_lower.split() if len(w) > 4]

    for source in sources:
        # sources is a list of dicts with keys: file_name, page, score, text_preview
        chunk_text = source.get("text_preview", "").lower()
        if not chunk_text:
            continue
        matches = sum(1 for term in key_terms if term in chunk_text)
        if matches >= max(1, len(key_terms) // 2):
            return True
    return False


def run_evaluation(
    pipeline,
    qa_pairs: list[dict],
    limit: Optional[int] = None,
) -> dict:
    """
    Run the full evaluation suite against the Q&A pairs.

    Args:
        pipeline: Initialized AgenticRAGPipeline instance.
        qa_pairs (list[dict]): Ground truth Q&A pairs.
        limit (Optional[int]): Limit evaluation to first N pairs.

    Returns:
        dict: Evaluation results with per-question scores and aggregates.
    """
    from src.generation.generator import get_llm

    llm = get_llm()
    pairs = qa_pairs[:limit] if limit else qa_pairs

    logger.info(f"Running agentic evaluation on {len(pairs)} Q&A pairs...")

    results = []
    total_latency = 0

    for i, pair in enumerate(pairs, 1):
        question = pair["question"]
        expected = pair["expected_answer"]
        difficulty = pair.get("difficulty", "Unknown")
        source_section = pair.get("source_section", "Unknown")

        logger.info(f"Evaluating [{i}/{len(pairs)}]: {question[:60]}...")

        start_time = time.time()

        try:
            # Run through agentic pipeline
            response = pipeline.query(question)
            latency = round(time.time() - start_time, 3)
            total_latency += latency

            answer = response["answer"]
            route = response["route"]
            tool_calls = response.get("tool_calls", [])
            sources = response.get("sources", [])
            iterations = response.get("iterations", 1)

            # Build context string from sources for faithfulness scoring
            context = " ".join(
                src.get("text_preview", "") for src in sources
            ) or "No context retrieved."

            # Score all dimensions
            hit = evaluate_retrieval_hit(sources, expected)
            faithfulness = evaluate_faithfulness(llm, answer, context)
            relevance = evaluate_relevance(llm, question, answer)
            correctness = evaluate_correctness(llm, question, expected, answer)
            route_accurate = evaluate_route_accuracy(route, difficulty)
            tool_precise = evaluate_tool_precision(tool_calls, route)

            result = {
                "question": question,
                "expected_answer": expected,
                "generated_answer": answer,
                "source_section": source_section,
                "difficulty": difficulty,
                "route": route,
                "route_reasoning": response.get("route_reasoning", ""),
                "route_confidence": response.get("route_confidence", 0),
                "iterations": iterations,
                "retrieval_hit": hit,
                "faithfulness_score": faithfulness,
                "relevance_score": relevance,
                "correctness_score": correctness,
                "route_accurate": route_accurate,
                "tool_precise": tool_precise,
                "num_tool_calls": len(tool_calls),
                "tool_calls": tool_calls,
                "sources": sources,
                "num_sources": len(sources),
                "escalated": route == "escalation",
                "latency_seconds": latency,
                "error": None,
            }

        except Exception as e:
            logger.error(f"Error evaluating question {i}: {e}")
            result = {
                "question": question,
                "expected_answer": expected,
                "generated_answer": None,
                "source_section": source_section,
                "difficulty": difficulty,
                "route": None,
                "route_reasoning": "",
                "route_confidence": 0,
                "iterations": 0,
                "retrieval_hit": False,
                "faithfulness_score": 0,
                "relevance_score": 0,
                "correctness_score": 0,
                "route_accurate": False,
                "tool_precise": False,
                "num_tool_calls": 0,
                "tool_calls": [],
                "sources": [],
                "num_sources": 0,
                "escalated": False,
                "latency_seconds": 0,
                "error": str(e),
            }

        results.append(result)

    # Compute aggregate metrics
    valid = [r for r in results if r["error"] is None]
    n = len(valid)

    metrics = {
        "total_questions": len(results),
        "evaluated_questions": n,
        # Inherited metrics
        "hit_rate": round(sum(r["retrieval_hit"] for r in valid) / n, 4) if n else 0,
        "avg_faithfulness": round(sum(r["faithfulness_score"] for r in valid) / n, 4) if n else 0,
        "avg_relevance": round(sum(r["relevance_score"] for r in valid) / n, 4) if n else 0,
        "avg_correctness": round(sum(r["correctness_score"] for r in valid) / n, 4) if n else 0,
        # Agent-specific metrics
        "route_accuracy": round(sum(r["route_accurate"] for r in valid) / n, 4) if n else 0,
        "tool_precision": round(sum(r["tool_precise"] for r in valid) / n, 4) if n else 0,
        "avg_iterations": round(sum(r["iterations"] for r in valid) / n, 2) if n else 0,
        "escalation_rate": round(sum(r["escalated"] for r in valid) / n, 4) if n else 0,
        "avg_latency_seconds": round(total_latency / n, 3) if n else 0,
        "by_difficulty": _aggregate_by_difficulty(valid),
        "by_route": _aggregate_by_route(valid),
    }

    return {"metrics": metrics, "results": results}


def _aggregate_by_difficulty(results: list[dict]) -> dict:
    """Aggregate correctness scores grouped by difficulty."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        groups[r["difficulty"]].append(r["correctness_score"])
    return {
        d: round(sum(s) / len(s), 4)
        for d, s in groups.items()
    }


def _aggregate_by_route(results: list[dict]) -> dict:
    """Aggregate correctness and count grouped by route."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        if r["route"]:
            groups[r["route"]].append(r["correctness_score"])
    return {
        route: {
            "count": len(scores),
            "avg_correctness": round(sum(scores) / len(scores), 4),
        }
        for route, scores in groups.items()
    }


def print_summary(metrics: dict) -> None:
    """Print a formatted evaluation summary."""
    print("\n" + "=" * 65)
    print(" Agentic RAG Pipeline — Evaluation Summary")
    print("=" * 65)
    print(f"  Total questions     : {metrics['total_questions']}")
    print(f"  Evaluated           : {metrics['evaluated_questions']}")
    print("-" * 65)
    print("  Retrieval & Answer Quality:")
    print(f"    Hit Rate          : {metrics['hit_rate'] * 100:.1f}%")
    print(f"    Avg Faithfulness  : {metrics['avg_faithfulness']:.2f} / 5")
    print(f"    Avg Relevance     : {metrics['avg_relevance']:.2f} / 5")
    print(f"    Avg Correctness   : {metrics['avg_correctness']:.2f} / 5")
    print("-" * 65)
    print("  Agent-Specific Metrics:")
    print(f"    Route Accuracy    : {metrics['route_accuracy'] * 100:.1f}%")
    print(f"    Tool Precision    : {metrics['tool_precision'] * 100:.1f}%")
    print(f"    Avg Iterations    : {metrics['avg_iterations']:.1f}")
    print(f"    Escalation Rate   : {metrics['escalation_rate'] * 100:.1f}%")
    print(f"    Avg Latency       : {metrics['avg_latency_seconds']:.3f}s")
    print("-" * 65)
    print("  Correctness by Difficulty:")
    for difficulty, score in metrics["by_difficulty"].items():
        print(f"    {difficulty:<12}: {score:.2f} / 5")
    print("-" * 65)
    print("  Results by Route:")
    for route, data in metrics["by_route"].items():
        print(
            f"    {route:<20}: {data['count']} questions | "
            f"avg correctness: {data['avg_correctness']:.2f}"
        )
    print("=" * 65 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the Meridian Capital Group Agentic RAG pipeline"
    )
    parser.add_argument("--qa", type=str, default=DEFAULT_QA_PATH)
    parser.add_argument("--docs", type=str, default=DEFAULT_DOCS_PATH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    try:
        from src.pipeline import AgenticRAGPipeline

        pipeline = AgenticRAGPipeline(docs_path=args.docs)
        qa_pairs = load_qa_pairs(args.qa)

        eval_results = run_evaluation(
            pipeline=pipeline,
            qa_pairs=qa_pairs,
            limit=args.limit,
        )

        print_summary(eval_results["metrics"])

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(eval_results, f, indent=2)
            logger.info(f"Full results saved to: {args.output}")

    except (FileNotFoundError, EnvironmentError) as e:
        print(f"Error: {e}")
        sys.exit(1)