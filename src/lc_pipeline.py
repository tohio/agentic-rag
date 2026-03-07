"""
pipeline.py
-----------
Orchestrates the full end-to-end agentic RAG pipeline using LangGraph.

This is the main entry point for the agentic-rag project. It wires
together all components and exposes a clean interface for the
Streamlit UI, evaluation suite, and CLI.

Initialization flow:
    1. Load embed model and LLM
    2. Check if Pinecone index exists
    3. If not: load docs → chunk → build index
    4. If yes: connect to existing index
    5. Build LangGraph StateGraph (build_graph)
    6. Ready for queries

Query flow (delegated to LangGraph):
    invoke(initial_state)
      → route_query
      → [retrieve | calculate | date_lookup | decompose | escalate]
      → check_retrieval (if retrieval route)
      → generate
      → update_memory
      → END

Usage:
    # Interactive mode
    python src/pipeline.py

    # Single query
    python src/pipeline.py --query "How many PTO days do I get?"

    # Force rebuild Pinecone index
    python src/pipeline.py --rebuild
"""

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_DOCS_PATH = "data/raw"


class AgenticRAGPipeline:
    """
    End-to-end agentic RAG pipeline powered by LangGraph.

    Manages initialization state, exposes a clean query interface,
    and handles warm starts (existing Pinecone index) vs cold starts
    (first run, index must be built).

    Attributes:
        docs_path (str): Path to source documents.
        rebuild (bool): Whether to force rebuild the Pinecone index.
        graph: Compiled LangGraph StateGraph.
        memory (ConversationMemory): Shared conversation memory.
        vectorstore: Initialized LangChain Pinecone vector store.
        llm: Initialized LangChain chat model.
    """

    def __init__(
        self,
        docs_path: str = DEFAULT_DOCS_PATH,
        rebuild: bool = False,
    ):
        self.docs_path = docs_path
        self.rebuild = rebuild
        self.graph = None
        self.memory = None
        self.vectorstore = None
        self.llm = None
        self._initialize()

    def _initialize(self) -> None:
        """
        Initialize all pipeline components and compile the LangGraph.

        Handles warm start (existing index) vs cold start (build index).
        On warm start, ingestion and embedding are skipped entirely.
        """
        logger.info("Initializing Agentic RAG pipeline (LangChain + LangGraph)...")

        from src.embedding.embedder import get_embed_model
        from src.generation.generator import get_llm
        from src.vectorstore.store import (
            get_vector_store,
            _index_exists,
            _get_config,
        )
        from src.memory.memory import ConversationMemory
        from src.agents.agent import build_graph
        from pinecone import Pinecone

        # Initialize LLM and embeddings
        embed_model = get_embed_model()
        self.llm = get_llm()

        # Decide whether to build or load the Pinecone index
        api_key, index_name, _, _, force_rebuild = _get_config()
        pc = Pinecone(api_key=api_key)
        should_rebuild = self.rebuild or force_rebuild

        if not should_rebuild and _index_exists(pc, index_name):
            logger.info("Warm start — connecting to existing Pinecone index")
            self.vectorstore = get_vector_store(embed_model=embed_model)
        else:
            logger.info("Cold start — building Pinecone index from documents")
            from src.ingestion.loader import load_documents
            from src.ingestion.chunker import chunk_documents

            raw_docs = load_documents(self.docs_path)
            lc_docs = chunk_documents(raw_docs)
            self.vectorstore = get_vector_store(
                documents=lc_docs,
                embed_model=embed_model,
                rebuild=should_rebuild,
            )

        # Initialize memory
        self.memory = ConversationMemory()

        # Compile the LangGraph StateGraph
        self.graph = build_graph(
            llm=self.llm,
            vectorstore=self.vectorstore,
            memory=self.memory,
        )

        logger.info("Agentic RAG pipeline ready (LangGraph compiled)")

    def query(self, question: str) -> dict:
        """
        Run a single query through the full LangGraph pipeline.

        Constructs the initial AgentState, invokes the compiled graph,
        and returns a structured response dict.

        Args:
            question (str): Natural language question from the user.

        Returns:
            dict: Structured response containing:
                - query       : original question
                - answer      : final generated answer
                - reasoning   : agent reasoning steps
                - tool_calls  : list of all tool calls made
                - sources     : retrieved document chunks
                - num_sources : number of sources
                - route       : classified query route
                - iterations  : number of graph iterations
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        logger.info(f"Processing query: '{question}'")

        from src.agents.agent import AgentState

        initial_state: AgentState = {
            "query": question.strip(),
            "route": "",
            "route_reasoning": "",
            "route_confidence": 0.0,
            "context": "",
            "retrieved_results": [],
            "retrieval_score": 0.0,
            "retrieval_attempts": 0,
            "tool_output": "",
            "sub_queries": [],
            "sub_query_results": [],
            "raw_answer": "",
            "answer": "",
            "reasoning": "",
            "tool_calls": [],
            "memory_context": self.memory.get_context_string(),
            "sources": [],
            "num_sources": 0,
            "iterations": 0,
            "error": None,
        }

        result = self.graph.invoke(initial_state)

        return {
            "query": result["query"],
            "answer": result["answer"],
            "reasoning": result.get("reasoning", ""),
            "tool_calls": result.get("tool_calls", []),
            "sources": result.get("sources", []),
            "num_sources": result.get("num_sources", 0),
            "route": result.get("route", "unknown"),
            "route_reasoning": result.get("route_reasoning", ""),
            "route_confidence": result.get("route_confidence", 0.0),
            "iterations": result.get("iterations", 1),
            "sub_queries": result.get("sub_queries", []),
        }

    def reset_memory(self) -> None:
        """Clear conversation memory to start a fresh session."""
        self.memory.clear()
        logger.info("Conversation memory reset")

    def get_graph_diagram(self) -> str:
        """
        Return an ASCII representation of the LangGraph structure.
        Useful for documentation and the Streamlit About tab.
        """
        return """
LangGraph Agent — State Flow
=============================
START
  ↓
[route_query]
  ↓ (conditional on route)
  ├─ hr_retrieval   → [retrieve] → [check_retrieval] ─┐
  ├─ calculation    → [calculate]                      │
  ├─ date_lookup    → [date_lookup]                    │
  ├─ multi_step     → [decompose_multi_step]            ├→ [generate] → [update_memory] → END
  ├─ escalation     → [handle_escalation] ─────────────┤
  └─ out_of_scope   → [handle_out_of_scope] ───────────┘

[check_retrieval]:
  score ≥ threshold → [generate]
  score < threshold → [retrieve]  (retry, max 2x)
"""

    def print_response(self, response: dict) -> None:
        """Pretty print a pipeline response to the console."""
        print("\n" + "=" * 65)
        print(f"QUESTION   : {response['query']}")
        print(f"ROUTE      : {response['route']} "
              f"(confidence: {response.get('route_confidence', 0):.2f})")
        print(f"ITERATIONS : {response['iterations']}")
        print("=" * 65)

        if response.get("sub_queries"):
            print(f"SUB-QUERIES:")
            for i, sq in enumerate(response["sub_queries"], 1):
                print(f"  {i}. {sq}")
            print("-" * 65)

        if response.get("reasoning"):
            print(f"REASONING  :\n{response['reasoning']}")
            print("-" * 65)

        print(f"ANSWER     : {response['answer']}")
        print("-" * 65)

        if response.get("tool_calls"):
            print(f"TOOLS USED : {len(response['tool_calls'])}")
            for tc in response["tool_calls"]:
                status = "✓" if tc["success"] else "✗"
                print(f"  [{status}] {tc['tool']}")

        print(f"SOURCES    : {response['num_sources']} chunk(s)")
        for src in response.get("sources", []):
            print(
                f"  - {src['file_name']} | "
                f"Page: {src['page_label']} | "
                f"Score: {src['similarity_score']}"
            )
        print("=" * 65 + "\n")


def run_interactive(pipeline: AgenticRAGPipeline) -> None:
    """Run the pipeline in interactive CLI mode."""
    print("\n" + "=" * 65)
    print(" Meridian Capital Group — Agentic HR Assistant")
    print(" LangChain + LangGraph | Type 'exit' to quit | 'reset' to clear memory")
    print(pipeline.get_graph_diagram())
    print("=" * 65 + "\n")

    while True:
        try:
            question = input("Ask a question: ").strip()

            if not question:
                continue
            if question.lower() in {"exit", "quit", "q"}:
                print("Goodbye.")
                break
            if question.lower() == "reset":
                pipeline.reset_memory()
                print("Memory cleared.\n")
                continue

            response = pipeline.query(question)
            pipeline.print_response(response)

        except KeyboardInterrupt:
            print("\nGoodbye.")
            break
        except ValueError as e:
            print(f"Input error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print(f"Something went wrong: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Meridian Capital Group Agentic HR RAG Pipeline"
    )
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--docs", type=str, default=DEFAULT_DOCS_PATH)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    try:
        pipeline = AgenticRAGPipeline(
            docs_path=args.docs,
            rebuild=args.rebuild,
        )

        if args.query:
            response = pipeline.query(args.query)
            pipeline.print_response(response)
        else:
            run_interactive(pipeline)

    except EnvironmentError as e:
        logger.error(f"Configuration error: {e}")
        print(f"\nConfiguration error: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"\nFile not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"\nPipeline error: {e}")
        sys.exit(1)
