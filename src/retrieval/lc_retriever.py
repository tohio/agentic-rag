"""
retriever.py
------------
Responsible for querying the Pinecone vector store and retrieving
the most relevant document chunks using LangChain's retriever interface.

Key difference from rag-pipeline/retriever.py:
    Uses LangChain's VectorStoreRetriever instead of LlamaIndex's
    VectorIndexRetriever. This integrates natively with LangGraph's
    state graph and LangChain's runnable interface.

LangChain retriever types supported:
    - similarity         : top-k cosine similarity (default)
    - mmr                : maximal marginal relevance (reduces redundancy)
    - similarity_score_threshold: filters by minimum similarity score

Configuration (in order of precedence):
    1. Function arguments
    2. Environment variables (RETRIEVAL_TOP_K, SIMILARITY_CUTOFF)
    3. Sensible defaults (top_k=5, similarity_cutoff=0.3)

Production note:
    Dense retrieval works well for semantic queries but can miss
    exact keyword matches. In production, Pinecone's hybrid search
    (dense + sparse BM25) typically outperforms either method alone.

Usage:
    from src.retrieval.retriever import get_retriever, retrieve
    retriever = get_retriever(vectorstore)
    results = retrieve(retriever, "What is the PTO policy?")
"""

import logging
import os
from typing import List, Optional

from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5
DEFAULT_SIMILARITY_CUTOFF = 0.3


def _get_config(
    top_k: Optional[int],
    similarity_cutoff: Optional[float],
) -> tuple[int, float]:
    """Resolve retrieval config from args > env vars > defaults."""
    resolved_top_k = top_k or int(os.getenv("RETRIEVAL_TOP_K", DEFAULT_TOP_K))
    resolved_cutoff = similarity_cutoff or float(
        os.getenv("SIMILARITY_CUTOFF", DEFAULT_SIMILARITY_CUTOFF)
    )
    return resolved_top_k, resolved_cutoff


def get_retriever(
    vectorstore: PineconeVectorStore,
    top_k: Optional[int] = None,
    similarity_cutoff: Optional[float] = None,
    search_type: str = "similarity",
):
    """
    Initialize and return a configured LangChain retriever.

    Args:
        vectorstore (PineconeVectorStore): Initialized LangChain vector store.
        top_k (Optional[int]): Number of chunks to retrieve.
        similarity_cutoff (Optional[float]): Minimum similarity score.
            Only applied when search_type='similarity_score_threshold'.
        search_type (str): Retrieval strategy:
            - 'similarity'                 : standard top-k (default)
            - 'mmr'                        : max marginal relevance
            - 'similarity_score_threshold' : filter by score

    Returns:
        VectorStoreRetriever: Configured LangChain retriever.
    """
    resolved_top_k, resolved_cutoff = _get_config(top_k, similarity_cutoff)

    logger.info(
        f"Initializing LangChain retriever — "
        f"search_type: {search_type} | top_k: {resolved_top_k}"
    )

    search_kwargs = {"k": resolved_top_k}

    if search_type == "similarity_score_threshold":
        search_kwargs["score_threshold"] = resolved_cutoff

    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )

    logger.info("Retriever ready")
    return retriever


def retrieve(
    retriever,
    query: str,
) -> List[Document]:
    """
    Retrieve the most relevant document chunks for a given query.

    Args:
        retriever: Initialized LangChain retriever.
        query (str): The user's natural language query.

    Returns:
        List[Document]: Retrieved LangChain Document chunks.

    Raises:
        ValueError: If query is empty or whitespace.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    logger.info(f"Retrieving chunks for query: '{query}'")

    results = retriever.invoke(query)

    logger.info(f"Retrieved {len(results)} chunk(s)")

    if not results:
        logger.warning(
            "No chunks retrieved. Consider lowering SIMILARITY_CUTOFF "
            "or checking your Pinecone index."
        )

    return results


def retrieve_with_scores(
    vectorstore: PineconeVectorStore,
    query: str,
    top_k: Optional[int] = None,
) -> List[tuple[Document, float]]:
    """
    Retrieve chunks with their similarity scores.

    LangChain's standard retriever does not return scores by default.
    This function uses similarity_search_with_score() directly on
    the vector store for cases where scores are needed — specifically
    for the agent's re-retrieval threshold check.

    Args:
        vectorstore (PineconeVectorStore): Initialized LangChain vector store.
        query (str): The user's natural language query.
        top_k (Optional[int]): Number of chunks to retrieve.

    Returns:
        List[Tuple[Document, float]]: List of (document, score) tuples
            sorted by score descending.

    Raises:
        ValueError: If query is empty.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    resolved_top_k, _ = _get_config(top_k, None)

    logger.info(f"Retrieving chunks with scores for query: '{query}'")

    results = vectorstore.similarity_search_with_score(
        query=query,
        k=resolved_top_k,
    )

    logger.info(
        f"Retrieved {len(results)} chunk(s) | "
        f"max score: {results[0][1]:.4f}" if results else "no results"
    )

    return results


def retrieve_with_metadata(
    vectorstore: PineconeVectorStore,
    query: str,
    top_k: Optional[int] = None,
    similarity_cutoff: Optional[float] = None,
) -> dict:
    """
    Retrieve chunks and return a structured result dict for the agent.

    This is the agent-facing retrieval function. Returns structured
    metadata the router and agent use to decide whether retrieval
    was successful or a different strategy should be attempted.

    Args:
        vectorstore (PineconeVectorStore): Initialized vector store.
        query (str): The user's natural language query.
        top_k (Optional[int]): Number of chunks to retrieve.
        similarity_cutoff (Optional[float]): Minimum score threshold.

    Returns:
        dict: Structured retrieval result:
            - query        : original query
            - results      : List[Tuple[Document, float]]
            - documents    : List[Document] (without scores)
            - num_results  : number of chunks retrieved
            - max_score    : highest similarity score
            - success      : True if results were found above cutoff
            - context      : formatted context string for the LLM
    """
    _, resolved_cutoff = _get_config(top_k, similarity_cutoff)

    results_with_scores = retrieve_with_scores(vectorstore, query, top_k)

    # Filter by similarity cutoff
    filtered = [
        (doc, score) for doc, score in results_with_scores
        if score >= resolved_cutoff
    ]

    if not filtered and results_with_scores:
        logger.warning(
            f"All {len(results_with_scores)} chunk(s) below cutoff "
            f"{resolved_cutoff}. Top score: {results_with_scores[0][1]:.4f}"
        )

    documents = [doc for doc, _ in filtered]
    context = format_retrieved_context(filtered)

    return {
        "query": query,
        "results": filtered,
        "documents": documents,
        "num_results": len(filtered),
        "max_score": round(filtered[0][1], 4) if filtered else 0.0,
        "success": len(filtered) > 0,
        "context": context,
    }


def format_retrieved_context(
    results: List[tuple[Document, float]],
) -> str:
    """
    Format retrieved chunks into a labeled context string for the LLM.

    Args:
        results (List[Tuple[Document, float]]): Retrieved (doc, score) pairs.

    Returns:
        str: Formatted context string ready for LLM prompt injection.
    """
    if not results:
        return "No relevant context found."

    context_parts = []
    for i, (doc, score) in enumerate(results, 1):
        file_name = doc.metadata.get("file_name", "unknown")
        page = doc.metadata.get("page_label", "unknown")

        context_parts.append(
            f"[Chunk {i} | Source: {file_name} | "
            f"Page: {page} | Score: {score:.3f}]\n"
            f"{doc.page_content}"
        )

    return "\n\n---\n\n".join(context_parts)


def get_retrieval_metadata(
    results: List[tuple[Document, float]],
) -> List[dict]:
    """
    Extract metadata from retrieved results for logging and evaluation.

    Args:
        results (List[Tuple[Document, float]]): Retrieved (doc, score) pairs.

    Returns:
        List[dict]: Metadata per retrieved chunk.
    """
    return [
        {
            "chunk_index": i,
            "file_name": doc.metadata.get("file_name", "unknown"),
            "page_label": doc.metadata.get("page_label", "unknown"),
            "similarity_score": round(score, 4),
            "text_preview": doc.page_content[:120].replace("\n", " ") + "...",
        }
        for i, (doc, score) in enumerate(results, 1)
    ]


# -------------------------------------------------------------------
# Quick test
# python -m src.retrieval.retriever
# -------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    from src.ingestion.loader import load_documents
    from src.ingestion.chunker import chunk_documents
    from src.embedding.embedder import get_embed_model
    from src.vectorstore.store import get_vector_store
    load_dotenv()

    test_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    test_query = sys.argv[2] if len(sys.argv) > 2 else "What is the PTO policy?"

    try:
        raw_docs = load_documents(test_path)
        lc_docs = chunk_documents(raw_docs)
        embed_model = get_embed_model()
        vectorstore = get_vector_store(documents=lc_docs, embed_model=embed_model)

        result = retrieve_with_metadata(vectorstore, test_query)
        metadata = get_retrieval_metadata(result["results"])

        print(f"\n--- Retriever Test Results ---")
        print(f"Query      : {result['query']}")
        print(f"Success    : {result['success']}")
        print(f"Chunks     : {result['num_results']}")
        print(f"Max Score  : {result['max_score']}")
        print(f"\nChunk Details:")
        for meta in metadata:
            print(
                f"  [{meta['chunk_index']}] "
                f"Score: {meta['similarity_score']} | "
                f"Page: {meta['page_label']}"
            )
            print(f"  Preview: {meta['text_preview']}\n")

        print(f"--- Context Preview ---")
        print(result["context"][:500] + "...")

    except (FileNotFoundError, ValueError, EnvironmentError) as e:
        print(f"Error: {e}")
        sys.exit(1)
