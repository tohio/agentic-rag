"""
embedder.py
-----------
Responsible for initializing and configuring the embedding model
using LangChain's OpenAI embeddings integration.

Intentionally uses LangChain instead of LlamaIndex (used in rag-pipeline)
to demonstrate framework diversity and because LangChain embeddings
integrate natively with LangChain's Pinecone vector store and retriever.

Model: OpenAI text-embedding-3-small (default)
    - 1536 dimensions
    - Fast and cost-effective for retrieval tasks

Configuration (in order of precedence):
    1. Environment variables (OPENAI_API_KEY, EMBEDDING_MODEL)
    2. Sensible defaults (model: text-embedding-3-small)

Production note:
    For high-volume workloads, consider text-embedding-3-large for
    higher accuracy or a locally hosted HuggingFace model to eliminate
    per-token API costs entirely.

Usage:
    from src.embedding.embedder import get_embed_model, embed_texts
    embed_model = get_embed_model()
    vectors = embed_texts(["some text"], embed_model)
"""

import logging
import os
from typing import List

from langchain_openai import OpenAIEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def get_embed_model() -> OpenAIEmbeddings:
    """
    Initialize and return the LangChain OpenAI embedding model.

    Returns:
        OpenAIEmbeddings: Configured LangChain embedding model instance.

    Raises:
        EnvironmentError: If OPENAI_API_KEY is not set.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Please add it to your .env file."
        )

    model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    logger.info(f"Initializing LangChain embedding model: {model_name}")

    embed_model = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=api_key,
    )

    logger.info(f"Embedding model ready: {model_name}")
    return embed_model


def embed_texts(
    texts: List[str],
    embed_model: OpenAIEmbeddings,
) -> List[List[float]]:
    """
    Generate embeddings for a list of text strings.

    Args:
        texts (List[str]): List of text strings to embed.
        embed_model (OpenAIEmbeddings): Initialized embedding model.

    Returns:
        List[List[float]]: List of embedding vectors.

    Raises:
        ValueError: If texts list is empty.
    """
    if not texts:
        raise ValueError("No texts provided for embedding.")

    logger.info(f"Embedding {len(texts)} text(s)...")
    embeddings = embed_model.embed_documents(texts)
    logger.info(
        f"Successfully embedded {len(texts)} text(s) | "
        f"dimensions: {len(embeddings[0])}"
    )
    return embeddings


def embed_query(query: str, embed_model: OpenAIEmbeddings) -> List[float]:
    """
    Generate an embedding for a single query string.

    Uses embed_query() instead of embed_documents() — LangChain
    uses different encoding for queries vs documents in some models.

    Args:
        query (str): Query string to embed.
        embed_model (OpenAIEmbeddings): Initialized embedding model.

    Returns:
        List[float]: Query embedding vector.

    Raises:
        ValueError: If query is empty.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    return embed_model.embed_query(query)


# -------------------------------------------------------------------
# Quick test
# python -m src.embedding.embedder
# -------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()

    try:
        embed_model = get_embed_model()
        test_texts = [
            "Employees receive 15 days of PTO in their first year.",
            "The 401(k) match is up to 5% of base salary.",
        ]
        vectors = embed_texts(test_texts, embed_model)

        print(f"\n--- Embedder Test Results ---")
        print(f"Texts embedded    : {len(vectors)}")
        print(f"Embedding dims    : {len(vectors[0])}")
        print(f"Sample (first 5)  : {vectors[0][:5]}")

        query_vec = embed_query("What is the PTO policy?", embed_model)
        print(f"Query embedding dims: {len(query_vec)}")

    except (EnvironmentError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)
