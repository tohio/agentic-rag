"""
store.py
--------
Responsible for creating, persisting, and loading the Pinecone vector store
using LangChain's Pinecone integration.

Strategy: Load if exists, create if not.
    - First run: creates a Pinecone index, embeds documents, and upserts vectors
    - Subsequent runs: connects to the existing index instantly
    - Force rebuild: set FORCE_REBUILD=true in .env or pass rebuild=True

Configuration (in order of precedence):
    1. Environment variables (see .env.sample for all options)
    2. Sensible defaults

Required environment variables:
    PINECONE_API_KEY        : your Pinecone API key
    PINECONE_INDEX_NAME     : name of your Pinecone index
    PINECONE_CLOUD          : cloud provider (aws | gcp | azure)
    PINECONE_REGION         : cloud region (e.g. us-east-1)

Key difference from rag-pipeline:
    Uses LangChain's PineconeVectorStore instead of LlamaIndex's.
    This integrates natively with LangChain retrievers and the
    LangGraph agent state graph.

Production note:
    For multi-tenant production use, configure Pinecone namespaces
    to isolate data per client or department.

Usage:
    from src.vectorstore.store import get_vector_store
    from langchain_core.documents import Document

    docs = [Document(page_content="text", metadata={"source": "handbook"})]
    vectorstore = get_vector_store(documents=docs)
"""

import logging
import os
from typing import List, Optional

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_INDEX_NAME = "agentic-rag-meridian"
DEFAULT_CLOUD = "aws"
DEFAULT_REGION = "us-east-1"
EMBEDDING_DIMENSION = 1536  # text-embedding-3-small


def _get_config() -> tuple[str, str, str, str, bool]:
    """
    Resolve Pinecone configuration from environment variables.

    Returns:
        Tuple of (api_key, index_name, cloud, region, force_rebuild).

    Raises:
        EnvironmentError: If PINECONE_API_KEY is not set.
    """
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "PINECONE_API_KEY is not set. "
            "Please add it to your .env file."
        )

    index_name = os.getenv("PINECONE_INDEX_NAME", DEFAULT_INDEX_NAME)
    cloud = os.getenv("PINECONE_CLOUD", DEFAULT_CLOUD)
    region = os.getenv("PINECONE_REGION", DEFAULT_REGION)
    force_rebuild = os.getenv("FORCE_REBUILD", "false").lower() == "true"

    return api_key, index_name, cloud, region, force_rebuild


def _get_pinecone_client(api_key: str) -> Pinecone:
    """Initialize and return a Pinecone client."""
    return Pinecone(api_key=api_key)


def _index_exists(pc: Pinecone, index_name: str) -> bool:
    """
    Check whether a Pinecone index exists and contains vectors.

    Args:
        pc (Pinecone): Initialized Pinecone client.
        index_name (str): Name of the Pinecone index.

    Returns:
        bool: True if index exists and has vectors, False otherwise.
    """
    try:
        existing = [idx.name for idx in pc.list_indexes()]
        if index_name not in existing:
            return False

        stats = pc.Index(index_name).describe_index_stats()
        total = stats.get("total_vector_count", 0)
        logger.info(
            f"Existing Pinecone index: '{index_name}' | vectors: {total}"
        )
        return total > 0
    except Exception as e:
        logger.warning(f"Could not check Pinecone index: {e}")
        return False


def _create_pinecone_index(
    pc: Pinecone,
    index_name: str,
    cloud: str,
    region: str,
) -> None:
    """Create a new Pinecone serverless index."""
    logger.info(
        f"Creating Pinecone index: '{index_name}' | "
        f"cloud: {cloud} | region: {region} | dims: {EMBEDDING_DIMENSION}"
    )
    pc.create_index(
        name=index_name,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud=cloud, region=region),
    )
    logger.info(f"Pinecone index '{index_name}' created")


def build_vector_store(
    documents: List[Document],
    embed_model: OpenAIEmbeddings,
    rebuild: bool = False,
) -> PineconeVectorStore:
    """
    Build a new LangChain Pinecone vector store from documents.

    Creates the Pinecone index if it does not exist, embeds all
    documents, and upserts them into the index.

    Args:
        documents (List[Document]): LangChain Document objects to index.
        embed_model (OpenAIEmbeddings): Initialized LangChain embeddings.
        rebuild (bool): If True, wipe existing vectors before indexing.

    Returns:
        PineconeVectorStore: Ready-to-use LangChain vector store.

    Raises:
        ValueError: If no documents are provided.
        EnvironmentError: If PINECONE_API_KEY is not set.
    """
    if not documents:
        raise ValueError("No documents provided to build vector store.")

    api_key, index_name, cloud, region, _ = _get_config()
    pc = _get_pinecone_client(api_key)

    existing = [idx.name for idx in pc.list_indexes()]

    if index_name not in existing:
        _create_pinecone_index(pc, index_name, cloud, region)
    elif rebuild:
        logger.info(f"Rebuild — deleting all vectors in '{index_name}'")
        pc.Index(index_name).delete(delete_all=True)

    logger.info(
        f"Upserting {len(documents)} document(s) into '{index_name}'..."
    )

    # API key is picked up automatically from PINECONE_API_KEY env var
    vectorstore = PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embed_model,
        index_name=index_name,
    )

    logger.info(f"Vector store built in Pinecone: '{index_name}'")
    return vectorstore


def load_vector_store(embed_model: OpenAIEmbeddings) -> PineconeVectorStore:
    """
    Connect to an existing Pinecone index as a LangChain vector store.

    Args:
        embed_model (OpenAIEmbeddings): Initialized LangChain embeddings.

    Returns:
        PineconeVectorStore: Loaded LangChain vector store.

    Raises:
        FileNotFoundError: If the Pinecone index does not exist or is empty.
        EnvironmentError: If PINECONE_API_KEY is not set.
    """
    api_key, index_name, _, _, _ = _get_config()
    pc = _get_pinecone_client(api_key)

    if not _index_exists(pc, index_name):
        raise FileNotFoundError(
            f"Pinecone index '{index_name}' does not exist or is empty. "
            f"Run the pipeline first to build the index."
        )

    logger.info(f"Loading existing Pinecone index: '{index_name}'")

    # API key is picked up automatically from PINECONE_API_KEY env var
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embed_model,
    )

    logger.info("Pinecone vector store loaded")
    return vectorstore


def get_vector_store(
    documents: Optional[List[Document]] = None,
    embed_model: Optional[OpenAIEmbeddings] = None,
    rebuild: bool = False,
) -> PineconeVectorStore:
    """
    Main entry point — load vector store if it exists, build if not.

    Args:
        documents (Optional[List[Document]]): LangChain Document objects.
            Required if building a new index.
        embed_model (Optional[OpenAIEmbeddings]): Initialized embeddings.
            Required for both building and loading.
        rebuild (bool): Force rebuild even if an index exists.

    Returns:
        PineconeVectorStore: Ready-to-use LangChain vector store.

    Raises:
        ValueError: If embed_model is not provided.
        EnvironmentError: If PINECONE_API_KEY is not set.
    """
    if embed_model is None:
        raise ValueError("embed_model is required.")

    api_key, index_name, _, _, force_rebuild = _get_config()
    pc = _get_pinecone_client(api_key)
    should_rebuild = rebuild or force_rebuild

    if not should_rebuild and _index_exists(pc, index_name):
        logger.info("Existing Pinecone index detected — loading")
        return load_vector_store(embed_model)

    if not documents:
        raise ValueError(
            "No existing Pinecone index found and no documents provided. "
            "Run the full ingestion pipeline first."
        )

    logger.info("Building new Pinecone vector store...")
    return build_vector_store(documents, embed_model, rebuild=should_rebuild)


# -------------------------------------------------------------------
# Quick test
# python -m src.vectorstore.store
# -------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from dotenv import load_dotenv
    from src.ingestion.loader import load_documents
    from src.ingestion.chunker import chunk_documents
    from src.embedding.embedder import get_embed_model
    load_dotenv()

    test_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw"

    try:
        raw_docs = load_documents(test_path)
        lc_docs = chunk_documents(raw_docs)
        embed_model = get_embed_model()

        vectorstore = get_vector_store(
            documents=lc_docs,
            embed_model=embed_model,
        )

        print(f"\n--- Store Test Results ---")
        print(f"Vector store type : {type(vectorstore).__name__}")
        print(f"Index name        : {os.getenv('PINECONE_INDEX_NAME', DEFAULT_INDEX_NAME)}")
        print(f"Ready for retrieval.")

    except (FileNotFoundError, ValueError, EnvironmentError) as e:
        print(f"Error: {e}")
        sys.exit(1)