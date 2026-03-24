"""
chunker.py
----------
Responsible for splitting loaded documents into smaller chunks
suitable for embedding and retrieval.

Key difference from rag-pipeline/chunker.py:
    Returns List[langchain.schema.Document] instead of
    List[llama_index.core.schema.BaseNode], so chunks integrate
    natively with LangChain's Pinecone vector store and retrievers.

Strategy: Fixed size token-based chunking with overlap using
    LangChain's RecursiveCharacterTextSplitter.
    - Splits on paragraph, sentence, then character boundaries
    - Preserves semantic coherence better than a naive fixed split
    - Overlap preserves context across chunk boundaries

Configuration (in order of precedence):
    1. Passed directly as arguments to chunk_documents()
    2. Environment variables (CHUNK_SIZE, CHUNK_OVERLAP)
    3. Sensible defaults (chunk_size=384, chunk_overlap=40)

Usage:
    from src.ingestion.chunker import chunk_documents
    from llama_index.core.schema import Document as LlamaDocument

    llama_docs = load_documents("data/raw")
    lc_docs = chunk_documents(llama_docs)
    # lc_docs is now List[langchain.schema.Document]
"""

import logging
import os
from typing import List, Optional

from langchain_core.documents import Document as LangChainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core.schema import Document as LlamaDocument

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Slightly smaller than rag-pipeline for improved retrieval
# precision in multi-hop agentic scenarios
DEFAULT_CHUNK_SIZE = 384
DEFAULT_CHUNK_OVERLAP = 40


def _get_config(
    chunk_size: Optional[int],
    chunk_overlap: Optional[int],
) -> tuple[int, int]:
    """Resolve chunk size and overlap from args > env vars > defaults."""
    resolved_size = chunk_size or int(
        os.getenv("CHUNK_SIZE", DEFAULT_CHUNK_SIZE)
    )
    resolved_overlap = chunk_overlap or int(
        os.getenv("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP)
    )

    if resolved_overlap >= resolved_size:
        raise ValueError(
            f"CHUNK_OVERLAP ({resolved_overlap}) must be less than "
            f"CHUNK_SIZE ({resolved_size})"
        )
    return resolved_size, resolved_overlap


def chunk_documents(
    documents: List[LlamaDocument],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[LangChainDocument]:
    """
    Split LlamaIndex documents into LangChain Document chunks.

    Accepts LlamaIndex Document objects from loader.py (which uses
    LlamaIndex's SimpleDirectoryReader for PDF loading) and returns
    LangChain Document objects ready for the LangChain/Pinecone pipeline.

    This bridge approach lets us reuse the LlamaIndex PDF loader
    while feeding LangChain-native objects to the rest of the pipeline.

    Args:
        documents (List[LlamaDocument]): Loaded documents from loader.py.
        chunk_size (Optional[int]): Characters per chunk.
        chunk_overlap (Optional[int]): Overlapping characters between chunks.

    Returns:
        List[LangChainDocument]: LangChain Document chunks ready for
            embedding and indexing into Pinecone.

    Raises:
        ValueError: If documents list is empty or overlap >= chunk size.
    """
    if not documents:
        raise ValueError("No documents provided to chunk.")

    resolved_size, resolved_overlap = _get_config(chunk_size, chunk_overlap)

    logger.info(
        f"Chunking {len(documents)} document(s) | "
        f"chunk_size={resolved_size} | chunk_overlap={resolved_overlap}"
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=resolved_size,
        chunk_overlap=resolved_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    lc_docs = []
    for doc in documents:
        # Convert LlamaIndex metadata to LangChain format
        metadata = {
            "file_name": doc.metadata.get("file_name", "unknown"),
            "file_path": doc.metadata.get("file_path", "unknown"),
            "page_label": doc.metadata.get("page_label", "unknown"),
            "source": doc.metadata.get("file_name", "unknown"),
        }

        chunks = splitter.split_text(doc.text)
        for chunk in chunks:
            lc_docs.append(
                LangChainDocument(
                    page_content=chunk,
                    metadata=metadata,
                )
            )

    logger.info(
        f"Produced {len(lc_docs)} chunk(s) from {len(documents)} document(s)"
    )
    _log_chunk_stats(lc_docs)
    return lc_docs


def _log_chunk_stats(docs: List[LangChainDocument]) -> None:
    """Log basic statistics about the produced chunks."""
    if not docs:
        return
    lengths = [len(doc.page_content) for doc in docs]
    avg_len = sum(lengths) / len(lengths)
    logger.info(
        f"Chunk stats — avg chars: {avg_len:.0f} | "
        f"min: {min(lengths)} | max: {max(lengths)}"
    )


def get_chunk_metadata(docs: List[LangChainDocument]) -> List[dict]:
    """
    Extract metadata from chunked LangChain Documents for debugging.

    Args:
        docs (List[LangChainDocument]): Chunked LangChain documents.

    Returns:
        List[dict]: Metadata for each chunk.
    """
    return [
        {
            "chunk_index": i,
            "file_name": doc.metadata.get("file_name", "unknown"),
            "page_label": doc.metadata.get("page_label", "unknown"),
            "char_count": len(doc.page_content),
            "text_preview": doc.page_content[:120].replace("\n", " ") + "...",
        }
        for i, doc in enumerate(docs)
    ]


# -------------------------------------------------------------------
# Quick test
# python -m src.ingestion.chunker
# -------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    from src.ingestion.loader import load_documents
    load_dotenv()

    test_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw"

    try:
        llama_docs = load_documents(test_path)
        lc_docs = chunk_documents(llama_docs)
        metadata = get_chunk_metadata(lc_docs)

        print(f"\n--- Chunker Test Results ---")
        print(f"Total chunks: {len(lc_docs)}")
        print(f"Document type: {type(lc_docs[0]).__name__}")
        print(f"\nFirst 5 chunks:")
        for meta in metadata[:5]:
            print(
                f"\n  Chunk {meta['chunk_index']} | "
                f"Page: {meta['page_label']} | "
                f"Chars: {meta['char_count']}"
            )
            print(f"  Preview: {meta['text_preview']}")

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)
