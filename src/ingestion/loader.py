"""
loader.py
---------
Responsible for loading documents from disk into LlamaIndex Document objects.

Identical in structure to rag-pipeline/loader.py with one addition:
    - Supports loading multiple file types in preparation for
      agentic tool-based ingestion (web pages, DOCX) in future iterations.

Supported formats:
    - PDF (.pdf)

Future formats (called out for agentic tool use):
    - Markdown (.md)
    - Web pages (URL) — for web search tool integration
    - DOCX (.docx)

Usage:
    from src.ingestion.loader import load_documents
    docs = load_documents("data/raw/meridian-capital-handbook.pdf")
"""

import logging
from pathlib import Path
from typing import List

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_documents(path: str) -> List[Document]:
    """
    Load documents from a file path or directory.

    Args:
        path (str): Path to a single PDF file or a directory containing PDFs.

    Returns:
        List[Document]: A list of LlamaIndex Document objects.

    Raises:
        FileNotFoundError: If the provided path does not exist.
        ValueError: If no documents could be loaded from the path.
    """
    resolved = Path(path).resolve()

    if not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")

    logger.info(f"Loading documents from: {resolved}")

    if resolved.is_file():
        documents = _load_single_file(resolved)
    elif resolved.is_dir():
        documents = _load_directory(resolved)
    else:
        raise ValueError(f"Path is neither a file nor a directory: {resolved}")

    if not documents:
        raise ValueError(f"No documents loaded from path: {resolved}")

    logger.info(f"Successfully loaded {len(documents)} document(s)")
    return documents


def _load_single_file(file_path: Path) -> List[Document]:
    """Load a single PDF file."""
    suffix = file_path.suffix.lower()

    if suffix != ".pdf":
        raise ValueError(
            f"Unsupported file type: '{suffix}'. "
            f"Currently supported formats: .pdf"
        )

    logger.info(f"Loading single file: {file_path.name}")
    reader = SimpleDirectoryReader(input_files=[str(file_path)])
    documents = reader.load_data()
    logger.info(f"Loaded {len(documents)} page(s) from {file_path.name}")
    return documents


def _load_directory(dir_path: Path) -> List[Document]:
    """Load all PDF files from a directory (non-recursive)."""
    pdf_files = list(dir_path.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(f"No PDF files found in directory: {dir_path}")

    logger.info(f"Found {len(pdf_files)} PDF file(s) in {dir_path}")

    reader = SimpleDirectoryReader(
        input_dir=str(dir_path),
        required_exts=[".pdf"],
        recursive=False,
    )
    documents = reader.load_data()
    logger.info(f"Loaded {len(documents)} total page(s) from directory")
    return documents


def get_document_metadata(documents: List[Document]) -> List[dict]:
    """Extract metadata from a list of loaded documents."""
    return [
        {
            "doc_id": doc.doc_id,
            "file_name": doc.metadata.get("file_name", "unknown"),
            "file_path": doc.metadata.get("file_path", "unknown"),
            "page_label": doc.metadata.get("page_label", "unknown"),
            "text_length": len(doc.text),
        }
        for doc in documents
    ]


if __name__ == "__main__":
    import sys
    test_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw"

    try:
        docs = load_documents(test_path)
        metadata = get_document_metadata(docs)
        print(f"\n--- Loader Test Results ---")
        print(f"Total documents loaded: {len(docs)}")
        for i, meta in enumerate(metadata, 1):
            print(f"\nDocument {i}:")
            for k, v in meta.items():
                print(f"  {k}: {v}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)
