"""
rag_service.py — AgriLearn AI
Retrieval-Augmented Generation pipeline for agricultural PDF documents.

Workflow:
  1. Load PDFs from a configured folder using PyPDFLoader.
  2. Split text into overlapping chunks using RecursiveCharacterTextSplitter.
  3. Generate embeddings using GoogleGenerativeAIEmbeddings (gemini-embedding-001).
  4. Store and persist the vector index using FAISS.
  5. Retrieve the most relevant chunks for a user question at query time.
  6. Return context text + source metadata (filename, page) for citation display.

Dependencies (install via requirements.txt):
  pip install pypdf faiss-cpu langchain-community

Safety notes:
  - Retrieved document text is passed to Gemini as information only.
  - The RAG pipeline does NOT bypass the application's safety rules.
  - Sources are cited only when relevant content was actually retrieved.
  - The pipeline never claims a document says something it does not contain.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from dataclasses import dataclass


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """A single retrieved text chunk with its source metadata."""
    content: str
    source_file: str   # Original PDF filename (no path)
    page_number: int   # 1-based page number


@dataclass
class RetrievalResult:
    """The full result of a retrieval operation."""
    context_text: str          # All chunks joined for the LLM prompt
    sources: list[RetrievedChunk]
    total_chunks_retrieved: int
    index_exists: bool


# ── Availability check ─────────────────────────────────────────────────────────

def check_rag_dependencies() -> tuple[bool, str]:
    """
    Check whether all RAG dependencies are installed.

    Returns (available: bool, message: str).
    """
    missing = []
    try:
        import pypdf  # noqa: F401
    except ImportError:
        missing.append("pypdf")
    try:
        import faiss  # noqa: F401
    except ImportError:
        missing.append("faiss-cpu")
    try:
        import langchain_community  # noqa: F401
    except ImportError:
        missing.append("langchain-community")

    if missing:
        return False, (
            f"RAG dependencies not installed: {', '.join(missing)}. "
            f"Run: pip install {' '.join(missing)}"
        )
    return True, "OK"


# ── RAG Service ────────────────────────────────────────────────────────────────

class RAGService:
    """
    Manages the full RAG pipeline for AgriLearn AI.

    Usage:
        rag = RAGService(config)
        rag.build_index()                          # run once after adding PDFs
        result = rag.retrieve("What is sowing?")   # at query time
    """

    def __init__(self, config):
        """
        Args:
            config: AppConfig instance with .api_key, .embedding_model_name, .rag
        """
        self._config = config
        self._vector_store = None   # In-memory FAISS index (loaded from disk)
        self._pdf_dir = Path(config.rag["pdf_dir"])
        self._store_dir = Path(config.rag["vector_store_dir"])
        self._chunk_size = config.rag["chunk_size"]
        self._chunk_overlap = config.rag["chunk_overlap"]
        self._top_k = config.rag["top_k"]

    # ── Embeddings ─────────────────────────────────────────────────────────────

    def _get_embeddings(self):
        """Initialise GoogleGenerativeAIEmbeddings with the configured model."""
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=self._config.embedding_model_name,
            google_api_key=self._config.api_key,
        )

    # ── Index building ─────────────────────────────────────────────────────────

    def build_index(self, force_rebuild: bool = False) -> tuple[bool, str]:
        """
        Build (or rebuild) the FAISS vector index from PDFs in the pdf_dir.

        Args:
            force_rebuild: If True, rebuild even if the index already exists.

        Returns:
            (success: bool, message: str)
        """
        available, msg = check_rag_dependencies()
        if not available:
            return False, msg

        if not self._config.is_configured:
            return False, "Gemini API key not configured. Cannot generate embeddings."

        # Find PDFs
        if not self._pdf_dir.exists():
            return False, (
                f"PDF folder not found: '{self._pdf_dir}'. "
                "Create the folder and add agricultural PDF documents."
            )

        pdf_files = list(self._pdf_dir.glob("*.pdf"))
        if not pdf_files:
            return False, (
                f"No PDF files found in '{self._pdf_dir}'. "
                "Add agricultural PDF documents to that folder and run the index builder again."
            )

        # Skip if index already exists and rebuild not forced
        if not force_rebuild and self._index_exists():
            ok, msg2 = self._load_index()
            if ok:
                return True, f"Loaded existing index ({len(pdf_files)} PDFs available)."

        # Load documents
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import FAISS

        all_docs = []
        failed = []

        for pdf_path in sorted(pdf_files):
            try:
                loader = PyPDFLoader(str(pdf_path))
                pages = loader.load()
                # Normalise metadata: use filename (not full path), 1-based pages
                for page in pages:
                    page.metadata["source_file"] = pdf_path.name
                    # PyPDFLoader sets page as 0-based int
                    raw_page = page.metadata.get("page", 0)
                    page.metadata["page_number"] = int(raw_page) + 1
                all_docs.extend(pages)
            except Exception as e:
                failed.append(f"{pdf_path.name}: {e}")

        if not all_docs:
            return False, f"Could not extract text from any PDFs. Errors: {failed}"

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(all_docs)

        if not chunks:
            return False, "Text splitting produced no chunks. Check that the PDFs contain readable text."

        # Generate embeddings and build FAISS index
        try:
            embeddings = self._get_embeddings()
            vector_store = FAISS.from_documents(chunks, embeddings)
        except Exception as e:
            return False, f"Failed to generate embeddings or build index: {e}"

        # Persist the index
        try:
            self._store_dir.mkdir(parents=True, exist_ok=True)
            vector_store.save_local(str(self._store_dir))
            self._vector_store = vector_store
        except Exception as e:
            return False, f"Failed to save index to disk: {e}"

        warning = f" (Skipped {len(failed)} files with errors)" if failed else ""
        return True, (
            f"Index built successfully from {len(pdf_files)} PDF(s), "
            f"{len(chunks)} chunks.{warning}"
        )

    def _index_exists(self) -> bool:
        """True if a persisted FAISS index exists on disk."""
        return (
            (self._store_dir / "index.faiss").exists()
            and (self._store_dir / "index.pkl").exists()
        )

    def _load_index(self) -> tuple[bool, str]:
        """Load an existing FAISS index from disk."""
        if not self._index_exists():
            return False, "No index found on disk."

        try:
            from langchain_community.vectorstores import FAISS
            embeddings = self._get_embeddings()
            self._vector_store = FAISS.load_local(
                str(self._store_dir),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            return True, "Index loaded."
        except Exception as e:
            self._vector_store = None
            return False, f"Failed to load index: {e}"

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def retrieve(self, question: str) -> RetrievalResult:
        """
        Retrieve the most relevant document chunks for a question.

        Returns a RetrievalResult. If the index is not available or no relevant
        content is found, context_text will be empty and sources will be empty.
        """
        # Check dependencies quietly — if not available, return empty result
        available, _ = check_rag_dependencies()
        if not available:
            return RetrievalResult("", [], 0, False)

        # Load index if not already loaded
        if self._vector_store is None:
            if self._index_exists():
                ok, _ = self._load_index()
                if not ok:
                    return RetrievalResult("", [], 0, False)
            else:
                return RetrievalResult("", [], 0, False)

        try:
            docs_and_scores = self._vector_store.similarity_search_with_score(
                question, k=self._top_k
            )
        except Exception:
            return RetrievalResult("", [], 0, True)

        if not docs_and_scores:
            return RetrievalResult("", [], 0, True)

        chunks: list[RetrievedChunk] = []
        context_parts: list[str] = []

        for doc, _score in docs_and_scores:
            source_file = doc.metadata.get("source_file", "Unknown document")
            page_number = doc.metadata.get("page_number", 1)
            content = doc.page_content.strip()

            if not content:
                continue

            chunk = RetrievedChunk(
                content=content,
                source_file=source_file,
                page_number=page_number,
            )
            chunks.append(chunk)
            context_parts.append(
                f"[Source: {source_file}, Page {page_number}]\n{content}"
            )

        context_text = "\n\n".join(context_parts)
        return RetrievalResult(
            context_text=context_text,
            sources=chunks,
            total_chunks_retrieved=len(chunks),
            index_exists=True,
        )

    def get_indexed_pdf_count(self) -> int:
        """Return the number of PDFs in the pdf_dir (useful for the UI)."""
        if not self._pdf_dir.exists():
            return 0
        return len(list(self._pdf_dir.glob("*.pdf")))

    def is_index_ready(self) -> bool:
        """True if the vector store is loaded or can be loaded from disk."""
        if self._vector_store is not None:
            return True
        return self._index_exists()
