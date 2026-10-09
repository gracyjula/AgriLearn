"""
tests/test_rag.py — AgriLearn AI
Automated tests for the RAG pipeline.

All tests mock the Gemini API and FAISS dependencies so no real API key
or GPU hardware is required.

Run with:
    pytest tests/test_rag.py -v
"""

import sys
import os
import io
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from config import AppConfig
from rag_service import RAGService, RetrievalResult, RetrievedChunk, check_rag_dependencies


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_config(tmp_pdf_dir: str = "", tmp_store_dir: str = "") -> AppConfig:
    """Build an AppConfig pointing at temporary directories."""
    cfg = AppConfig.__new__(AppConfig)
    cfg.api_key = "test-api-key-placeholder"
    cfg.model_name = "gemini-3.5-flash"
    cfg.embedding_model_name = "gemini-embedding-001"
    cfg.temperature = 0.3
    cfg.rag = {
        "pdf_dir": tmp_pdf_dir or "docs/pdfs",
        "vector_store_dir": tmp_store_dir or "docs/vector_store",
        "chunk_size": 500,
        "chunk_overlap": 50,
        "top_k": 2,
    }
    cfg.app_title = "AgriLearn AI"
    cfg.app_subtitle = "Test"
    cfg.app_description = "Test instance"
    return cfg


def make_fake_pdf_bytes() -> bytes:
    """
    Return minimal valid PDF bytes containing agricultural text.
    Uses PyPDF-compatible structure.
    """
    content = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj
4 0 obj << /Length 120 >>
stream
BT /F1 12 Tf 72 720 Td
(Germination is the process by which a plant grows from a seed.) Tj
0 -20 Td
(Sowing involves placing seeds in soil at the correct depth.) Tj
ET
endstream
endobj
5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000274 00000 n
0000000446 00000 n
trailer << /Size 6 /Root 1 0 R >>
startxref
530
%%EOF"""
    return content


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — PDF ingestion: detect PDFs in the configured folder
# ─────────────────────────────────────────────────────────────────────────────

def test_pdf_ingestion_detects_files():
    """RAGService must detect PDF files placed in the pdf_dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_dir = Path(tmpdir) / "pdfs"
        pdf_dir.mkdir()

        # Place a fake PDF
        (pdf_dir / "agri_basics.pdf").write_bytes(b"%PDF-1.4 fake content")
        (pdf_dir / "irrigation_guide.pdf").write_bytes(b"%PDF-1.4 fake content")

        cfg = make_config(tmp_pdf_dir=str(pdf_dir), tmp_store_dir=str(tmpdir + "/store"))
        rag = RAGService(cfg)

        count = rag.get_indexed_pdf_count()
        assert count == 2, f"Expected 2 PDFs, found {count}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — PDF ingestion: missing folder returns count 0
# ─────────────────────────────────────────────────────────────────────────────

def test_pdf_ingestion_missing_dir_returns_zero():
    """If pdf_dir does not exist, get_indexed_pdf_count() should return 0."""
    cfg = make_config(
        tmp_pdf_dir="/nonexistent_path_xyz/pdfs",
        tmp_store_dir="/nonexistent_path_xyz/store",
    )
    rag = RAGService(cfg)
    assert rag.get_indexed_pdf_count() == 0


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — Index build fails gracefully when no PDFs are present
# ─────────────────────────────────────────────────────────────────────────────

def test_build_index_no_pdfs_returns_error():
    """build_index() must return (False, message) when pdf_dir has no PDFs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_dir = Path(tmpdir) / "pdfs"
        pdf_dir.mkdir()  # Empty folder — no PDFs

        cfg = make_config(tmp_pdf_dir=str(pdf_dir), tmp_store_dir=str(tmpdir + "/store"))
        rag = RAGService(cfg)

        # Mock RAG deps check so it reports available
        with patch("rag_service.check_rag_dependencies", return_value=(True, "OK")):
            ok, msg = rag.build_index()

        assert ok is False, "Expected build_index to fail with no PDFs"
        assert "no pdf" in msg.lower() or "found" in msg.lower(), (
            f"Expected informative message, got: {msg}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — Source metadata preserved: source_file and page_number
# ─────────────────────────────────────────────────────────────────────────────

def test_retrieved_chunks_have_source_metadata():
    """Retrieved chunks must carry source_file and page_number metadata."""
    chunk = RetrievedChunk(
        content="Germination starts when the seed absorbs water.",
        source_file="agri_basics.pdf",
        page_number=3,
    )
    assert chunk.source_file == "agri_basics.pdf"
    assert chunk.page_number == 3
    assert "Germination" in chunk.content


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — Retrieval returns empty result when index does not exist
# ─────────────────────────────────────────────────────────────────────────────

def test_retrieval_no_index_returns_empty():
    """
    retrieve() must return an empty RetrievalResult (not raise) when no
    vector index has been built.
    """
    cfg = make_config(
        tmp_pdf_dir="/nonexistent/pdfs",
        tmp_store_dir="/nonexistent/store",
    )
    rag = RAGService(cfg)

    result = rag.retrieve("What is sowing?")

    assert isinstance(result, RetrievalResult)
    assert result.context_text == ""
    assert result.sources == []
    assert result.total_chunks_retrieved == 0
    assert result.index_exists is False


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 — Retrieval returns context text from mocked vector store
# ─────────────────────────────────────────────────────────────────────────────

def test_retrieval_returns_context_from_mock_store():
    """
    retrieve() must return context_text and source metadata when the
    vector store returns matching documents.
    """
    from langchain_core.documents import Document

    fake_doc = Document(
        page_content="Sowing is the process of planting seeds in prepared soil.",
        metadata={"source_file": "sowing_guide.pdf", "page_number": 2},
    )

    mock_store = MagicMock()
    mock_store.similarity_search_with_score.return_value = [(fake_doc, 0.85)]

    cfg = make_config()
    rag = RAGService(cfg)
    rag._vector_store = mock_store  # Inject mock

    result = rag.retrieve("What is sowing?")

    assert result.total_chunks_retrieved == 1
    assert "Sowing" in result.context_text
    assert result.sources[0].source_file == "sowing_guide.pdf"
    assert result.sources[0].page_number == 2


# ─────────────────────────────────────────────────────────────────────────────
# Test 7 — Retrieval deduplicates sources for citation display
# ─────────────────────────────────────────────────────────────────────────────

def test_retrieval_multiple_chunks_same_source():
    """
    When multiple chunks from the same page are retrieved, the source list
    should still contain entries for each chunk (deduplication happens in UI).
    """
    from langchain_core.documents import Document

    docs = [
        (Document(
            page_content="Germination stage 1: water absorption.",
            metadata={"source_file": "crop_guide.pdf", "page_number": 1},
        ), 0.9),
        (Document(
            page_content="Germination stage 2: radicle emergence.",
            metadata={"source_file": "crop_guide.pdf", "page_number": 1},
        ), 0.88),
    ]

    mock_store = MagicMock()
    mock_store.similarity_search_with_score.return_value = docs

    cfg = make_config()
    rag = RAGService(cfg)
    rag._vector_store = mock_store

    result = rag.retrieve("Explain germination stages")

    assert result.total_chunks_retrieved == 2
    assert len(result.sources) == 2
    assert all(s.source_file == "crop_guide.pdf" for s in result.sources)


# ─────────────────────────────────────────────────────────────────────────────
# Test 8 — Missing PDF folder handled gracefully in build_index
# ─────────────────────────────────────────────────────────────────────────────

def test_build_index_missing_pdf_dir_returns_error():
    """build_index() must return (False, message) when pdf_dir does not exist."""
    cfg = make_config(
        tmp_pdf_dir="/nonexistent_xyz/pdfs",
        tmp_store_dir="/nonexistent_xyz/store",
    )
    rag = RAGService(cfg)

    with patch("rag_service.check_rag_dependencies", return_value=(True, "OK")):
        ok, msg = rag.build_index()

    assert ok is False
    assert "not found" in msg.lower() or "pdf" in msg.lower(), (
        f"Expected informative error, got: {msg}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 9 — API failure in GeminiService generate() with RAG context
# ─────────────────────────────────────────────────────────────────────────────

def test_gemini_service_rag_api_failure_handled():
    """
    When the Gemini API raises an exception during RAG-grounded generation,
    GeminiService.generate() must return (error_message, False) and not crash.

    We patch the chain's invoke method directly, since `prompt | llm` builds
    a new RunnableSequence that wraps the LLM rather than calling __or__.
    """
    from gemini_service import GeminiService

    cfg = make_config()
    svc = GeminiService(cfg)

    # Make _get_llm() return a non-None mock so the code proceeds past the None check
    svc._llm = MagicMock()

    # Patch build_rag_prompt to return a mock whose | operator raises an exception
    mock_chain = MagicMock()
    mock_chain.invoke = MagicMock(
        side_effect=Exception("404 models/gemini-3.5-flash not found")
    )
    mock_prompt = MagicMock()
    mock_prompt.__or__ = MagicMock(return_value=mock_chain)

    with patch("gemini_service.build_rag_prompt", return_value=mock_prompt):
        response_text, success = svc.generate(
            question="What is crop germination?",
            rag_context="[Source: doc.pdf, Page 1]\nGermination is when seeds sprout.",
        )

    assert success is False, (
        f"Expected success=False when API raises exception, got response: {response_text}"
    )
    assert response_text, "Expected a non-empty error message"


# ─────────────────────────────────────────────────────────────────────────────
# Test 10 — GeminiService reports NotFound error clearly
# ─────────────────────────────────────────────────────────────────────────────

def test_gemini_service_notfound_error_message():
    """
    A 404 NotFound error from the Gemini API must produce a message that
    mentions the model name and suggests a fix.
    """
    from gemini_service import GeminiService, _classify_api_error

    cfg = make_config()

    exc = Exception("404 models/gemini-1.5-flash is not found for API version v1beta")
    msg = _classify_api_error(exc, "gemini-1.5-flash")

    assert "not found" in msg.lower() or "model" in msg.lower(), (
        f"Expected model-not-found message, got: {msg}"
    )
    assert "gemini-1.5-flash" in msg, "Expected model name in error message"
    assert "gemini-3.5-flash" in msg or "gemini-3.6-flash" in msg, (
        "Expected a suggested replacement model in the error message"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 11 — GeminiService reports auth error clearly
# ─────────────────────────────────────────────────────────────────────────────

def test_gemini_service_auth_error_message():
    """
    A 401 authentication error must produce a message that mentions the API key
    without exposing the key value.
    """
    from gemini_service import _classify_api_error

    exc = Exception("401 UNAUTHENTICATED: Request had invalid authentication credentials")
    msg = _classify_api_error(exc, "gemini-3.5-flash")

    assert "authentication" in msg.lower() or "api key" in msg.lower(), (
        f"Expected auth error message, got: {msg}"
    )
    # Critically — the actual key value must not appear
    assert "test-api-key-placeholder" not in msg


# ─────────────────────────────────────────────────────────────────────────────
# Test 12 — is_index_ready returns False when no index exists
# ─────────────────────────────────────────────────────────────────────────────

def test_is_index_ready_false_when_no_index():
    """is_index_ready() must return False when no FAISS index has been saved."""
    cfg = make_config(
        tmp_pdf_dir="/nonexistent/pdfs",
        tmp_store_dir="/nonexistent/store",
    )
    rag = RAGService(cfg)
    assert rag.is_index_ready() is False


# ─────────────────────────────────────────────────────────────────────────────
# Test 13 — check_rag_dependencies correctly detects missing packages
# ─────────────────────────────────────────────────────────────────────────────

def test_check_rag_dependencies_missing():
    """
    check_rag_dependencies() must return (False, message) when a required
    package is not importable.
    """
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pypdf":
            raise ImportError("No module named 'pypdf'")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        available, msg = check_rag_dependencies()

    assert available is False
    assert "pypdf" in msg.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Test 14 — Safety is NOT bypassed by retrieved document content
# ─────────────────────────────────────────────────────────────────────────────

def test_safety_check_not_bypassed_by_rag_framing():
    """
    A user framing a prohibited request as 'what does the document say'
    should still be caught by the safety check if it requests a recommendation.
    """
    from safety import check_input

    # This is still a fertilizer recommendation request, just wrapped differently
    result = check_input(
        "Based on the document, recommend how much fertilizer to apply to my crop."
    )
    assert result.is_safe is False, (
        "Fertilizer recommendation should be refused even when framed as a RAG question"
    )
