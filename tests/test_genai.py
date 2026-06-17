"""Tests for Phase 5: RAG, document loader, sentiment analyzer, vector store."""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from genai.document_loader import SECFilingLoader, FilingSection
from genai.rag_chain import format_docs
from genai.sentiment_analyzer import FilingToneAnalyzer


# ──────────────────────────────────────────────
# Test 1: HTML cleaning
# ──────────────────────────────────────────────

def test_document_loader_cleans_html():
    """Raw HTML should be stripped of tags and entities."""
    loader = SECFilingLoader()
    html = (
        "<html><body><h1>Item&nbsp;1A</h1>"
        "<p>Risk&amp;Factors: <b>going concern</b> issues.</p></body></html>"
    )
    text = loader._clean_html(html)

    assert "<" not in text, "HTML tags should be stripped"
    assert ">" not in text, "HTML tags should be stripped"
    assert "&nbsp;" not in text, "HTML entities should be decoded"
    assert "&amp;" not in text, "HTML entities should be decoded"
    assert "going concern" in text


# ──────────────────────────────────────────────
# Test 2: Section extraction finds risk_factors
# ──────────────────────────────────────────────

def test_section_extraction_finds_risk_factors():
    """Mock 10-K text with 'item 1a risk factors' should yield a risk_factors section."""
    loader = SECFilingLoader()

    mock_text = (
        "Some table of contents text...\n\n"
        "ITEM 1A. RISK FACTORS\n\n"
        "The company faces substantial competition in all markets. "
        "There can be no assurance that the company will maintain its market share. "
        "Global economic conditions may adversely affect our results.\n\n"
        "ITEM 7. MANAGEMENT DISCUSSION AND ANALYSIS\n\n"
        "Revenue increased 12% year over year driven by strong demand."
    )

    sections = loader.extract_sections(
        text=mock_text,
        cik="320193",
        company_name="Apple Inc.",
        ticker="AAPL",
        accession_number="0000320193-23-000106",
        period_of_report="2023-09-30",
        form_type="10-K",
    )

    section_keys = [s.section_key for s in sections]
    assert "risk_factors" in section_keys, (
        f"Expected risk_factors in {section_keys}"
    )

    rf = next(s for s in sections if s.section_key == "risk_factors")
    assert len(rf.text) > 50
    assert rf.cik == "320193"
    assert rf.ticker == "AAPL"


# ──────────────────────────────────────────────
# Test 3: sections_to_documents adds metadata
# ──────────────────────────────────────────────

def test_sections_to_documents_adds_metadata():
    """FilingSection → LangChain Documents should carry cik, ticker, section_key."""
    loader = SECFilingLoader()
    section = FilingSection(
        cik="320193",
        company_name="Apple Inc.",
        ticker="AAPL",
        accession_number="0000320193-23-000106",
        period_of_report="2023-09-30",
        form_type="10-K",
        section_name="Item 1A - Risk Factors",
        section_key="risk_factors",
        text="The company faces competition. " * 50,
    )

    docs = loader.sections_to_documents([section])

    assert len(docs) >= 1
    for doc in docs:
        assert isinstance(doc, Document)
        assert doc.metadata["cik"] == "320193"
        assert doc.metadata["ticker"] == "AAPL"
        assert doc.metadata["section_key"] == "risk_factors"
        assert "source" in doc.metadata
        assert doc.metadata["source"].startswith("AAPL_")


# ──────────────────────────────────────────────
# Test 4: Lexicon scan detects going concern
# ──────────────────────────────────────────────

def test_lexicon_scan_detects_going_concern():
    """Text containing 'going concern' should set going_concern_detected=True."""
    analyzer = FilingToneAnalyzer()
    text = (
        "The auditors have expressed substantial doubt about the company's "
        "ability to continue as a going concern. There can be no assurance "
        "the company will not cease operations within 12 months."
    )

    result = analyzer.lexicon_scan(text)

    assert result["going_concern_detected"] is True, (
        "Expected going_concern_detected=True"
    )
    assert result["categories"]["going_concern"]["total_mentions"] > 0


# ──────────────────────────────────────────────
# Test 5: Risk score higher for risky text
# ──────────────────────────────────────────────

def test_lexicon_scan_scores_risk():
    """High-risk text should score higher than clean/neutral text."""
    analyzer = FilingToneAnalyzer()

    risky_text = (
        "There is substantial doubt about our ability to continue as a going concern. "
        "We face securities litigation and class action lawsuits. "
        "The SEC investigation may result in material weakness findings. "
        "Cannot assure investors there will be no restatement of financial results. "
        "No assurance we will not cease operations. "
        "We may not be able to continue revenue recognition practices."
    ) * 3

    clean_text = (
        "The company achieved record revenue this quarter driven by strong growth. "
        "We maintain a competitive advantage across all product lines. "
        "Robust demand continues across international markets."
    ) * 3

    risky_score = analyzer.lexicon_scan(risky_text)["language_risk_score"]
    clean_score = analyzer.lexicon_scan(clean_text)["language_risk_score"]

    assert risky_score > clean_score, (
        f"Expected risky_score ({risky_score}) > clean_score ({clean_score})"
    )
    assert risky_score > 0


# ──────────────────────────────────────────────
# Test 6: Vector store initializes
# ──────────────────────────────────────────────

def test_vector_store_initializes(tmp_path):
    """ChromaDB collection should initialize without error (no OpenAI calls)."""
    from genai.vector_store import FilingVectorStore

    mock_embeddings = MagicMock()
    mock_chroma = MagicMock()
    mock_chroma._collection.count.return_value = 0
    mock_chroma._collection.get.return_value = {"metadatas": []}

    with patch("genai.vector_store.OpenAIEmbeddings", return_value=mock_embeddings), \
         patch("genai.vector_store.Chroma", return_value=mock_chroma):

        vs = FilingVectorStore()
        result = vs.get_or_create()

    assert result is mock_chroma
    stats = vs.get_collection_stats()
    assert stats["total_chunks"] == 0
    assert stats["collection_name"] == "finsight360_filings"


# ──────────────────────────────────────────────
# Test 7: format_docs produces citations
# ──────────────────────────────────────────────

def test_format_docs_with_citations():
    """Two Document objects should produce formatted string with company name and period."""
    docs = [
        Document(
            page_content="Revenue increased 12% year over year.",
            metadata={
                "company_name": "Apple Inc.",
                "form_type": "10-K",
                "period_of_report": "2023-09-30",
                "section_name": "Item 7 - MD&A",
                "ticker": "AAPL",
            },
        ),
        Document(
            page_content="We face substantial competition from other technology companies.",
            metadata={
                "company_name": "Apple Inc.",
                "form_type": "10-K",
                "period_of_report": "2023-09-30",
                "section_name": "Item 1A - Risk Factors",
                "ticker": "AAPL",
            },
        ),
    ]

    result = format_docs(docs)

    assert "Apple Inc." in result
    assert "2023-09-30" in result
    assert "Item 7 - MD&A" in result
    assert "Item 1A - Risk Factors" in result
    assert "Revenue increased" in result
    assert "substantial competition" in result
    assert "---" in result  # separator between chunks
