"""
Natural language Q&A interface — single entry point for all GenAI capabilities.
Lazy-loads components to minimise startup time.
"""
import asyncio

from genai.document_loader import SECFilingLoader
from genai.rag_chain import FinSightRAGChain
from genai.risk_narrator import RiskNarrator
from genai.sentiment_analyzer import FilingToneAnalyzer
from genai.vector_store import FilingVectorStore
from utils.logger import get_logger

log = get_logger("qa_interface")


class FinSightQA:
    """Single entry point for all GenAI capabilities."""

    def __init__(self, db):
        self.db = db
        self._vs: FilingVectorStore | None = None
        self._rag: FinSightRAGChain | None = None
        self._tone: FilingToneAnalyzer | None = None
        self._narrator: RiskNarrator | None = None
        self._loader: SECFilingLoader | None = None

    @property
    def vs(self) -> FilingVectorStore:
        if not self._vs:
            self._vs = FilingVectorStore()
            self._vs.get_or_create()
        return self._vs

    @property
    def rag(self) -> FinSightRAGChain:
        if not self._rag:
            self._rag = FinSightRAGChain(self.vs)
        return self._rag

    @property
    def tone(self) -> FilingToneAnalyzer:
        if not self._tone:
            self._tone = FilingToneAnalyzer()
        return self._tone

    @property
    def narrator(self) -> RiskNarrator:
        if not self._narrator:
            self._narrator = RiskNarrator(self.db, self.rag, self.tone)
        return self._narrator

    def index_company(self, ticker: str) -> dict:
        """Index a company's SEC filings into ChromaDB."""
        conn = self.db.get_connection()
        company = conn.execute(
            "SELECT cik, name FROM companies WHERE ticker = ?",
            [ticker.upper()],
        ).df()

        if company.empty:
            return {
                "status": "error",
                "message": f"Company {ticker} not found. Run ingest first.",
            }

        cik = str(company.iloc[0]["cik"])

        if not self._loader:
            self._loader = SECFilingLoader()

        docs = asyncio.run(self._loader.load_company_filings(self.db, cik))
        added = self.vs.add_documents(docs)
        stats = self.vs.get_collection_stats()

        return {
            "status": "success",
            "ticker": ticker,
            "cik": cik,
            "docs_added": added,
            "total_in_store": stats["total_chunks"],
        }

    def ask(self, question: str, ticker: str | None = None) -> str:
        """Ask any natural language question about SEC filings."""
        cik = None
        if ticker:
            conn = self.db.get_connection()
            result = conn.execute(
                "SELECT cik FROM companies WHERE ticker = ?",
                [ticker.upper()],
            ).df()
            if not result.empty:
                cik = str(result.iloc[0]["cik"])

        result = self.rag.ask(question, cik=cik)
        return result["answer"]

    def generate_risk_brief(self, ticker: str) -> dict:
        """Generate full risk brief for a company."""
        conn = self.db.get_connection()
        company = conn.execute(
            "SELECT cik FROM companies WHERE ticker = ?",
            [ticker.upper()],
        ).df()

        if company.empty:
            return {"status": "error", "message": f"{ticker} not found"}

        cik = str(company.iloc[0]["cik"])
        return self.narrator.generate_brief(cik)

    def get_store_stats(self) -> dict:
        return self.vs.get_collection_stats()
