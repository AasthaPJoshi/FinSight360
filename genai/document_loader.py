"""
Loads SEC 10-K filing text sections and chunks them for RAG.

10-K sections targeted:
- Item 1: Business description
- Item 1A: Risk Factors (most important for fraud detection)
- Item 7: MD&A
- Item 7A: Quantitative disclosures about market risk
- Item 8: Financial statements
"""
import asyncio
import re
from dataclasses import dataclass
from pathlib import Path

import httpx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config.settings import settings
from utils.logger import get_logger

log = get_logger("document_loader")

RAW_FILINGS_DIR = Path("data/raw/filings")
RAW_FILINGS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class FilingSection:
    cik: str
    company_name: str
    ticker: str
    accession_number: str
    period_of_report: str
    form_type: str
    section_name: str
    section_key: str
    text: str
    word_count: int = 0

    def __post_init__(self):
        self.word_count = len(self.text.split())


SECTION_PATTERNS = {
    "business": r"item\s*1[\.\s]*business",
    "risk_factors": r"item\s*1a[\.\s]*risk\s*factor",
    "mda": r"item\s*7[\.\s]*management.{0,20}discussion",
    "market_risk": r"item\s*7a[\.\s]*quantitative",
    "financial_statements": r"item\s*8[\.\s]*financial\s*statement",
}

SECTION_NAMES = {
    "business": "Item 1 - Business",
    "risk_factors": "Item 1A - Risk Factors",
    "mda": "Item 7 - MD&A",
    "market_risk": "Item 7A - Market Risk",
    "financial_statements": "Item 8 - Financial Statements",
}


class SECFilingLoader:
    """Downloads and parses 10-K filing text from SEC EDGAR."""

    def __init__(self):
        self.headers = {
            "User-Agent": settings.edgar_user_agent,
            "Accept-Encoding": "gzip,deflate",
        }
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    async def get_filing_text(
        self, cik: str, accession_number: str
    ) -> str | None:
        """Download 10-K full text from SEC EDGAR with local caching."""
        cache_path = RAW_FILINGS_DIR / f"{cik}_{accession_number}.txt"
        if cache_path.exists():
            log.info("filing_cache_hit", cik=cik)
            return cache_path.read_text(encoding="utf-8", errors="ignore")

        acc_clean = accession_number.replace("-", "")
        index_url = (
            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}"
            f"/{acc_clean}/{accession_number}-index.htm"
        )

        try:
            async with httpx.AsyncClient(
                headers=self.headers, timeout=60, follow_redirects=True
            ) as client:
                await asyncio.sleep(0.12)
                resp = await client.get(index_url)
                if resp.status_code != 200:
                    log.warning("filing_index_not_found", cik=cik, url=index_url)
                    return None

                primary_doc = self._find_primary_doc(resp.text)
                if not primary_doc:
                    return None

                doc_url = (
                    f"https://www.sec.gov/Archives/edgar/data"
                    f"/{int(cik)}/{acc_clean}/{primary_doc}"
                )
                await asyncio.sleep(0.12)
                doc_resp = await client.get(doc_url)
                if doc_resp.status_code != 200:
                    return None

                text = self._clean_html(doc_resp.text)
                cache_path.write_text(text, encoding="utf-8")
                log.info("filing_downloaded", cik=cik, chars=len(text))
                return text

        except Exception as e:
            log.warning("filing_download_failed", cik=cik, error=str(e))
            return None

    def extract_sections(
        self,
        text: str,
        cik: str,
        company_name: str,
        ticker: str,
        accession_number: str,
        period_of_report: str,
        form_type: str = "10-K",
    ) -> list[FilingSection]:
        """Extract key sections from 10-K text using regex boundary detection."""
        sections = []
        text_lower = text.lower()

        section_positions: dict[str, int] = {}
        for key, pattern in SECTION_PATTERNS.items():
            matches = list(re.finditer(pattern, text_lower))
            if matches:
                section_positions[key] = matches[-1].start()

        if not section_positions:
            sections.append(
                FilingSection(
                    cik=cik,
                    company_name=company_name,
                    ticker=ticker,
                    accession_number=accession_number,
                    period_of_report=period_of_report,
                    form_type=form_type,
                    section_name="Full Filing",
                    section_key="full",
                    text=text[:50000],
                )
            )
            return sections

        sorted_sections = sorted(section_positions.items(), key=lambda x: x[1])

        for i, (key, start_pos) in enumerate(sorted_sections):
            if i + 1 < len(sorted_sections):
                end_pos = sorted_sections[i + 1][1]
            else:
                end_pos = start_pos + 15000

            section_text = text[start_pos : min(end_pos, start_pos + 15000)]
            section_text = self._clean_text(section_text)

            if len(section_text) < 100:
                continue

            sections.append(
                FilingSection(
                    cik=cik,
                    company_name=company_name,
                    ticker=ticker,
                    accession_number=accession_number,
                    period_of_report=period_of_report,
                    form_type=form_type,
                    section_name=SECTION_NAMES.get(key, key),
                    section_key=key,
                    text=section_text,
                )
            )

        log.info(
            "sections_extracted",
            cik=cik,
            count=len(sections),
            sections=[s.section_key for s in sections],
        )
        return sections

    def sections_to_documents(
        self, sections: list[FilingSection]
    ) -> list[Document]:
        """Convert FilingSections to LangChain Documents with metadata."""
        docs = []
        for section in sections:
            chunks = self.splitter.split_text(section.text)
            for i, chunk in enumerate(chunks):
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "cik": section.cik,
                            "company_name": section.company_name,
                            "ticker": section.ticker,
                            "accession_number": section.accession_number,
                            "period_of_report": section.period_of_report,
                            "form_type": section.form_type,
                            "section_name": section.section_name,
                            "section_key": section.section_key,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "source": (
                                f"{section.ticker}_"
                                f"{section.period_of_report}_"
                                f"{section.section_key}"
                            ),
                        },
                    )
                )
        return docs

    def _find_primary_doc(self, index_html: str) -> str | None:
        patterns = [
            r'href="([^"]+\.htm)"[^>]*>(?:10-K|Annual)',
            r'<td>10-K</td>.*?href="([^"]+\.htm)"',
            r'href="([^"]+10k[^"]*\.htm)"',
            r'href="([^"]+annual[^"]*\.htm)"',
        ]
        for pattern in patterns:
            match = re.search(pattern, index_html, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).split("/")[-1]
        return None

    def _clean_html(self, html: str) -> str:
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"&nbsp;", " ", text)
        text = re.sub(r"&amp;", "&", text)
        text = re.sub(r"&lt;", "<", text)
        text = re.sub(r"&gt;", ">", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"_{3,}", "", text)
        text = re.sub(r"-{3,}", "", text)
        return text.strip()

    async def load_company_filings(
        self, db, cik: str, max_filings: int = 3
    ) -> list[Document]:
        """Load last N 10-K filings for a company into LangChain Documents."""
        conn = db.get_connection()

        filings = conn.execute(
            """
            SELECT f.accession_number, f.period_of_report,
                   f.form_type, c.name, c.ticker
            FROM filings f
            JOIN companies c ON f.cik = c.cik
            WHERE f.cik = ? AND f.form_type = '10-K'
            ORDER BY f.period_of_report DESC
            LIMIT ?
            """,
            [cik, max_filings],
        ).df()

        if filings.empty:
            log.warning("no_filings_found", cik=cik)
            return []

        all_docs = []
        for _, row in filings.iterrows():
            text = await self.get_filing_text(cik, str(row["accession_number"]))
            if not text:
                continue

            sections = self.extract_sections(
                text=text,
                cik=cik,
                company_name=str(row["name"]),
                ticker=str(row.get("ticker", "")),
                accession_number=str(row["accession_number"]),
                period_of_report=str(row["period_of_report"]),
                form_type=str(row["form_type"]),
            )
            docs = self.sections_to_documents(sections)
            all_docs.extend(docs)
            log.info(
                "filing_loaded",
                cik=cik,
                period=str(row["period_of_report"]),
                docs=len(docs),
            )

        return all_docs
