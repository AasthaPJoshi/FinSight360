"""
LangChain RAG pipeline for SEC filing Q&A.

Architecture (LCEL):
retriever → context_formatter → prompt → LLM → output_parser
"""
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

from config.settings import settings
from genai.vector_store import FilingVectorStore
from utils.logger import get_logger

log = get_logger("rag_chain")

ANALYST_SYSTEM_PROMPT = """You are a senior financial risk analyst at a \
top-tier investment bank with expertise in forensic accounting and SEC \
filing analysis. You have access to excerpts from SEC 10-K annual filings.

Your role:
- Answer questions about company financial risk based ONLY on the \
  provided filing excerpts
- Cite the specific section and filing period for each claim
- Flag any language that indicates elevated risk (hedging, uncertainty, \
  legal proceedings, going concern language, restatements)
- Be precise and factual — never speculate beyond the source material
- Format responses as a structured analyst brief

If the context doesn't contain enough information to answer, \
say "Insufficient filing data — recommend requesting full 10-K review."
"""

ANALYST_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", ANALYST_SYSTEM_PROMPT),
    ("human", """
FILING EXCERPTS:
{context}

ANALYST QUERY: {question}

Provide a structured response with:
1. Direct answer based on filing text
2. Specific citations (company, period, section)
3. Risk implications (if any)
4. Confidence level: [HIGH/MEDIUM/LOW] based on context quality
"""),
])

RISK_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", ANALYST_SYSTEM_PROMPT),
    ("human", """
FILING EXCERPTS FOR {company_name} ({ticker}):
{context}

TASK: Generate a comprehensive risk intelligence brief for this company.

## Executive Summary
[2-3 sentences on overall risk profile]

## Key Risk Factors Identified
[Bullet points from Item 1A - Risk Factors section]

## MD&A Red Flags
[Any concerning language in MD&A]

## Financial Risk Signals
[Specific financial metrics or trends mentioned]

## Language Risk Indicators
[Unusual hedging, uncertainty, legal/regulatory language]

## Analyst Recommendation
[ESCALATE / MONITOR / CLEAN with brief rationale]

ML Risk Score Context: {ml_risk_score}/100 | Risk Tier: {risk_tier}
Dominant Signal: {dominant_signal}
"""),
])


def format_docs(docs: list[Document]) -> str:
    """Format retrieved documents into context string with citations."""
    if not docs:
        return "No relevant filing excerpts found."

    formatted = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        citation = (
            f"[{i}] {meta.get('company_name', 'Unknown')} "
            f"| {meta.get('form_type', '')} "
            f"| Period: {meta.get('period_of_report', '')} "
            f"| {meta.get('section_name', '')}"
        )
        formatted.append(f"{citation}\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted)


class FinSightRAGChain:
    """
    Primary RAG chain for FinSight360.
    Supports: general Q&A, company-specific risk summaries,
    cross-company comparisons, section-specific queries.
    """

    def __init__(self, vector_store: FilingVectorStore):
        self.vs = vector_store
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0.1,
            openai_api_key=settings.openai_api_key,
        )
        self.output_parser = StrOutputParser()

    def ask(
        self,
        question: str,
        cik: str | None = None,
        section: str | None = None,
    ) -> dict:
        """Ask any question about SEC filings."""
        log.info("rag_query", question=question[:80], cik=cik, section=section)

        retrieved_docs = self.vs.similarity_search(
            question, k=6, filter_cik=cik, filter_section=section
        )
        context = format_docs(retrieved_docs)

        chain = ANALYST_PROMPT_TEMPLATE | self.llm | self.output_parser
        response = chain.invoke({"question": question, "context": context})

        sources = list(
            {
                f"{d.metadata.get('ticker', '')} "
                f"{d.metadata.get('period_of_report', '')} "
                f"- {d.metadata.get('section_name', '')}"
                for d in retrieved_docs
            }
        )

        log.info(
            "rag_response_generated",
            response_length=len(response),
            sources_used=len(sources),
        )
        return {
            "question": question,
            "answer": response,
            "sources": sources,
            "docs_retrieved": len(retrieved_docs),
        }

    def generate_risk_summary(
        self,
        cik: str,
        company_name: str,
        ticker: str,
        ml_risk_score: float,
        risk_tier: str,
        dominant_signal: str,
    ) -> str:
        """Generate a full analyst risk brief for one company."""
        log.info(
            "risk_summary_requested",
            cik=cik,
            company=company_name,
            ml_score=ml_risk_score,
        )

        risk_docs = self.vs.similarity_search(
            f"risk factors financial performance outlook {company_name}",
            k=4,
            filter_cik=cik,
        )
        mda_docs = self.vs.similarity_search(
            f"management discussion analysis results operations {company_name}",
            k=3,
            filter_cik=cik,
        )

        all_docs = risk_docs + mda_docs
        context = format_docs(all_docs)

        if not all_docs:
            return (
                f"## {company_name} ({ticker}) — Risk Brief\n\n"
                f"**No filing text indexed for this company.**\n"
                f"ML Risk Score: {ml_risk_score}/100 | "
                f"Tier: {risk_tier} | Signal: {dominant_signal}\n\n"
                f"*Run `finsight index --ticker {ticker}` to index filings.*"
            )

        chain = RISK_SUMMARY_PROMPT | self.llm | self.output_parser
        brief = chain.invoke(
            {
                "company_name": company_name,
                "ticker": ticker,
                "context": context,
                "ml_risk_score": ml_risk_score,
                "risk_tier": risk_tier,
                "dominant_signal": dominant_signal,
            }
        )

        log.info("risk_brief_generated", cik=cik, length=len(brief))
        return brief
