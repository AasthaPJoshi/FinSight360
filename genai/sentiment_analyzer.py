"""
Analyzes tone and language patterns in SEC filings.

Two-layer approach:
1. Fast lexicon scan: counts risk phrases per category
2. LLM contextual analysis: understands negations and nuance
"""
import re

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config.settings import settings
from utils.logger import get_logger

log = get_logger("sentiment_analyzer")

# Risk language patterns from forensic accounting literature
RISK_LANGUAGE_PATTERNS = {
    "going_concern": [
        "going concern",
        "substantial doubt",
        "ability to continue",
        "liquidation",
        "cease operations",
    ],
    "restatement": [
        "restate",
        "restatement",
        "material weakness",
        "significant deficiency",
        "internal control",
    ],
    "legal_risk": [
        "securities litigation",
        "class action",
        "SEC investigation",
        "regulatory inquiry",
        "subpoena",
        "indictment",
    ],
    "uncertainty_hedging": [
        "may not",
        "could result",
        "cannot assure",
        "no assurance",
        "might not",
        "there can be no guarantee",
    ],
    "revenue_risk": [
        "revenue recognition",
        "channel stuffing",
        "bill and hold",
        "accelerated recognition",
        "contingent revenue",
    ],
    "auditor_risk": [
        "change in auditor",
        "auditor disagreement",
        "qualified opinion",
        "adverse opinion",
    ],
}

POSITIVE_LANGUAGE = [
    "record revenue",
    "strong growth",
    "exceeded expectations",
    "market leader",
    "competitive advantage",
    "robust demand",
]


class FilingToneAnalyzer:
    """Lexicon-based + LLM-based tone analysis for SEC filings."""

    def __init__(self):
        self._llm: ChatOpenAI | None = None

    @property
    def llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=settings.llm_model,
                temperature=0,
                openai_api_key=settings.openai_api_key,
            )
        return self._llm

    def lexicon_scan(self, text: str) -> dict:
        """Fast regex scan for risk language patterns. Returns counts per category."""
        text_lower = text.lower()

        results: dict = {}
        total_risk_hits = 0

        for category, phrases in RISK_LANGUAGE_PATTERNS.items():
            hits = []
            for phrase in phrases:
                count = len(re.findall(re.escape(phrase), text_lower))
                if count > 0:
                    hits.append({"phrase": phrase, "count": count})
                    total_risk_hits += count
            results[category] = {
                "hit_count": len(hits),
                "total_mentions": sum(h["count"] for h in hits),
                "phrases_found": hits,
            }

        positive_hits = sum(
            len(re.findall(re.escape(p), text_lower)) for p in POSITIVE_LANGUAGE
        )

        word_count = max(len(text.split()), 1)
        risk_density = (total_risk_hits / word_count) * 1000
        language_risk_score = min(100, round(risk_density * 8, 1))

        return {
            "categories": results,
            "total_risk_mentions": total_risk_hits,
            "positive_mentions": positive_hits,
            "language_risk_score": language_risk_score,
            "sentiment_ratio": round(positive_hits / max(total_risk_hits, 1), 2),
            "going_concern_detected": results["going_concern"]["total_mentions"] > 0,
            "restatement_risk": results["restatement"]["total_mentions"] > 0,
            "legal_risk_detected": results["legal_risk"]["total_mentions"] > 1,
        }

    def llm_tone_analysis(self, text_sample: str, company_name: str) -> str:
        """LLM-based contextual tone analysis — catches negations lexicon misses."""
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a forensic accounting expert analyzing SEC filing language.",
            ),
            (
                "human",
                """Analyze the tone and risk language in this SEC filing excerpt for {company}.

TEXT:
{text}

Provide:
1. Overall tone: [OPTIMISTIC / NEUTRAL / CAUTIOUS / ALARMING]
2. Top 3 concerning phrases (with context explaining WHY they're concerning)
3. Any positive language that seems inconsistent with financial data
4. Language risk rating: [LOW / MEDIUM / HIGH / CRITICAL]
5. One-sentence summary for analyst report

Be concise — this is for an analyst dashboard.""",
            ),
        ])

        chain = prompt | self.llm | StrOutputParser()
        sample = text_sample[:3000]
        return chain.invoke({"company": company_name, "text": sample})

    def analyze_filing_docs(
        self, docs: list[Document], company_name: str
    ) -> dict:
        """Run full tone analysis on a company's filing documents."""
        if not docs:
            return {
                "lexicon_results": {},
                "llm_analysis": "No filing text available.",
                "language_risk_score": 0,
                "going_concern_detected": False,
                "restatement_risk": False,
                "legal_risk_detected": False,
            }

        full_text = " ".join([d.page_content for d in docs])
        lexicon_results = self.lexicon_scan(full_text)

        risk_docs = [
            d for d in docs if d.metadata.get("section_key") == "risk_factors"
        ]
        analysis_text = (
            risk_docs[0].page_content if risk_docs else full_text[:3000]
        )
        llm_analysis = self.llm_tone_analysis(analysis_text, company_name)

        return {
            "lexicon_results": lexicon_results,
            "llm_analysis": llm_analysis,
            "language_risk_score": lexicon_results["language_risk_score"],
            "going_concern_detected": lexicon_results["going_concern_detected"],
            "restatement_risk": lexicon_results["restatement_risk"],
            "legal_risk_detected": lexicon_results["legal_risk_detected"],
        }
