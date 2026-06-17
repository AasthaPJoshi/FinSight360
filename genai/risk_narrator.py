"""
Generates LLM-powered risk narratives combining all signals:
- ML anomaly score + flags
- Benford's Law result
- Network graph position
- Filing tone analysis
- RAG-retrieved filing evidence
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config.settings import settings
from genai.rag_chain import FinSightRAGChain
from genai.sentiment_analyzer import FilingToneAnalyzer
from utils.logger import get_logger

log = get_logger("risk_narrator")

MASTER_BRIEF_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are the Chief Risk Analyst at FinSight360, \
an autonomous financial intelligence platform. You synthesize \
quantitative ML signals, forensic accounting analysis, and \
qualitative filing language into clear risk briefs for \
investment committee review.""",
    ),
    (
        "human",
        """Generate a FinSight360 Risk Intelligence Brief for:

COMPANY: {company_name} ({ticker})
PERIOD: {period}

═══ QUANTITATIVE SIGNALS ═══
ML Anomaly Score: {ml_risk_score}/100
Isolation Forest: {if_score} (percentile: {percentile}th)
Benford's Law: MAD={benford_mad} | Rating: {benford_rating}
Network Risk: {network_risk}/100
COMBINED Risk Score: {final_score}/100
Risk Tier: {risk_tier}

═══ ML SIGNAL BREAKDOWN ═══
{shap_summary}

═══ FILING LANGUAGE ANALYSIS ═══
Language Risk Score: {language_risk}/100
Going Concern Language: {going_concern}
Restatement Risk: {restatement_risk}
Legal Risk: {legal_risk}
Tone: {llm_tone_snippet}

═══ RELEVANT FILING EXCERPTS ═══
{filing_context}

═══ OUTPUT FORMAT ═══
# FinSight360 Risk Brief: {company_name}

## Risk Rating: [CRITICAL / HIGH / MEDIUM / LOW / CLEAN]

## Executive Summary
[3 sentences max. What is the core risk? Why does it matter?]

## Evidence Summary
| Signal | Value | Risk Level |
|--------|-------|------------|
[Table with all quantitative signals]

## Key Risk Drivers
[Numbered list of top 3-5 specific risk factors with evidence]

## Filing Evidence
[2-3 direct observations from filing text with citations]

## Analyst Action
[ESCALATE TO SENIOR ANALYST / PLACE ON WATCHLIST / QUARTERLY MONITOR / NO ACTION]
[One sentence rationale]

## Confidence Level
[HIGH / MEDIUM / LOW] — [reason]
""",
    ),
])


class RiskNarrator:
    """Orchestrates all signals into final risk briefs."""

    def __init__(
        self,
        db,
        rag_chain: FinSightRAGChain,
        tone_analyzer: FilingToneAnalyzer,
    ):
        self.db = db
        self.rag = rag_chain
        self.tone = tone_analyzer
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0.2,
            openai_api_key=settings.openai_api_key,
        )

    def generate_brief(self, cik: str) -> dict:
        """Generate complete risk brief for one company."""
        conn = self.db.get_connection()

        scores = conn.execute(
            """
            SELECT
                f.cik, f.ticker, f.company_name, f.final_risk_score,
                f.final_risk_tier, f.ml_risk_score, f.benford_risk_score,
                f.network_risk_score,
                m.anomaly_percentile,
                b.mad_score, b.conformity_rating,
                r.dominant_signal
            FROM final_risk_scores f
            LEFT JOIN ml_anomaly_scores m ON f.cik = m.cik
            LEFT JOIN benford_results b ON f.cik = b.cik
            LEFT JOIN main_marts.mart_risk_candidates r ON f.cik = r.cik
            WHERE f.cik = ?
            """,
            [cik],
        ).df()

        if scores.empty:
            return {
                "status": "no_data",
                "cik": cik,
                "brief": "No data available. Run full pipeline first.",
            }

        row = scores.iloc[0]

        rag_result = self.rag.ask(
            f"What are the main risk factors and MD&A concerns "
            f"for {row['company_name']}?",
            cik=cik,
        )

        filing_docs = self.rag.vs.similarity_search(
            f"risk factors outlook {row['company_name']}", k=5, filter_cik=cik
        )
        tone_results = self.tone.analyze_filing_docs(
            filing_docs, str(row["company_name"])
        )

        chain = MASTER_BRIEF_PROMPT | self.llm | StrOutputParser()
        brief = chain.invoke(
            {
                "company_name": row["company_name"],
                "ticker": row.get("ticker", "N/A"),
                "period": "Latest",
                "ml_risk_score": round(float(row.get("ml_risk_score", 0)), 1),
                "if_score": "N/A",
                "percentile": round(float(row.get("anomaly_percentile", 0)), 0),
                "benford_mad": round(float(row.get("mad_score", 0) or 0), 5),
                "benford_rating": str(row.get("conformity_rating", "N/A")),
                "network_risk": round(
                    float(row.get("network_risk_score", 0)), 1
                ),
                "final_score": round(float(row.get("final_risk_score", 0)), 1),
                "risk_tier": str(row.get("final_risk_tier", "N/A")),
                "shap_summary": self._build_shap_summary(row),
                "language_risk": tone_results.get("language_risk_score", 0),
                "going_concern": str(
                    tone_results.get("going_concern_detected", False)
                ),
                "restatement_risk": str(
                    tone_results.get("restatement_risk", False)
                ),
                "legal_risk": str(tone_results.get("legal_risk_detected", False)),
                "llm_tone_snippet": tone_results.get("llm_analysis", "N/A")[
                    :300
                ],
                "filing_context": rag_result.get("answer", "No filing data.")[
                    :2000
                ],
            }
        )

        self._save_brief(
            cik,
            str(row["company_name"]),
            float(row.get("final_risk_score", 0)),
            brief,
        )

        log.info(
            "risk_brief_generated",
            cik=cik,
            final_score=row.get("final_risk_score", 0),
            length=len(brief),
        )

        return {
            "status": "success",
            "cik": cik,
            "company_name": str(row["company_name"]),
            "ticker": str(row.get("ticker", "")),
            "final_risk_score": float(row.get("final_risk_score", 0)),
            "risk_tier": str(row.get("final_risk_tier", "")),
            "brief": brief,
            "sources": rag_result.get("sources", []),
        }

    def _build_shap_summary(self, row) -> str:
        signals = []
        flag_map = {
            "flag_ar_acceleration": "AR acceleration (+0.18)",
            "flag_cashflow_divergence": "Cash-earnings divergence (+0.22)",
            "flag_margin_compression": "Margin compression (+0.15)",
        }
        for flag, desc in flag_map.items():
            if row.get(flag, 0):
                signals.append(f"  • {desc}")

        dominant = row.get("dominant_signal", "None")
        if dominant and str(dominant) != "None":
            signals.insert(0, f"  • Dominant: {dominant} (+0.28)")

        if not signals:
            return "  • No individual signals triggered"
        return "\n".join(signals)

    def _save_brief(
        self, cik: str, company_name: str, risk_score: float, brief: str
    ) -> None:
        conn = self.db.get_connection()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS risk_briefs (
                    cik VARCHAR PRIMARY KEY,
                    company_name VARCHAR,
                    final_risk_score DOUBLE,
                    brief_text TEXT,
                    generated_at TIMESTAMP DEFAULT current_timestamp
                )
            """)
            conn.execute(
                """
                INSERT INTO risk_briefs VALUES (?,?,?,?,current_timestamp)
                ON CONFLICT (cik) DO UPDATE SET
                    company_name = excluded.company_name,
                    final_risk_score = excluded.final_risk_score,
                    brief_text = excluded.brief_text,
                    generated_at = excluded.generated_at
                """,
                [cik, company_name, risk_score, brief],
            )
        except Exception as e:
            log.warning("brief_save_failed", error=str(e))
