"""
Benford's Law forensic analysis engine.

Benford's Law: in naturally occurring financial data, the leading digit d
appears with probability log10(1 + 1/d). Deviation from this distribution
is a classic forensic accounting signal used by the SEC and Big 4 auditors.
"""
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, field
from utils.logger import get_logger

log = get_logger("benford")

BENFORD_DISTRIBUTION = {
    1: 0.30103, 2: 0.17609, 3: 0.12494,
    4: 0.09691, 5: 0.07918, 6: 0.06695,
    7: 0.05799, 8: 0.05115, 9: 0.04576
}


@dataclass
class BenfordResult:
    cik: str
    company_name: str
    ticker: str
    n_observations: int
    observed_frequencies: dict
    expected_frequencies: dict
    chi_square_statistic: float
    chi_square_p_value: float
    mad_score: float
    conformity_rating: str    # "Acceptable" | "Marginal" | "Non-conforming"
    benford_risk_score: float  # 0-100
    suspicious_digits: list
    interpretation: str


class BenfordAnalyzer:
    """
    Applies Benford's Law to financial statement data.

    Thresholds per Nigrini (2012) — standard forensic accounting reference:
    MAD < 0.006: Acceptable conformity
    MAD 0.006-0.012: Marginal (investigate further)
    MAD > 0.012: Non-conforming (significant red flag)
    """

    MAD_ACCEPTABLE = 0.006
    MAD_MARGINAL = 0.012

    def extract_leading_digits(self, values: pd.Series) -> pd.Series:
        """Extract leading digit from absolute values, ignore zeros/nulls."""
        clean = values.dropna()
        clean = clean[clean != 0].abs()
        magnitudes = np.floor(np.log10(clean))
        normalized = clean / (10 ** magnitudes)
        leading = normalized.astype(int)
        return leading[leading.between(1, 9)]

    def analyze_series(self, values: pd.Series, cik: str,
                       company_name: str, ticker: str) -> BenfordResult:
        """Run full Benford analysis on a series of financial values."""
        digits = self.extract_leading_digits(values)
        n = len(digits)

        if n < 50:
            log.warning("benford_insufficient_data", cik=cik, n=n)
            return BenfordResult(
                cik=cik, company_name=company_name, ticker=ticker,
                n_observations=n,
                observed_frequencies={d: 0.0 for d in range(1, 10)},
                expected_frequencies=BENFORD_DISTRIBUTION.copy(),
                chi_square_statistic=0.0, chi_square_p_value=1.0,
                mad_score=0.0, conformity_rating="Insufficient Data",
                benford_risk_score=0.0, suspicious_digits=[],
                interpretation=(
                    "Insufficient data for Benford analysis "
                    "(need ≥50 observations)"
                )
            )

        counts = digits.value_counts().reindex(range(1, 10), fill_value=0)
        observed_freq = (counts / n).to_dict()

        expected_counts = [BENFORD_DISTRIBUTION[d] * n for d in range(1, 10)]
        observed_counts = [counts[d] for d in range(1, 10)]
        chi2, p_value = stats.chisquare(observed_counts, f_exp=expected_counts)

        mad = np.mean([
            abs(observed_freq.get(d, 0) - BENFORD_DISTRIBUTION[d])
            for d in range(1, 10)
        ])

        if mad < self.MAD_ACCEPTABLE:
            conformity = "Acceptable"
        elif mad < self.MAD_MARGINAL:
            conformity = "Marginal"
        else:
            conformity = "Non-conforming"

        mad_score = min(100, (mad / 0.020) * 60)
        pval_score = max(0, (1 - p_value) * 40)
        benford_risk = round(mad_score + pval_score, 1)

        suspicious = [
            d for d in range(1, 10)
            if abs(observed_freq.get(d, 0) - BENFORD_DISTRIBUTION[d])
               > 2 * BENFORD_DISTRIBUTION[d] * 0.15
        ]

        interpretation = self._interpret(conformity, mad, p_value, suspicious, n)

        log.info("benford_analysis_complete", cik=cik,
                 mad=round(mad, 4), conformity=conformity,
                 risk_score=benford_risk, n=n)

        return BenfordResult(
            cik=cik, company_name=company_name, ticker=ticker,
            n_observations=n,
            observed_frequencies=observed_freq,
            expected_frequencies=BENFORD_DISTRIBUTION.copy(),
            chi_square_statistic=round(chi2, 4),
            chi_square_p_value=round(p_value, 4),
            mad_score=round(mad, 6),
            conformity_rating=conformity,
            benford_risk_score=benford_risk,
            suspicious_digits=suspicious,
            interpretation=interpretation
        )

    def analyze_company_financials(self,
                                   facts_df: pd.DataFrame,
                                   cik: str,
                                   company_name: str,
                                   ticker: str) -> BenfordResult:
        """
        Run Benford on all financial facts for a company.
        Combines revenues, assets, liabilities, expenses into one pool.
        """
        numeric_cols = facts_df.select_dtypes(include=[np.number]).columns
        all_values = facts_df[numeric_cols].values.flatten()
        series = pd.Series(all_values).dropna()
        return self.analyze_series(series, cik, company_name, ticker)

    def analyze_all_companies(self, db_manager) -> pd.DataFrame:
        """Run Benford analysis across all companies in DuckDB."""
        conn = db_manager.get_connection()

        facts = conn.execute("""
            SELECT ff.cik, ff.concept, ff.value, ff.period_end, ff.form,
                   c.name as company_name, c.ticker
            FROM financial_facts ff
            JOIN companies c ON ff.cik = c.cik
            WHERE ff.form IN ('10-K', '10-Q')
              AND ff.value != 0
        """).df()

        results = []
        for cik, group in facts.groupby("cik"):
            company_name = group["company_name"].iloc[0]
            ticker = group["ticker"].iloc[0] if group["ticker"].iloc[0] else cik
            result = self.analyze_company_financials(
                group[["value"]], str(cik),
                str(company_name), str(ticker)
            )
            results.append({
                "cik": result.cik,
                "company_name": result.company_name,
                "ticker": result.ticker,
                "n_observations": result.n_observations,
                "mad_score": result.mad_score,
                "chi_square_statistic": result.chi_square_statistic,
                "chi_square_p_value": result.chi_square_p_value,
                "conformity_rating": result.conformity_rating,
                "benford_risk_score": result.benford_risk_score,
                "suspicious_digits": str(result.suspicious_digits),
                "interpretation": result.interpretation
            })

        return pd.DataFrame(results)

    def _interpret(self, conformity: str, mad: float, p_value: float,
                   suspicious: list, n: int) -> str:
        if conformity == "Acceptable":
            return (
                f"Financial data ({n} observations) conforms to Benford's Law "
                f"(MAD={mad:.4f}). No forensic concerns from digit analysis."
            )
        elif conformity == "Marginal":
            digits_str = ", ".join(map(str, suspicious)) if suspicious else "none"
            return (
                f"Marginal Benford conformity (MAD={mad:.4f}, p={p_value:.3f}). "
                f"Suspicious digit positions: {digits_str}. "
                f"Recommend cross-checking with cash flow signals."
            )
        else:
            digits_str = (
                ", ".join(map(str, suspicious)) if suspicious else "multiple"
            )
            return (
                f"NON-CONFORMING: Significant Benford deviation (MAD={mad:.4f}, "
                f"p={p_value:.3f}). Suspicious digits: {digits_str}. "
                f"This pattern is consistent with financial data manipulation. "
                f"Escalate for detailed review."
            )
