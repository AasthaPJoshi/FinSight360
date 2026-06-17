"""
Bias analysis for FinSight360 ML models.

Checks whether risk scores are systematically higher/lower
for certain industry groups, geographies, or SIC codes.
Addresses EU AI Act Article 10 bias monitoring requirements.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("bias_analyzer")


@dataclass
class BiasReport:
    group_column: str
    overall_mean: float
    overall_std: float
    group_stats: pd.DataFrame
    flagged_groups: list[str]
    bias_detected: bool
    bias_severity: str  # "NONE" | "LOW" | "MEDIUM" | "HIGH"
    interpretation: str


class BiasAnalyzer:
    """
    Analyzes systematic bias in FinSight360 risk scores
    across industry groups, geographic regions, and SIC codes.
    """

    BIAS_THRESHOLD_STD = 1.5

    def analyze_by_group(
        self,
        df: pd.DataFrame,
        group_col: str,
        score_col: str = "final_risk_score",
    ) -> BiasReport:
        """Analyze risk score distribution by a grouping variable."""
        if df.empty or group_col not in df.columns or score_col not in df.columns:
            return BiasReport(
                group_column=group_col,
                overall_mean=0.0,
                overall_std=0.0,
                group_stats=pd.DataFrame(),
                flagged_groups=[],
                bias_detected=False,
                bias_severity="NONE",
                interpretation="Insufficient data",
            )

        overall_mean = float(df[score_col].mean())
        overall_std = float(df[score_col].std())

        group_stats = (
            df.groupby(group_col)[score_col]
            .agg(
                mean_score="mean",
                std_score="std",
                company_count="count",
                median_score="median",
            )
            .reset_index()
        )
        group_stats["high_risk_count"] = (
            df[df[score_col] > 70].groupby(group_col)[score_col].count().reindex(
                group_stats[group_col]
            ).fillna(0).values
        )
        group_stats["high_risk_rate"] = (
            group_stats["high_risk_count"] / group_stats["company_count"]
        )

        group_stats["z_score"] = (
            (group_stats["mean_score"] - overall_mean) / max(overall_std, 0.001)
        )
        group_stats["deviation_from_mean"] = (
            group_stats["mean_score"] - overall_mean
        ).round(1)
        group_stats["bias_flag"] = (
            group_stats["z_score"].abs() > self.BIAS_THRESHOLD_STD
        )
        group_stats["bias_direction"] = group_stats["z_score"].apply(
            lambda z: (
                "Over-scored"
                if z > self.BIAS_THRESHOLD_STD
                else ("Under-scored" if z < -self.BIAS_THRESHOLD_STD else "Normal")
            )
        )

        flagged = group_stats[group_stats["bias_flag"]][group_col].tolist()
        max_z = float(group_stats["z_score"].abs().max()) if not group_stats.empty else 0.0

        if max_z > 3.0:
            severity = "HIGH"
        elif max_z > 2.0:
            severity = "MEDIUM"
        elif max_z > self.BIAS_THRESHOLD_STD:
            severity = "LOW"
        else:
            severity = "NONE"

        interpretation = self._interpret(
            group_col, flagged, severity, overall_mean, group_stats
        )

        log.info(
            "bias_analysis_complete",
            group_col=group_col,
            flagged=len(flagged),
            severity=severity,
        )

        return BiasReport(
            group_column=group_col,
            overall_mean=round(overall_mean, 2),
            overall_std=round(overall_std, 2),
            group_stats=group_stats.sort_values("z_score", ascending=False),
            flagged_groups=flagged,
            bias_detected=len(flagged) > 0,
            bias_severity=severity,
            interpretation=interpretation,
        )

    def run_full_bias_audit(
        self, df: pd.DataFrame
    ) -> dict[str, BiasReport]:
        """Run bias analysis across all relevant grouping dimensions."""
        reports: dict[str, BiasReport] = {}
        dimensions = {"industry_group": "Industry Sector Bias"}
        if "state_of_incorporation" in df.columns:
            dimensions["state_of_incorporation"] = "Geographic Bias"

        for col in dimensions:
            if col in df.columns:
                reports[col] = self.analyze_by_group(df, col)

        return reports

    def generate_bias_summary_table(
        self, reports: dict[str, BiasReport]
    ) -> pd.DataFrame:
        rows = []
        for dim, report in reports.items():
            rows.append(
                {
                    "Dimension": dim.replace("_", " ").title(),
                    "Groups Analyzed": len(report.group_stats),
                    "Groups Flagged": len(report.flagged_groups),
                    "Bias Severity": report.bias_severity,
                    "Overall Mean Score": report.overall_mean,
                    "Flagged Groups": (
                        ", ".join(report.flagged_groups[:3])
                        if report.flagged_groups
                        else "None"
                    ),
                }
            )
        return pd.DataFrame(rows)

    def _interpret(
        self,
        group_col: str,
        flagged: list,
        severity: str,
        overall_mean: float,
        group_stats: pd.DataFrame,
    ) -> str:
        if not flagged:
            return (
                f"No systematic bias detected across {group_col} groups. "
                f"Risk scores are distributed consistently (mean={overall_mean:.1f})."
            )
        over = group_stats[group_stats["bias_direction"] == "Over-scored"][
            group_col
        ].tolist()
        under = group_stats[group_stats["bias_direction"] == "Under-scored"][
            group_col
        ].tolist()
        parts = [f"Bias severity: {severity}."]
        if over:
            parts.append(
                f"Over-scored groups: {', '.join(str(g) for g in over[:3])}."
            )
        if under:
            parts.append(
                f"Under-scored groups: {', '.join(str(g) for g in under[:3])}."
            )
        parts.append(
            "Recommend: review feature engineering for these groups "
            "and validate with domain expert before production deployment."
        )
        return " ".join(parts)
