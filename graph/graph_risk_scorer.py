"""Compute final combined risk scores: ML (50%) + Benford (30%) + Network (20%)."""
import pandas as pd
import structlog

import networkx as nx

from storage.database import Database

log = structlog.get_logger(__name__)

ML_WEIGHT = 0.50
BENFORD_WEIGHT = 0.30
NETWORK_WEIGHT = 0.20


class GraphRiskScorer:
    def __init__(self, db: Database):
        self.db = db

    def compute_final_scores(self, networkx_graph: nx.MultiDiGraph) -> pd.DataFrame:
        conn = self.db.get_connection()

        ml_df = self._load_ml_scores(conn)
        benford_df = self._load_benford_scores(conn)

        if ml_df.empty:
            log.warning("no_ml_scores", msg="final_risk_scores will be empty")
            return pd.DataFrame()

        df = ml_df.merge(benford_df, on="cik", how="left")
        df["benford_risk_score"] = df["benford_risk_score"].fillna(0.0)

        network_scores = self._extract_network_scores(networkx_graph)
        network_df = pd.DataFrame(
            list(network_scores.items()), columns=["cik", "network_risk_score"]
        )
        df = df.merge(network_df, on="cik", how="left")
        df["network_risk_score"] = df["network_risk_score"].fillna(0.0)

        df["final_risk_score"] = (
            df["ml_risk_score"] * ML_WEIGHT
            + df["benford_risk_score"] * BENFORD_WEIGHT
            + df["network_risk_score"] * NETWORK_WEIGHT
        ).clip(0, 100).round(2)

        df["final_risk_tier"] = df["final_risk_score"].apply(self._score_to_tier)

        result = df[[
            "cik", "ticker", "company_name",
            "ml_risk_score", "benford_risk_score", "network_risk_score",
            "final_risk_score", "final_risk_tier",
        ]].copy()

        self._write_to_db(conn, result)
        log.info("final_scores_computed", n=len(result))
        return result

    def _load_ml_scores(self, conn) -> pd.DataFrame:
        try:
            return conn.execute("""
                SELECT cik, ticker, company_name,
                       ml_risk_score, risk_tier, is_anomaly
                FROM ml_anomaly_scores
            """).df()
        except Exception:
            return pd.DataFrame()

    def _load_benford_scores(self, conn) -> pd.DataFrame:
        try:
            return conn.execute("""
                SELECT cik, benford_risk_score
                FROM benford_results
            """).df()
        except Exception:
            return pd.DataFrame(columns=["cik", "benford_risk_score"])

    def _extract_network_scores(self, G: nx.MultiDiGraph) -> dict[str, float]:
        scores = {}
        for node, data in G.nodes(data=True):
            if data.get("node_type") == "Company":
                scores[node] = data.get("network_risk_score", 0.0)
        return scores

    @staticmethod
    def _score_to_tier(score: float) -> str:
        if score >= 70:
            return "HIGH"
        if score >= 45:
            return "MEDIUM"
        if score >= 20:
            return "LOW"
        return "CLEAN"

    def _write_to_db(self, conn, df: pd.DataFrame) -> None:
        try:
            conn.execute("DROP TABLE IF EXISTS final_risk_scores")
            conn.execute("CREATE TABLE final_risk_scores AS SELECT * FROM df")
            log.info("final_risk_scores_saved", rows=len(df))
        except Exception as e:
            log.error("final_risk_scores_write_failed", error=str(e))
