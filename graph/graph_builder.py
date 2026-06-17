"""Build an in-memory corporate knowledge graph using NetworkX."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import networkx as nx
import structlog

from storage.database import Database

log = structlog.get_logger(__name__)

AUDITORS_JSON = Path(__file__).parent.parent / "data" / "reference" / "sp500_auditors.json"
RELATIONSHIPS_JSON = Path(__file__).parent.parent / "data" / "reference" / "sp500_relationships.json"


@dataclass
class GraphStats:
    n_companies: int
    n_auditors: int
    n_subsidiaries: int
    n_edges: int
    n_communities: int = 0
    high_risk_nodes: int = 0


class CorporateGraphBuilder:
    def __init__(self, db: Database):
        self.db = db
        self.G: nx.MultiDiGraph = nx.MultiDiGraph()

    def build(self) -> nx.MultiDiGraph:
        self.G = nx.MultiDiGraph()
        self._add_company_nodes()
        self._add_auditor_relationships()
        self._add_corporate_relationships()
        self._add_risk_attributes()
        log.info(
            "graph_built",
            nodes=self.G.number_of_nodes(),
            edges=self.G.number_of_edges(),
        )
        return self.G

    def _add_company_nodes(self) -> None:
        conn = self.db.get_connection()
        try:
            rows = conn.execute("""
                SELECT
                    c.cik,
                    c.ticker,
                    c.name AS company_name,
                    c.sic_description AS industry_group,
                    COALESCE(m.ml_risk_score, 0.0) AS ml_risk_score,
                    COALESCE(m.risk_tier, 'UNKNOWN') AS risk_tier,
                    COALESCE(m.is_anomaly, false) AS is_anomaly,
                    COALESCE(r.risk_score, 0.0) AS rule_risk_score
                FROM companies c
                LEFT JOIN ml_anomaly_scores m ON c.cik = m.cik
                LEFT JOIN main_marts.mart_risk_candidates r ON c.cik = r.cik
            """).fetchall()
        except Exception:
            try:
                rows = conn.execute("""
                    SELECT
                        c.cik,
                        c.ticker,
                        c.name AS company_name,
                        c.sic_description AS industry_group,
                        0.0 AS ml_risk_score,
                        'UNKNOWN' AS risk_tier,
                        false AS is_anomaly,
                        0.0 AS rule_risk_score
                    FROM companies c
                """).fetchall()
            except Exception:
                rows = []

        for row in rows:
            cik, ticker, name, industry, ml_score, tier, is_anomaly, rule_score = row
            self.G.add_node(
                str(cik),
                node_type="Company",
                ticker=ticker or "",
                company_name=name or "",
                industry_group=industry or "",
                ml_risk_score=float(ml_score or 0),
                risk_tier=tier or "UNKNOWN",
                is_anomaly=bool(is_anomaly),
                rule_risk_score=float(rule_score or 0),
                network_risk_score=0.0,
            )

    def _add_auditor_relationships(self) -> None:
        if not AUDITORS_JSON.exists():
            log.warning("auditors_json_missing", path=str(AUDITORS_JSON))
            return

        with open(AUDITORS_JSON) as f:
            auditors: dict = json.load(f)

        for cik, info in auditors.items():
            auditor_id = info["auditor_id"]
            auditor_name = info["auditor"]

            if auditor_id not in self.G:
                self.G.add_node(
                    auditor_id,
                    node_type="Auditor",
                    auditor_name=auditor_name,
                    network_risk_score=0.0,
                )

            if str(cik) in self.G:
                self.G.add_edge(
                    str(cik),
                    auditor_id,
                    edge_type="AUDITED_BY",
                    audit_opinion=info.get("audit_opinion", ""),
                    audit_fee_millions=info.get("audit_fee_millions", 0.0),
                )

    def _add_corporate_relationships(self) -> None:
        if not RELATIONSHIPS_JSON.exists():
            log.warning("relationships_json_missing", path=str(RELATIONSHIPS_JSON))
            return

        with open(RELATIONSHIPS_JSON) as f:
            relationships: list = json.load(f)

        edge_type_map = {
            "SUBSIDIARY": "SUBSIDIARY_OF",
            "PARTNERS_WITH": "PARTNERS_WITH",
            "CUSTOMER_OF": "CUSTOMER_OF",
            "JOINT_VENTURE_WITH": "JOINT_VENTURE_WITH",
        }

        for rel in relationships:
            parent_cik = str(rel["parent_cik"])
            child_cik = str(rel["subsidiary_cik"])
            rel_type = edge_type_map.get(rel["relationship_type"], rel["relationship_type"])

            if parent_cik not in self.G:
                self.G.add_node(
                    parent_cik,
                    node_type="Company",
                    company_name=rel.get("parent_name", ""),
                    network_risk_score=0.0,
                )
            if child_cik not in self.G:
                self.G.add_node(
                    child_cik,
                    node_type="Company",
                    company_name=rel.get("subsidiary_name", ""),
                    network_risk_score=0.0,
                )

            self.G.add_edge(
                parent_cik,
                child_cik,
                edge_type=rel_type,
                ownership_pct=rel.get("ownership_pct"),
            )

    def _add_risk_attributes(self) -> None:
        """Propagate network risk: high-risk neighbors increase node's network score."""
        for node in self.G.nodes():
            neighbors = list(self.G.predecessors(node)) + list(self.G.successors(node))
            if not neighbors:
                continue
            neighbor_risks = [
                self.G.nodes[n].get("ml_risk_score", 0.0)
                for n in neighbors
                if self.G.nodes[n].get("node_type") == "Company"
            ]
            if neighbor_risks:
                avg_neighbor_risk = sum(neighbor_risks) / len(neighbor_risks)
                self.G.nodes[node]["network_risk_score"] = min(avg_neighbor_risk * 0.3, 100.0)

    def get_auditor_sharing_pairs(self) -> list[dict]:
        """Return pairs of companies that share the same auditor."""
        auditor_to_companies: dict[str, list[str]] = {}
        for u, v, data in self.G.edges(data=True):
            if data.get("edge_type") == "AUDITED_BY":
                auditor_to_companies.setdefault(v, []).append(u)

        pairs = []
        for auditor_id, companies in auditor_to_companies.items():
            auditor_name = self.G.nodes[auditor_id].get("auditor_name", auditor_id)
            for i in range(len(companies)):
                for j in range(i + 1, len(companies)):
                    pairs.append({
                        "company_a": companies[i],
                        "company_b": companies[j],
                        "shared_auditor_id": auditor_id,
                        "shared_auditor_name": auditor_name,
                    })
        return pairs

    def get_stats(self) -> GraphStats:
        company_nodes = [n for n, d in self.G.nodes(data=True) if d.get("node_type") == "Company"]
        auditor_nodes = [n for n, d in self.G.nodes(data=True) if d.get("node_type") == "Auditor"]
        subsidiary_edges = [
            (u, v) for u, v, d in self.G.edges(data=True)
            if d.get("edge_type") == "SUBSIDIARY_OF"
        ]
        high_risk = [
            n for n in company_nodes
            if self.G.nodes[n].get("ml_risk_score", 0) >= 70
        ]
        return GraphStats(
            n_companies=len(company_nodes),
            n_auditors=len(auditor_nodes),
            n_subsidiaries=len(subsidiary_edges),
            n_edges=self.G.number_of_edges(),
            high_risk_nodes=len(high_risk),
        )
