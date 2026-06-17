"""Query engine for the corporate knowledge graph — Neo4j or NetworkX fallback."""
from typing import Optional

import networkx as nx
import structlog

log = structlog.get_logger(__name__)

CYPHER_QUERIES = {
    "high_risk_companies": """
        MATCH (c:Company)
        WHERE c.ml_risk_score >= $threshold
        RETURN c.cik AS cik, c.ticker AS ticker, c.company_name AS company_name,
               c.ml_risk_score AS ml_risk_score, c.risk_tier AS risk_tier
        ORDER BY c.ml_risk_score DESC
        LIMIT $limit
    """,
    "auditor_sharing_risk": """
        MATCH (c1:Company)-[:AUDITED_BY]->(a:Auditor)<-[:AUDITED_BY]-(c2:Company)
        WHERE c1.cik < c2.cik
        RETURN c1.ticker AS ticker_a, c2.ticker AS ticker_b,
               a.auditor_name AS shared_auditor,
               c1.ml_risk_score AS risk_a, c2.ml_risk_score AS risk_b,
               (c1.ml_risk_score + c2.ml_risk_score) / 2 AS avg_risk
        ORDER BY avg_risk DESC
        LIMIT 50
    """,
    "subsidiary_chain": """
        MATCH path = (parent:Company)-[:SUBSIDIARY_OF*1..3]->(child:Company)
        WHERE parent.cik = $cik
        RETURN [node IN nodes(path) | node.ticker] AS chain,
               length(path) AS depth
        ORDER BY depth DESC
    """,
    "network_neighbors": """
        MATCH (c:Company {cik: $cik})-[r]-(neighbor)
        RETURN neighbor.cik AS neighbor_cik,
               type(r) AS relationship,
               neighbor.ml_risk_score AS neighbor_risk
        ORDER BY neighbor_risk DESC
    """,
    "top_pagerank_nodes": """
        MATCH (c:Company)
        RETURN c.cik AS cik, c.ticker AS ticker, c.company_name AS company_name,
               c.ml_risk_score AS ml_risk_score
        ORDER BY c.ml_risk_score DESC
        LIMIT $limit
    """,
}


class GraphQueryEngine:
    def __init__(
        self,
        neo4j_loader=None,
        networkx_graph: Optional[nx.MultiDiGraph] = None,
    ):
        self.neo4j = neo4j_loader
        self.G = networkx_graph
        self._use_neo4j = neo4j_loader is not None and getattr(neo4j_loader, "connected", False)

    def query_high_risk_companies(
        self, threshold: float = 70.0, limit: int = 20
    ) -> list[dict]:
        if self._use_neo4j:
            try:
                with self.neo4j._driver.session() as session:
                    result = session.run(
                        CYPHER_QUERIES["high_risk_companies"],
                        threshold=threshold,
                        limit=limit,
                    )
                    return [dict(record) for record in result]
            except Exception as e:
                log.warning("neo4j_query_failed", error=str(e), fallback="networkx")

        if self.G is None:
            return []

        results = []
        for node, data in self.G.nodes(data=True):
            if data.get("node_type") != "Company":
                continue
            score = data.get("ml_risk_score", 0.0)
            if score >= threshold:
                results.append({
                    "cik": node,
                    "ticker": data.get("ticker", ""),
                    "company_name": data.get("company_name", ""),
                    "ml_risk_score": score,
                    "risk_tier": data.get("risk_tier", ""),
                })

        results.sort(key=lambda x: x["ml_risk_score"], reverse=True)
        return results[:limit]

    def query_auditor_sharing_risk(self) -> list[dict]:
        if self._use_neo4j:
            try:
                with self.neo4j._driver.session() as session:
                    result = session.run(CYPHER_QUERIES["auditor_sharing_risk"])
                    return [dict(record) for record in result]
            except Exception as e:
                log.warning("neo4j_auditor_query_failed", error=str(e))
        return []

    def get_all_cypher_queries(self) -> dict:
        return CYPHER_QUERIES
