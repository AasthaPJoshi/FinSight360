"""Community detection and centrality analysis for the corporate knowledge graph."""
import networkx as nx
import pandas as pd
import structlog

try:
    import community as community_louvain
    _LOUVAIN_AVAILABLE = True
except ImportError:
    _LOUVAIN_AVAILABLE = False

log = structlog.get_logger(__name__)


class ClusterDetector:
    def __init__(self, graph: nx.MultiDiGraph):
        self.G = graph
        self._centrality: dict = {}
        self._pagerank: dict = {}

    def detect_communities(self) -> dict[str, int]:
        """Run Louvain community detection on the undirected projection."""
        G_undirected = self.G.to_undirected()
        G_simple = nx.Graph(G_undirected)

        if G_simple.number_of_nodes() == 0:
            return {}

        if not _LOUVAIN_AVAILABLE:
            log.warning("louvain_unavailable", msg="Falling back to connected components")
            partition = {}
            for i, component in enumerate(nx.connected_components(G_simple)):
                for node in component:
                    partition[node] = i
            return partition

        try:
            partition = community_louvain.best_partition(G_simple, random_state=42)
        except Exception as e:
            log.warning("louvain_failed", error=str(e))
            partition = {}
            for i, component in enumerate(nx.connected_components(G_simple)):
                for node in component:
                    partition[node] = i
        return partition

    def compute_centrality(self) -> dict[str, dict]:
        """Compute PageRank, betweenness centrality, and degree centrality."""
        if self.G.number_of_nodes() == 0:
            return {}

        G_simple = nx.DiGraph()
        for u, v, data in self.G.edges(data=True):
            G_simple.add_edge(u, v)

        pagerank = nx.pagerank(G_simple, alpha=0.85) if G_simple.number_of_edges() > 0 else {}
        betweenness = (
            nx.betweenness_centrality(G_simple)
            if G_simple.number_of_nodes() > 1
            else {}
        )
        in_degree = dict(G_simple.in_degree())
        out_degree = dict(G_simple.out_degree())

        centrality = {}
        for node in self.G.nodes():
            centrality[node] = {
                "pagerank": pagerank.get(node, 0.0),
                "betweenness": betweenness.get(node, 0.0),
                "in_degree": in_degree.get(node, 0),
                "out_degree": out_degree.get(node, 0),
            }

        self._centrality = centrality
        self._pagerank = pagerank
        return centrality

    def identify_risk_clusters(self, partition: dict[str, int]) -> pd.DataFrame:
        """Summarize risk metrics per community cluster."""
        if not partition:
            return pd.DataFrame()

        records = []
        community_ids = set(partition.values())

        for community_id in sorted(community_ids):
            members = [n for n, c in partition.items() if c == community_id]
            company_members = [
                n for n in members
                if self.G.nodes[n].get("node_type") == "Company"
            ]

            risk_scores = [
                self.G.nodes[n].get("ml_risk_score", 0.0)
                for n in company_members
            ]
            anomaly_count = sum(
                1 for n in company_members
                if self.G.nodes[n].get("is_anomaly", False)
            )
            pagerank_scores = [
                self._pagerank.get(n, 0.0) for n in members
            ]

            records.append({
                "community_id": community_id,
                "size": len(members),
                "company_count": len(company_members),
                "avg_ml_risk_score": sum(risk_scores) / len(risk_scores) if risk_scores else 0.0,
                "max_ml_risk_score": max(risk_scores) if risk_scores else 0.0,
                "anomaly_count": anomaly_count,
                "avg_pagerank": sum(pagerank_scores) / len(pagerank_scores) if pagerank_scores else 0.0,
                "cluster_risk_level": (
                    "HIGH" if (sum(risk_scores) / len(risk_scores) if risk_scores else 0) >= 60
                    else "MEDIUM" if (sum(risk_scores) / len(risk_scores) if risk_scores else 0) >= 30
                    else "LOW"
                ),
            })

        return pd.DataFrame(records).sort_values("avg_ml_risk_score", ascending=False)

    def find_critical_connectors(self, top_n: int = 10) -> list[dict]:
        """Find nodes with highest betweenness centrality (systemic risk connectors)."""
        if not self._centrality:
            self.compute_centrality()

        sorted_nodes = sorted(
            self._centrality.items(),
            key=lambda x: x[1]["betweenness"],
            reverse=True,
        )[:top_n]

        results = []
        for node, metrics in sorted_nodes:
            node_data = self.G.nodes[node]
            results.append({
                "node_id": node,
                "node_type": node_data.get("node_type", "Unknown"),
                "name": node_data.get("company_name") or node_data.get("auditor_name") or node,
                "betweenness": metrics["betweenness"],
                "pagerank": metrics["pagerank"],
                "in_degree": metrics["in_degree"],
                "out_degree": metrics["out_degree"],
                "ml_risk_score": node_data.get("ml_risk_score", 0.0),
            })
        return results
