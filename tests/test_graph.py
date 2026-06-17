"""Tests for Phase 4: Corporate Knowledge Graph."""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import networkx as nx
import pandas as pd
import pytest

from graph.graph_builder import CorporateGraphBuilder, GraphStats
from graph.cluster_detector import ClusterDetector
from graph.graph_queries import GraphQueryEngine
from graph.graph_risk_scorer import GraphRiskScorer


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

def _make_mock_db(rows=None):
    """Create a mock Database with controllable query results."""
    mock_db = MagicMock()
    mock_conn = MagicMock()
    mock_db.get_connection.return_value = mock_conn

    if rows is None:
        rows = [
            ("CIK001", "AAPL", "Apple Inc.", "Technology", 80.0, "HIGH", True, 75.0),
            ("CIK002", "MSFT", "Microsoft Corp.", "Technology", 40.0, "MEDIUM", False, 35.0),
            ("CIK003", "JPM", "JPMorgan Chase", "Financials", 15.0, "CLEAN", False, 10.0),
        ]

    mock_conn.execute.return_value.fetchall.return_value = rows
    return mock_db, mock_conn


def _make_auditors_json(tmp_path: Path, cik_auditor_map: dict) -> Path:
    """Write a temporary sp500_auditors.json and return its path."""
    p = tmp_path / "sp500_auditors.json"
    p.write_text(json.dumps(cik_auditor_map))
    return p


def _make_simple_graph(n_nodes: int = 5) -> nx.MultiDiGraph:
    """Build a simple graph with Company nodes for testing."""
    G = nx.MultiDiGraph()
    for i in range(n_nodes):
        G.add_node(
            f"CIK{i:03d}",
            node_type="Company",
            ticker=f"T{i}",
            company_name=f"Company {i}",
            ml_risk_score=float(i * 15),
            risk_tier="CLEAN" if i < 3 else "HIGH",
            is_anomaly=(i >= 4),
            network_risk_score=0.0,
        )
    for i in range(n_nodes - 1):
        G.add_edge(f"CIK{i:03d}", f"CIK{(i+1):03d}", edge_type="PARTNERS_WITH")
    return G


# ──────────────────────────────────────────────
# Test 1: Graph builds with company nodes
# ──────────────────────────────────────────────

def test_graph_builds_with_company_nodes(tmp_path):
    """CorporateGraphBuilder should create ≥3 Company nodes from mock DB rows."""
    mock_db, _ = _make_mock_db()

    auditors_path = tmp_path / "sp500_auditors.json"
    auditors_path.write_text("{}")
    rels_path = tmp_path / "sp500_relationships.json"
    rels_path.write_text("[]")

    with patch("graph.graph_builder.AUDITORS_JSON", auditors_path), \
         patch("graph.graph_builder.RELATIONSHIPS_JSON", rels_path):
        builder = CorporateGraphBuilder(mock_db)
        G = builder.build()

    company_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "Company"]
    assert len(company_nodes) >= 3, f"Expected ≥3 Company nodes, got {len(company_nodes)}"

    stats = builder.get_stats()
    assert isinstance(stats, GraphStats)
    assert stats.n_companies >= 3


# ──────────────────────────────────────────────
# Test 2: Auditor sharing pairs
# ──────────────────────────────────────────────

def test_auditor_sharing_pairs(tmp_path):
    """Two companies sharing the same auditor should produce ≥1 sharing pair."""
    mock_db, _ = _make_mock_db(rows=[
        ("CIK001", "AAPL", "Apple Inc.", "Technology", 80.0, "HIGH", True, 75.0),
        ("CIK002", "MSFT", "Microsoft Corp.", "Technology", 40.0, "MEDIUM", False, 35.0),
    ])

    auditors = {
        "CIK001": {"auditor": "PricewaterhouseCoopers", "auditor_id": "PWC_001",
                   "audit_opinion": "Unqualified", "audit_fee_millions": 45.0},
        "CIK002": {"auditor": "PricewaterhouseCoopers", "auditor_id": "PWC_001",
                   "audit_opinion": "Unqualified", "audit_fee_millions": 38.0},
    }
    auditors_path = tmp_path / "sp500_auditors.json"
    auditors_path.write_text(json.dumps(auditors))
    rels_path = tmp_path / "sp500_relationships.json"
    rels_path.write_text("[]")

    with patch("graph.graph_builder.AUDITORS_JSON", auditors_path), \
         patch("graph.graph_builder.RELATIONSHIPS_JSON", rels_path):
        builder = CorporateGraphBuilder(mock_db)
        builder.build()
        pairs = builder.get_auditor_sharing_pairs()

    assert len(pairs) >= 1, "Expected ≥1 auditor-sharing pair"
    assert pairs[0]["shared_auditor_name"] == "PricewaterhouseCoopers"


# ──────────────────────────────────────────────
# Test 3: Cluster detector finds communities
# ──────────────────────────────────────────────

def test_cluster_detector_finds_communities():
    """ClusterDetector should find ≥1 community in a 5-node graph."""
    G = _make_simple_graph(n_nodes=5)
    detector = ClusterDetector(G)
    partition = detector.detect_communities()

    assert isinstance(partition, dict)
    assert len(partition) > 0, "Expected a non-empty partition"
    n_communities = len(set(partition.values()))
    assert n_communities >= 1, f"Expected ≥1 community, got {n_communities}"

    centrality = detector.compute_centrality()
    assert len(centrality) == G.number_of_nodes()
    for node, metrics in centrality.items():
        assert "pagerank" in metrics
        assert "betweenness" in metrics

    cluster_df = detector.identify_risk_clusters(partition)
    assert isinstance(cluster_df, pd.DataFrame)
    if not cluster_df.empty:
        assert "community_id" in cluster_df.columns
        assert "avg_ml_risk_score" in cluster_df.columns


# ──────────────────────────────────────────────
# Test 4: Final risk score range
# ──────────────────────────────────────────────

def test_final_risk_score_range(tmp_path):
    """All final_risk_score values should be in [0, 100]."""
    mock_db = MagicMock()
    mock_conn = MagicMock()
    mock_db.get_connection.return_value = mock_conn

    ml_df = pd.DataFrame({
        "cik": ["CIK001", "CIK002", "CIK003"],
        "ticker": ["AAPL", "MSFT", "JPM"],
        "company_name": ["Apple", "Microsoft", "JPMorgan"],
        "ml_risk_score": [85.0, 40.0, 10.0],
        "risk_tier": ["HIGH", "MEDIUM", "CLEAN"],
        "is_anomaly": [True, False, False],
    })
    benford_df = pd.DataFrame({
        "cik": ["CIK001", "CIK002"],
        "benford_risk_score": [60.0, 20.0],
    })

    call_count = [0]
    def side_effect(query):
        result_mock = MagicMock()
        if call_count[0] == 0:
            result_mock.df.return_value = ml_df
        else:
            result_mock.df.return_value = benford_df
        call_count[0] += 1
        return result_mock

    mock_conn.execute.side_effect = side_effect

    G = _make_simple_graph(n_nodes=3)
    for i, cik in enumerate(["CIK001", "CIK002", "CIK003"]):
        G.add_node(cik, node_type="Company", ml_risk_score=float(i * 30),
                   network_risk_score=float(i * 5))

    scorer = GraphRiskScorer(mock_db)
    result = scorer.compute_final_scores(G)

    assert isinstance(result, pd.DataFrame)
    if not result.empty:
        assert all(0.0 <= s <= 100.0 for s in result["final_risk_score"]), \
            f"Scores out of range: {result['final_risk_score'].tolist()}"
        assert "final_risk_tier" in result.columns


# ──────────────────────────────────────────────
# Test 5: GraphQueryEngine falls back to NetworkX
# ──────────────────────────────────────────────

def test_graph_queries_fallback():
    """Without Neo4j, GraphQueryEngine should fall back to NetworkX and return results."""
    G = _make_simple_graph(n_nodes=5)

    engine = GraphQueryEngine(neo4j_loader=None, networkx_graph=G)
    assert not engine._use_neo4j

    results = engine.query_high_risk_companies(threshold=50.0, limit=10)
    assert isinstance(results, list)

    high_risk_nodes = [
        n for n, d in G.nodes(data=True)
        if d.get("ml_risk_score", 0) >= 50.0
    ]
    assert len(results) == len(high_risk_nodes), (
        f"Expected {len(high_risk_nodes)} results, got {len(results)}"
    )

    cypher_queries = engine.get_all_cypher_queries()
    assert isinstance(cypher_queries, dict)
    assert len(cypher_queries) == 5

    auditor_results = engine.query_auditor_sharing_risk()
    assert auditor_results == []
