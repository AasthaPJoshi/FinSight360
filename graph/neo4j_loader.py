"""Load the corporate knowledge graph into Neo4j (optional — graceful fallback)."""
import structlog
import networkx as nx

log = structlog.get_logger(__name__)


class Neo4jLoader:
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "finsight360"):
        self.uri = uri
        self.user = user
        self.password = password
        self._driver = None
        self.connected = False

    def connect(self) -> bool:
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self._driver.verify_connectivity()
            self.connected = True
            log.info("neo4j_connected", uri=self.uri)
            return True
        except Exception as e:
            log.warning("neo4j_unavailable", error=str(e), msg="Continuing without Neo4j")
            self.connected = False
            return False

    def setup_schema(self) -> None:
        if not self.connected or self._driver is None:
            return
        with self._driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Company) REQUIRE c.cik IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Auditor) REQUIRE a.auditor_id IS UNIQUE")
            session.run("CREATE INDEX IF NOT EXISTS FOR (c:Company) ON (c.ticker)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (c:Company) ON (c.risk_tier)")

    def load_graph(self, G: nx.MultiDiGraph) -> dict:
        if not self.connected or self._driver is None:
            return {"status": "skipped", "reason": "not_connected"}

        nodes_created = 0
        edges_created = 0

        with self._driver.session() as session:
            for node_id, data in G.nodes(data=True):
                node_type = data.get("node_type", "Company")
                props = {k: v for k, v in data.items() if v is not None}
                props["id"] = node_id

                if node_type == "Auditor":
                    session.run(
                        "MERGE (a:Auditor {auditor_id: $auditor_id}) SET a += $props",
                        auditor_id=node_id,
                        props=props,
                    )
                else:
                    session.run(
                        "MERGE (c:Company {cik: $cik}) SET c += $props",
                        cik=node_id,
                        props=props,
                    )
                nodes_created += 1

            for u, v, data in G.edges(data=True):
                edge_type = data.get("edge_type", "RELATED_TO")
                props = {k: v for k, v in data.items() if k != "edge_type" and v is not None}

                u_type = G.nodes[u].get("node_type", "Company")
                v_type = G.nodes[v].get("node_type", "Company")

                u_label = "Auditor" if u_type == "Auditor" else "Company"
                v_label = "Auditor" if v_type == "Auditor" else "Company"
                u_key = "auditor_id" if u_label == "Auditor" else "cik"
                v_key = "auditor_id" if v_label == "Auditor" else "cik"

                cypher = (
                    f"MATCH (a:{u_label} {{{u_key}: $u_id}}), "
                    f"(b:{v_label} {{{v_key}: $v_id}}) "
                    f"MERGE (a)-[r:{edge_type}]->(b) SET r += $props"
                )
                session.run(cypher, u_id=u, v_id=v, props=props)
                edges_created += 1

        log.info("neo4j_load_complete", nodes=nodes_created, edges=edges_created)
        return {"status": "success", "nodes_created": nodes_created, "edges_created": edges_created}

    def close(self) -> None:
        if self._driver is not None:
            try:
                self._driver.close()
            except Exception:
                pass
        self.connected = False
