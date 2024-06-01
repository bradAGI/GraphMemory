import duckdb
import json
import os
import logging
from contextlib import contextmanager
from graphrag.models import Node, Edge, Neighbor
from typing import List


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphRAG:
    def __init__(self, database=None, vector_length=3):
        self.database = database
        self.vector_length = vector_length
        self.conn = duckdb.connect(database=self.database)
        self._load_vss_extension()
        self._configure_database()
        
        if database and os.path.exists(database):
            self.load_database(database)
        
        # Check if 'nodes' and 'edges' tables exist, and create them if they do not
        nodes_exist = self.conn.execute("SELECT 1 FROM information_schema.tables WHERE table_name = 'nodes';").fetchone()
        edges_exist = self.conn.execute("SELECT 1 FROM information_schema.tables WHERE table_name = 'edges';").fetchone()
        
        if not nodes_exist or not edges_exist:
            self.create_tables()
            logger.info("Tables created or verified successfully.")
    
    def load_database(self, path):
        if not os.path.exists(path):
            logger.error(f"Database file not found: {path}")
            return
        try:
            self.conn.execute(f"ATTACH DATABASE '{path}' AS vss;")
        except duckdb.Error as e:
            logger.error(f"Error loading database: {e}")

    def _configure_database(self):
        try:
            self.conn.execute("SET hnsw_enable_experimental_persistence=true;")
        except duckdb.Error as e:
            logger.error(f"Error setting configuration: {e}")

    def _load_vss_extension(self):
        try:
            self.conn.execute("INSTALL vss;")
            self.conn.execute("LOAD vss;")
        except duckdb.Error as e:
            logger.error(f"Error loading VSS extension: {e}")

    def set_vector_length(self, vector_length):
        self.vector_length = vector_length
        logger.info(f"Vector length set to: {self.vector_length}")

    def create_tables(self):
        self.conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_id;")
        self.conn.execute(f"""
        CREATE TABLE IF NOT EXISTS nodes (
            id INTEGER DEFAULT nextval('seq_id'),
            data JSON,
            vector FLOAT[{self.vector_length}],
            PRIMARY KEY (id)
        );
        """)
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS edges (
            source_id INTEGER,
            target_id INTEGER,
            weight FLOAT,
            relation TEXT,
            PRIMARY KEY (source_id, target_id)
        );
        """)
        logger.info("Tables 'nodes' and 'edges' created or already exist.")
        self.conn.commit()  # Ensure changes are committed

    def insert_node(self, node: Node) -> int:
        if not self._validate_vector(node.vector):
            logger.error("Invalid vector: Must be a list of float values.")
            return -1
        try:
            with self.transaction():
                result = self.conn.execute(
                    "INSERT INTO nodes (data, vector) VALUES (?, ?) RETURNING id;",
                    (json.dumps(node.data), node.vector)
                ).fetchone()
                if result:
                    logger.info(f"Node inserted with ID: {result[0]}")
                return result[0] if result else -1
        except duckdb.Error as e:
            logger.error(f"Error during insert node: {e}")
            return -1

    # Update insert_edge to use Edge model
    def insert_edge(self, edge: Edge):
        try:
            with self.transaction():
                # Check if source and target nodes exist
                source_exists = self.conn.execute("SELECT 1 FROM nodes WHERE id = ?", (edge.source_id,)).fetchone()
                target_exists = self.conn.execute("SELECT 1 FROM nodes WHERE id = ?", (edge.target_id,)).fetchone()
                if not source_exists or not target_exists:
                    raise ValueError("Source or target node does not exist.")
                
                self.conn.execute("INSERT INTO edges (source_id, target_id, weight, relation) VALUES (?, ?, ?, ?);", (edge.source_id, edge.target_id, edge.weight, edge.relation))
        except duckdb.Error as e:
            logger.error(f"Error during insert edge: {e}")
        except ValueError as e:
            logger.error(f"Error during insert edge: {e}")
            raise  # Re-raise the ValueError

    def bulk_insert_nodes(self, nodes: List[Node]) -> List[Node]:
        try:
            with self.transaction():
                for node in nodes:
                    result = self.conn.execute(
                        "INSERT INTO nodes (data, vector) VALUES (?, ?) RETURNING id;",
                        (json.dumps(node.data), node.vector)
                    ).fetchone()
                    if result:
                        node.id = result[0]  # Set the node ID
                return nodes
        except duckdb.Error as e:
            logger.error(f"Error during bulk insert nodes: {e}")
            return []

    def bulk_insert_edges(self, edges: List[Edge]):
        try:
            with self.transaction():
                self.conn.executemany(
                    "INSERT INTO edges (source_id, target_id, weight, relation) VALUES (?, ?, ?, ?);",
                    [(edge.source_id, edge.target_id, edge.weight, edge.relation) for edge in edges]
                )
        except duckdb.Error as e:
            logger.error(f"Error during bulk insert edges: {e}")

    def delete_node(self, node_id: int):
        try:
            with self.transaction():
                self.conn.execute("DELETE FROM nodes WHERE id = ?;", (node_id,))
                self.conn.execute("DELETE FROM edges WHERE source_id = ? OR target_id = ?;", (node_id, node_id))
        except duckdb.Error as e:
            logger.error(f"Error deleting node: {e}")

    def delete_edge(self, source_id: int, target_id: int):
        try:
            with self.transaction():
                self.conn.execute("DELETE FROM edges WHERE source_id = ? AND target_id = ?;", (source_id, target_id))
        except duckdb.Error as e:
            logger.error(f"Error deleting edge: {e}")

    def create_index(self):
        try:
            self.conn.execute("CREATE INDEX IF NOT EXISTS vss_idx ON nodes USING HNSW(vector);")
        except duckdb.Error as e:
            logger.error(f"Error creating index: {e}")

    # Update nearest_neighbors to use vector and limit directly
    def nearest_neighbors(self, vector: List[float], limit: int) -> List[Neighbor]:
        if not self._validate_vector(vector):
            logger.error("Invalid vector: Must be a list of float values.")
            return []

        query = f"""
        SELECT id, data, vector, array_distance(vector, CAST(? AS FLOAT[{self.vector_length}])) AS distance
        FROM nodes
        ORDER BY distance LIMIT ?;
        """
        try:
            results = self.conn.execute(query, (vector, limit)).fetchall()
            return [
                Neighbor(
                    node=Node(id=row[0], data=json.loads(row[1]), vector=row[2]),
                    distance=row[3]
                ) for row in results
            ]
        except duckdb.Error as e:
            logger.error(f"Error fetching nearest neighbors: {e}")
            return []

    def connected_nodes(self, node_id: int) -> List[Node]:
        query = """
        SELECT n.id, n.data, n.vector
        FROM nodes n
        WHERE n.id IN (
                SELECT target_id FROM edges WHERE source_id = CAST(? AS INTEGER)
                UNION
                SELECT source_id FROM edges WHERE target_id = CAST(? AS INTEGER)
            );
        """
        try:
            logger.info(f"Executing query to fetch connected nodes for node_id: {node_id}")
            results = self.conn.execute(query, (node_id, node_id)).fetchall()
            if results:
                connected_nodes = [Node(id=row[0], data=json.loads(row[1]), vector=row[2]) for row in results]
                logger.info(f"Found {len(connected_nodes)} connected nodes.")
            else:
                connected_nodes = []
                logger.info("No connected nodes found.")
            return connected_nodes
        except duckdb.Error as e:
            logger.error(f"Error fetching connected nodes: {e}")
            return []

    def nodes_to_json(self):
        try:
            nodes = self.conn.execute("SELECT id, data, vector FROM nodes;").fetchall()
            return [{"id": row[0], "data": json.loads(row[1]), "vector": row[2]} for row in nodes]
        except duckdb.Error as e:
            logger.error(f"Error fetching nodes: {e}")
            return []

    def edges_to_json(self):
        try:
            edges = self.conn.execute("SELECT source_id, target_id, weight, relation FROM edges;").fetchall()
            return [{"source_id": row[0], "target_id": row[1], "weight": row[2], "relation": row[3]} for row in edges]
        except duckdb.Error as e:
            logger.error(f"Error fetching edges: {e}")
            return []
    
    # Get node by id
    def get_node(self, node_id: int):
        try:
            node = self.conn.execute("SELECT id, data, vector FROM nodes WHERE id = ?;", (node_id,)).fetchone()
            return {"id": node[0], "data": json.loads(node[1]), "vector": node[2]}
        except duckdb.Error as e:
            logger.error(f"Error fetching node: {e}")
            return {}

    def print_json(self):
        nodes_json = self.nodes_to_json()
        edges_json = self.edges_to_json()
        print("Nodes JSON:", json.dumps(nodes_json, indent=2))
        print("Edges JSON:", json.dumps(edges_json, indent=2))

    def _validate_vector(self, vector):
        return isinstance(vector, list) and len(vector) == self.vector_length and all(isinstance(x, float) for x in vector)

    @contextmanager
    def transaction(self):
        try:
            self.conn.execute("BEGIN TRANSACTION;")
            yield
            self.conn.execute("COMMIT;")
        except Exception as e:
            self.conn.execute("ROLLBACK;")
            raise e

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.conn.close()
        logger.info("Database connection closed.")

