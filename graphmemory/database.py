import csv
import duckdb
import io
import functools
import json
import logging
import os
import re
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from typing import Any, Dict, List, Union, List

from graphmemory.models import Edge, NearestNode, Node, SearchResult

logger = logging.getLogger(__name__)

# Transient DuckDB errors that warrant a retry
TRANSIENT_ERRORS = (
    duckdb.IOException,
    duckdb.ConnectionException,
)


def with_retry(max_retries=3, base_delay=0.1):
    """Decorator that retries a method on transient DuckDB errors with exponential backoff.

    Uses instance attributes ``max_retries`` and ``retry_base_delay`` when available,
    falling back to the decorator arguments.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            retries = getattr(self, 'max_retries', max_retries)
            delay = getattr(self, 'retry_base_delay', base_delay)
            last_error = None
            for attempt in range(retries + 1):
                try:
                    return func(self, *args, **kwargs)
                except TRANSIENT_ERRORS as e:
                    last_error = e
                    if attempt < retries:
                        sleep_time = delay * (2 ** attempt)
                        logger.warning(
                            f"Transient error in {func.__name__} "
                            f"(attempt {attempt + 1}/{retries + 1}): {e}. "
                            f"Retrying in {sleep_time:.2f}s..."
                        )
                        time.sleep(sleep_time)
                        self._reconnect()
                    else:
                        logger.error(
                            f"All {retries + 1} attempts failed for "
                            f"{func.__name__}: {e}"
                        )
                        raise
            raise last_error
        return wrapper
    return decorator


class GraphMemory:
    """Graph database backed by DuckDB with connection pooling and reconnection logic.

    Connection lifecycle:
        1. ``__init__``: Opens a DuckDB connection, loads VSS extension, creates tables.
        2. ``cursor()``: Returns a new DuckDB cursor for individual operations, enabling
           concurrent read access from multiple threads.
        3. ``_reconnect()``: Closes the current connection and re-establishes it.
           Called automatically by ``@with_retry`` on transient errors.
        4. ``close()`` / ``__exit__``: Closes the connection.

    Reconnection: Methods decorated with ``@with_retry`` automatically retry on
    transient DuckDB errors (``IOException``, ``ConnectionException``) with
    exponential backoff, reconnecting between attempts.

    Thread safety: A ``threading.RLock`` protects connection-level operations
    (reconnect, close, transactions). Use ``cursor()`` for concurrent read operations.
    """

    DISTANCE_METRICS = {
        'l2': {'function': 'array_distance', 'hnsw_metric': 'l2sq'},
        'cosine': {'function': 'array_cosine_distance', 'hnsw_metric': 'cosine'},
        'inner_product': {'function': 'array_negative_inner_product', 'hnsw_metric': 'ip'},
    }

    def __init__(self, database=None, vector_length=3, distance_metric='l2', max_retries=3, retry_base_delay=0.1):
        if distance_metric not in self.DISTANCE_METRICS:
            raise ValueError(
                f"Invalid distance_metric '{distance_metric}'. "
                f"Supported: {', '.join(self.DISTANCE_METRICS.keys())}"
            )
        self.database = database
        self.vector_length = vector_length
        self.distance_metric = distance_metric
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self._lock = threading.RLock()
        self._fts_initialized = False
        self._fts_dirty = True
        self._closed = False
        self.conn = duckdb.connect(database=self.database)
        self._load_vss_extension()
        self._load_fts_extension()
        self._configure_database()

        # Check if 'nodes' and 'edges' tables exist, and create them if they do not
        nodes_exist = self.conn.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = 'nodes';").fetchone()
        edges_exist = self.conn.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = 'edges';").fetchone()

        if not nodes_exist or not edges_exist:
            self._create_tables()
            logger.info("Tables created or verified successfully.")

    def cursor(self):
        """Return a new DuckDB cursor for individual operations.

        Each cursor operates independently, allowing concurrent read access
        from multiple threads. The caller should use the cursor within the
        scope of a single operation.
        """
        with self._lock:
            if self._closed:
                raise duckdb.ConnectionException("Connection is closed")
            return self.conn.cursor()

    def _reconnect(self):
        """Close and re-establish the DuckDB connection.

        Called automatically by ``@with_retry`` when transient errors occur.
        Thread-safe via the internal lock.
        """
        with self._lock:
            if self._closed:
                raise duckdb.ConnectionException(
                    "Cannot reconnect a closed GraphMemory instance"
                )
            logger.info("Reconnecting to DuckDB...")
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = duckdb.connect(database=self.database)
            self._load_vss_extension()
            self._load_fts_extension()
            self._configure_database()
            self._fts_initialized = False
            self._fts_dirty = True
            logger.info("Reconnection successful.")

    def close(self):
        """Close the database connection. Thread-safe."""
        with self._lock:
            if not self._closed:
                self._closed = True
                self.conn.close()
                logger.info("Database connection closed.")

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

    def _load_fts_extension(self):
        try:
            self.conn.execute("INSTALL fts;")
            self.conn.execute("LOAD fts;")
        except duckdb.Error as e:
            logger.error(f"Error loading FTS extension: {e}")

    def set_vector_length(self, vector_length):
        self.vector_length = vector_length
        logger.info(f"Vector length set to: {self.vector_length}")

    def _create_tables(self):
        self.conn.execute(f"""
        CREATE TABLE IF NOT EXISTS nodes (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            type TEXT,
            properties JSON,
            vector FLOAT[{self.vector_length}]
        );
        """)
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS edges (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            source_id UUID,
            target_id UUID,
            relation TEXT,
            weight FLOAT,
            FOREIGN KEY (source_id) REFERENCES nodes(id),
            FOREIGN KEY (target_id) REFERENCES nodes(id)
        );
        """)
        logger.info("Tables 'nodes' and 'edges' created or already exist.")
        self.conn.commit()

    @with_retry()
    def insert_node(self, node: Node) -> uuid.UUID:
        if node.vector and not self._validate_vector(node.vector):
            logger.error("Invalid vector: Must be a list of float values.")
            return None
        try:
            with self.transaction():
                cur = self.cursor()
                result = cur.execute(
                    "INSERT INTO nodes (id, type, properties, vector) VALUES (?, ?, ?, ?) RETURNING id;",
                    (str(node.id), node.type, json.dumps(node.properties), node.vector if node.vector else [0.0] * self.vector_length)
                ).fetchone()
                if result:
                    self._fts_dirty = True
                    logger.info(f"Node inserted with ID: {result[0]}")
                return result[0] if result else None
        except duckdb.Error as e:
            logger.error(f"Error during insert node: {e}")
            return None

    @with_retry()
    def insert_edge(self, edge: Edge):
        try:
            with self.transaction():
                cur = self.cursor()
                source_exists = cur.execute(
                    "SELECT 1 FROM nodes WHERE id = ?", (str(edge.source_id),)).fetchone()
                target_exists = cur.execute(
                    "SELECT 1 FROM nodes WHERE id = ?", (str(edge.target_id),)).fetchone()
                if not source_exists or not target_exists:
                    raise ValueError("Source or target node does not exist.")

                cur.execute("INSERT INTO edges (id, source_id, target_id, relation, weight) VALUES (?, ?, ?, ?, ?);", (
                    str(edge.id), str(edge.source_id), str(edge.target_id), edge.relation, edge.weight))
        except duckdb.Error as e:
            logger.error(f"Error during insert edge: {e}")
        except ValueError as e:
            logger.error(f"Error during insert edge: {e}")
            raise

    @with_retry()
    def bulk_insert_nodes(self, nodes: list[Node]) -> list[Node]:
        inserted = []
        try:
            with self.transaction():
                cur = self.cursor()
                for node in nodes:
                    if node.vector and not self._validate_vector(node.vector):
                        logger.error(f"Invalid vector for node {node.id}: Must be a list of float values with length {self.vector_length}.")
                        continue
                    result = cur.execute(
                        "INSERT INTO nodes (id, type, properties, vector) VALUES (?, ?, ?, ?) RETURNING id;",
                        (str(node.id), node.type, json.dumps(node.properties), node.vector if node.vector else None)
                    ).fetchone()
                    if result:
                        node.id = result[0]
                        inserted.append(node)
                if inserted:
                    self._fts_dirty = True
                return inserted
        except duckdb.Error as e:
            logger.error(f"Error during bulk insert nodes: {e}")
            return []

    @with_retry()
    def bulk_insert_edges(self, edges: list[Edge]):
        try:
            with self.transaction():
                cur = self.cursor()
                cur.executemany(
                    "INSERT INTO edges (id, source_id, target_id, relation, weight) VALUES (?, ?, ?, ?, ?);",
                    [(str(edge.id), str(edge.source_id), str(edge.target_id), edge.relation, edge.weight)
                     for edge in edges]
                )
        except duckdb.Error as e:
            logger.error(f"Error during bulk insert edges: {e}")

    @with_retry()
    def delete_node(self, node_id: uuid.UUID):
        try:
            with self.transaction():
                cur = self.cursor()
                cur.execute(
                    "DELETE FROM edges WHERE source_id = ? OR target_id = ?;", (str(node_id), str(node_id)))
                cur.execute(
                    "DELETE FROM nodes WHERE id = ?;", (str(node_id),))
                self._fts_dirty = True
        except duckdb.Error as e:
            logger.error(f"Error deleting node: {e}")

    @with_retry()
    def bulk_delete_nodes(self, node_ids: List[uuid.UUID]):
        if not node_ids:
            return
        try:
            id_strs = [str(nid) for nid in node_ids]
            placeholders = ', '.join(['?'] * len(id_strs))
            with self.transaction():
                cur = self.cursor()
                cur.execute(
                    f"DELETE FROM edges WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders});",
                    id_strs + id_strs)
                cur.execute(
                    f"DELETE FROM nodes WHERE id IN ({placeholders});", id_strs)
                self._fts_dirty = True
        except duckdb.Error as e:
            logger.error(f"Error during bulk delete nodes: {e}")

    @with_retry()
    def bulk_delete_edges(self, edge_ids: List[uuid.UUID]):
        if not edge_ids:
            return
        try:
            with self.transaction():
                cur = self.cursor()
                id_strs = [str(eid) for eid in edge_ids]
                placeholders = ', '.join(['?'] * len(id_strs))
                cur.execute(
                    f"DELETE FROM edges WHERE id IN ({placeholders});", id_strs)
        except duckdb.Error as e:
            logger.error(f"Error during bulk delete edges: {e}")

    @with_retry()
    def delete_edge(self, source_id: uuid.UUID, target_id: uuid.UUID):
        try:
            with self.transaction():
                cur = self.cursor()
                cur.execute(
                    "DELETE FROM edges WHERE source_id = ? AND target_id = ?;", (str(source_id), str(target_id)))
        except duckdb.Error as e:
            logger.error(f"Error deleting edge: {e}")

    @with_retry()
    def update_node(self, node_id: uuid.UUID, **kwargs) -> bool:
        allowed_fields = {"type", "properties", "vector"}
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        if not updates:
            logger.error("No valid fields to update.")
            return False
        if "vector" in updates and not self._validate_vector(updates["vector"]):
            logger.error("Invalid vector: Must be a list of float values with correct length.")
            return False
        try:
            with self.transaction():
                cur = self.cursor()
                set_clauses = []
                params = []
                for field, value in updates.items():
                    set_clauses.append(f"{field} = ?")
                    params.append(json.dumps(value) if field == "properties" else value)
                params.append(str(node_id))
                query = f"UPDATE nodes SET {', '.join(set_clauses)} WHERE id = ?;"
                cur.execute(query, params)
                updated = cur.execute(
                    "SELECT 1 FROM nodes WHERE id = ?;", (str(node_id),)).fetchone()
                if not updated:
                    logger.error(f"Node {node_id} not found.")
                    return False
                self._fts_dirty = True
                logger.info(f"Node {node_id} updated.")
                return True
        except duckdb.Error as e:
            logger.error(f"Error updating node: {e}")
            return False

    @with_retry()
    def update_edge(self, edge_id: uuid.UUID, **kwargs) -> bool:
        allowed_fields = {"relation", "weight"}
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        if not updates:
            logger.error("No valid fields to update.")
            return False
        try:
            with self.transaction():
                cur = self.cursor()
                set_clauses = []
                params = []
                for field, value in updates.items():
                    set_clauses.append(f"{field} = ?")
                    params.append(value)
                params.append(str(edge_id))
                query = f"UPDATE edges SET {', '.join(set_clauses)} WHERE id = ?;"
                cur.execute(query, params)
                updated = cur.execute(
                    "SELECT 1 FROM edges WHERE id = ?;", (str(edge_id),)).fetchone()
                if not updated:
                    logger.error(f"Edge {edge_id} not found.")
                    return False
                logger.info(f"Edge {edge_id} updated.")
                return True
        except duckdb.Error as e:
            logger.error(f"Error updating edge: {e}")
            return False

    @with_retry()
    def create_index(self):
        with self._lock:
            try:
                hnsw_metric = self.DISTANCE_METRICS[self.distance_metric]['hnsw_metric']
                self.conn.execute(
                    f"CREATE INDEX IF NOT EXISTS vss_idx ON nodes USING HNSW(vector) WITH (metric = '{hnsw_metric}');")
            except duckdb.Error as e:
                logger.error(f"Error creating index: {e}")

    @with_retry()
    def nearest_nodes(self, vector: list[float], limit: int) -> list[NearestNode]:
        if not self._validate_vector(vector):
            logger.error("Invalid vector: Must be a list of float values.")
            return []

        dist_func = self.DISTANCE_METRICS[self.distance_metric]['function']
        dist_func = self.DISTANCE_METRICS[self.distance_metric]["function"]
        query = f"""
        SELECT id, type, properties, vector, {dist_func}(vector, CAST(? AS FLOAT[{self.vector_length}])) AS distance
        FROM nodes
        WHERE vector IS NOT NULL
        ORDER BY distance LIMIT ?;
        """
        with self._lock:
            try:
                cur = self.cursor()
                results = cur.execute(query, (vector, limit)).fetchall()
                return [
                    NearestNode(
                        node=Node(id=row[0], type=row[1], properties=json.loads(row[2]), vector=row[3]),
                        distance=row[4]
                    ) for row in results
                ]
            except duckdb.Error as e:
                logger.error(f"Error fetching nearest neighbors: {e}")
                return []

    @with_retry()
    def connected_nodes(self, node_id: uuid.UUID) -> list[Node]:
        query = """
        SELECT n.id, n.type, n.properties, n.vector
        FROM nodes n
        WHERE n.id IN (
                SELECT target_id FROM edges WHERE source_id = ?
                UNION
                SELECT source_id FROM edges WHERE target_id = ?
            );
        """
        with self._lock:
            try:
                logger.info(
                    f"Executing query to fetch connected nodes for node_id: {node_id}")
                cur = self.cursor()
                results = cur.execute(query, (str(node_id), str(node_id))).fetchall()
                if results:
                    connected_nodes = [Node(id=uuid.UUID(str(row[0])), type=row[1], properties=json.loads(row[2]), vector=row[3]) for row in results]
                    logger.info(f"Found {len(connected_nodes)} connected nodes.")
                else:
                    connected_nodes = []
                    logger.info("No connected nodes found.")
                return connected_nodes
            except duckdb.Error as e:
                logger.error(f"Error fetching connected nodes: {e}")
                return []

    @with_retry()
    def nodes_to_json(self, limit: int | None = None, offset: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            try:
                query = "SELECT id, type, properties, vector FROM nodes"
                params = []
                if limit is not None:
                    query += " LIMIT ?"
                    params.append(limit)
                if offset is not None:
                    query += " OFFSET ?"
                    params.append(offset)
                cur = self.cursor()
                nodes = cur.execute(query + ";", params).fetchall()
                return [{"id": str(row[0]), "type": row[1], "properties": json.loads(row[2]), "vector": row[3]} for row in nodes]
            except duckdb.Error as e:
                logger.error(f"Error fetching nodes: {e}")
                return []

    @with_retry()
    def get_edge(self, edge_id: uuid.UUID) -> Edge | None:
        with self._lock:
            try:
                cur = self.cursor()
                row = cur.execute(
                    "SELECT id, source_id, target_id, relation, weight FROM edges WHERE id = ?;",
                    (str(edge_id),)
                ).fetchone()
                if row:
                    return Edge(id=row[0], source_id=row[1], target_id=row[2], relation=row[3], weight=row[4])
                return None
            except duckdb.Error as e:
                logger.error(f"Error fetching edge: {e}")
                return None

    @with_retry()
    def get_edges_by_relation(self, relation: str) -> list[Edge]:
        with self._lock:
            try:
                cur = self.cursor()
                rows = cur.execute(
                    "SELECT id, source_id, target_id, relation, weight FROM edges WHERE relation = ?;",
                    (relation,)
                ).fetchall()
                return [Edge(id=row[0], source_id=row[1], target_id=row[2], relation=row[3], weight=row[4]) for row in rows]
            except duckdb.Error as e:
                logger.error(f"Error fetching edges by relation: {e}")
                return []

    @with_retry()
    def edges_by_attribute(self, attribute: str, value) -> list[Edge]:
        allowed_attributes = {"relation", "weight", "source_id", "target_id"}
        if attribute not in allowed_attributes:
            logger.error(f"Invalid edge attribute: {attribute}")
            return []
        with self._lock:
            try:
                cur = self.cursor()
                query = f"SELECT id, source_id, target_id, relation, weight FROM edges WHERE {attribute} = ?;"
                rows = cur.execute(query, (str(value) if isinstance(value, uuid.UUID) else value,)).fetchall()
                return [Edge(id=row[0], source_id=row[1], target_id=row[2], relation=row[3], weight=row[4]) for row in rows]
            except duckdb.Error as e:
                logger.error(f"Error fetching edges by attribute: {e}")
                return []

    @with_retry()
    def edges_to_json(self, limit: int | None = None, offset: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            try:
                query = "SELECT id, source_id, target_id, relation, weight FROM edges"
                params = []
                if limit is not None:
                    query += " LIMIT ?"
                    params.append(limit)
                if offset is not None:
                    query += " OFFSET ?"
                    params.append(offset)
                cur = self.cursor()
                edges = cur.execute(query + ";", params).fetchall()
                return [{"id": str(row[0]), "source_id": str(row[1]), "target_id": str(row[2]), "relation": row[3], "weight": row[4]} for row in edges]
            except duckdb.Error as e:
                logger.error(f"Error fetching edges: {e}")
                return []

    @with_retry()
    def get_node(self, node_id: uuid.UUID) -> Node:
        with self._lock:
            try:
                cur = self.cursor()
                node = cur.execute(
                    "SELECT id, type, properties, vector FROM nodes WHERE id = ?;", (str(node_id),)).fetchone()
                if node:
                    return Node(id=node[0], type=node[1], properties=json.loads(node[2]), vector=node[3])
                else:
                    return None
            except duckdb.Error as e:
                logger.error(f"Error fetching node: {e}")
                return None

    _VALID_ATTRIBUTE_RE = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')

    @with_retry()
    def nodes_by_attribute(self, attribute, value, limit: int | None = None, offset: int | None = None) -> list[Node]:
        with self._lock:
            try:
                if not self._VALID_ATTRIBUTE_RE.match(attribute):
                    raise ValueError(f"Invalid attribute name: {attribute!r}")
                query = f"SELECT id, type, properties, vector FROM nodes WHERE json_extract(properties, '$.{attribute}') = ?"
                params: list = [json.dumps(value)]
                if limit is not None:
                    query += " LIMIT ?"
                    params.append(limit)
                if offset is not None:
                    query += " OFFSET ?"
                    params.append(offset)
                cur = self.cursor()
                nodes = cur.execute(query + ";", params).fetchall()
                if nodes:
                    return [Node(id=row[0], type=row[1], properties=json.loads(row[2]), vector=row[3]) for row in nodes]
                else:
                    return []
            except duckdb.Error as e:
                logger.error(f"Error fetching nodes: {e}")
                return []

    @with_retry()
    def get_nodes_vector(self, node_id: uuid.UUID) -> list[float]:
        with self._lock:
            try:
                cur = self.cursor()
                vector = cur.execute(
                    "SELECT vector FROM nodes WHERE id = ?;", (str(node_id),)).fetchone()
                return vector[0] if vector else []
            except duckdb.Error as e:
                logger.error(f"Error fetching vector: {e}")
                return []

    def print_json(self):
        nodes_json = self.nodes_to_json()
        edges_json = self.edges_to_json()
        print("Nodes JSON:", json.dumps(nodes_json, indent=2))
        print("Edges JSON:", json.dumps(edges_json, indent=2))

    @with_retry()
    def export_graph(self, format: str = 'json'):
        """Export the full graph. Formats: json, json_string, csv, graphml."""
        if format == 'json':
            return {"nodes": self.nodes_to_json(), "edges": self.edges_to_json()}
        elif format == 'json_string':
            return json.dumps({"nodes": self.nodes_to_json(), "edges": self.edges_to_json()}, indent=2)
        elif format == 'csv':
            return self._export_csv()
        elif format == 'graphml':
            return self._export_graphml()
        else:
            raise ValueError(f"Unsupported export format: '{format}'. Use 'json', 'json_string', 'csv', or 'graphml'.")

    def import_graph(self, data, format: str = 'json') -> None:
        """Import a full graph. Formats: json, json_string, csv, graphml."""
        if format == 'json':
            self._import_json(data)
        elif format == 'json_string':
            self._import_json(json.loads(data))
        elif format == 'csv':
            self._import_csv(data)
        elif format == 'graphml':
            self._import_graphml(data)
        else:
            raise ValueError(f"Unsupported import format: '{format}'. Use 'json', 'json_string', 'csv', or 'graphml'.")

    def _export_csv(self):
        nodes = self.nodes_to_json()
        edges = self.edges_to_json()
        nodes_buf = io.StringIO()
        w = csv.writer(nodes_buf)
        w.writerow(['id', 'type', 'properties', 'vector'])
        for n in nodes:
            w.writerow([n['id'], n['type'], json.dumps(n['properties']), json.dumps(n['vector'])])
        edges_buf = io.StringIO()
        w = csv.writer(edges_buf)
        w.writerow(['id', 'source_id', 'target_id', 'relation', 'weight'])
        for e in edges:
            w.writerow([e['id'], e['source_id'], e['target_id'], e['relation'], e['weight']])
        return {'nodes': nodes_buf.getvalue(), 'edges': edges_buf.getvalue()}

    def _export_graphml(self):
        graphml = ET.Element('graphml', xmlns='http://graphml.graphstruct.org/xmlns')
        ET.SubElement(graphml, 'key', id='type', attrib={'for': 'node', 'attr.name': 'type', 'attr.type': 'string'})
        ET.SubElement(graphml, 'key', id='properties', attrib={'for': 'node', 'attr.name': 'properties', 'attr.type': 'string'})
        ET.SubElement(graphml, 'key', id='vector', attrib={'for': 'node', 'attr.name': 'vector', 'attr.type': 'string'})
        ET.SubElement(graphml, 'key', id='relation', attrib={'for': 'edge', 'attr.name': 'relation', 'attr.type': 'string'})
        ET.SubElement(graphml, 'key', id='weight', attrib={'for': 'edge', 'attr.name': 'weight', 'attr.type': 'double'})
        ET.SubElement(graphml, 'key', id='edge_id', attrib={'for': 'edge', 'attr.name': 'edge_id', 'attr.type': 'string'})
        graph = ET.SubElement(graphml, 'graph', id='G', edgedefault='directed')
        for n in self.nodes_to_json():
            ne = ET.SubElement(graph, 'node', id=n['id'])
            ET.SubElement(ne, 'data', key='type').text = n['type'] or ''
            ET.SubElement(ne, 'data', key='properties').text = json.dumps(n['properties'])
            ET.SubElement(ne, 'data', key='vector').text = json.dumps(n['vector'])
        for e in self.edges_to_json():
            ee = ET.SubElement(graph, 'edge', source=e['source_id'], target=e['target_id'])
            ET.SubElement(ee, 'data', key='edge_id').text = e['id']
            ET.SubElement(ee, 'data', key='relation').text = e['relation'] or ''
            ET.SubElement(ee, 'data', key='weight').text = str(e['weight']) if e['weight'] is not None else ''
        return ET.tostring(graphml, encoding='unicode', xml_declaration=True)

    def _import_json(self, data):
        from graphmemory.models import Node, Edge
        nodes = [Node(id=uuid.UUID(n['id']), type=n.get('type'), properties=n.get('properties', {}), vector=n.get('vector', [])) for n in data.get('nodes', [])]
        if nodes:
            self.bulk_insert_nodes(nodes)
        edges = [Edge(id=uuid.UUID(e['id']), source_id=uuid.UUID(e['source_id']), target_id=uuid.UUID(e['target_id']), relation=e.get('relation'), weight=e.get('weight')) for e in data.get('edges', [])]
        if edges:
            self.bulk_insert_edges(edges)

    def _import_csv(self, data):
        from graphmemory.models import Node, Edge
        nodes_reader = csv.DictReader(io.StringIO(data['nodes']))
        nodes = []
        for row in nodes_reader:
            vector = json.loads(row['vector']) if row['vector'] else []
            nodes.append(Node(id=uuid.UUID(row['id']), type=row['type'] if row['type'] else None, properties=json.loads(row['properties']), vector=vector))
        if nodes:
            self.bulk_insert_nodes(nodes)
        edges_reader = csv.DictReader(io.StringIO(data['edges']))
        edges = []
        for row in edges_reader:
            weight = float(row['weight']) if row['weight'] else None
            edges.append(Edge(id=uuid.UUID(row['id']), source_id=uuid.UUID(row['source_id']), target_id=uuid.UUID(row['target_id']), relation=row['relation'] if row['relation'] else None, weight=weight))
        if edges:
            self.bulk_insert_edges(edges)

    def _import_graphml(self, data):
        from graphmemory.models import Node, Edge
        root = ET.fromstring(data)
        ns = {'g': 'http://graphml.graphstruct.org/xmlns'}
        graph = root.find('g:graph', ns)
        nodes = []
        for ne in graph.findall('g:node', ns):
            nd = {}
            for d in ne.findall('g:data', ns):
                nd[d.get('key')] = d.text or ''
            nodes.append(Node(id=uuid.UUID(ne.get('id')), type=nd.get('type') or None, properties=json.loads(nd.get('properties', '{}')), vector=json.loads(nd.get('vector', '[]'))))
        if nodes:
            self.bulk_insert_nodes(nodes)
        edges = []
        for ee in graph.findall('g:edge', ns):
            ed = {}
            for d in ee.findall('g:data', ns):
                ed[d.get('key')] = d.text or ''
            eid = ed.get('edge_id', str(uuid.uuid4()))
            rel = ed.get('relation') or None
            wstr = ed.get('weight', '')
            wt = float(wstr) if wstr else None
            edges.append(Edge(id=uuid.UUID(eid), source_id=uuid.UUID(ee.get('source')), target_id=uuid.UUID(ee.get('target')), relation=rel, weight=wt))
        if edges:
            self.bulk_insert_edges(edges)

    def cypher(self, cypher_query):
        sql_query = self._cypher_to_sql(cypher_query)
        with self._lock:
            try:
                cur = self.cursor()
                results = cur.execute(sql_query).fetchall()
                logger.debug(f"Query results: {results}")
                return results
            except duckdb.Error as e:
                logger.error(f"Error executing SQL query: {e}")
                return []

    def _cypher_to_sql(self, cypher_query):
        node_pattern = re.compile(r"\((\w+)(?::(\w+))?(?:\s*{([^}]+)})?\)")
        rel_pattern = re.compile(r"\[(\w+)?(?::(\w+))?(?:\s*{([^}]+)})?\]")

        def parse_properties(prop_string):
            properties = {}
            if prop_string:
                props = prop_string.split(',')
                for prop in props:
                    key, value = prop.split(':')
                    value = value.strip().strip('"\'')
                    if value.isdigit():
                        value = int(value)
                    elif re.match(r"^\d+?\.\d+?$", value):
                        value = float(value)
                    properties[key.strip()] = value
            return properties

        match_clause = re.search(r'MATCH\s+(.*)\s+RETURN', cypher_query, re.IGNORECASE)
        if not match_clause:
            raise ValueError("Invalid Cypher query: missing MATCH or RETURN clause")
        match_content = match_clause.group(1).strip()

        return_clause = re.search(r'RETURN\s+(.*)', cypher_query, re.IGNORECASE)
        if not return_clause:
            raise ValueError("Invalid Cypher query: missing RETURN clause")
        return_content = return_clause.group(1).strip().split(',')

        elements = re.split(r'(\[.*?\])', match_content)

        nodes = []
        relationships = []
        for elem in elements:
            if '(' in elem:
                match = node_pattern.search(elem)
                if match:
                    alias, label, prop_string = match.groups()
                    nodes.append({
                        "alias": alias,
                        "label": label,
                        "properties": parse_properties(prop_string)
                    })
            elif '[' in elem:
                match = rel_pattern.search(elem)
                if match:
                    alias, label, prop_string = match.groups()
                    relationships.append({
                        "alias": alias or f"r{len(relationships)+1}",
                        "label": label,
                        "properties": parse_properties(prop_string)
                    })

        sql_query = "SELECT "
        sql_parts = []

        for item in return_content:
            item = item.strip()
            if '.' in item:
                alias, field = item.split('.')
                if field == "embedding":
                    sql_parts.append(f"{alias}.{field}")
            else:
                sql_parts.append("*")

        if not sql_parts:
            sql_parts.append("*")

        from_clause = []
        where_conditions = []

        for i, node in enumerate(nodes):
            alias, label, properties = node.values()
            if i == 0:
                from_clause.append(f"nodes AS {alias}")
            else:
                prev_node = nodes[i-1]['alias']
                rel = relationships[i-1]
                rel_alias, rel_label, rel_properties = rel.values()
                from_clause.append(f"JOIN nodes AS {alias} ON {prev_node}.id = {rel_alias}.start_node_id AND {alias}.id = {rel_alias}.end_node_id")

            if label:
                where_conditions.append(f"{alias}.type = '{label}'")
            for prop, val in properties.items():
                if prop == "embedding":
                    sql_parts.append(f"{alias}.embedding")
                else:
                    if isinstance(val, (int, float)):
                        where_conditions.append(f"json_extract({alias}.properties, '$.{prop}') = json('{val}')")
                    else:
                        where_conditions.append(f"json_extract({alias}.properties, '$.{prop}') = json('{json.dumps(val)}')")

        for rel in relationships:
            rel_alias, rel_label, rel_properties = rel.values()
            if rel_label:
                where_conditions.append(f"{rel_alias}.type = '{rel_label}'")
            for prop, val in rel_properties.items():
                if isinstance(val, (int, float)):
                    where_conditions.append(f"json_extract({rel_alias}.properties, '$.{prop}') = json('{val}')")
                else:
                    where_conditions.append(f"json_extract({rel_alias}.properties, '$.{prop}') = json('{json.dumps(val)}')")

        sql_query += ", ".join(sql_parts)
        sql_query += " FROM " + " ".join(from_clause)

        if where_conditions:
            sql_query += " WHERE " + " AND ".join(where_conditions)

        return sql_query + ";"

    def _rebuild_fts_index(self):
        """Rebuild the FTS index on the nodes properties column only if data has changed."""
        if self._fts_initialized and not self._fts_dirty:
            return
        try:
            self.conn.execute(
                "PRAGMA create_fts_index('nodes', 'id', 'properties', stemmer='porter', overwrite=1)"
            )
            self._fts_initialized = True
            self._fts_dirty = False
        except duckdb.Error as e:
            logger.error(f"Error creating FTS index: {e}")

    def reindex(self):
        """Force a rebuild of the FTS index, regardless of dirty state."""
        with self._lock:
            self._fts_dirty = True
            self._rebuild_fts_index()

    def search_nodes(self, query_text: str, limit: int = 10) -> list[SearchResult]:
        """Full-text search on node properties using DuckDB FTS extension (BM25 scoring)."""
        if not query_text or not query_text.strip():
            return []

        with self._lock:
            self._rebuild_fts_index()

            query = """
            SELECT n.id, n.type, n.properties, n.vector, n.score
            FROM (
                SELECT *, fts_main_nodes.match_bm25(id, ?) AS score
                FROM nodes
            ) n
            WHERE n.score IS NOT NULL
            ORDER BY n.score
            LIMIT ?;
            """
            try:
                results = self.conn.execute(query, (query_text, limit)).fetchall()
                return [
                    SearchResult(
                        node=Node(id=row[0], type=row[1], properties=json.loads(row[2]), vector=row[3]),
                        score=-row[4]  # BM25 returns negative scores; negate so higher = more relevant
                    ) for row in results
                ]
            except duckdb.Error as e:
                logger.error(f"Error during FTS search: {e}")
                return []

    def hybrid_search(self, query_text: str, query_vector: list[float], limit: int = 10,
                      text_weight: float = 0.5, vector_weight: float = 0.5) -> list[SearchResult]:
        """Hybrid search combining FTS (BM25) and vector similarity (VSS) results."""
        if not self._validate_vector(query_vector):
            logger.error("Invalid vector for hybrid search.")
            return []

        with self._lock:
            # Collect FTS results
            fts_results = {}
            if query_text and query_text.strip():
                self._rebuild_fts_index()
                fts_query = """
                SELECT n.id, n.type, n.properties, n.vector, n.score
                FROM (
                    SELECT *, fts_main_nodes.match_bm25(id, ?) AS score
                    FROM nodes
                ) n
                WHERE n.score IS NOT NULL;
                """
                try:
                    rows = self.conn.execute(fts_query, (query_text,)).fetchall()
                    for row in rows:
                        fts_results[str(row[0])] = {
                            "node": Node(id=row[0], type=row[1], properties=json.loads(row[2]), vector=row[3]),
                            "fts_score": -row[4]  # Negate BM25 negative score
                        }
                except duckdb.Error as e:
                    logger.error(f"Error during FTS in hybrid search: {e}")

            # Collect vector similarity results
            vss_results = {}
            vss_query = f"""
            SELECT id, type, properties, vector,
                   array_distance(vector, CAST(? AS FLOAT[{self.vector_length}])) AS distance
            FROM nodes
            WHERE vector IS NOT NULL
            ORDER BY distance;
            """
            try:
                rows = self.conn.execute(vss_query, (query_vector,)).fetchall()
                for row in rows:
                    vss_results[str(row[0])] = {
                        "node": Node(id=row[0], type=row[1], properties=json.loads(row[2]), vector=row[3]),
                        "distance": row[4]
                    }
            except duckdb.Error as e:
                logger.error(f"Error during VSS in hybrid search: {e}")

        # Normalize and combine scores
        all_ids = set(fts_results.keys()) | set(vss_results.keys())
        if not all_ids:
            return []

        max_fts = max((v["fts_score"] for v in fts_results.values()), default=1.0) or 1.0
        max_dist = max((v["distance"] for v in vss_results.values()), default=1.0) or 1.0

        combined = []
        for node_id in all_ids:
            fts_entry = fts_results.get(node_id)
            vss_entry = vss_results.get(node_id)

            node = fts_entry["node"] if fts_entry else vss_entry["node"]

            norm_fts = (fts_entry["fts_score"] / max_fts) if fts_entry else 0.0
            norm_vss = (1.0 - vss_entry["distance"] / max_dist) if vss_entry else 0.0

            score = text_weight * norm_fts + vector_weight * norm_vss
            combined.append(SearchResult(node=node, score=score))

        combined.sort(key=lambda r: r.score, reverse=True)
        return combined[:limit]

    def _validate_vector(self, vector):
        return isinstance(vector, list) and len(vector) == self.vector_length and all(isinstance(x, float) for x in vector)

    @contextmanager
    def transaction(self):
        with self._lock:
            try:
                self.conn.execute("BEGIN TRANSACTION;")
                yield
                self.conn.execute("COMMIT;")
            except Exception as e:
                self.conn.execute("ROLLBACK;")
                raise e

    def to_networkx(self):
        """Export this graph to a NetworkX DiGraph.

        Requires the ``networkx`` package (install via ``pip install graphmemory[algorithms]``).

        Returns:
            A ``networkx.DiGraph`` with all nodes and edges from this instance.
        """
        from graphmemory.algorithms import to_networkx
        return to_networkx(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
