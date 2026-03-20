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

from graphmemory.models import DuplicateCluster, Edge, EdgeMergeResult, MergeResult, MergeStrategy, NearestNode, Node, RetrievalContext, RetrievalResult, SearchResult, TraversalResult

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

    def __init__(self, database=None, vector_length=3, distance_metric='l2', max_retries=3, retry_base_delay=0.1,
                 hnsw_ef_construction=128, hnsw_ef_search=64, hnsw_m=16, auto_index=True):
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
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search
        self.hnsw_m = hnsw_m
        self._lock = threading.RLock()
        self._fts_initialized = False
        self._fts_dirty = True
        self._hnsw_indexed = False
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

        if auto_index:
            self._ensure_hnsw_index()

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
            self._hnsw_indexed = False
            self._ensure_hnsw_index()
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
        self._hnsw_indexed = False
        self._ensure_hnsw_index()
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
            self.compact_index()
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
            self.compact_index()
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

    def _find_matching_node(
        self, cur, node: Node, match_keys: list[str], match_type: bool,
        similarity_threshold: float = 1.0,
        vector_threshold: float | None = None,
    ) -> Node | None:
        """Find an existing node matching the given property keys and optional type.

        When ``similarity_threshold`` is 1.0 (default), matching is exact.
        Lower values enable fuzzy matching via DuckDB's ``jaro_winkler_similarity``.
        When ``vector_threshold`` is set and the node has a vector, candidates must
        also have a cosine distance within that threshold.
        """
        fuzzy = similarity_threshold < 1.0

        # Separate param lists for SELECT expressions vs WHERE clauses,
        # since DuckDB binds positional params in statement order.
        select_extra: list[str] = []
        select_params: list = []
        where_parts: list[str] = []
        where_params: list = []

        if match_type and node.type is not None:
            where_parts.append("type = ?")
            where_params.append(node.type)

        for key in match_keys:
            if not self._VALID_ATTRIBUTE_RE.match(key):
                raise ValueError(f"Invalid match key: {key!r}")
            value = (node.properties or {}).get(key)
            if value is None:
                where_parts.append(f"json_extract(properties, '$.{key}') IS NULL")
            elif fuzzy and isinstance(value, str):
                alias = f"sim_{key}"
                select_extra.append(
                    f"jaro_winkler_similarity(json_extract_string(properties, '$.{key}'), ?) AS {alias}"
                )
                select_params.append(value)
                where_parts.append(f"{alias} >= ?")
                where_params.append(similarity_threshold)
            else:
                where_parts.append(f"json_extract(properties, '$.{key}') = ?")
                where_params.append(json.dumps(value))

        if vector_threshold is not None and node.vector:
            where_parts.append(f"array_cosine_distance(vector, CAST(? AS FLOAT[{self.vector_length}])) <= ?")
            where_params.extend([node.vector, vector_threshold])

        if not where_parts and not select_extra:
            return None

        select_cols = "id, type, properties, vector"
        if select_extra:
            select_cols += ", " + ", ".join(select_extra)

        where_clause = " AND ".join(where_parts) if where_parts else "TRUE"

        order_clause = ""
        if fuzzy:
            sim_cols = [f"sim_{k}" for k in match_keys
                        if isinstance((node.properties or {}).get(k), str)]
            if sim_cols:
                order_clause = " ORDER BY " + " + ".join(sim_cols) + " DESC"

        query = f"SELECT {select_cols} FROM nodes WHERE {where_clause}{order_clause} LIMIT 1;"
        params = select_params + where_params
        row = cur.execute(query, params).fetchone()
        if row:
            return Node(id=row[0], type=row[1], properties=json.loads(row[2]), vector=row[3])
        return None

    @staticmethod
    def _merge_properties(existing: dict, incoming: dict, strategy: MergeStrategy) -> dict:
        """Apply the merge strategy to combine existing and incoming properties."""
        if strategy == MergeStrategy.REPLACE:
            return incoming or {}
        elif strategy == MergeStrategy.UPDATE:
            return {**(existing or {}), **(incoming or {})}
        elif strategy == MergeStrategy.KEEP:
            return existing or {}
        return incoming or {}

    def _safe_update_node(self, cur, node_id: str, node_type, properties: dict, vector) -> None:
        """Update a node, working around DuckDB FK constraints on UPDATE.

        DuckDB internally deletes+reinserts rows on UPDATE, which triggers FK
        violations when edges reference the node. This method temporarily removes
        and restores those edges.
        """
        try:
            cur.execute(
                "UPDATE nodes SET type = ?, properties = ?, vector = ? WHERE id = ?;",
                (node_type, json.dumps(properties), vector, node_id)
            )
        except duckdb.ConstraintException:
            # Stash edges, update node, restore edges
            edges = cur.execute(
                "SELECT id, source_id, target_id, relation, weight FROM edges "
                "WHERE source_id = ? OR target_id = ?;",
                (node_id, node_id)
            ).fetchall()
            for eid, *_ in edges:
                cur.execute("DELETE FROM edges WHERE id = ?;", (eid,))
            cur.execute(
                "UPDATE nodes SET type = ?, properties = ?, vector = ? WHERE id = ?;",
                (node_type, json.dumps(properties), vector, node_id)
            )
            for eid, src, tgt, rel, wt in edges:
                cur.execute(
                    "INSERT INTO edges (id, source_id, target_id, relation, weight) "
                    "VALUES (?, ?, ?, ?, ?);",
                    (eid, src, tgt, rel, wt)
                )

    @staticmethod
    def normalize_relation(relation: str) -> str:
        """Lowercase, strip, and collapse whitespace/separators to underscores."""
        s = relation.strip().lower()
        s = re.sub(r'[\s\-\.]+', '_', s)
        s = re.sub(r'_+', '_', s)
        return s.strip('_')

    @with_retry()
    def merge_node(self, node: Node, match_keys: list[str],
                   match_type: bool = True,
                   strategy: MergeStrategy = MergeStrategy.UPDATE,
                   update_vector: bool = True,
                   similarity_threshold: float = 1.0,
                   vector_threshold: float | None = None) -> MergeResult:
        """Insert a node or update it if a match is found by property keys.

        Args:
            node: The node to merge.
            match_keys: Property names to match on (e.g. ["name"]).
            match_type: Also require node.type to match (default True).
            strategy: How to merge properties (REPLACE, UPDATE, KEEP).
            update_vector: Whether to replace the vector on match.

        Returns:
            MergeResult with the resulting node and whether it was created.
        """
        if not match_keys:
            raise ValueError("match_keys must not be empty")
        for key in match_keys:
            if not self._VALID_ATTRIBUTE_RE.match(key):
                raise ValueError(f"Invalid match key: {key!r}")
        if node.vector and not self._validate_vector(node.vector):
            raise ValueError("Invalid vector: must be a list of floats with correct length")

        try:
            with self.transaction():
                cur = self.cursor()
                existing = self._find_matching_node(
                    cur, node, match_keys, match_type,
                    similarity_threshold=similarity_threshold,
                    vector_threshold=vector_threshold,
                )
                if existing:
                    merged_props = self._merge_properties(existing.properties, node.properties, strategy)
                    vector = node.vector if update_vector and node.vector else existing.vector
                    node_type = node.type if node.type is not None else existing.type
                    self._safe_update_node(cur, str(existing.id), node_type, merged_props, vector)
                    self._fts_dirty = True
                    result_node = Node(id=existing.id, type=node_type, properties=merged_props, vector=vector)
                    return MergeResult(node=result_node, created=False)
                else:
                    result = cur.execute(
                        "INSERT INTO nodes (id, type, properties, vector) VALUES (?, ?, ?, ?) RETURNING id;",
                        (str(node.id), node.type, json.dumps(node.properties), node.vector if node.vector else [0.0] * self.vector_length)
                    ).fetchone()
                    self._fts_dirty = True
                    result_node = Node(id=result[0], type=node.type, properties=node.properties, vector=node.vector)
                    return MergeResult(node=result_node, created=True)
        except duckdb.Error as e:
            logger.error(f"Error during merge node: {e}")
            raise

    @with_retry()
    def bulk_merge_nodes(self, nodes: list[Node], match_keys: list[str],
                         match_type: bool = True,
                         strategy: MergeStrategy = MergeStrategy.UPDATE,
                         update_vector: bool = True,
                         similarity_threshold: float = 1.0,
                         vector_threshold: float | None = None) -> list[MergeResult]:
        """Merge multiple nodes, inserting new ones and updating matches.

        Runs in a single transaction for atomicity.
        """
        if not match_keys:
            raise ValueError("match_keys must not be empty")
        for key in match_keys:
            if not self._VALID_ATTRIBUTE_RE.match(key):
                raise ValueError(f"Invalid match key: {key!r}")

        results: list[MergeResult] = []
        try:
            with self.transaction():
                cur = self.cursor()
                for node in nodes:
                    if node.vector and not self._validate_vector(node.vector):
                        logger.error(f"Invalid vector for node, skipping: {node.id}")
                        continue
                    existing = self._find_matching_node(
                        cur, node, match_keys, match_type,
                        similarity_threshold=similarity_threshold,
                        vector_threshold=vector_threshold,
                    )
                    if existing:
                        merged_props = self._merge_properties(existing.properties, node.properties, strategy)
                        vector = node.vector if update_vector and node.vector else existing.vector
                        node_type = node.type if node.type is not None else existing.type
                        self._safe_update_node(cur, str(existing.id), node_type, merged_props, vector)
                        result_node = Node(id=existing.id, type=node_type, properties=merged_props, vector=vector)
                        results.append(MergeResult(node=result_node, created=False))
                    else:
                        result = cur.execute(
                            "INSERT INTO nodes (id, type, properties, vector) VALUES (?, ?, ?, ?) RETURNING id;",
                            (str(node.id), node.type, json.dumps(node.properties), node.vector if node.vector else [0.0] * self.vector_length)
                        ).fetchone()
                        result_node = Node(id=result[0], type=node.type, properties=node.properties, vector=node.vector)
                        results.append(MergeResult(node=result_node, created=True))
                if results:
                    self._fts_dirty = True
                return results
        except duckdb.Error as e:
            logger.error(f"Error during bulk merge nodes: {e}")
            raise

    def _find_matching_edge(self, cur, edge: Edge) -> Edge | None:
        """Find an existing edge matching (source_id, target_id, relation).

        Relations are compared in normalized form (lowercase, underscored).
        """
        normalized = self.normalize_relation(edge.relation)
        row = cur.execute(
            "SELECT id, source_id, target_id, relation, weight FROM edges WHERE source_id = ? AND target_id = ? AND relation = ? LIMIT 1;",
            (str(edge.source_id), str(edge.target_id), normalized)
        ).fetchone()
        if row:
            return Edge(id=row[0], source_id=row[1], target_id=row[2], relation=row[3], weight=row[4])
        return None

    @with_retry()
    def merge_edge(self, edge: Edge, update_weight: bool = True) -> EdgeMergeResult:
        """Insert an edge or update it if a match on (source_id, target_id, relation) exists.

        Args:
            edge: The edge to merge.
            update_weight: Whether to update the weight on match.

        Returns:
            EdgeMergeResult with the resulting edge and whether it was created.
        """
        try:
            with self.transaction():
                cur = self.cursor()
                source_exists = cur.execute("SELECT 1 FROM nodes WHERE id = ?", (str(edge.source_id),)).fetchone()
                target_exists = cur.execute("SELECT 1 FROM nodes WHERE id = ?", (str(edge.target_id),)).fetchone()
                if not source_exists or not target_exists:
                    raise ValueError("Source or target node does not exist.")

                existing = self._find_matching_edge(cur, edge)
                if existing:
                    if update_weight and edge.weight is not None:
                        cur.execute("UPDATE edges SET weight = ? WHERE id = ?;", (edge.weight, str(existing.id)))
                        result_edge = Edge(id=existing.id, source_id=existing.source_id, target_id=existing.target_id, relation=existing.relation, weight=edge.weight)
                    else:
                        result_edge = existing
                    return EdgeMergeResult(edge=result_edge, created=False)
                else:
                    normalized = self.normalize_relation(edge.relation)
                    cur.execute(
                        "INSERT INTO edges (id, source_id, target_id, relation, weight) VALUES (?, ?, ?, ?, ?);",
                        (str(edge.id), str(edge.source_id), str(edge.target_id), normalized, edge.weight)
                    )
                    result_edge = Edge(id=edge.id, source_id=edge.source_id, target_id=edge.target_id, relation=normalized, weight=edge.weight)
                    return EdgeMergeResult(edge=result_edge, created=True)
        except duckdb.Error as e:
            logger.error(f"Error during merge edge: {e}")
            raise

    @with_retry()
    def bulk_merge_edges(self, edges: list[Edge], update_weight: bool = True) -> list[EdgeMergeResult]:
        """Merge multiple edges, inserting new ones and updating matches.

        Deduplicates on (source_id, target_id, relation). Runs in a single transaction.
        """
        results: list[EdgeMergeResult] = []
        try:
            with self.transaction():
                cur = self.cursor()
                for edge in edges:
                    source_exists = cur.execute("SELECT 1 FROM nodes WHERE id = ?", (str(edge.source_id),)).fetchone()
                    target_exists = cur.execute("SELECT 1 FROM nodes WHERE id = ?", (str(edge.target_id),)).fetchone()
                    if not source_exists or not target_exists:
                        logger.error(f"Skipping edge {edge.id}: source or target node does not exist.")
                        continue
                    existing = self._find_matching_edge(cur, edge)
                    if existing:
                        if update_weight and edge.weight is not None:
                            cur.execute("UPDATE edges SET weight = ? WHERE id = ?;", (edge.weight, str(existing.id)))
                            result_edge = Edge(id=existing.id, source_id=existing.source_id, target_id=existing.target_id, relation=existing.relation, weight=edge.weight)
                        else:
                            result_edge = existing
                        results.append(EdgeMergeResult(edge=result_edge, created=False))
                    else:
                        normalized = self.normalize_relation(edge.relation)
                        cur.execute(
                            "INSERT INTO edges (id, source_id, target_id, relation, weight) VALUES (?, ?, ?, ?, ?);",
                            (str(edge.id), str(edge.source_id), str(edge.target_id), normalized, edge.weight)
                        )
                        result_edge = Edge(id=edge.id, source_id=edge.source_id, target_id=edge.target_id, relation=normalized, weight=edge.weight)
                        results.append(EdgeMergeResult(edge=result_edge, created=True))
                return results
        except duckdb.Error as e:
            logger.error(f"Error during bulk merge edges: {e}")
            raise

    @with_retry()
    def resolve_duplicates(
        self,
        match_keys: list[str] | None = None,
        match_type: bool = True,
        similarity_threshold: float = 0.9,
        vector_threshold: float | None = None,
        strategy: MergeStrategy = MergeStrategy.UPDATE,
    ) -> list[DuplicateCluster]:
        """Scan all nodes and merge clusters of likely duplicates.

        For each unprocessed node, finds fuzzy matches among remaining nodes.
        The first node encountered becomes the "survivor"; duplicates have their
        edges reassigned and are then deleted.

        Args:
            match_keys: Property names to compare (default ``["name"]``).
            match_type: Also require ``node.type`` to match (default ``True``).
            similarity_threshold: Jaro-Winkler threshold for string properties.
            vector_threshold: Max cosine distance for vector similarity (optional).
            strategy: How to merge properties from duplicates into the survivor.

        Returns:
            List of :class:`~graphmemory.models.DuplicateCluster` results.
        """
        if match_keys is None:
            match_keys = ["name"]
        for key in match_keys:
            if not self._VALID_ATTRIBUTE_RE.match(key):
                raise ValueError(f"Invalid match key: {key!r}")

        clusters: list[DuplicateCluster] = []
        try:
            cur = self.cursor()
            all_rows = cur.execute(
                "SELECT id, type, properties, vector FROM nodes ORDER BY id;"
            ).fetchall()
            all_nodes = [
                Node(id=r[0], type=r[1], properties=json.loads(r[2]), vector=r[3])
                for r in all_rows
            ]

            seen: set[str] = set()
            for node in all_nodes:
                nid = str(node.id)
                if nid in seen:
                    continue
                seen.add(nid)

                # Build fuzzy query for candidates (separate param lists for ordering)
                select_extra: list[str] = []
                select_params: list = []
                where_parts: list[str] = ["id != ?"]
                where_params: list = [nid]

                if match_type and node.type is not None:
                    where_parts.append("type = ?")
                    where_params.append(node.type)

                for key in match_keys:
                    value = (node.properties or {}).get(key)
                    if value is None:
                        continue
                    if isinstance(value, str):
                        alias = f"sim_{key}"
                        select_extra.append(
                            f"jaro_winkler_similarity(json_extract_string(properties, '$.{key}'), ?) AS {alias}"
                        )
                        select_params.append(value)
                        where_parts.append(f"{alias} >= ?")
                        where_params.append(similarity_threshold)

                if vector_threshold is not None and node.vector:
                    where_parts.append(
                        f"array_cosine_distance(vector, CAST(? AS FLOAT[{self.vector_length}])) <= ?"
                    )
                    where_params.extend([node.vector, vector_threshold])

                if not select_extra:
                    continue

                # Exclude already-processed nodes
                if seen - {nid}:
                    placeholders = ", ".join("?" for _ in seen if _ != nid)
                    where_parts.append(f"id NOT IN ({placeholders})")
                    where_params.extend(s for s in seen if s != nid)

                select_cols = "id, type, properties, vector"
                if select_extra:
                    select_cols += ", " + ", ".join(select_extra)

                query = f"SELECT {select_cols} FROM nodes WHERE {' AND '.join(where_parts)};"
                dup_rows = cur.execute(query, select_params + where_params).fetchall()

                if not dup_rows:
                    continue

                duplicates: list[Node] = []
                survivor_props = dict(node.properties or {})
                survivor_vector = node.vector
                survivor_type = node.type
                edges_to_rewrite: list[tuple] = []

                for row in dup_rows:
                    dup = Node(id=row[0], type=row[1], properties=json.loads(row[2]), vector=row[3])
                    dup_id = str(dup.id)
                    seen.add(dup_id)
                    duplicates.append(dup)

                    survivor_props = self._merge_properties(survivor_props, dup.properties, strategy)
                    if not survivor_vector and dup.vector:
                        survivor_vector = dup.vector
                    if not survivor_type and dup.type:
                        survivor_type = dup.type

                    dup_edges = cur.execute(
                        "SELECT id, source_id, target_id, relation, weight FROM edges "
                        "WHERE source_id = ? OR target_id = ?;",
                        (dup_id, dup_id)
                    ).fetchall()
                    for eid, src, tgt, rel, wt in dup_edges:
                        new_src = nid if src == dup_id else src
                        new_tgt = nid if tgt == dup_id else tgt
                        edges_to_rewrite.append((eid, new_src, new_tgt, rel, wt))

                # Delete edges referencing duplicates
                for dup in duplicates:
                    cur.execute(
                        "DELETE FROM edges WHERE source_id = ? OR target_id = ?;",
                        (str(dup.id), str(dup.id))
                    )
                # Also temporarily remove edges referencing survivor (DuckDB
                # internally does delete+reinsert on UPDATE, triggering FK checks)
                survivor_edges = cur.execute(
                    "SELECT id, source_id, target_id, relation, weight FROM edges "
                    "WHERE source_id = ? OR target_id = ?;",
                    (nid, nid)
                ).fetchall()
                for eid, *_ in survivor_edges:
                    cur.execute("DELETE FROM edges WHERE id = ?;", (eid,))

                # Delete duplicate nodes
                for dup in duplicates:
                    cur.execute("DELETE FROM nodes WHERE id = ?;", (str(dup.id),))

                # Update survivor with merged properties (safe now, no FK refs)
                cur.execute(
                    "UPDATE nodes SET type = ?, properties = ?, vector = ? WHERE id = ?;",
                    (survivor_type, json.dumps(survivor_props), survivor_vector, nid)
                )

                # Re-insert all edges, verifying both endpoints still exist
                rewritten_eids = {e[0] for e in edges_to_rewrite}
                all_edges_to_insert = []
                for eid, src, tgt, rel, wt in edges_to_rewrite:
                    if src == tgt:
                        continue  # skip self-loops
                    all_edges_to_insert.append((eid, src, tgt, rel, wt))
                for eid, src, tgt, rel, wt in survivor_edges:
                    if eid in rewritten_eids:
                        continue
                    all_edges_to_insert.append((eid, src, tgt, rel, wt))

                for eid, src, tgt, rel, wt in all_edges_to_insert:
                    src_exists = cur.execute(
                        "SELECT 1 FROM nodes WHERE id = ?", (str(src),)
                    ).fetchone()
                    tgt_exists = cur.execute(
                        "SELECT 1 FROM nodes WHERE id = ?", (str(tgt),)
                    ).fetchone()
                    if src_exists and tgt_exists:
                        cur.execute(
                            "INSERT INTO edges (id, source_id, target_id, relation, weight) "
                            "VALUES (?, ?, ?, ?, ?);",
                            (eid, src, tgt, rel, wt)
                        )

                survivor = Node(id=node.id, type=survivor_type, properties=survivor_props, vector=survivor_vector)
                clusters.append(DuplicateCluster(survivor=survivor, merged=duplicates))

            if clusters:
                self._fts_dirty = True

            logger.info(
                "Resolved %d duplicate clusters (%d nodes merged).",
                len(clusters),
                sum(len(c.merged) for c in clusters),
            )
            return clusters
        except duckdb.Error as e:
            logger.error(f"Error during resolve_duplicates: {e}")
            raise

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

    def _ensure_hnsw_index(self):
        """Create HNSW index if not already present. Called automatically on init."""
        if self._hnsw_indexed:
            return
        try:
            nodes_exist = self.conn.execute(
                "SELECT 1 FROM information_schema.tables WHERE table_name = 'nodes';"
            ).fetchone()
            if nodes_exist:
                self.create_index()
        except duckdb.Error:
            pass

    @with_retry()
    def create_index(self, ef_construction: int | None = None, ef_search: int | None = None, m: int | None = None):
        """Create or recreate the HNSW vector index.

        Args:
            ef_construction: Candidate vertices during build (default from init).
            ef_search: Candidate vertices during search (default from init).
            m: Max neighbors per vertex (default from init).
        """
        ef_c = ef_construction or self.hnsw_ef_construction
        ef_s = ef_search or self.hnsw_ef_search
        m_val = m or self.hnsw_m
        hnsw_metric = self.DISTANCE_METRICS[self.distance_metric]['hnsw_metric']
        with self._lock:
            try:
                # Drop existing index first to allow metric/param changes
                self.conn.execute("DROP INDEX IF EXISTS vss_idx;")
                self.conn.execute(
                    f"CREATE INDEX vss_idx ON nodes USING HNSW(vector) "
                    f"WITH (metric = '{hnsw_metric}', ef_construction = {ef_c}, ef_search = {ef_s}, M = {m_val});"
                )
                self._hnsw_indexed = True
                logger.info(f"HNSW index created (metric={hnsw_metric}, ef_construction={ef_c}, ef_search={ef_s}, M={m_val}).")
            except duckdb.Error as e:
                logger.error(f"Error creating HNSW index: {e}")

    def compact_index(self):
        """Compact the HNSW index to reclaim space after deletions."""
        with self._lock:
            try:
                self.conn.execute("PRAGMA hnsw_compact_index('vss_idx');")
                logger.info("HNSW index compacted.")
            except duckdb.Error as e:
                logger.error(f"Error compacting HNSW index: {e}")

    @with_retry()
    def nearest_nodes(self, vector: list[float], limit: int) -> list[NearestNode]:
        if not self._validate_vector(vector):
            logger.error("Invalid vector: Must be a list of float values.")
            return []

        dist_func = self.DISTANCE_METRICS[self.distance_metric]['function']
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

    def query(self) -> 'QueryBuilder':
        """Return a new QueryBuilder for constructing parameterized graph queries.

        Example::

            results = graph.query().match(type="Person").where(name="Alice").execute()
            results = graph.query().match().traverse(depth=2).limit(10).execute()
        """
        return QueryBuilder(self)

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
            dist_func = self.DISTANCE_METRICS[self.distance_metric]['function']
            vss_query = f"""
            SELECT id, type, properties, vector,
                   {dist_func}(vector, CAST(? AS FLOAT[{self.vector_length}])) AS distance
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

    def _expand_graph(self, seed_node_ids: set[str], max_hops: int) -> tuple[dict[str, Node], list[Edge]]:
        """Expand from seed nodes via multi-hop traversal, returning all discovered nodes and edges."""
        visited_ids = set(seed_node_ids)
        all_nodes: dict[str, Node] = {}
        all_edges: list[Edge] = []
        seen_edge_ids: set[str] = set()

        for nid in seed_node_ids:
            node = self.get_node(uuid.UUID(nid))
            if node:
                all_nodes[str(node.id)] = node

        frontier = set(seed_node_ids)
        for _hop in range(max_hops):
            if not frontier:
                break
            next_frontier: set[str] = set()
            for nid in frontier:
                with self._lock:
                    try:
                        cur = self.cursor()
                        rows = cur.execute(
                            "SELECT id, source_id, target_id, relation, weight FROM edges "
                            "WHERE source_id = ? OR target_id = ?;",
                            (nid, nid)
                        ).fetchall()
                    except duckdb.Error as e:
                        logger.error(f"Error expanding graph from node {nid}: {e}")
                        continue

                for row in rows:
                    edge = Edge(id=row[0], source_id=row[1], target_id=row[2], relation=row[3], weight=row[4])
                    eid = str(edge.id)
                    if eid not in seen_edge_ids:
                        seen_edge_ids.add(eid)
                        all_edges.append(edge)

                    neighbor_id = str(edge.target_id) if str(edge.source_id) == nid else str(edge.source_id)
                    if neighbor_id not in visited_ids:
                        visited_ids.add(neighbor_id)
                        next_frontier.add(neighbor_id)
                        neighbor_node = self.get_node(uuid.UUID(neighbor_id))
                        if neighbor_node:
                            all_nodes[neighbor_id] = neighbor_node

            frontier = next_frontier

        return all_nodes, all_edges

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: ~4 characters per token."""
        return len(text) // 4

    @staticmethod
    def _format_node_context(node: Node) -> str:
        """Format a single node into a readable context string."""
        parts = []
        label = f"[{node.type}]" if node.type else "[Node]"
        parts.append(f"{label} (id: {node.id})")
        if node.properties:
            for key, value in node.properties.items():
                parts.append(f"  {key}: {value}")
        return "\n".join(parts)

    @staticmethod
    def _format_edge_context(edge: Edge, nodes: dict[str, Node]) -> str:
        """Format an edge into a readable relationship string."""
        source = nodes.get(str(edge.source_id))
        target = nodes.get(str(edge.target_id))
        source_label = (source.type or "Node") if source else "?"
        target_label = (target.type or "Node") if target else "?"
        relation = edge.relation or "related_to"
        weight_str = f" (weight: {edge.weight})" if edge.weight is not None else ""
        return f"({source_label}:{edge.source_id}) -[{relation}{weight_str}]-> ({target_label}:{edge.target_id})"

    def _assemble_context(self, query: str, nodes: dict[str, Node], edges: list[Edge],
                          seed_ids: set[str], max_tokens: int) -> RetrievalResult:
        """Assemble nodes and edges into a prompt-ready context window respecting token limits."""
        contexts: list[RetrievalContext] = []

        # Build adjacency for hop distance calculation
        adjacency: dict[str, list[dict[str, Any]]] = {nid: [] for nid in nodes}
        for edge in edges:
            sid, tid = str(edge.source_id), str(edge.target_id)
            edge_dict = {"id": str(edge.id), "relation": edge.relation, "weight": edge.weight,
                         "source_id": sid, "target_id": tid}
            if sid in adjacency:
                adjacency[sid].append(edge_dict)
            if tid in adjacency:
                adjacency[tid].append(edge_dict)

        # BFS from seed nodes to compute hop distance
        hop_distances: dict[str, int] = {nid: 0 for nid in seed_ids if nid in nodes}
        bfs_queue = [nid for nid in seed_ids if nid in nodes]
        bfs_visited = set(bfs_queue)
        while bfs_queue:
            current = bfs_queue.pop(0)
            for edge_dict in adjacency.get(current, []):
                neighbor = edge_dict["target_id"] if edge_dict["source_id"] == current else edge_dict["source_id"]
                if neighbor not in bfs_visited and neighbor in nodes:
                    bfs_visited.add(neighbor)
                    hop_distances[neighbor] = hop_distances[current] + 1
                    bfs_queue.append(neighbor)

        for nid, node in nodes.items():
            hop = hop_distances.get(nid, len(nodes))
            node_edges = adjacency.get(nid, [])
            contexts.append(RetrievalContext(node=node, relationships=node_edges, hop_distance=hop))
        contexts.sort(key=lambda c: c.hop_distance)

        # Assemble text respecting token limit
        text_parts = [f"## Context for query: {query}\n"]
        text_parts.append("### Entities\n")

        included_node_ids: set[str] = set()
        for ctx in contexts:
            node_text = self._format_node_context(ctx.node)
            candidate = "\n".join(text_parts) + node_text + "\n\n"
            if self._estimate_tokens(candidate) > max_tokens:
                break
            text_parts.append(node_text)
            text_parts.append("")
            included_node_ids.add(str(ctx.node.id))

        if edges:
            rel_header = "### Relationships\n"
            candidate = "\n".join(text_parts) + rel_header
            if self._estimate_tokens(candidate) <= max_tokens:
                text_parts.append(rel_header)
                for edge in edges:
                    if str(edge.source_id) in included_node_ids or str(edge.target_id) in included_node_ids:
                        edge_text = self._format_edge_context(edge, nodes)
                        candidate = "\n".join(text_parts) + edge_text + "\n"
                        if self._estimate_tokens(candidate) > max_tokens:
                            break
                        text_parts.append(edge_text)

        context_text = "\n".join(text_parts).strip()

        return RetrievalResult(
            query=query,
            contexts=contexts,
            context_text=context_text,
            token_estimate=self._estimate_tokens(context_text),
            seed_node_count=len(seed_ids),
            total_node_count=len(nodes),
        )

    def retrieve(self, query: str, query_vector: list[float],
                 max_hops: int = 2, max_tokens: int = 4000,
                 search_limit: int = 10, text_weight: float = 0.5,
                 vector_weight: float = 0.5) -> RetrievalResult:
        """Full GraphRAG retrieval pipeline.

        Chains: hybrid search -> graph expansion via multi-hop traversal -> context assembly.
        """
        search_results = self.hybrid_search(
            query_text=query,
            query_vector=query_vector,
            limit=search_limit,
            text_weight=text_weight,
            vector_weight=vector_weight,
        )

        if not search_results:
            return RetrievalResult(query=query)

        seed_ids = {str(sr.node.id) for sr in search_results}
        all_nodes, all_edges = self._expand_graph(seed_ids, max_hops)
        return self._assemble_context(query, all_nodes, all_edges, seed_ids, max_tokens)

    def ask(self, query: str, query_vector: list[float],
            llm_callable: Any = None,
            max_hops: int = 2, max_tokens: int = 4000,
            search_limit: int = 10, text_weight: float = 0.5,
            vector_weight: float = 0.5,
            system_prompt: str = "Answer the question using only the provided context. "
                                 "If the context does not contain enough information, say so.") -> dict[str, Any]:
        """End-to-end question answering: retrieval + LLM generation.

        Args:
            query: The question to answer.
            query_vector: Vector embedding of the query.
            llm_callable: A callable that takes (system_prompt: str, user_prompt: str) -> str.
                          If None, returns only the retrieval result without generation.
        """
        retrieval = self.retrieve(
            query=query,
            query_vector=query_vector,
            max_hops=max_hops,
            max_tokens=max_tokens,
            search_limit=search_limit,
            text_weight=text_weight,
            vector_weight=vector_weight,
        )

        answer = None
        if llm_callable is not None and retrieval.context_text:
            user_prompt = f"{retrieval.context_text}\n\n## Question\n{query}"
            try:
                answer = llm_callable(system_prompt, user_prompt)
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                answer = None

        return {"retrieval": retrieval, "answer": answer}

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


class QueryBuilder:
    """Fluent, composable query builder for GraphMemory.

    All filters use parameterized queries to prevent SQL injection.
    Returns typed results (list[Node], list[Edge], or list[TraversalResult]).

    Example::

        graph.query().match(type="Person").where(name="Alice").execute()
        graph.query().match().where(age=30).order_by("name").limit(10).execute()
        graph.query().match(type="Person").traverse(depth=2).execute()
    """

    _VALID_ATTR_RE = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')
    _VALID_ORDER_FIELDS = {'id', 'type'}

    def __init__(self, graph: GraphMemory):
        self._graph = graph
        self._type: str | None = None
        self._where_conditions: list[tuple[str, Any]] = []
        self._traverse_depth: int | None = None
        self._traverse_source_id: uuid.UUID | None = None
        self._order_by_field: str | None = None
        self._order_by_property: str | None = None
        self._order_asc: bool = True
        self._limit_val: int | None = None
        self._offset_val: int | None = None
        self._return_edges: bool = False

    def match(self, type: str | None = None) -> 'QueryBuilder':
        """Filter nodes by type label."""
        self._type = type
        return self

    def where(self, **conditions) -> 'QueryBuilder':
        """Add property-based WHERE conditions (parameterized)."""
        for key, value in conditions.items():
            if not self._VALID_ATTR_RE.match(key):
                raise ValueError(f"Invalid attribute name: {key!r}")
            self._where_conditions.append((key, value))
        return self

    def traverse(self, source_id: uuid.UUID | None = None, depth: int = 1) -> 'QueryBuilder':
        """Traverse edges from a node (or all matched nodes) to a configurable depth."""
        if depth < 1:
            raise ValueError("Traverse depth must be >= 1")
        self._traverse_depth = depth
        self._traverse_source_id = source_id
        return self

    def order_by(self, field: str, ascending: bool = True) -> 'QueryBuilder':
        """Order results by a node column ('id', 'type') or a property key."""
        if field in self._VALID_ORDER_FIELDS:
            self._order_by_field = field
            self._order_by_property = None
        else:
            if not self._VALID_ATTR_RE.match(field):
                raise ValueError(f"Invalid order_by field: {field!r}")
            self._order_by_property = field
            self._order_by_field = None
        self._order_asc = ascending
        return self

    def limit(self, n: int) -> 'QueryBuilder':
        """Limit the number of results."""
        if n < 0:
            raise ValueError("Limit must be >= 0")
        self._limit_val = n
        return self

    def offset(self, n: int) -> 'QueryBuilder':
        """Skip the first n results."""
        if n < 0:
            raise ValueError("Offset must be >= 0")
        self._offset_val = n
        return self

    def edges(self) -> 'QueryBuilder':
        """Return edges instead of nodes."""
        self._return_edges = True
        return self

    def execute(self) -> list[Node] | list[Edge] | list[TraversalResult]:
        """Execute the query and return typed results."""
        if self._traverse_depth is not None:
            return self._execute_traversal()
        if self._return_edges:
            return self._execute_edges()
        return self._execute_nodes()

    def _execute_nodes(self) -> list[Node]:
        query = "SELECT id, type, properties, vector FROM nodes"
        params: list[Any] = []
        where_parts: list[str] = []

        if self._type is not None:
            where_parts.append("type = ?")
            params.append(self._type)

        for attr, value in self._where_conditions:
            where_parts.append(f"json_extract(properties, '$.{attr}') = ?")
            params.append(json.dumps(value))

        if where_parts:
            query += " WHERE " + " AND ".join(where_parts)

        if self._order_by_field:
            query += f" ORDER BY {self._order_by_field}"
            query += " ASC" if self._order_asc else " DESC"
        elif self._order_by_property:
            query += f" ORDER BY json_extract(properties, '$.{self._order_by_property}')"
            query += " ASC" if self._order_asc else " DESC"

        if self._limit_val is not None:
            query += " LIMIT ?"
            params.append(self._limit_val)
        if self._offset_val is not None:
            query += " OFFSET ?"
            params.append(self._offset_val)

        query += ";"

        with self._graph._lock:
            try:
                cur = self._graph.cursor()
                rows = cur.execute(query, params).fetchall()
                return [
                    Node(id=row[0], type=row[1], properties=json.loads(row[2]), vector=row[3])
                    for row in rows
                ]
            except duckdb.Error as e:
                logger.error(f"Error executing query: {e}")
                return []

    def _execute_edges(self) -> list[Edge]:
        query = "SELECT e.id, e.source_id, e.target_id, e.relation, e.weight FROM edges e"
        params: list[Any] = []
        where_parts: list[str] = []

        if self._type is not None or self._where_conditions:
            query += " JOIN nodes n ON e.source_id = n.id"
            if self._type is not None:
                where_parts.append("n.type = ?")
                params.append(self._type)
            for attr, value in self._where_conditions:
                where_parts.append(f"json_extract(n.properties, '$.{attr}') = ?")
                params.append(json.dumps(value))

        if where_parts:
            query += " WHERE " + " AND ".join(where_parts)

        if self._limit_val is not None:
            query += " LIMIT ?"
            params.append(self._limit_val)
        if self._offset_val is not None:
            query += " OFFSET ?"
            params.append(self._offset_val)

        query += ";"

        with self._graph._lock:
            try:
                cur = self._graph.cursor()
                rows = cur.execute(query, params).fetchall()
                return [
                    Edge(id=row[0], source_id=row[1], target_id=row[2], relation=row[3], weight=row[4])
                    for row in rows
                ]
            except duckdb.Error as e:
                logger.error(f"Error executing edge query: {e}")
                return []

    def _execute_traversal(self) -> list[TraversalResult]:
        source_ids = self._resolve_source_ids()
        if not source_ids:
            return []

        visited: dict[str, tuple[int, list[uuid.UUID]]] = {}
        for sid in source_ids:
            visited[str(sid)] = (0, [sid])

        current_level = list(source_ids)
        for depth in range(1, self._traverse_depth + 1):
            if not current_level:
                break
            placeholders = ', '.join(['?'] * len(current_level))
            id_strs = [str(nid) for nid in current_level]
            query = f"""
            SELECT DISTINCT target_id, source_id FROM edges WHERE source_id IN ({placeholders})
            UNION
            SELECT DISTINCT source_id, target_id FROM edges WHERE target_id IN ({placeholders})
            """
            with self._graph._lock:
                try:
                    cur = self._graph.cursor()
                    rows = cur.execute(query, id_strs + id_strs).fetchall()
                except duckdb.Error as e:
                    logger.error(f"Error during traversal: {e}")
                    break

            next_level = []
            for row in rows:
                neighbor_id = row[0]
                parent_id = str(row[1])
                nid_str = str(neighbor_id)
                if nid_str not in visited:
                    parent_path = visited.get(parent_id, (0, []))[1]
                    visited[nid_str] = (depth, parent_path + [neighbor_id])
                    next_level.append(neighbor_id)
            current_level = next_level

        source_id_strs = {str(sid) for sid in source_ids}
        result_entries = [
            (nid_str, depth, path)
            for nid_str, (depth, path) in visited.items()
            if nid_str not in source_id_strs
        ]

        if not result_entries:
            return []

        result_id_strs = [entry[0] for entry in result_entries]
        placeholders = ', '.join(['?'] * len(result_id_strs))
        node_query = f"SELECT id, type, properties, vector FROM nodes WHERE id IN ({placeholders});"
        with self._graph._lock:
            try:
                cur = self._graph.cursor()
                rows = cur.execute(node_query, result_id_strs).fetchall()
            except duckdb.Error as e:
                logger.error(f"Error fetching traversal nodes: {e}")
                return []

        node_map = {
            str(row[0]): Node(id=row[0], type=row[1], properties=json.loads(row[2]), vector=row[3])
            for row in rows
        }

        results = []
        for nid_str, depth, path in result_entries:
            if nid_str in node_map:
                results.append(TraversalResult(node=node_map[nid_str], depth=depth, path=path))

        results.sort(key=lambda r: r.depth)

        if self._offset_val is not None:
            results = results[self._offset_val:]
        if self._limit_val is not None:
            results = results[:self._limit_val]

        return results

    def _resolve_source_ids(self) -> list[uuid.UUID]:
        if self._traverse_source_id is not None:
            return [self._traverse_source_id]
        return [n.id for n in self._execute_nodes()]
