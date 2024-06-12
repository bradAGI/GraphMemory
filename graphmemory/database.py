import duckdb
import json
import os
import logging
from contextlib import contextmanager
from graphmemory.models import Node, Edge, NearestNode
from typing import List, Any
from typing import Dict as D
import uuid


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphMemory:
    def __init__(self, database=None, vector_length=3):
        self.database = database
        self.vector_length = vector_length
        self.conn = duckdb.connect(database=self.database)
        self._load_vss_extension()
        self._configure_database()

        if database and os.path.exists(database):
            self.load_database(database)

        # Check if 'nodes' and 'edges' tables exist, and create them if they do not
        nodes_exist = self.conn.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = 'nodes';").fetchone()
        edges_exist = self.conn.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = 'edges';").fetchone()

        if not nodes_exist or not edges_exist:
            self._create_tables()
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

    def _create_tables(self):
        # Correctly format the SQL string to include vector_length
        self.conn.execute(f"""
        CREATE TABLE IF NOT EXISTS nodes (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            type TEXT,
            properties JSON,
            vector FLOAT[{self.vector_length}]
        );
        """)
        self.conn.execute(f"""
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

    def insert_node(self, node: Node) -> uuid.UUID:
        if node.vector and not self._validate_vector(node.vector):
            logger.error("Invalid vector: Must be a list of float values.")
            return None
        try:
            with self.transaction():
                result = self.conn.execute(
                    "INSERT INTO nodes (id, type, properties, vector) VALUES (?, ?, ?, ?) RETURNING id;",
                    (str(node.id), node.type, json.dumps(node.properties), node.vector if node.vector else [0.0] * self.vector_length)
                ).fetchone()
                if result:
                    logger.info(f"Node inserted with ID: {result[0]}")
                return result[0] if result else None
        except duckdb.Error as e:
            logger.error(f"Error during insert node: {e}")
            return None

    def insert_edge(self, edge: Edge):
        try:
            with self.transaction():
                # Check if source and target nodes exist
                source_exists = self.conn.execute(
                    "SELECT 1 FROM nodes WHERE id = ?", (str(edge.source_id),)).fetchone()
                target_exists = self.conn.execute(
                    "SELECT 1 FROM nodes WHERE id = ?", (str(edge.target_id),)).fetchone()
                if not source_exists or not target_exists:
                    raise ValueError("Source or target node does not exist.")

                self.conn.execute("INSERT INTO edges (id, source_id, target_id, relation, weight) VALUES (?, ?, ?, ?, ?);", (
                    str(edge.id), str(edge.source_id), str(edge.target_id), edge.relation, edge.weight))
        except duckdb.Error as e:
            logger.error(f"Error during insert edge: {e}")
        except ValueError as e:
            logger.error(f"Error during insert edge: {e}")
            raise

    def bulk_insert_nodes(self, nodes: List[Node]) -> List[Node]:
        try:
            with self.transaction():
                for node in nodes:
                    result = self.conn.execute(
                        "INSERT INTO nodes (id, type, properties, vector) VALUES (?, ?, ?, ?) RETURNING id;",
                        (str(node.id), node.type, json.dumps(node.properties), node.vector if node.vector else None)
                    ).fetchone()
                    if result:
                        node.id = result[0]
                return nodes
        except duckdb.Error as e:
            logger.error(f"Error during bulk insert nodes: {e}")
            return []

    def bulk_insert_edges(self, edges: List[Edge]):
        try:
            with self.transaction():
                self.conn.executemany(
                    "INSERT INTO edges (id, source_id, target_id, relation, weight) VALUES (?, ?, ?, ?, ?);",
                    [(str(edge.id), str(edge.source_id), str(edge.target_id), edge.relation, edge.weight)
                     for edge in edges]
                )
        except duckdb.Error as e:
            logger.error(f"Error during bulk insert edges: {e}")

    def delete_node(self, node_id: uuid.UUID):
        try:
            with self.transaction():
                self.conn.execute(
                    "DELETE FROM nodes WHERE id = ?;", (str(node_id),))
                self.conn.execute(
                    "DELETE FROM edges WHERE source_id = ? OR target_id = ?;", (str(node_id), str(node_id)))
        except duckdb.Error as e:
            logger.error(f"Error deleting node: {e}")

    def delete_edge(self, source_id: uuid.UUID, target_id: uuid.UUID):
        try:
            with self.transaction():
                self.conn.execute(
                    "DELETE FROM edges WHERE source_id = ? AND target_id = ?;", (str(source_id), str(target_id)))
        except duckdb.Error as e:
            logger.error(f"Error deleting edge: {e}")

    def create_index(self):
        try:
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS vss_idx ON nodes USING HNSW(vector);")
        except duckdb.Error as e:
            logger.error(f"Error creating index: {e}")

    def nearest_nodes(self, vector: List[float], limit: int) -> List[NearestNode]:
        if not self._validate_vector(vector):
            logger.error("Invalid vector: Must be a list of float values.")
            return []

        query = f"""
        SELECT id, type, properties, vector, array_distance(vector, CAST(? AS FLOAT[{self.vector_length}])) AS distance
        FROM nodes
        WHERE vector IS NOT NULL
        ORDER BY distance LIMIT ?;
        """
        try:
            results = self.conn.execute(query, (vector, limit)).fetchall()
            return [
                NearestNode(
                    node=Node(id=row[0], type=row[1], properties=json.loads(row[2]), vector=row[3]),
                    distance=row[4]
                ) for row in results
            ]
        except duckdb.Error as e:
            logger.error(f"Error fetching nearest neighbors: {e}")
            return []

    def connected_nodes(self, node_id: uuid.UUID) -> List[Node]:
        query = """
        SELECT n.id, n.type, n.properties, n.vector
        FROM nodes n
        WHERE n.id IN (
                SELECT target_id FROM edges WHERE source_id = ?
                UNION
                SELECT source_id FROM edges WHERE target_id = ?
            );
        """
        try:
            logger.info(
                f"Executing query to fetch connected nodes for node_id: {node_id}")
            results = self.conn.execute(query, (str(node_id), str(node_id))).fetchall()
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

    def nodes_to_json(self) -> List[D[str, Any]]:
        try:
            nodes = self.conn.execute(
                "SELECT id, type, properties, vector FROM nodes;").fetchall()
            return [{"id": row[0], "type": row[1], "properties": json.loads(row[2]), "vector": row[3]} for row in nodes]
        except duckdb.Error as e:
            logger.error(f"Error fetching nodes: {e}")
            return []

    def edges_to_json(self) -> List[D[str, Any]]:
        try:
            edges = self.conn.execute(
                "SELECT id, source_id, target_id, relation, weight FROM edges;").fetchall()
            return [{"id": str(row[0]), "source_id": str(row[1]), "target_id": str(row[2]), "relation": row[3], "weight": row[4]} for row in edges]
        except duckdb.Error as e:
            logger.error(f"Error fetching edges: {e}")
            return []

    def get_node(self, node_id: uuid.UUID) -> Node:
        try:
            node = self.conn.execute(
                "SELECT id, type, properties, vector FROM nodes WHERE id = ?;", (str(node_id),)).fetchone()
            if node:
                return Node(id=node[0], type=node[1], properties=json.loads(node[2]), vector=node[3])
            else:
                return None
        except duckdb.Error as e:
            logger.error(f"Error fetching node: {e}")
            return None

    def nodes_by_attribute(self, attribute, value) -> List[Node]:
        try:
            query = f"SELECT id, type, properties, vector FROM nodes WHERE json_extract(properties, '$.{attribute}') = ?;"
            nodes = self.conn.execute(query, (json.dumps(value),)).fetchall()
            if nodes:
                return [Node(id=row[0], type=row[1], properties=json.loads(row[2]), vector=row[3]) for row in nodes]
            else:
                return []
        except duckdb.Error as e:
            logger.error(f"Error fetching nodes: {e}")
            return []

    def get_nodes_vector(self, node_id: int) -> List[float]:
        try:
            vector = self.conn.execute(
                "SELECT vector FROM nodes WHERE id = ?;", (node_id,)).fetchone()
            return vector[0] if vector else []
        except duckdb.Error as e:
            logger.error(f"Error fetching vector: {e}")
            return []

    def print_json(self):
        nodes_json = self.nodes_to_json()
        edges_json = self.edges_to_json()
        print("Nodes JSON:", json.dumps(nodes_json, indent=2))
        print("Edges JSON:", json.dumps(edges_json, indent=2))

    def cypher(self, cypher_query):
        sql_query = self._cypher_to_sql(cypher_query)
        try:
            results = self.conn.execute(sql_query).fetchall()
            logger.debug(f"Query results: {results}")
            return results
        except duckdb.Error as e:
            logger.error(f"Error executing SQL query: {e}")
            return []
   

    def _cypher_to_sql(self, cypher_query):
        import re
        import json  # Added import for json
        # Define regex patterns for nodes, relationships, and properties
        node_pattern = re.compile(r"\((\w+)(?::(\w+))?(?:\s*{([^}]+)})?\)")
        rel_pattern = re.compile(r"\[(\w+)?(?::(\w+))?(?:\s*{([^}]+)})?\]")
        
        # Helper function to parse properties
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

        # Extract MATCH clause
        match_clause = re.search(r'MATCH\s+(.*)\s+RETURN', cypher_query, re.IGNORECASE)
        if not match_clause:
            raise ValueError("Invalid Cypher query: missing MATCH or RETURN clause")
        match_content = match_clause.group(1).strip()

        # Extract RETURN clause
        return_clause = re.search(r'RETURN\s+(.*)', cypher_query, re.IGNORECASE)
        if not return_clause:
            raise ValueError("Invalid Cypher query: missing RETURN clause")
        return_content = return_clause.group(1).strip().split(',')

        # Parse nodes and relationships together
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

        # Start building the SQL query
        sql_query = "SELECT "
        sql_parts = []

        # Determine what is being returned
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

        # Process nodes and relationships in sequence
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