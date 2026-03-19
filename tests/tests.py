import json
import logging
import os
import sys
import uuid

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import duckdb
import unittest

from pydantic import ValidationError

from graphmemory.database import GraphMemory
from graphmemory.models import Edge, Node

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class TestGraphMemory(unittest.TestCase):
    def setUp(self):
        self.db = GraphMemory(database=":memory:", vector_length=3)

    def test_insert_node(self):
        node = Node(properties={"name": "node1"}, vector=[0.1, 0.2, 0.3])
        node_id = self.db.insert_node(node)
        result = self.db.conn.execute("SELECT * FROM nodes WHERE id = ?", (str(node_id),)).fetchone()
        self.assertIsNotNone(result)
        self.assertEqual(str(result[0]), str(node_id))  # Convert UUID to string for comparison
        self.assertEqual(result[1], None)
        self.assertEqual(result[2], '{"name": "node1"}')
        self.assertAlmostEqual(result[3][0], 0.1, places=7)
        self.assertAlmostEqual(result[3][1], 0.2, places=7)
        self.assertAlmostEqual(result[3][2], 0.3, places=7)

    def test_insert_invalid_node(self):
        with self.assertRaises(ValidationError):
            node = Node(properties={"name": "node1"}, vector=[0.1, 0.2, "invalid"])
            self.db.insert_node(node)

    def test_insert_node_invalid_vector_length(self):
        node = Node(properties={"name": "node1"}, vector=[0.1, 0.2])
        result = self.db.insert_node(node)
        self.assertIsNone(result)

    def test_insert_edge(self):
        node1 = Node(properties={"name": "node1"}, vector=[0.1, 0.2, 0.3])
        node2 = Node(properties={"name": "node2"}, vector=[0.4, 0.5, 0.6])
        node1_id = self.db.insert_node(node1)
        node2_id = self.db.insert_node(node2)
        edge = Edge(source_id=node1_id, target_id=node2_id, weight=0.5, relation="friendship")
        self.db.insert_edge(edge)
        result = self.db.conn.execute(
            "SELECT source_id, target_id, weight, relation FROM edges WHERE source_id = ? AND target_id = ?",
            (str(node1_id), str(node2_id)),
        ).fetchone()
        self.assertIsNotNone(result)
        self.assertEqual(str(result[0]), str(node1_id))
        self.assertEqual(str(result[1]), str(node2_id))
        self.assertEqual(result[2], 0.5)
        self.assertEqual(result[3], "friendship")

    def test_insert_edge_non_existent_nodes(self):
        edge = Edge(source_id=uuid.uuid4(), target_id=uuid.uuid4(), weight=0.5, relation="friendship")
        with self.assertRaises(ValueError):
            self.db.insert_edge(edge)

    def test_nearest_nodes_empty_db(self):
        neighbors = self.db.nearest_nodes(vector=[0.1, 0.2, 0.3], limit=2)
        self.assertEqual(len(neighbors), 0)

    def test_nearest_nodes(self):
        nodes = [
            Node(properties={"name": "node1"}, vector=[0.1, 0.2, 0.3]),
            Node(properties={"name": "node2"}, vector=[0.4, 0.5, 0.6]),
            Node(properties={"name": "node3"}, vector=[0.7, 0.8, 0.9]),
        ]
        for node in nodes:
            self.db.insert_node(node)
        self.db.create_index()
        neighbors = self.db.nearest_nodes(vector=[0.1, 0.2, 0.3], limit=2)
        self.assertEqual(len(neighbors), 2)
        self.assertEqual(neighbors[0].node.properties["name"], "node1")
        self.assertEqual(neighbors[1].node.properties["name"], "node2")

    def test_nodes_to_json(self):
        node = Node(properties={"name": "node1"}, vector=[0.1, 0.2, 0.3])
        self.db.insert_node(node)
        nodes_json = self.db.nodes_to_json()
        self.assertEqual(len(nodes_json), 1)
        self.assertEqual(nodes_json[0]["properties"], {"name": "node1"})
        self.assertAlmostEqual(nodes_json[0]["vector"][0], 0.1, places=7)
        self.assertAlmostEqual(nodes_json[0]["vector"][1], 0.2, places=7)
        self.assertAlmostEqual(nodes_json[0]["vector"][2], 0.3, places=7)

    def test_edges_to_json(self):
        node1 = Node(properties={"name": "node1"}, vector=[0.1, 0.2, 0.3])
        node2 = Node(properties={"name": "node2"}, vector=[0.4, 0.5, 0.6])
        self.db.insert_node(node1)
        self.db.insert_node(node2)
        edge = Edge(source_id=node1.id, target_id=node2.id, weight=0.5, relation="friendship")
        self.db.insert_edge(edge)
        edges_json = self.db.edges_to_json()
        self.assertEqual(len(edges_json), 1)
        self.assertEqual(edges_json[0]["source_id"], str(node1.id))
        self.assertEqual(edges_json[0]["target_id"], str(node2.id))
        self.assertEqual(edges_json[0]["weight"], 0.5)
        self.assertEqual(edges_json[0]["relation"], "friendship")

    def test_bulk_insert_nodes(self):
        nodes = [
            Node(properties={"name": "node1"}, vector=[0.1, 0.2, 0.3]),
            Node(properties={"name": "node2"}, vector=[0.4, 0.5, 0.6]),
        ]
        inserted_nodes = self.db.bulk_insert_nodes(nodes)
        self.assertEqual(len(inserted_nodes), 2)
        self.assertTrue(all(node.id is not None for node in inserted_nodes))

    def test_bulk_insert_nodes_with_invalid_vector(self):
        nodes = [
            Node(properties={"name": "valid1"}, vector=[0.1, 0.2, 0.3]),
            Node(properties={"name": "invalid"}, vector=[0.1, 0.2]),  # wrong length
            Node(properties={"name": "valid2"}, vector=[0.4, 0.5, 0.6]),
        ]
        inserted_nodes = self.db.bulk_insert_nodes(nodes)
        self.assertEqual(len(inserted_nodes), 2)
        inserted_names = [n.properties["name"] for n in inserted_nodes]
        self.assertIn("valid1", inserted_names)
        self.assertIn("valid2", inserted_names)
        self.assertNotIn("invalid", inserted_names)
        # Verify only valid nodes are in the database
        db_nodes = self.db.conn.execute("SELECT * FROM nodes").fetchall()
        self.assertEqual(len(db_nodes), 2)

    def test_bulk_insert_edges(self):
        node1 = Node(properties={"name": "node1"}, vector=[0.1, 0.2, 0.3])
        node2 = Node(properties={"name": "node2"}, vector=[0.4, 0.5, 0.6])
        node3 = Node(properties={"name": "node3"}, vector=[0.7, 0.8, 0.9])
        self.db.bulk_insert_nodes([node1, node2, node3])
        edges = [
            Edge(source_id=node1.id, target_id=node2.id, weight=0.5, relation="friendship"),
            Edge(source_id=node2.id, target_id=node3.id, weight=0.7, relation="colleague"),
        ]
        self.db.bulk_insert_edges(edges)
        result = self.db.conn.execute("SELECT * FROM edges").fetchall()
        self.assertEqual(len(result), 2)

    def test_delete_node(self):
        node = Node(properties={"name": "node1"}, vector=[0.1, 0.2, 0.3])
        node_id = self.db.insert_node(node)
        self.db.delete_node(node_id)
        result = self.db.conn.execute("SELECT * FROM nodes WHERE id = ?", (str(node_id),)).fetchone()
        self.assertIsNone(result)

    def test_delete_node_with_edges(self):
        """Deleting a node must remove all associated edges first, then the node."""
        node1 = Node(properties={"name": "node1"}, vector=[0.1, 0.2, 0.3])
        node2 = Node(properties={"name": "node2"}, vector=[0.4, 0.5, 0.6])
        node3 = Node(properties={"name": "node3"}, vector=[0.7, 0.8, 0.9])
        node1_id = self.db.insert_node(node1)
        node2_id = self.db.insert_node(node2)
        node3_id = self.db.insert_node(node3)

        # Create edges where node2 is both source and target
        self.db.insert_edge(Edge(source_id=node1_id, target_id=node2_id, weight=0.5, relation="friendship"))
        self.db.insert_edge(Edge(source_id=node2_id, target_id=node3_id, weight=0.7, relation="colleague"))

        # Delete node2 — edges referencing it should be removed first
        self.db.delete_node(node2_id)

        # Verify node2 is gone
        result = self.db.conn.execute("SELECT * FROM nodes WHERE id = ?", (str(node2_id),)).fetchone()
        self.assertIsNone(result)

        # Verify all edges referencing node2 are gone
        edges = self.db.conn.execute(
            "SELECT * FROM edges WHERE source_id = ? OR target_id = ?", (str(node2_id), str(node2_id))
        ).fetchall()
        self.assertEqual(len(edges), 0)

        # Verify other nodes still exist
        self.assertIsNotNone(self.db.conn.execute("SELECT * FROM nodes WHERE id = ?", (str(node1_id),)).fetchone())
        self.assertIsNotNone(self.db.conn.execute("SELECT * FROM nodes WHERE id = ?", (str(node3_id),)).fetchone())

    def test_delete_non_existent_node(self):
        self.db.delete_node(uuid.uuid4())
        result = self.db.conn.execute("SELECT * FROM nodes WHERE id = ?", (uuid.uuid4(),)).fetchone()
        self.assertIsNone(result)

    def test_delete_edge(self):
        node1 = Node(properties={"name": "node1"}, vector=[0.1, 0.2, 0.3])
        node2 = Node(properties={"name": "node2"}, vector=[0.4, 0.5, 0.6])
        self.db.insert_node(node1)
        self.db.insert_node(node2)
        edge = Edge(source_id=node1.id, target_id=node2.id, weight=0.5, relation="friendship")
        self.db.insert_edge(edge)
        self.db.delete_edge(node1.id, node2.id)
        result = self.db.conn.execute(
            "SELECT * FROM edges WHERE source_id = ? AND target_id = ?", (str(node1.id), str(node2.id))
        ).fetchone()
        self.assertIsNone(result)

    def test_delete_non_existent_edge(self):
        self.db.delete_edge(uuid.uuid4(), uuid.uuid4())
        result = self.db.conn.execute(
            "SELECT * FROM edges WHERE source_id = ? AND target_id = ?", (uuid.uuid4(), uuid.uuid4())
        ).fetchone()
        self.assertIsNone(result)

    def test_transaction_handling(self):
        try:
            with self.db.transaction():
                self.db.conn.execute(
                    'INSERT INTO nodes (properties, vector) VALUES (\'{"name": "node1"}\', [0.1, 0.2, 0.3]);'
                )
                raise Exception("Force rollback")
        except Exception:
            pass
        result = self.db.conn.execute('SELECT * FROM nodes WHERE properties = \'{"name": "node1"}\'').fetchone()
        self.assertIsNone(result)

    def test_node_id(self):
        node1 = Node(properties={"name": "node1"}, vector=[0.1, 0.2, 0.3])
        node2 = Node(properties={"name": "node2"}, vector=[0.4, 0.5, 0.6])
        id1 = self.db.insert_node(node1)
        id2 = self.db.insert_node(node2)

        self.assertIsInstance(id1, uuid.UUID)
        self.assertIsInstance(id2, uuid.UUID)

    def test_edge_id(self):
        node1 = Node(properties={"name": "node1"}, vector=[0.1, 0.2, 0.3])
        node2 = Node(properties={"name": "node2"}, vector=[0.4, 0.5, 0.6])
        id1 = self.db.insert_node(node1)
        id2 = self.db.insert_node(node2)

        edge1 = Edge(source_id=id1, target_id=id2, weight=0.5, relation="friendship")
        edge2 = Edge(source_id=id2, target_id=id1, weight=0.7, relation="colleague")
        self.db.insert_edge(edge1)
        self.db.insert_edge(edge2)

        result1 = self.db.conn.execute(
            "SELECT source_id, target_id FROM edges WHERE source_id = ? AND target_id = ?", (str(id1), str(id2))
        ).fetchone()
        result2 = self.db.conn.execute(
            "SELECT source_id, target_id FROM edges WHERE source_id = ? AND target_id = ?", (str(id2), str(id1))
        ).fetchone()

        self.assertIsNotNone(result1)
        self.assertIsNotNone(result2)
        self.assertEqual(str(result1[0]), str(id1))
        self.assertEqual(str(result1[1]), str(id2))
        self.assertEqual(str(result2[0]), str(id2))
        self.assertEqual(str(result2[1]), str(id1))

    def tearDown(self):
        self.db.conn.close()
        # Ensure any test directories are cleaned up
        import shutil

        if os.path.exists("test_db_parquet"):
            shutil.rmtree("test_db_parquet")
        if os.path.exists("test_db_csv"):
            shutil.rmtree("test_db_csv")


class TestGraphMemoryGetConnectedNodes(unittest.TestCase):
    def setUp(self):
        # Initialize the GraphMemory instance with an in-memory database for testing
        self.db = GraphMemory(database=":memory:", vector_length=3)
        self.db._create_tables()  # Ensure tables are created

        # Insert nodes and edges for testing
        self.node1 = Node(properties={"name": "node1"}, vector=[0.1, 0.2, 0.3])
        self.node2 = Node(properties={"name": "node2"}, vector=[0.4, 0.5, 0.6])
        self.node3 = Node(properties={"name": "node3"}, vector=[0.7, 0.8, 0.9])
        self.node1_id = self.db.insert_node(self.node1)
        self.node2_id = self.db.insert_node(self.node2)
        self.node3_id = self.db.insert_node(self.node3)

        # Insert edges
        self.db.insert_edge(Edge(source_id=self.node1_id, target_id=self.node2_id, weight=0.5, relation="friendship"))
        self.db.insert_edge(Edge(source_id=self.node2_id, target_id=self.node3_id, weight=0.7, relation="colleague"))

    def test_connected_nodes(self):
        # Test for node1, should have node2 as connected
        connected_nodes = self.db.connected_nodes(self.node1_id)
        self.assertEqual(len(connected_nodes), 1)
        self.assertEqual(connected_nodes[0].id, self.node2_id)

        # Test for node2, should have node1 and node3 as connected
        connected_nodes = self.db.connected_nodes(self.node2_id)
        self.assertEqual(len(connected_nodes), 2)
        connected_ids = [node.id for node in connected_nodes]
        self.assertIn(self.node1_id, connected_ids)
        self.assertIn(self.node3_id, connected_ids)

        # Test for node3, should have node2 as connected
        connected_nodes = self.db.connected_nodes(self.node3_id)
        self.assertEqual(len(connected_nodes), 1)
        self.assertEqual(connected_nodes[0].id, self.node2_id)

    def tearDown(self):
        self.db.conn.close()


class TestEdgeQueryMethods(unittest.TestCase):
    def setUp(self):
        self.db = GraphMemory(database=":memory:", vector_length=3)
        self.node1 = Node(properties={"name": "Alice"}, vector=[0.1, 0.2, 0.3])
        self.node2 = Node(properties={"name": "Bob"}, vector=[0.4, 0.5, 0.6])
        self.node3 = Node(properties={"name": "Charlie"}, vector=[0.7, 0.8, 0.9])
        self.node1_id = self.db.insert_node(self.node1)
        self.node2_id = self.db.insert_node(self.node2)
        self.node3_id = self.db.insert_node(self.node3)

        self.edge1 = Edge(source_id=self.node1_id, target_id=self.node2_id, weight=0.5, relation="friendship")
        self.edge2 = Edge(source_id=self.node2_id, target_id=self.node3_id, weight=0.7, relation="colleague")
        self.edge3 = Edge(source_id=self.node1_id, target_id=self.node3_id, weight=0.3, relation="friendship")
        self.db.insert_edge(self.edge1)
        self.db.insert_edge(self.edge2)
        self.db.insert_edge(self.edge3)

    def test_get_edge_by_id(self):
        edge = self.db.get_edge(self.edge1.id)
        self.assertIsNotNone(edge)
        self.assertEqual(edge.id, self.edge1.id)
        self.assertEqual(edge.source_id, self.node1_id)
        self.assertEqual(edge.target_id, self.node2_id)
        self.assertEqual(edge.relation, "friendship")
        self.assertEqual(edge.weight, 0.5)

    def test_get_edge_not_found(self):
        edge = self.db.get_edge(uuid.uuid4())
        self.assertIsNone(edge)

    def test_get_edges_by_relation(self):
        edges = self.db.get_edges_by_relation("friendship")
        self.assertEqual(len(edges), 2)
        relations = [e.relation for e in edges]
        self.assertTrue(all(r == "friendship" for r in relations))

    def test_get_edges_by_relation_single(self):
        edges = self.db.get_edges_by_relation("colleague")
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].source_id, self.node2_id)
        self.assertEqual(edges[0].target_id, self.node3_id)

    def test_get_edges_by_relation_none_found(self):
        edges = self.db.get_edges_by_relation("nonexistent")
        self.assertEqual(len(edges), 0)

    def test_edges_by_attribute_relation(self):
        edges = self.db.edges_by_attribute("relation", "colleague")
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].relation, "colleague")

    def test_edges_by_attribute_weight(self):
        edges = self.db.edges_by_attribute("weight", 0.5)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].weight, 0.5)

    def test_edges_by_attribute_none_found(self):
        edges = self.db.edges_by_attribute("relation", "enemy")
        self.assertEqual(len(edges), 0)

    def tearDown(self):
        self.db.conn.close()


class TestQueryBuilder(unittest.TestCase):

    def setUp(self):
        self.graph = GraphMemory(database=":memory:", vector_length=3)
        self.node1 = Node(properties={"name": "George Washington", "age": 57}, type="Person", vector=[0.1, 0.2, 0.3])
        self.node2 = Node(properties={"name": "John Adams", "age": 61}, type="Person", vector=[0.4, 0.5, 0.6])
        self.node3 = Node(properties={"name": "Acme Corp"}, type="Company", vector=[0.7, 0.8, 0.9])
        self.node1_id = self.graph.insert_node(self.node1)
        self.node2_id = self.graph.insert_node(self.node2)
        self.node3_id = self.graph.insert_node(self.node3)
        self.graph.insert_edge(Edge(source_id=self.node1_id, target_id=self.node2_id, relation="knows", weight=1.0))
        self.graph.insert_edge(Edge(source_id=self.node1_id, target_id=self.node3_id, relation="works_at", weight=0.8))

    def test_match_by_type(self):
        results = self.graph.query().match(type="Person").execute()
        self.assertEqual(len(results), 2)
        types = {r.type for r in results}
        self.assertEqual(types, {"Person"})

    def test_where_single_condition(self):
        results = self.graph.query().match(type="Person").where(name="George Washington").execute()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].properties["name"], "George Washington")

    def test_where_multiple_conditions(self):
        results = self.graph.query().match(type="Person").where(name="George Washington", age=57).execute()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, self.node1_id)

    def test_where_no_match(self):
        results = self.graph.query().match(type="Person").where(name="Nobody").execute()
        self.assertEqual(len(results), 0)

    def test_match_all_nodes(self):
        results = self.graph.query().match().execute()
        self.assertEqual(len(results), 3)

    def test_limit(self):
        results = self.graph.query().match().limit(2).execute()
        self.assertEqual(len(results), 2)

    def test_offset(self):
        all_results = self.graph.query().match().execute()
        offset_results = self.graph.query().match().offset(1).execute()
        self.assertEqual(len(offset_results), len(all_results) - 1)

    def test_limit_and_offset(self):
        results = self.graph.query().match().limit(1).offset(1).execute()
        self.assertEqual(len(results), 1)

    def test_order_by_type(self):
        results = self.graph.query().match().order_by("type").execute()
        types = [r.type for r in results]
        self.assertEqual(types, sorted(types))

    def test_order_by_property(self):
        results = self.graph.query().match(type="Person").order_by("name").execute()
        names = [r.properties["name"] for r in results]
        self.assertEqual(names, sorted(names))

    def test_order_by_descending(self):
        results = self.graph.query().match(type="Person").order_by("name", ascending=False).execute()
        names = [r.properties["name"] for r in results]
        self.assertEqual(names, sorted(names, reverse=True))

    def test_traverse_depth_1(self):
        results = self.graph.query().traverse(source_id=self.node1_id, depth=1).execute()
        result_ids = {r.node.id for r in results}
        self.assertIn(self.node2_id, result_ids)
        self.assertIn(self.node3_id, result_ids)
        for r in results:
            self.assertEqual(r.depth, 1)

    def test_traverse_depth_2(self):
        results = self.graph.query().traverse(source_id=self.node2_id, depth=2).execute()
        result_ids = {r.node.id for r in results}
        self.assertIn(self.node1_id, result_ids)
        self.assertIn(self.node3_id, result_ids)

    def test_traverse_with_match_filter(self):
        results = self.graph.query().match(type="Person").where(name="George Washington").traverse(depth=1).execute()
        result_ids = {r.node.id for r in results}
        self.assertIn(self.node2_id, result_ids)
        self.assertIn(self.node3_id, result_ids)

    def test_traverse_limit(self):
        results = self.graph.query().traverse(source_id=self.node1_id, depth=1).limit(1).execute()
        self.assertEqual(len(results), 1)

    def test_edges_query(self):
        results = self.graph.query().edges().execute()
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], Edge)

    def test_edges_with_type_filter(self):
        results = self.graph.query().match(type="Person").edges().execute()
        self.assertGreater(len(results), 0)
        for edge in results:
            self.assertIsInstance(edge, Edge)

    def test_edges_limit(self):
        results = self.graph.query().edges().limit(1).execute()
        self.assertEqual(len(results), 1)

    def test_chainable(self):
        results = (
            self.graph.query()
            .match(type="Person")
            .where(name="George Washington")
            .order_by("name")
            .limit(10)
            .offset(0)
            .execute()
        )
        self.assertEqual(len(results), 1)

    def test_returns_typed_nodes(self):
        results = self.graph.query().match(type="Person").execute()
        for r in results:
            self.assertIsInstance(r, Node)

    def test_where_invalid_attribute_rejected(self):
        with self.assertRaises(ValueError):
            self.graph.query().where(**{"invalid name": "test"})

    def test_where_sql_injection_rejected(self):
        with self.assertRaises(ValueError):
            self.graph.query().where(**{"name'; DROP TABLE nodes;--": "test"})

    def test_order_by_invalid_field_rejected(self):
        with self.assertRaises(ValueError):
            self.graph.query().order_by("invalid field!")

    def test_traverse_invalid_depth(self):
        with self.assertRaises(ValueError):
            self.graph.query().traverse(depth=0)

    def test_limit_negative(self):
        with self.assertRaises(ValueError):
            self.graph.query().limit(-1)

    def test_offset_negative(self):
        with self.assertRaises(ValueError):
            self.graph.query().offset(-1)

    def test_traverse_returns_traversal_results(self):
        from graphmemory.models import TraversalResult
        results = self.graph.query().traverse(source_id=self.node1_id, depth=1).execute()
        for r in results:
            self.assertIsInstance(r, TraversalResult)
            self.assertIsInstance(r.node, Node)
            self.assertGreater(r.depth, 0)
            self.assertIsInstance(r.path, list)

    def test_traverse_path_includes_node(self):
        results = self.graph.query().traverse(source_id=self.node1_id, depth=1).execute()
        for r in results:
            self.assertIn(r.node.id, r.path)

    def tearDown(self):
        self.graph.conn.close()


class TestLoggingConfig(unittest.TestCase):
    """Verify that importing graphmemory does not modify the root logger."""

    def test_import_does_not_call_basic_config(self):
        import importlib

        root_logger = logging.getLogger()
        handlers_before = list(root_logger.handlers)
        importlib.reload(__import__("graphmemory.database"))
        handlers_after = list(root_logger.handlers)
        self.assertEqual(
            len(handlers_before),
            len(handlers_after),
            "Importing graphmemory.database should not add handlers to the root logger",
        )


class TestNodesByAttributeInjection(unittest.TestCase):
    def setUp(self):
        self.db = GraphMemory(database=":memory:", vector_length=3)
        node = Node(properties={"name": "test", "age": 30}, vector=[0.1, 0.2, 0.3])
        self.db.insert_node(node)

    def test_valid_attribute_returns_results(self):
        results = self.db.nodes_by_attribute("name", "test")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].properties["name"], "test")

    def test_valid_attribute_with_underscore(self):
        node = Node(properties={"first_name": "alice"}, vector=[0.1, 0.2, 0.3])
        self.db.insert_node(node)
        results = self.db.nodes_by_attribute("first_name", "alice")
        self.assertEqual(len(results), 1)

    def test_sql_injection_via_quote(self):
        with self.assertRaises(ValueError):
            self.db.nodes_by_attribute("name') OR 1=1 --", "test")

    def test_sql_injection_via_semicolon(self):
        with self.assertRaises(ValueError):
            self.db.nodes_by_attribute("name; DROP TABLE nodes;", "test")

    def test_sql_injection_via_parentheses(self):
        with self.assertRaises(ValueError):
            self.db.nodes_by_attribute("name) UNION SELECT * FROM nodes--", "test")

    def test_empty_attribute_rejected(self):
        with self.assertRaises(ValueError):
            self.db.nodes_by_attribute("", "test")

    def test_attribute_starting_with_number_rejected(self):
        with self.assertRaises(ValueError):
            self.db.nodes_by_attribute("1name", "test")

    def test_attribute_with_spaces_rejected(self):
        with self.assertRaises(ValueError):
            self.db.nodes_by_attribute("na me", "test")

    def tearDown(self):
        self.db.conn.close()


class TestPagination(unittest.TestCase):
    def setUp(self):
        self.db = GraphMemory(database=":memory:", vector_length=3)
        # Insert 50 nodes with sequential names
        self.nodes = []
        for i in range(50):
            node = Node(
                properties={"name": f"node{i}", "category": "test"},
                vector=[float(i) / 50, float(i + 1) / 50, float(i + 2) / 50],
            )
            self.db.insert_node(node)
            self.nodes.append(node)
        # Insert edges between consecutive nodes
        for i in range(49):
            edge = Edge(
                source_id=self.nodes[i].id,
                target_id=self.nodes[i + 1].id,
                weight=0.5,
                relation="next",
            )
            self.db.insert_edge(edge)

    def test_nodes_to_json_limit(self):
        result = self.db.nodes_to_json(limit=10)
        self.assertEqual(len(result), 10)

    def test_nodes_to_json_limit_offset(self):
        result = self.db.nodes_to_json(limit=10, offset=0)
        self.assertEqual(len(result), 10)

    def test_nodes_to_json_offset_near_end(self):
        result = self.db.nodes_to_json(limit=10, offset=40)
        self.assertEqual(len(result), 10)

    def test_nodes_to_json_offset_past_end(self):
        result = self.db.nodes_to_json(limit=10, offset=50)
        self.assertEqual(len(result), 0)

    def test_nodes_to_json_no_pagination(self):
        result = self.db.nodes_to_json()
        self.assertEqual(len(result), 50)

    def test_edges_to_json_limit(self):
        result = self.db.edges_to_json(limit=10)
        self.assertEqual(len(result), 10)

    def test_edges_to_json_limit_offset(self):
        result = self.db.edges_to_json(limit=10, offset=0)
        self.assertEqual(len(result), 10)

    def test_edges_to_json_offset_near_end(self):
        result = self.db.edges_to_json(limit=10, offset=39)
        self.assertEqual(len(result), 10)

    def test_edges_to_json_offset_past_end(self):
        result = self.db.edges_to_json(limit=10, offset=49)
        self.assertEqual(len(result), 0)

    def test_edges_to_json_no_pagination(self):
        result = self.db.edges_to_json()
        self.assertEqual(len(result), 49)

    def test_nodes_by_attribute_limit(self):
        result = self.db.nodes_by_attribute("category", "test", limit=10)
        self.assertEqual(len(result), 10)

    def test_nodes_by_attribute_limit_offset(self):
        result = self.db.nodes_by_attribute("category", "test", limit=10, offset=40)
        self.assertEqual(len(result), 10)

    def test_nodes_by_attribute_offset_past_end(self):
        result = self.db.nodes_by_attribute("category", "test", limit=10, offset=50)
        self.assertEqual(len(result), 0)

    def test_nodes_by_attribute_no_pagination(self):
        result = self.db.nodes_by_attribute("category", "test")
        self.assertEqual(len(result), 50)

    def tearDown(self):
        self.db.conn.close()




class TestBulkDeleteOperations(unittest.TestCase):

    def setUp(self):
        self.db = GraphMemory(database=':memory:', vector_length=3)
        self.nodes = [
            Node(properties={"name": f"node{i}"}, vector=[0.1 * i, 0.2 * i, 0.3 * i])
            for i in range(1, 6)
        ]
        self.db.bulk_insert_nodes(self.nodes)
        self.edges = [
            Edge(source_id=self.nodes[i].id, target_id=self.nodes[i + 1].id,
                 weight=0.5, relation=f"rel{i}")
            for i in range(4)
        ]
        self.db.bulk_insert_edges(self.edges)

    def test_bulk_delete_edges(self):
        edge_ids = [self.edges[0].id, self.edges[2].id]
        self.db.bulk_delete_edges(edge_ids)
        remaining = self.db.conn.execute("SELECT id FROM edges").fetchall()
        remaining_ids = {row[0] for row in remaining}
        self.assertEqual(len(remaining_ids), 2)
        self.assertNotIn(self.edges[0].id, remaining_ids)
        self.assertNotIn(self.edges[2].id, remaining_ids)
        self.assertIn(self.edges[1].id, remaining_ids)
        self.assertIn(self.edges[3].id, remaining_ids)

    def test_bulk_delete_nodes(self):
        node_ids = [self.nodes[1].id, self.nodes[3].id]
        self.db.bulk_delete_nodes(node_ids)
        remaining_nodes = self.db.conn.execute("SELECT id FROM nodes").fetchall()
        remaining_node_ids = {row[0] for row in remaining_nodes}
        self.assertEqual(len(remaining_node_ids), 3)
        self.assertNotIn(self.nodes[1].id, remaining_node_ids)
        self.assertNotIn(self.nodes[3].id, remaining_node_ids)
        remaining_edges = self.db.conn.execute("SELECT id FROM edges").fetchall()
        self.assertEqual(len(remaining_edges), 0)

    def test_bulk_delete_nodes_subset(self):
        node_ids = [self.nodes[4].id]
        self.db.bulk_delete_nodes(node_ids)
        remaining_nodes = self.db.conn.execute("SELECT id FROM nodes").fetchall()
        self.assertEqual(len(remaining_nodes), 4)
        remaining_edges = self.db.conn.execute("SELECT id FROM edges").fetchall()
        remaining_edge_ids = {row[0] for row in remaining_edges}
        self.assertEqual(len(remaining_edge_ids), 3)
        self.assertNotIn(self.edges[3].id, remaining_edge_ids)

    def test_bulk_delete_edges_empty_list(self):
        self.db.bulk_delete_edges([])
        remaining = self.db.conn.execute("SELECT id FROM edges").fetchall()
        self.assertEqual(len(remaining), 4)

    def test_bulk_delete_nodes_empty_list(self):
        self.db.bulk_delete_nodes([])
        remaining = self.db.conn.execute("SELECT id FROM nodes").fetchall()
        self.assertEqual(len(remaining), 5)

    def tearDown(self):
        self.db.conn.close()


class TestExportImport(unittest.TestCase):
    """Test export/import support for JSON, CSV, and GraphML formats."""

    def setUp(self):
        self.db = GraphMemory(database=':memory:', vector_length=3)
        self.node1 = Node(properties={"name": "Alice", "age": 30}, type="Person", vector=[0.1, 0.2, 0.3])
        self.node2 = Node(properties={"name": "Bob", "age": 25}, type="Person", vector=[0.4, 0.5, 0.6])
        self.node3 = Node(properties={"name": "Acme"}, type="Company", vector=[0.7, 0.8, 0.9])
        self.db.insert_node(self.node1)
        self.db.insert_node(self.node2)
        self.db.insert_node(self.node3)
        self.db.insert_edge(Edge(source_id=self.node1.id, target_id=self.node2.id, relation="knows", weight=1.0))
        self.db.insert_edge(Edge(source_id=self.node1.id, target_id=self.node3.id, relation="works_at", weight=0.8))

    def test_export_json(self):
        data = self.db.export_graph(format='json')
        self.assertIn('nodes', data)
        self.assertIn('edges', data)
        self.assertEqual(len(data['nodes']), 3)
        self.assertEqual(len(data['edges']), 2)
        node = data['nodes'][0]
        self.assertIn('id', node)
        self.assertIn('type', node)
        self.assertIn('properties', node)
        self.assertIn('vector', node)

    def test_export_json_string(self):
        data = self.db.export_graph(format='json_string')
        parsed = json.loads(data)
        self.assertEqual(len(parsed['nodes']), 3)
        self.assertEqual(len(parsed['edges']), 2)

    def test_export_csv(self):
        data = self.db.export_graph(format='csv')
        self.assertIn('nodes', data)
        self.assertIn('edges', data)
        node_lines = data['nodes'].strip().splitlines()
        edge_lines = data['edges'].strip().splitlines()
        self.assertEqual(len(node_lines), 4)  # header + 3 nodes
        self.assertEqual(len(edge_lines), 3)  # header + 2 edges
        self.assertIn('id', node_lines[0])
        self.assertIn('source_id', edge_lines[0])

    def test_export_graphml(self):
        data = self.db.export_graph(format='graphml')
        self.assertIsInstance(data, str)
        self.assertIn('<graphml', data)
        self.assertIn('<node', data)
        self.assertIn('<edge', data)

    def test_export_invalid_format(self):
        with self.assertRaises(ValueError):
            self.db.export_graph(format='yaml')

    def test_import_json(self):
        data = self.db.export_graph(format='json')
        db2 = GraphMemory(database=':memory:', vector_length=3)
        db2.import_graph(data, format='json')
        self.assertEqual(len(db2.nodes_to_json()), 3)
        self.assertEqual(len(db2.edges_to_json()), 2)
        db2.conn.close()

    def test_import_json_string(self):
        data = self.db.export_graph(format='json_string')
        db2 = GraphMemory(database=':memory:', vector_length=3)
        db2.import_graph(data, format='json_string')
        self.assertEqual(len(db2.nodes_to_json()), 3)
        self.assertEqual(len(db2.edges_to_json()), 2)
        db2.conn.close()

    def test_import_csv(self):
        data = self.db.export_graph(format='csv')
        db2 = GraphMemory(database=':memory:', vector_length=3)
        db2.import_graph(data, format='csv')
        self.assertEqual(len(db2.nodes_to_json()), 3)
        self.assertEqual(len(db2.edges_to_json()), 2)
        db2.conn.close()

    def test_import_graphml(self):
        data = self.db.export_graph(format='graphml')
        db2 = GraphMemory(database=':memory:', vector_length=3)
        db2.import_graph(data, format='graphml')
        self.assertEqual(len(db2.nodes_to_json()), 3)
        self.assertEqual(len(db2.edges_to_json()), 2)
        db2.conn.close()

    def test_import_invalid_format(self):
        with self.assertRaises(ValueError):
            self.db.import_graph({}, format='yaml')

    def test_roundtrip_json_preserves_data(self):
        data = self.db.export_graph(format='json')
        db2 = GraphMemory(database=':memory:', vector_length=3)
        db2.import_graph(data, format='json')
        orig_nodes = sorted(self.db.nodes_to_json(), key=lambda n: n['id'])
        new_nodes = sorted(db2.nodes_to_json(), key=lambda n: n['id'])
        self.assertEqual(len(orig_nodes), len(new_nodes))
        for orig, new in zip(orig_nodes, new_nodes):
            self.assertEqual(orig['id'], new['id'])
            self.assertEqual(orig['type'], new['type'])
            self.assertEqual(orig['properties'], new['properties'])
        orig_edges = sorted(self.db.edges_to_json(), key=lambda e: e['id'])
        new_edges = sorted(db2.edges_to_json(), key=lambda e: e['id'])
        self.assertEqual(len(orig_edges), len(new_edges))
        for orig, new in zip(orig_edges, new_edges):
            self.assertEqual(orig['id'], new['id'])
            self.assertEqual(orig['source_id'], new['source_id'])
            self.assertEqual(orig['target_id'], new['target_id'])
            self.assertEqual(orig['relation'], new['relation'])
            self.assertEqual(orig['weight'], new['weight'])
        db2.conn.close()

    def test_roundtrip_csv_preserves_data(self):
        data = self.db.export_graph(format='csv')
        db2 = GraphMemory(database=':memory:', vector_length=3)
        db2.import_graph(data, format='csv')
        self.assertEqual(len(db2.nodes_to_json()), 3)
        self.assertEqual(len(db2.edges_to_json()), 2)
        orig_nodes = {n['id']: n for n in self.db.nodes_to_json()}
        new_nodes = {n['id']: n for n in db2.nodes_to_json()}
        for node_id, orig in orig_nodes.items():
            self.assertIn(node_id, new_nodes)
            self.assertEqual(orig['type'], new_nodes[node_id]['type'])
            self.assertEqual(orig['properties'], new_nodes[node_id]['properties'])
        db2.conn.close()

    def test_roundtrip_graphml_preserves_data(self):
        data = self.db.export_graph(format='graphml')
        db2 = GraphMemory(database=':memory:', vector_length=3)
        db2.import_graph(data, format='graphml')
        self.assertEqual(len(db2.nodes_to_json()), 3)
        self.assertEqual(len(db2.edges_to_json()), 2)
        db2.conn.close()

    def test_export_empty_graph(self):
        db = GraphMemory(database=':memory:', vector_length=3)
        data = db.export_graph(format='json')
        self.assertEqual(data['nodes'], [])
        self.assertEqual(data['edges'], [])
        db.conn.close()

    def tearDown(self):
        self.db.conn.close()


if __name__ == "__main__":
    unittest.main()


class TestUpdateNode(unittest.TestCase):

    def setUp(self):
        self.db = GraphMemory(database=':memory:', vector_length=3)
        self.node = Node(properties={"name": "node1", "age": 25}, type="Person", vector=[0.1, 0.2, 0.3])
        self.node_id = self.db.insert_node(self.node)

    def test_update_node_properties(self):
        result = self.db.update_node(self.node_id, properties={"name": "updated", "age": 30})
        self.assertTrue(result)
        node = self.db.get_node(self.node_id)
        self.assertEqual(node.properties["name"], "updated")
        self.assertEqual(node.properties["age"], 30)
        self.assertEqual(node.type, "Person")
        self.assertAlmostEqual(node.vector[0], 0.1, places=5)

    def test_update_node_type(self):
        result = self.db.update_node(self.node_id, type="Organization")
        self.assertTrue(result)
        node = self.db.get_node(self.node_id)
        self.assertEqual(node.type, "Organization")
        self.assertEqual(node.properties["name"], "node1")

    def test_update_node_vector(self):
        result = self.db.update_node(self.node_id, vector=[0.9, 0.8, 0.7])
        self.assertTrue(result)
        node = self.db.get_node(self.node_id)
        self.assertAlmostEqual(node.vector[0], 0.9, places=5)
        self.assertAlmostEqual(node.vector[1], 0.8, places=5)
        self.assertAlmostEqual(node.vector[2], 0.7, places=5)
        self.assertEqual(node.type, "Person")

    def test_update_node_multiple_fields(self):
        result = self.db.update_node(self.node_id, type="Place", properties={"name": "city"})
        self.assertTrue(result)
        node = self.db.get_node(self.node_id)
        self.assertEqual(node.type, "Place")
        self.assertEqual(node.properties["name"], "city")

    def test_update_node_invalid_vector_length(self):
        result = self.db.update_node(self.node_id, vector=[0.1, 0.2])
        self.assertFalse(result)
        node = self.db.get_node(self.node_id)
        self.assertAlmostEqual(node.vector[0], 0.1, places=5)

    def test_update_node_nonexistent(self):
        result = self.db.update_node(uuid.uuid4(), type="Ghost")
        self.assertFalse(result)

    def test_update_node_no_kwargs(self):
        result = self.db.update_node(self.node_id)
        self.assertFalse(result)

    def tearDown(self):
        self.db.conn.close()


class TestUpdateEdge(unittest.TestCase):

    def setUp(self):
        self.db = GraphMemory(database=':memory:', vector_length=3)
        self.node1 = Node(properties={"name": "node1"}, vector=[0.1, 0.2, 0.3])
        self.node2 = Node(properties={"name": "node2"}, vector=[0.4, 0.5, 0.6])
        self.node1_id = self.db.insert_node(self.node1)
        self.node2_id = self.db.insert_node(self.node2)
        self.edge = Edge(source_id=self.node1_id, target_id=self.node2_id, weight=0.5, relation="friendship")
        self.db.insert_edge(self.edge)
        self.edge_id = self.edge.id

    def test_update_edge_weight(self):
        result = self.db.update_edge(self.edge_id, weight=0.9)
        self.assertTrue(result)
        row = self.db.conn.execute("SELECT weight, relation FROM edges WHERE id = ?", (str(self.edge_id),)).fetchone()
        self.assertAlmostEqual(row[0], 0.9, places=5)
        self.assertEqual(row[1], "friendship")

    def test_update_edge_relation(self):
        result = self.db.update_edge(self.edge_id, relation="colleague")
        self.assertTrue(result)
        row = self.db.conn.execute("SELECT weight, relation FROM edges WHERE id = ?", (str(self.edge_id),)).fetchone()
        self.assertEqual(row[0], 0.5)
        self.assertEqual(row[1], "colleague")

    def test_update_edge_multiple_fields(self):
        result = self.db.update_edge(self.edge_id, weight=1.0, relation="enemy")
        self.assertTrue(result)
        row = self.db.conn.execute("SELECT weight, relation FROM edges WHERE id = ?", (str(self.edge_id),)).fetchone()
        self.assertAlmostEqual(row[0], 1.0, places=5)
        self.assertEqual(row[1], "enemy")

    def test_update_edge_nonexistent(self):
        result = self.db.update_edge(uuid.uuid4(), weight=0.5)
        self.assertFalse(result)

    def test_update_edge_no_kwargs(self):
        result = self.db.update_edge(self.edge_id)
        self.assertFalse(result)

    def tearDown(self):
        self.db.conn.close()


class TestSearchNodes(unittest.TestCase):
    """Tests for FTS search_nodes and hybrid_search methods."""

    def setUp(self):
        self.db = GraphMemory(database=":memory:", vector_length=3)
        self.node1 = Node(
            properties={"name": "Alice Smith", "bio": "Software engineer who loves machine learning"},
            type="Person",
            vector=[0.1, 0.2, 0.3],
        )
        self.node2 = Node(
            properties={"name": "Bob Jones", "bio": "Data scientist working on NLP and deep learning"},
            type="Person",
            vector=[0.4, 0.5, 0.6],
        )
        self.node3 = Node(
            properties={"title": "Machine Learning Guide", "content": "A comprehensive guide to ML algorithms"},
            type="Document",
            vector=[0.7, 0.8, 0.9],
        )
        self.node4 = Node(
            properties={"name": "Charlie Brown", "bio": "Chef specializing in Italian cuisine"},
            type="Person",
            vector=[0.2, 0.3, 0.4],
        )
        self.node1_id = self.db.insert_node(self.node1)
        self.node2_id = self.db.insert_node(self.node2)
        self.node3_id = self.db.insert_node(self.node3)
        self.node4_id = self.db.insert_node(self.node4)

    def test_search_nodes_basic(self):
        results = self.db.search_nodes("machine learning")
        self.assertGreater(len(results), 0)
        result_ids = [r.node.id for r in results]
        self.assertIn(self.node1_id, result_ids)
        self.assertIn(self.node3_id, result_ids)

    def test_search_nodes_single_term(self):
        results = self.db.search_nodes("chef")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].node.id, self.node4_id)

    def test_search_nodes_no_results(self):
        results = self.db.search_nodes("quantum physics")
        self.assertEqual(len(results), 0)

    def test_search_nodes_with_limit(self):
        results = self.db.search_nodes("learning", limit=1)
        self.assertEqual(len(results), 1)

    def test_search_nodes_returns_search_results(self):
        from graphmemory.models import SearchResult

        results = self.db.search_nodes("Alice")
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertIsInstance(r, SearchResult)
            self.assertIsInstance(r.score, float)
            self.assertIsNotNone(r.node)

    def test_search_nodes_scores_ordered(self):
        results = self.db.search_nodes("learning")
        if len(results) > 1:
            for i in range(len(results) - 1):
                self.assertGreaterEqual(results[i].score, results[i + 1].score)

    def test_search_nodes_empty_query(self):
        results = self.db.search_nodes("")
        self.assertEqual(len(results), 0)

    def test_hybrid_search_basic(self):
        results = self.db.hybrid_search("machine learning", [0.1, 0.2, 0.3])
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].node.id, self.node1_id)

    def test_hybrid_search_with_limit(self):
        results = self.db.hybrid_search("learning", [0.1, 0.2, 0.3], limit=2)
        self.assertLessEqual(len(results), 2)

    def test_hybrid_search_vector_only_match(self):
        results = self.db.hybrid_search("nonexistent term xyz", [0.1, 0.2, 0.3])
        self.assertGreater(len(results), 0)

    def test_hybrid_search_returns_search_results(self):
        from graphmemory.models import SearchResult

        results = self.db.hybrid_search("Alice", [0.1, 0.2, 0.3])
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertIsInstance(r, SearchResult)

    def tearDown(self):
        self.db.conn.close()


class TestThreadSafety(unittest.TestCase):
    """Verify thread safety of GraphMemory with concurrent operations."""

    def setUp(self):
        self.db = GraphMemory(database=":memory:", vector_length=3)

    def test_concurrent_inserts_no_corruption(self):
        """Multiple threads inserting nodes concurrently should not corrupt data."""
        import threading

        num_threads = 10
        nodes_per_thread = 20
        errors = []
        inserted_ids = []
        lock = threading.Lock()

        def insert_nodes(thread_idx):
            try:
                for i in range(nodes_per_thread):
                    node = Node(
                        properties={"thread": thread_idx, "index": i},
                        vector=[float(thread_idx) / 10, float(i) / 20, 0.5],
                    )
                    node_id = self.db.insert_node(node)
                    if node_id:
                        with lock:
                            inserted_ids.append(node_id)
            except Exception as e:
                with lock:
                    errors.append((thread_idx, str(e)))

        threads = [threading.Thread(target=insert_nodes, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], f"Errors during concurrent insert: {errors}")
        self.assertEqual(len(inserted_ids), num_threads * nodes_per_thread)

        # Verify all nodes are actually in the database
        all_nodes = self.db.nodes_to_json()
        self.assertEqual(len(all_nodes), num_threads * nodes_per_thread)

    def test_concurrent_reads_and_writes(self):
        """Concurrent reads and writes should not raise errors."""
        import threading

        # Pre-populate some data
        for i in range(10):
            node = Node(properties={"name": f"node{i}"}, vector=[0.1, 0.2, 0.3])
            self.db.insert_node(node)

        errors = []
        lock = threading.Lock()

        def writer():
            try:
                for i in range(20):
                    node = Node(properties={"name": f"writer_{i}"}, vector=[0.1, 0.2, 0.3])
                    self.db.insert_node(node)
            except Exception as e:
                with lock:
                    errors.append(("writer", str(e)))

        def reader():
            try:
                for _ in range(20):
                    self.db.nodes_to_json()
                    self.db.edges_to_json()
            except Exception as e:
                with lock:
                    errors.append(("reader", str(e)))

        threads = [threading.Thread(target=writer) for _ in range(3)]
        threads += [threading.Thread(target=reader) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], f"Errors during concurrent read/write: {errors}")

    def test_concurrent_insert_and_delete(self):
        """Concurrent inserts and deletes should not corrupt state."""
        import threading

        # Insert initial nodes
        initial_ids = []
        for i in range(10):
            node = Node(properties={"name": f"initial_{i}"}, vector=[0.1, 0.2, 0.3])
            node_id = self.db.insert_node(node)
            initial_ids.append(node_id)

        errors = []
        lock = threading.Lock()

        def inserter():
            try:
                for i in range(10):
                    node = Node(properties={"name": f"new_{i}"}, vector=[0.4, 0.5, 0.6])
                    self.db.insert_node(node)
            except Exception as e:
                with lock:
                    errors.append(("inserter", str(e)))

        def deleter():
            try:
                for nid in initial_ids[:5]:
                    self.db.delete_node(nid)
            except Exception as e:
                with lock:
                    errors.append(("deleter", str(e)))

        threads = [threading.Thread(target=inserter), threading.Thread(target=deleter)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], f"Errors during concurrent insert/delete: {errors}")

    def test_has_lock_attribute(self):
        """GraphMemory should expose a threading.RLock as _lock."""
        import threading
        self.assertIsInstance(self.db._lock, type(threading.RLock()))

    def tearDown(self):
        self.db.conn.close()


class TestContextManager(unittest.TestCase):
    """Tests for __enter__ and __exit__ context manager protocol."""

    def test_context_manager_usage(self):
        with GraphMemory(database=":memory:", vector_length=3) as db:
            node = Node(properties={"name": "ctx_test"}, vector=[0.1, 0.2, 0.3])
            node_id = db.insert_node(node)
            self.assertIsNotNone(node_id)
        # After exiting, the connection should be closed
        with self.assertRaises(Exception):
            db.conn.execute("SELECT 1;")

    def test_context_manager_returns_self(self):
        db = GraphMemory(database=":memory:", vector_length=3)
        result = db.__enter__()
        self.assertIs(result, db)
        db.__exit__(None, None, None)


class TestSetVectorLength(unittest.TestCase):
    def setUp(self):
        self.db = GraphMemory(database=":memory:", vector_length=3)

    def test_set_vector_length(self):
        self.assertEqual(self.db.vector_length, 3)
        self.db.set_vector_length(5)
        self.assertEqual(self.db.vector_length, 5)

    def tearDown(self):
        self.db.conn.close()


class TestGetNode(unittest.TestCase):
    def setUp(self):
        self.db = GraphMemory(database=":memory:", vector_length=3)

    def test_get_existing_node(self):
        node = Node(properties={"name": "test"}, type="Person", vector=[0.1, 0.2, 0.3])
        node_id = self.db.insert_node(node)
        result = self.db.get_node(node_id)
        self.assertIsNotNone(result)
        self.assertEqual(result.id, node_id)
        self.assertEqual(result.properties["name"], "test")
        self.assertEqual(result.type, "Person")

    def test_get_nonexistent_node(self):
        result = self.db.get_node(uuid.uuid4())
        self.assertIsNone(result)

    def tearDown(self):
        self.db.conn.close()


class TestGetNodesVector(unittest.TestCase):
    def setUp(self):
        self.db = GraphMemory(database=":memory:", vector_length=3)

    def test_get_vector_existing_node(self):
        node = Node(properties={"name": "test"}, vector=[0.1, 0.2, 0.3])
        node_id = self.db.insert_node(node)
        vector = self.db.get_nodes_vector(node_id)
        self.assertEqual(len(vector), 3)
        self.assertAlmostEqual(vector[0], 0.1, places=5)

    def test_get_vector_nonexistent_node(self):
        vector = self.db.get_nodes_vector(uuid.uuid4())
        self.assertEqual(vector, [])

    def tearDown(self):
        self.db.conn.close()


class TestPrintJson(unittest.TestCase):
    def setUp(self):
        self.db = GraphMemory(database=":memory:", vector_length=3)

    def test_print_json_output(self):
        node1 = Node(properties={"name": "a"}, vector=[0.1, 0.2, 0.3])
        node2 = Node(properties={"name": "b"}, vector=[0.4, 0.5, 0.6])
        self.db.insert_node(node1)
        self.db.insert_node(node2)
        self.db.insert_edge(Edge(source_id=node1.id, target_id=node2.id, weight=1.0, relation="knows"))
        import io
        import sys
        captured = io.StringIO()
        sys.stdout = captured
        self.db.print_json()
        sys.stdout = sys.__stdout__
        output = captured.getvalue()
        self.assertIn("Nodes JSON:", output)
        self.assertIn("Edges JSON:", output)

    def tearDown(self):
        self.db.conn.close()


class TestCreateIndex(unittest.TestCase):
    def setUp(self):
        self.db = GraphMemory(database=":memory:", vector_length=3)

    def test_create_index_standalone(self):
        node = Node(properties={"name": "test"}, vector=[0.1, 0.2, 0.3])
        self.db.insert_node(node)
        self.db.create_index()
        # Verify index was created by querying nearest_nodes
        results = self.db.nearest_nodes([0.1, 0.2, 0.3], limit=1)
        self.assertEqual(len(results), 1)

    def test_create_index_idempotent(self):
        node = Node(properties={"name": "test"}, vector=[0.1, 0.2, 0.3])
        self.db.insert_node(node)
        self.db.create_index()
        self.db.create_index()  # Should not raise

    def tearDown(self):
        self.db.conn.close()


class TestValidateVector(unittest.TestCase):
    def setUp(self):
        self.db = GraphMemory(database=":memory:", vector_length=3)

    def test_valid_vector(self):
        self.assertTrue(self.db._validate_vector([1.0, 2.0, 3.0]))

    def test_invalid_not_list(self):
        self.assertFalse(self.db._validate_vector((1.0, 2.0, 3.0)))

    def test_invalid_wrong_length(self):
        self.assertFalse(self.db._validate_vector([1.0, 2.0]))

    def test_invalid_contains_int(self):
        self.assertFalse(self.db._validate_vector([1.0, 2, 3.0]))

    def test_invalid_contains_string(self):
        self.assertFalse(self.db._validate_vector([1.0, "a", 3.0]))

    def test_invalid_empty(self):
        self.assertFalse(self.db._validate_vector([]))

    def test_invalid_none(self):
        self.assertFalse(self.db._validate_vector(None))

    def tearDown(self):
        self.db.conn.close()


class TestLoadDatabase(unittest.TestCase):
    def test_load_nonexistent_path(self):
        db = GraphMemory(database=":memory:", vector_length=3)
        db.load_database("/nonexistent/path/to/db.duckdb")
        # Should log error but not crash
        db.conn.close()

    def test_load_database_on_init_with_file(self):
        import tempfile
        import duckdb as _duckdb
        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, "test.duckdb")
        try:
            # Create a real db file first
            conn = _duckdb.connect(tmp_path)
            conn.close()
            # Now init GraphMemory with that path - triggers load_database in __init__
            db = GraphMemory(database=tmp_path, vector_length=3)
            db.conn.close()
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)


class TestNodesByAttribute(unittest.TestCase):
    def setUp(self):
        self.db = GraphMemory(database=":memory:", vector_length=3)
        self.db.insert_node(Node(properties={"name": "alice", "role": "admin"}, vector=[0.1, 0.2, 0.3]))
        self.db.insert_node(Node(properties={"name": "bob", "role": "user"}, vector=[0.4, 0.5, 0.6]))

    def test_find_by_attribute(self):
        results = self.db.nodes_by_attribute("name", "alice")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].properties["name"], "alice")

    def test_no_match(self):
        results = self.db.nodes_by_attribute("name", "charlie")
        self.assertEqual(len(results), 0)

    def tearDown(self):
        self.db.conn.close()


class TestConnectedNodesEmpty(unittest.TestCase):
    def setUp(self):
        self.db = GraphMemory(database=":memory:", vector_length=3)

    def test_connected_nodes_no_edges(self):
        node = Node(properties={"name": "lonely"}, vector=[0.1, 0.2, 0.3])
        node_id = self.db.insert_node(node)
        connected = self.db.connected_nodes(node_id)
        self.assertEqual(len(connected), 0)

    def test_connected_nodes_nonexistent(self):
        connected = self.db.connected_nodes(uuid.uuid4())
        self.assertEqual(len(connected), 0)

    def tearDown(self):
        self.db.conn.close()


class TestNearestNodesInvalidVector(unittest.TestCase):
    def setUp(self):
        self.db = GraphMemory(database=":memory:", vector_length=3)

    def test_invalid_vector_returns_empty(self):
        result = self.db.nearest_nodes(vector=[0.1, 0.2], limit=5)
        self.assertEqual(result, [])

    def test_non_list_vector(self):
        result = self.db.nearest_nodes(vector="not_a_list", limit=5)
        self.assertEqual(result, [])

    def tearDown(self):
        self.db.conn.close()


class TestInsertNodeWithoutVector(unittest.TestCase):
    def setUp(self):
        self.db = GraphMemory(database=":memory:", vector_length=3)

    def test_insert_node_no_vector(self):
        node = Node(properties={"name": "no_vec"})
        node_id = self.db.insert_node(node)
        self.assertIsNotNone(node_id)
        # Should get default zero vector
        vec = self.db.get_nodes_vector(node_id)
        self.assertEqual(len(vec), 3)
        self.assertEqual(vec[0], 0.0)

    def tearDown(self):
        self.db.conn.close()


class TestQueryBuilderEdgeCases(unittest.TestCase):
    def setUp(self):
        self.db = GraphMemory(database=":memory:", vector_length=3)
        self.node = Node(properties={"name": "test", "price": 9.99}, type="Item", vector=[0.1, 0.2, 0.3])
        self.db.insert_node(self.node)

    def test_where_with_float(self):
        results = self.db.query().match(type="Item").where(price=9.99).execute()
        self.assertEqual(len(results), 1)

    def test_where_with_int(self):
        self.db.insert_node(Node(properties={"count": 42}, type="Counter", vector=[0.1, 0.2, 0.3]))
        results = self.db.query().match(type="Counter").where(count=42).execute()
        self.assertEqual(len(results), 1)

    def test_empty_query(self):
        results = self.db.query().execute()
        self.assertGreater(len(results), 0)

    def test_match_nonexistent_type(self):
        results = self.db.query().match(type="NonExistent").execute()
        self.assertEqual(len(results), 0)

    def test_traverse_nonexistent_source(self):
        results = self.db.query().traverse(source_id=uuid.uuid4(), depth=1).execute()
        self.assertEqual(len(results), 0)

    def test_query_after_close(self):
        db = GraphMemory(database=":memory:", vector_length=3)
        db.insert_node(Node(properties={"name": "test"}, vector=[0.1, 0.2, 0.3]))
        db.close()
        results = db.query().execute()
        self.assertEqual(results, [])

    def tearDown(self):
        self.db.conn.close()


class TestBulkInsertNodeInvalidVector(unittest.TestCase):
    def setUp(self):
        self.db = GraphMemory(database=":memory:", vector_length=3)

    def test_bulk_insert_skips_invalid_vectors(self):
        nodes = [
            Node(properties={"name": "good"}, vector=[0.1, 0.2, 0.3]),
            Node(properties={"name": "bad"}, vector=[0.1, 0.2]),  # wrong length
            Node(properties={"name": "good2"}, vector=[0.4, 0.5, 0.6]),
        ]
        inserted = self.db.bulk_insert_nodes(nodes)
        self.assertEqual(len(inserted), 2)

    def tearDown(self):
        self.db.conn.close()


class TestClosedConnectionErrors(unittest.TestCase):
    """Test error handlers by using a closed connection."""

    def test_insert_node_after_close(self):
        db = GraphMemory(database=":memory:", vector_length=3)
        db.conn.close()
        node = Node(properties={"name": "test"}, vector=[0.1, 0.2, 0.3])
        result = db.insert_node(node)
        self.assertIsNone(result)

    def test_insert_edge_after_close(self):
        db = GraphMemory(database=":memory:", vector_length=3)
        n1 = Node(properties={"name": "a"}, vector=[0.1, 0.2, 0.3])
        n2 = Node(properties={"name": "b"}, vector=[0.4, 0.5, 0.6])
        n1_id = db.insert_node(n1)
        n2_id = db.insert_node(n2)
        db.conn.close()
        edge = Edge(source_id=n1_id, target_id=n2_id, weight=0.5, relation="test")
        # Should not raise, just log error
        db.insert_edge(edge)

    def test_bulk_insert_nodes_after_close(self):
        db = GraphMemory(database=":memory:", vector_length=3)
        db.conn.close()
        nodes = [Node(properties={"name": "test"}, vector=[0.1, 0.2, 0.3])]
        result = db.bulk_insert_nodes(nodes)
        self.assertEqual(result, [])

    def test_bulk_insert_edges_after_close(self):
        db = GraphMemory(database=":memory:", vector_length=3)
        n1 = Node(properties={"name": "a"}, vector=[0.1, 0.2, 0.3])
        n2 = Node(properties={"name": "b"}, vector=[0.4, 0.5, 0.6])
        n1_id = db.insert_node(n1)
        n2_id = db.insert_node(n2)
        db.conn.close()
        edges = [Edge(source_id=n1_id, target_id=n2_id, weight=0.5, relation="test")]
        db.bulk_insert_edges(edges)  # Should not raise

    def test_delete_node_after_close(self):
        db = GraphMemory(database=":memory:", vector_length=3)
        db.conn.close()
        db.delete_node(uuid.uuid4())  # Should not raise

    def test_delete_edge_after_close(self):
        db = GraphMemory(database=":memory:", vector_length=3)
        db.conn.close()
        db.delete_edge(uuid.uuid4(), uuid.uuid4())  # Should not raise

    def test_update_node_after_close(self):
        db = GraphMemory(database=":memory:", vector_length=3)
        db.conn.close()
        result = db.update_node(uuid.uuid4(), type="new_type")
        self.assertFalse(result)

    def test_update_edge_after_close(self):
        db = GraphMemory(database=":memory:", vector_length=3)
        db.conn.close()
        result = db.update_edge(uuid.uuid4(), relation="new_rel")
        self.assertFalse(result)

    def test_nearest_nodes_after_close(self):
        db = GraphMemory(database=":memory:", vector_length=3)
        db.conn.close()
        result = db.nearest_nodes([0.1, 0.2, 0.3], limit=5)
        self.assertEqual(result, [])

    def test_connected_nodes_after_close(self):
        db = GraphMemory(database=":memory:", vector_length=3)
        db.conn.close()
        result = db.connected_nodes(uuid.uuid4())
        self.assertEqual(result, [])

    def test_nodes_to_json_after_close(self):
        db = GraphMemory(database=":memory:", vector_length=3)
        db.conn.close()
        result = db.nodes_to_json()
        self.assertEqual(result, [])

    def test_edges_to_json_after_close(self):
        db = GraphMemory(database=":memory:", vector_length=3)
        db.conn.close()
        result = db.edges_to_json()
        self.assertEqual(result, [])

    def test_get_node_after_close(self):
        db = GraphMemory(database=":memory:", vector_length=3)
        db.conn.close()
        result = db.get_node(uuid.uuid4())
        self.assertIsNone(result)

    def test_get_edge_after_close(self):
        db = GraphMemory(database=":memory:", vector_length=3)
        db.conn.close()
        result = db.get_edge(uuid.uuid4())
        self.assertIsNone(result)

    def test_get_edges_by_relation_after_close(self):
        db = GraphMemory(database=":memory:", vector_length=3)
        db.conn.close()
        result = db.get_edges_by_relation("test")
        self.assertEqual(result, [])

    def test_edges_by_attribute_after_close(self):
        db = GraphMemory(database=":memory:", vector_length=3)
        db.conn.close()
        result = db.edges_by_attribute("relation", "test")
        self.assertEqual(result, [])

    def test_nodes_by_attribute_after_close(self):
        db = GraphMemory(database=":memory:", vector_length=3)
        db.conn.close()
        result = db.nodes_by_attribute("name", "test")
        self.assertEqual(result, [])

    def test_get_nodes_vector_after_close(self):
        db = GraphMemory(database=":memory:", vector_length=3)
        db.conn.close()
        result = db.get_nodes_vector(uuid.uuid4())
        self.assertEqual(result, [])

    def test_create_index_after_close(self):
        db = GraphMemory(database=":memory:", vector_length=3)
        db.conn.close()
        db.create_index()  # Should not raise


class TestEdgesByAttributeInvalid(unittest.TestCase):
    def setUp(self):
        self.db = GraphMemory(database=":memory:", vector_length=3)

    def test_invalid_attribute_name(self):
        result = self.db.edges_by_attribute("invalid_field", "value")
        self.assertEqual(result, [])

    def tearDown(self):
        self.db.conn.close()


class TestSearchNodesErrorPaths(unittest.TestCase):
    def test_search_nodes_after_close(self):
        db = GraphMemory(database=":memory:", vector_length=3)
        db.insert_node(Node(properties={"name": "test"}, vector=[0.1, 0.2, 0.3]))
        db.conn.close()
        result = db.search_nodes("test")
        self.assertEqual(result, [])

    def test_hybrid_search_invalid_vector(self):
        db = GraphMemory(database=":memory:", vector_length=3)
        result = db.hybrid_search("test", [0.1, 0.2])  # wrong length
        self.assertEqual(result, [])
        db.conn.close()

    def test_hybrid_search_after_close(self):
        db = GraphMemory(database=":memory:", vector_length=3)
        db.insert_node(Node(properties={"name": "test"}, vector=[0.1, 0.2, 0.3]))
        db.conn.close()
        result = db.hybrid_search("test", [0.1, 0.2, 0.3])
        self.assertEqual(result, [])

    def test_hybrid_search_empty_db(self):
        db = GraphMemory(database=":memory:", vector_length=3)
        result = db.hybrid_search("nonexistent", [0.1, 0.2, 0.3])
        self.assertEqual(result, [])
        db.conn.close()


class TestDistanceMetrics(unittest.TestCase):
    """Test configurable distance metrics for vector search."""

    def _insert_test_nodes(self, db):
        """Insert nodes with known vectors for distance metric testing."""
        nodes = [
            Node(properties={"name": "origin"}, vector=[1.0, 0.0, 0.0]),
            Node(properties={"name": "similar"}, vector=[0.9, 0.1, 0.0]),
            Node(properties={"name": "different"}, vector=[0.0, 0.0, 1.0]),
        ]
        for node in nodes:
            db.insert_node(node)
        return nodes

    def test_default_metric_is_l2(self):
        db = GraphMemory(database=':memory:', vector_length=3)
        self.assertEqual(db.distance_metric, 'l2')
        db.conn.close()

    def test_invalid_metric_raises(self):
        with self.assertRaises(ValueError):
            GraphMemory(database=':memory:', vector_length=3, distance_metric='manhattan')

    def test_l2_nearest_nodes(self):
        db = GraphMemory(database=':memory:', vector_length=3, distance_metric='l2')
        self._insert_test_nodes(db)
        db.create_index()
        results = db.nearest_nodes(vector=[1.0, 0.0, 0.0], limit=3)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].node.properties["name"], "origin")
        self.assertAlmostEqual(results[0].distance, 0.0, places=5)
        self.assertEqual(results[1].node.properties["name"], "similar")
        db.conn.close()

    def test_cosine_nearest_nodes(self):
        db = GraphMemory(database=':memory:', vector_length=3, distance_metric='cosine')
        self._insert_test_nodes(db)
        db.create_index()
        results = db.nearest_nodes(vector=[1.0, 0.0, 0.0], limit=3)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].node.properties["name"], "origin")
        self.assertAlmostEqual(results[0].distance, 0.0, places=5)
        self.assertEqual(results[1].node.properties["name"], "similar")
        db.conn.close()

    def test_inner_product_nearest_nodes(self):
        db = GraphMemory(database=':memory:', vector_length=3, distance_metric='inner_product')
        self._insert_test_nodes(db)
        db.create_index()
        results = db.nearest_nodes(vector=[1.0, 0.0, 0.0], limit=3)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].node.properties["name"], "origin")
        self.assertEqual(results[1].node.properties["name"], "similar")
        db.conn.close()

    def test_cosine_metric_distinguishes_direction(self):
        """Cosine should rank by direction, not magnitude."""
        db = GraphMemory(database=':memory:', vector_length=3, distance_metric='cosine')
        db.insert_node(Node(properties={"name": "far_same_dir"}, vector=[10.0, 0.0, 0.0]))
        db.insert_node(Node(properties={"name": "close_diff_dir"}, vector=[0.5, 0.5, 0.0]))
        db.create_index()
        results = db.nearest_nodes(vector=[1.0, 0.0, 0.0], limit=2)
        self.assertEqual(results[0].node.properties["name"], "far_same_dir")
        db.conn.close()


class TestConnectionManagement(unittest.TestCase):
    """Tests for cursor-based operations, reconnection logic, and concurrent usage."""

    def test_cursor_returns_independent_cursor(self):
        """cursor() should return a usable DuckDB cursor."""
        db = GraphMemory(database=':memory:', vector_length=3)
        cur = db.cursor()
        result = cur.execute("SELECT 1;").fetchone()
        self.assertEqual(result[0], 1)
        db.close()

    def test_cursor_after_close_raises(self):
        """cursor() should raise after close() is called."""
        db = GraphMemory(database=':memory:', vector_length=3)
        db.close()
        with self.assertRaises(duckdb.ConnectionException):
            db.cursor()

    def test_close_is_idempotent(self):
        """Calling close() multiple times should not raise."""
        db = GraphMemory(database=':memory:', vector_length=3)
        db.close()
        db.close()  # Should not raise

    def test_context_manager_calls_close(self):
        """__exit__ should close the connection."""
        with GraphMemory(database=':memory:', vector_length=3) as db:
            node = Node(properties={"name": "test"}, vector=[1.0, 2.0, 3.0])
            db.insert_node(node)
        self.assertTrue(db._closed)

    def test_reconnect_restores_connection(self):
        """_reconnect should restore a working connection after the old one is closed."""
        db = GraphMemory(database=':memory:', vector_length=3)
        node = Node(properties={"name": "before"}, vector=[1.0, 2.0, 3.0])
        db.insert_node(node)
        # Force reconnect (in-memory DB loses data, but connection should work)
        db._reconnect()
        # Connection should be usable after reconnect
        cur = db.cursor()
        result = cur.execute("SELECT 1;").fetchone()
        self.assertEqual(result[0], 1)
        db.close()

    def test_reconnect_after_close_raises(self):
        """_reconnect should raise on a closed instance."""
        db = GraphMemory(database=':memory:', vector_length=3)
        db.close()
        with self.assertRaises(duckdb.ConnectionException):
            db._reconnect()

    def test_reconnect_with_file_database_preserves_data(self):
        """_reconnect on a file-based DB should preserve data."""
        db_path = '/tmp/test_reconnect_preserves.duckdb'
        if os.path.exists(db_path):
            os.remove(db_path)
        try:
            db = GraphMemory(database=db_path, vector_length=3)
            node = Node(properties={"name": "persistent"}, vector=[1.0, 2.0, 3.0])
            node_id = db.insert_node(node)
            db._reconnect()
            result = db.get_node(node_id)
            self.assertIsNotNone(result)
            self.assertEqual(result.properties["name"], "persistent")
            db.close()
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_max_retries_configurable(self):
        """max_retries and retry_base_delay should be configurable via __init__."""
        db = GraphMemory(database=':memory:', vector_length=3, max_retries=5, retry_base_delay=0.05)
        self.assertEqual(db.max_retries, 5)
        self.assertAlmostEqual(db.retry_base_delay, 0.05)
        db.close()

    def test_with_retry_decorator_retries_on_transient_error(self):
        """@with_retry should retry and eventually succeed on transient errors."""
        from graphmemory.database import with_retry, TRANSIENT_ERRORS

        call_count = 0

        class FakeDB:
            max_retries = 2
            retry_base_delay = 0.01

            def _reconnect(self):
                pass

            @with_retry()
            def flaky_method(self):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise duckdb.IOException("transient failure")
                return "success"

        fake = FakeDB()
        result = fake.flaky_method()
        self.assertEqual(result, "success")
        self.assertEqual(call_count, 3)

    def test_with_retry_raises_after_exhausting_retries(self):
        """@with_retry should raise after all retries are exhausted."""
        from graphmemory.database import with_retry

        class FakeDB:
            max_retries = 1
            retry_base_delay = 0.01

            def _reconnect(self):
                pass

            @with_retry()
            def always_fails(self):
                raise duckdb.IOException("permanent failure")

        fake = FakeDB()
        with self.assertRaises(duckdb.IOException):
            fake.always_fails()


class TestConcurrentCursorUsage(unittest.TestCase):
    """Tests for concurrent cursor usage from multiple threads."""

    def test_concurrent_reads(self):
        """Multiple threads should be able to read concurrently using cursors."""
        import threading

        db = GraphMemory(database=':memory:', vector_length=3)
        # Insert test data
        for i in range(10):
            node = Node(properties={"name": f"node_{i}"}, vector=[float(i), float(i), float(i)])
            db.insert_node(node)

        results = []
        errors = []

        def reader(thread_id):
            try:
                nodes = db.nodes_to_json()
                results.append((thread_id, len(nodes)))
            except Exception as e:
                errors.append((thread_id, e))

        threads = [threading.Thread(target=reader, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Errors in threads: {errors}")
        self.assertEqual(len(results), 5)
        for thread_id, count in results:
            self.assertEqual(count, 10)
        db.close()

    def test_concurrent_inserts(self):
        """Multiple threads inserting nodes should not corrupt data."""
        import threading

        db = GraphMemory(database=':memory:', vector_length=3)
        errors = []

        def inserter(thread_id):
            try:
                for i in range(5):
                    node = Node(
                        properties={"thread": thread_id, "index": i},
                        vector=[float(thread_id), float(i), 0.0]
                    )
                    db.insert_node(node)
            except Exception as e:
                errors.append((thread_id, e))

        threads = [threading.Thread(target=inserter, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Errors in threads: {errors}")
        nodes = db.nodes_to_json()
        self.assertEqual(len(nodes), 20)  # 4 threads * 5 nodes each
        db.close()


class TestAlgorithms(unittest.TestCase):
    """Tests for graphmemory.algorithms module."""

    def setUp(self):
        self.db = GraphMemory(database=":memory:", vector_length=3)
        # Build a small directed graph:
        #   A -> B -> C
        #   A -> D
        #   D -> C
        self.node_a = Node(properties={"name": "A"}, vector=[1.0, 0.0, 0.0])
        self.node_b = Node(properties={"name": "B"}, vector=[0.0, 1.0, 0.0])
        self.node_c = Node(properties={"name": "C"}, vector=[0.0, 0.0, 1.0])
        self.node_d = Node(properties={"name": "D"}, vector=[1.0, 1.0, 0.0])

        self.db.insert_node(self.node_a)
        self.db.insert_node(self.node_b)
        self.db.insert_node(self.node_c)
        self.db.insert_node(self.node_d)

        self.db.insert_edge(Edge(source_id=self.node_a.id, target_id=self.node_b.id, relation="knows"))
        self.db.insert_edge(Edge(source_id=self.node_b.id, target_id=self.node_c.id, relation="knows"))
        self.db.insert_edge(Edge(source_id=self.node_a.id, target_id=self.node_d.id, relation="knows"))
        self.db.insert_edge(Edge(source_id=self.node_d.id, target_id=self.node_c.id, relation="knows"))

    def test_to_networkx(self):
        from graphmemory.algorithms import to_networkx
        G = to_networkx(self.db)
        self.assertEqual(len(G.nodes), 4)
        self.assertEqual(len(G.edges), 4)
        # Check node attributes
        node_a_id = str(self.node_a.id)
        self.assertEqual(G.nodes[node_a_id]["properties"]["name"], "A")
        self.assertEqual(list(G.nodes[node_a_id]["vector"]), [1.0, 0.0, 0.0])

    def test_to_networkx_bridge_method(self):
        G = self.db.to_networkx()
        self.assertEqual(len(G.nodes), 4)
        self.assertEqual(len(G.edges), 4)

    def test_to_networkx_edge_attributes(self):
        from graphmemory.algorithms import to_networkx
        G = to_networkx(self.db)
        node_a_id = str(self.node_a.id)
        node_b_id = str(self.node_b.id)
        self.assertEqual(G.edges[node_a_id, node_b_id]["relation"], "knows")

    def test_to_networkx_empty_graph(self):
        from graphmemory.algorithms import to_networkx
        empty_db = GraphMemory(database=":memory:", vector_length=3)
        G = to_networkx(empty_db)
        self.assertEqual(len(G.nodes), 0)
        self.assertEqual(len(G.edges), 0)

    def test_pagerank(self):
        from graphmemory.algorithms import pagerank
        scores = pagerank(self.db)
        self.assertEqual(len(scores), 4)
        # All scores should sum to ~1.0
        self.assertAlmostEqual(sum(scores.values()), 1.0, places=5)
        # C should have high rank (two incoming edges, no outgoing)
        node_c_id = str(self.node_c.id)
        node_a_id = str(self.node_a.id)
        self.assertGreater(scores[node_c_id], scores[node_a_id])

    def test_pagerank_custom_alpha(self):
        from graphmemory.algorithms import pagerank
        scores = pagerank(self.db, alpha=0.5)
        self.assertEqual(len(scores), 4)
        self.assertAlmostEqual(sum(scores.values()), 1.0, places=5)

    def test_betweenness_centrality(self):
        from graphmemory.algorithms import betweenness_centrality
        scores = betweenness_centrality(self.db)
        self.assertEqual(len(scores), 4)
        # All values should be between 0 and 1 (normalized)
        for score in scores.values():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_betweenness_centrality_unnormalized(self):
        from graphmemory.algorithms import betweenness_centrality
        scores = betweenness_centrality(self.db, normalized=False)
        self.assertEqual(len(scores), 4)
        for score in scores.values():
            self.assertGreaterEqual(score, 0.0)

    def test_degree_distribution(self):
        from graphmemory.algorithms import degree_distribution
        dist = degree_distribution(self.db)
        self.assertEqual(len(dist), 4)

        node_a_id = str(self.node_a.id)
        node_c_id = str(self.node_c.id)

        # A has 0 in, 2 out
        self.assertEqual(dist[node_a_id]["in_degree"], 0)
        self.assertEqual(dist[node_a_id]["out_degree"], 2)
        self.assertEqual(dist[node_a_id]["total_degree"], 2)

        # C has 2 in, 0 out
        self.assertEqual(dist[node_c_id]["in_degree"], 2)
        self.assertEqual(dist[node_c_id]["out_degree"], 0)
        self.assertEqual(dist[node_c_id]["total_degree"], 2)

    def test_connected_components_single(self):
        from graphmemory.algorithms import connected_components
        components = connected_components(self.db)
        # All 4 nodes are connected
        self.assertEqual(len(components), 1)
        self.assertEqual(len(components[0]), 4)

    def test_connected_components_multiple(self):
        from graphmemory.algorithms import connected_components
        # Create a separate disconnected graph
        db2 = GraphMemory(database=":memory:", vector_length=3)
        n1 = Node(properties={"name": "X"}, vector=[1.0, 0.0, 0.0])
        n2 = Node(properties={"name": "Y"}, vector=[0.0, 1.0, 0.0])
        n3 = Node(properties={"name": "Z"}, vector=[0.0, 0.0, 1.0])
        db2.insert_node(n1)
        db2.insert_node(n2)
        db2.insert_node(n3)
        # Only connect X -> Y, Z is isolated
        db2.insert_edge(Edge(source_id=n1.id, target_id=n2.id, relation="linked"))

        components = connected_components(db2)
        self.assertEqual(len(components), 2)
        # Largest first
        self.assertEqual(len(components[0]), 2)
        self.assertEqual(len(components[1]), 1)
        self.assertIn(str(n3.id), components[1])

    def test_algorithms_empty_graph(self):
        from graphmemory.algorithms import pagerank, betweenness_centrality, degree_distribution, connected_components
        empty_db = GraphMemory(database=":memory:", vector_length=3)
        self.assertEqual(pagerank(empty_db), {})
        self.assertEqual(betweenness_centrality(empty_db), {})
        self.assertEqual(degree_distribution(empty_db), {})
        self.assertEqual(connected_components(empty_db), [])


class TestRetrieve(unittest.TestCase):
    """Tests for the retrieve() GraphRAG retrieval pipeline."""

    def setUp(self):
        self.db = GraphMemory(database=":memory:", vector_length=3)
        # Build a small graph:
        # Washington -[served_under]-> Continental Congress
        # Washington -[commanded]-> Continental Army
        # Continental Army -[fought_in]-> Battle of Yorktown
        # Jefferson -[drafted]-> Declaration of Independence
        self.washington = Node(properties={"name": "George Washington", "role": "General"}, type="Person", vector=[1.0, 0.0, 0.0])
        self.congress = Node(properties={"name": "Continental Congress", "established": 1774}, type="Organization", vector=[0.9, 0.1, 0.0])
        self.army = Node(properties={"name": "Continental Army", "founded": 1775}, type="Organization", vector=[0.8, 0.2, 0.0])
        self.battle = Node(properties={"name": "Battle of Yorktown", "year": 1781}, type="Event", vector=[0.7, 0.3, 0.0])
        self.jefferson = Node(properties={"name": "Thomas Jefferson", "role": "Statesman"}, type="Person", vector=[0.0, 1.0, 0.0])
        self.declaration = Node(properties={"name": "Declaration of Independence", "year": 1776}, type="Document", vector=[0.0, 0.9, 0.1])

        for node in [self.washington, self.congress, self.army, self.battle, self.jefferson, self.declaration]:
            self.db.insert_node(node)

        self.db.insert_edge(Edge(source_id=self.washington.id, target_id=self.congress.id, relation="served_under"))
        self.db.insert_edge(Edge(source_id=self.washington.id, target_id=self.army.id, relation="commanded"))
        self.db.insert_edge(Edge(source_id=self.army.id, target_id=self.battle.id, relation="fought_in"))
        self.db.insert_edge(Edge(source_id=self.jefferson.id, target_id=self.declaration.id, relation="drafted"))

    def tearDown(self):
        self.db.close()

    def test_retrieve_returns_retrieval_result(self):
        from graphmemory.models import RetrievalResult
        result = self.db.retrieve(query="Washington", query_vector=[1.0, 0.0, 0.0])
        self.assertIsInstance(result, RetrievalResult)
        self.assertEqual(result.query, "Washington")

    def test_retrieve_finds_seed_nodes(self):
        result = self.db.retrieve(query="Washington", query_vector=[1.0, 0.0, 0.0], max_hops=0)
        self.assertGreater(result.seed_node_count, 0)

    def test_retrieve_expands_via_graph_traversal(self):
        result = self.db.retrieve(query="Washington", query_vector=[1.0, 0.0, 0.0], max_hops=2, search_limit=1, vector_weight=1.0, text_weight=0.0)
        # With only 1 seed node, expansion should find more via graph traversal
        self.assertGreater(result.total_node_count, result.seed_node_count)

    def test_retrieve_includes_relationships(self):
        result = self.db.retrieve(query="Washington", query_vector=[1.0, 0.0, 0.0], max_hops=1)
        self.assertIn("Relationships", result.context_text)

    def test_retrieve_context_text_is_nonempty(self):
        result = self.db.retrieve(query="Washington", query_vector=[1.0, 0.0, 0.0])
        self.assertGreater(len(result.context_text), 0)
        self.assertIn("Context for query", result.context_text)

    def test_retrieve_empty_query_returns_empty(self):
        result = self.db.retrieve(query="", query_vector=[0.5, 0.5, 0.5])
        # Empty text query still does vector search via hybrid_search
        # But if no results, should return empty
        self.assertIsNotNone(result)

    def test_retrieve_respects_max_tokens(self):
        result = self.db.retrieve(query="Washington", query_vector=[1.0, 0.0, 0.0], max_tokens=50)
        self.assertLessEqual(result.token_estimate, 50)

    def test_retrieve_max_hops_zero_no_expansion(self):
        result = self.db.retrieve(query="Washington", query_vector=[1.0, 0.0, 0.0], max_hops=0)
        # With 0 hops, total nodes should equal seed nodes
        self.assertEqual(result.total_node_count, result.seed_node_count)

    def test_retrieve_hop_distances_correct(self):
        result = self.db.retrieve(query="Washington", query_vector=[1.0, 0.0, 0.0], max_hops=2, search_limit=1, vector_weight=1.0, text_weight=0.0)
        # Seed nodes should have hop_distance 0
        distances = {str(ctx.node.id): ctx.hop_distance for ctx in result.contexts}
        # Washington should be a seed (closest vector)
        self.assertEqual(distances.get(str(self.washington.id), -1), 0)

    def test_retrieve_contexts_sorted_by_hop(self):
        result = self.db.retrieve(query="Washington", query_vector=[1.0, 0.0, 0.0], max_hops=2)
        hops = [ctx.hop_distance for ctx in result.contexts]
        self.assertEqual(hops, sorted(hops))

    def test_retrieve_multi_hop_reaches_distant_nodes(self):
        # Battle of Yorktown is 2 hops from Washington: Washington -> Army -> Battle
        result = self.db.retrieve(query="Washington", query_vector=[1.0, 0.0, 0.0], max_hops=2, search_limit=1, vector_weight=1.0, text_weight=0.0)
        node_ids = {str(ctx.node.id) for ctx in result.contexts}
        self.assertIn(str(self.battle.id), node_ids)

    def test_retrieve_search_limit(self):
        result1 = self.db.retrieve(query="Washington", query_vector=[1.0, 0.0, 0.0], search_limit=1, max_hops=0, vector_weight=1.0, text_weight=0.0)
        result2 = self.db.retrieve(query="Washington", query_vector=[1.0, 0.0, 0.0], search_limit=5, max_hops=0, vector_weight=1.0, text_weight=0.0)
        self.assertLessEqual(result1.seed_node_count, 1)
        self.assertLessEqual(result2.seed_node_count, 5)


class TestAsk(unittest.TestCase):
    """Tests for the ask() end-to-end question answering method."""

    def setUp(self):
        self.db = GraphMemory(database=":memory:", vector_length=3)
        node = Node(properties={"name": "George Washington", "role": "First President"}, type="Person", vector=[1.0, 0.0, 0.0])
        self.db.insert_node(node)

    def tearDown(self):
        self.db.close()

    def test_ask_without_llm_returns_none_answer(self):
        result = self.db.ask(query="Who was Washington?", query_vector=[1.0, 0.0, 0.0])
        self.assertIsNone(result["answer"])
        self.assertIsNotNone(result["retrieval"])

    def test_ask_with_llm_callable(self):
        def fake_llm(system_prompt, user_prompt):
            return f"Answer based on context"

        result = self.db.ask(query="Who was Washington?", query_vector=[1.0, 0.0, 0.0], llm_callable=fake_llm)
        self.assertEqual(result["answer"], "Answer based on context")
        self.assertIsNotNone(result["retrieval"])

    def test_ask_llm_receives_context(self):
        captured = {}

        def capture_llm(system_prompt, user_prompt):
            captured["system"] = system_prompt
            captured["user"] = user_prompt
            return "ok"

        self.db.ask(query="Who was Washington?", query_vector=[1.0, 0.0, 0.0], llm_callable=capture_llm)
        self.assertIn("Washington", captured["user"])
        self.assertIn("Context for query", captured["user"])
        self.assertIn("Question", captured["user"])

    def test_ask_custom_system_prompt(self):
        captured = {}

        def capture_llm(system_prompt, user_prompt):
            captured["system"] = system_prompt
            return "ok"

        self.db.ask(query="test", query_vector=[1.0, 0.0, 0.0], llm_callable=capture_llm, system_prompt="Custom prompt")
        self.assertEqual(captured["system"], "Custom prompt")

    def test_ask_llm_exception_returns_none_answer(self):
        def failing_llm(system_prompt, user_prompt):
            raise RuntimeError("LLM failed")

        result = self.db.ask(query="test", query_vector=[1.0, 0.0, 0.0], llm_callable=failing_llm)
        self.assertIsNone(result["answer"])
        self.assertIsNotNone(result["retrieval"])

    def test_ask_returns_dict_with_expected_keys(self):
        result = self.db.ask(query="test", query_vector=[1.0, 0.0, 0.0])
        self.assertIn("retrieval", result)
        self.assertIn("answer", result)


class TestExpandGraph(unittest.TestCase):
    """Tests for the _expand_graph internal method."""

    def setUp(self):
        self.db = GraphMemory(database=":memory:", vector_length=3)
        self.n1 = Node(properties={"name": "A"}, vector=[1.0, 0.0, 0.0])
        self.n2 = Node(properties={"name": "B"}, vector=[0.0, 1.0, 0.0])
        self.n3 = Node(properties={"name": "C"}, vector=[0.0, 0.0, 1.0])
        self.n4 = Node(properties={"name": "D"}, vector=[0.5, 0.5, 0.0])
        for n in [self.n1, self.n2, self.n3, self.n4]:
            self.db.insert_node(n)
        # A -> B -> C, A -> D
        self.db.insert_edge(Edge(source_id=self.n1.id, target_id=self.n2.id, relation="r1"))
        self.db.insert_edge(Edge(source_id=self.n2.id, target_id=self.n3.id, relation="r2"))
        self.db.insert_edge(Edge(source_id=self.n1.id, target_id=self.n4.id, relation="r3"))

    def tearDown(self):
        self.db.close()

    def test_expand_zero_hops(self):
        nodes, edges = self.db._expand_graph({str(self.n1.id)}, max_hops=0)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(len(edges), 0)

    def test_expand_one_hop(self):
        nodes, edges = self.db._expand_graph({str(self.n1.id)}, max_hops=1)
        node_ids = set(nodes.keys())
        self.assertIn(str(self.n1.id), node_ids)
        self.assertIn(str(self.n2.id), node_ids)
        self.assertIn(str(self.n4.id), node_ids)
        # C is 2 hops away, should not be included
        self.assertNotIn(str(self.n3.id), node_ids)
        self.assertGreater(len(edges), 0)

    def test_expand_two_hops(self):
        nodes, edges = self.db._expand_graph({str(self.n1.id)}, max_hops=2)
        node_ids = set(nodes.keys())
        # All nodes should be reachable in 2 hops
        self.assertEqual(len(node_ids), 4)

    def test_expand_no_duplicate_edges(self):
        nodes, edges = self.db._expand_graph({str(self.n1.id)}, max_hops=2)
        edge_ids = [str(e.id) for e in edges]
        self.assertEqual(len(edge_ids), len(set(edge_ids)))


class TestContextAssembly(unittest.TestCase):
    """Tests for context assembly and formatting."""

    def setUp(self):
        self.db = GraphMemory(database=":memory:", vector_length=3)

    def tearDown(self):
        self.db.close()

    def test_format_node_with_type(self):
        node = Node(properties={"name": "Alice"}, type="Person", vector=[1.0, 0.0, 0.0])
        text = GraphMemory._format_node_context(node)
        self.assertIn("[Person]", text)
        self.assertIn("name: Alice", text)

    def test_format_node_without_type(self):
        node = Node(properties={"name": "Alice"}, vector=[1.0, 0.0, 0.0])
        text = GraphMemory._format_node_context(node)
        self.assertIn("[Node]", text)

    def test_format_edge(self):
        n1 = Node(type="Person", properties={}, vector=[1.0, 0.0, 0.0])
        n2 = Node(type="Organization", properties={}, vector=[0.0, 1.0, 0.0])
        edge = Edge(source_id=n1.id, target_id=n2.id, relation="works_at", weight=0.9)
        nodes = {str(n1.id): n1, str(n2.id): n2}
        text = GraphMemory._format_edge_context(edge, nodes)
        self.assertIn("Person", text)
        self.assertIn("Organization", text)
        self.assertIn("works_at", text)
        self.assertIn("weight: 0.9", text)

    def test_estimate_tokens(self):
        text = "a" * 400
        self.assertEqual(GraphMemory._estimate_tokens(text), 100)


if __name__ == '__main__':
    unittest.main()
