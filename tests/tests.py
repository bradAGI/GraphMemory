import json
import sys
import os
import uuid
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import duckdb
import unittest
from graphmemory.database import GraphMemory
from graphmemory.models import Node, Edge, NearestNode
from pydantic import ValidationError

class TestGraphMemory(unittest.TestCase):

    def setUp(self):
        self.db = GraphMemory(database=':memory:', vector_length=3)

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
        result = self.db.conn.execute("SELECT source_id, target_id, weight, relation FROM edges WHERE source_id = ? AND target_id = ?", (str(node1_id), str(node2_id))).fetchone()
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
            Node(properties={"name": "node3"}, vector=[0.7, 0.8, 0.9])
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
        self.assertEqual(nodes_json[0]['properties'], {"name": "node1"})
        self.assertAlmostEqual(nodes_json[0]['vector'][0], 0.1, places=7)
        self.assertAlmostEqual(nodes_json[0]['vector'][1], 0.2, places=7)
        self.assertAlmostEqual(nodes_json[0]['vector'][2], 0.3, places=7)

    def test_edges_to_json(self):
        node1 = Node(properties={"name": "node1"}, vector=[0.1, 0.2, 0.3])
        node2 = Node(properties={"name": "node2"}, vector=[0.4, 0.5, 0.6])
        self.db.insert_node(node1)
        self.db.insert_node(node2)
        edge = Edge(source_id=node1.id, target_id=node2.id, weight=0.5, relation="friendship")
        self.db.insert_edge(edge)
        edges_json = self.db.edges_to_json()
        self.assertEqual(len(edges_json), 1)
        self.assertEqual(edges_json[0]['source_id'], str(node1.id))
        self.assertEqual(edges_json[0]['target_id'], str(node2.id))
        self.assertEqual(edges_json[0]['weight'], 0.5)
        self.assertEqual(edges_json[0]['relation'], "friendship")

    def test_bulk_insert_nodes(self):
        nodes = [
            Node(properties={"name": "node1"}, vector=[0.1, 0.2, 0.3]),
            Node(properties={"name": "node2"}, vector=[0.4, 0.5, 0.6])
        ]
        inserted_nodes = self.db.bulk_insert_nodes(nodes)
        self.assertEqual(len(inserted_nodes), 2)
        self.assertTrue(all(node.id is not None for node in inserted_nodes))

    def test_bulk_insert_edges(self):
        node1 = Node(properties={"name": "node1"}, vector=[0.1, 0.2, 0.3])
        node2 = Node(properties={"name": "node2"}, vector=[0.4, 0.5, 0.6])
        node3 = Node(properties={"name": "node3"}, vector=[0.7, 0.8, 0.9])
        self.db.bulk_insert_nodes([node1, node2, node3])
        edges = [
            Edge(source_id=node1.id, target_id=node2.id, weight=0.5, relation="friendship"),
            Edge(source_id=node2.id, target_id=node3.id, weight=0.7, relation="colleague")
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
        result = self.db.conn.execute("SELECT * FROM edges WHERE source_id = ? AND target_id = ?", (str(node1.id), str(node2.id))).fetchone()
        self.assertIsNone(result)

    def test_delete_non_existent_edge(self):
        self.db.delete_edge(uuid.uuid4(), uuid.uuid4())
        result = self.db.conn.execute("SELECT * FROM edges WHERE source_id = ? AND target_id = ?", (uuid.uuid4(), uuid.uuid4())).fetchone()
        self.assertIsNone(result)

    def test_transaction_handling(self):
        try:
            with self.db.transaction():
                self.db.conn.execute("INSERT INTO nodes (properties, vector) VALUES ('{\"name\": \"node1\"}', [0.1, 0.2, 0.3]);")
                raise Exception("Force rollback")
        except:
            pass
        result = self.db.conn.execute("SELECT * FROM nodes WHERE properties = '{\"name\": \"node1\"}'").fetchone()
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
        
        result1 = self.db.conn.execute("SELECT source_id, target_id FROM edges WHERE source_id = ? AND target_id = ?", (str(id1), str(id2))).fetchone()
        result2 = self.db.conn.execute("SELECT source_id, target_id FROM edges WHERE source_id = ? AND target_id = ?", (str(id2), str(id1))).fetchone()
        
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
        if os.path.exists('test_db_parquet'):
            shutil.rmtree('test_db_parquet')
        if os.path.exists('test_db_csv'):
            shutil.rmtree('test_db_csv')

class TestGraphMemoryGetConnectedNodes(unittest.TestCase):

    def setUp(self):
        # Initialize the GraphMemory instance with an in-memory database for testing
        self.db = GraphMemory(database=':memory:', vector_length=3)
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

import unittest
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

class TestCypherToSQL(unittest.TestCase):
    node_id = None

    def setUp(self):
        # Set up the graph and logger if necessary
        self.graph = GraphMemory(database=':memory:', vector_length=3)
        example_node = Node(properties={"name": "George Washington", "age": 57}, type="Person")
        logger.info(f"Inserted Node ID: {self.node_id}")
        self.node_id = self.graph.insert_node(example_node)
        print("*****", self.node_id)

    def test_if_node_is_inserted(self):
        result = self.graph.get_node(self.node_id)
        self.assertIsNotNone(result)

    def test_cypher_to_sql(self):
        example_node = Node(properties={"name": "George Washington", "age": 57}, type="Person")
        self.node_id = self.graph.insert_node(example_node)

        cypher_query = "MATCH (n:Person {name: 'George Washington', age: 57}) RETURN n"
        expected_sql_query = (
            "SELECT * "
            "FROM nodes AS n "
            "WHERE n.type = 'Person' "
            "AND json_extract(n.properties, '$.name') = json('\"George Washington\"') "
            "AND json_extract(n.properties, '$.age') = json('57');"
        )

        sql_query = self.graph._cypher_to_sql(cypher_query)
        logger.info(f"Generated SQL Query: {sql_query}")
        self.assertEqual(sql_query.strip(), expected_sql_query.strip())

    def test_cypher_method(self):
        # Use the cypher method to query the node
        cypher_query = "MATCH (n:Person {name: 'George Washington', age: 57}) RETURN n"
        logger.info(f"Executing Cypher query: {cypher_query}")
        result = self.graph.cypher(cypher_query)
        logger.info(f"Query result: {result}")

        # Check if the result is as expected
        self.assertEqual(len(result), 1, f"Expected 1 result, but got {len(result)}")
        self.assertEqual(str(result[0][0]), str(self.node_id))  # Convert UUID to string for comparison

    def tearDown(self):
        self.graph.conn.close()

if __name__ == '__main__':
    unittest.main()
