import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import duckdb
import unittest
from src.graphrag import GraphRAG
from src.models import Node, Edge, Neighbor
from pydantic import ValidationError

class TestGraphRAG(unittest.TestCase):

    def setUp(self):
        self.db = GraphRAG(database=':memory:', vector_length=3)

    def test_insert_node(self):
        node = Node(data={"name": "node1"}, vector=[0.1, 0.2, 0.3])
        node_id = self.db.insert_node(node)
        result = self.db.conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        self.assertIsNotNone(result)
        self.assertEqual(result[0], node_id)
        self.assertEqual(result[1], '{"name": "node1"}')
        self.assertAlmostEqual(result[2][0], 0.1, places=7)
        self.assertAlmostEqual(result[2][1], 0.2, places=7)
        self.assertAlmostEqual(result[2][2], 0.3, places=7)

    def test_insert_invalid_node(self):
        with self.assertRaises(ValidationError):
            node = Node(data={"name": "node1"}, vector=[0.1, 0.2, "invalid"])
            self.db.insert_node(node)

    def test_insert_node_invalid_vector_length(self):
        node = Node(data={"name": "node1"}, vector=[0.1, 0.2])
        result = self.db.insert_node(node)
        self.assertEqual(result, -1)

    def test_insert_edge(self):
        node1 = Node(data={"name": "node1"}, vector=[0.1, 0.2, 0.3])
        node2 = Node(data={"name": "node2"}, vector=[0.4, 0.5, 0.6])
        node1_id = self.db.insert_node(node1)
        node2_id = self.db.insert_node(node2)
        edge = Edge(source_id=node1_id, target_id=node2_id, weight=0.5, relation="friendship")
        self.db.insert_edge(edge)
        result = self.db.conn.execute("SELECT source_id, target_id, weight, relation FROM edges WHERE source_id = ? AND target_id = ?", (node1_id, node2_id)).fetchone()
        self.assertIsNotNone(result)
        self.assertEqual(result[0], node1_id)
        self.assertEqual(result[1], node2_id)
        self.assertEqual(result[2], 0.5)
        self.assertEqual(result[3], "friendship")

    def test_insert_edge_non_existent_nodes(self):
        edge = Edge(source_id=999, target_id=1000, weight=0.5, relation="friendship")
        with self.assertRaises(ValueError):
            self.db.insert_edge(edge)

    def test_nearest_neighbors_empty_db(self):
        neighbors = self.db.nearest_neighbors(vector=[0.1, 0.2, 0.3], limit=2)
        self.assertEqual(len(neighbors), 0)

    def test_nearest_neighbors(self):
        nodes = [
            Node(data={"name": "node1"}, vector=[0.1, 0.2, 0.3]),
            Node(data={"name": "node2"}, vector=[0.4, 0.5, 0.6]),
            Node(data={"name": "node3"}, vector=[0.7, 0.8, 0.9])
        ]
        for node in nodes:
            self.db.insert_node(node)
        self.db.create_index()
        neighbors = self.db.nearest_neighbors(vector=[0.1, 0.2, 0.3], limit=2)
        self.assertEqual(len(neighbors), 2)
        self.assertEqual(neighbors[0].node.id, 1)
        self.assertEqual(neighbors[1].node.id, 2)

    def test_nodes_to_json(self):
        node = Node(data={"name": "node1"}, vector=[0.1, 0.2, 0.3])
        self.db.insert_node(node)
        nodes_json = self.db.nodes_to_json()
        self.assertEqual(len(nodes_json), 1)
        self.assertEqual(nodes_json[0]['id'], 1)
        self.assertEqual(nodes_json[0]['data'], {"name": "node1"})
        self.assertAlmostEqual(nodes_json[0]['vector'][0], 0.1, places=7)
        self.assertAlmostEqual(nodes_json[0]['vector'][1], 0.2, places=7)
        self.assertAlmostEqual(nodes_json[0]['vector'][2], 0.3, places=7)

    def test_edges_to_json(self):
        node1 = Node(data={"name": "node1"}, vector=[0.1, 0.2, 0.3])
        node2 = Node(data={"name": "node2"}, vector=[0.4, 0.5, 0.6])
        self.db.insert_node(node1)
        self.db.insert_node(node2)
        edge = Edge(source_id=1, target_id=2, weight=0.5, relation="friendship")
        self.db.insert_edge(edge)
        edges_json = self.db.edges_to_json()
        self.assertEqual(len(edges_json), 1)
        self.assertEqual(edges_json[0]['source_id'], 1)
        self.assertEqual(edges_json[0]['target_id'], 2)
        self.assertEqual(edges_json[0]['weight'], 0.5)
        self.assertEqual(edges_json[0]['relation'], "friendship")

    def test_bulk_insert_nodes(self):
        nodes = [
            Node(data={"name": "node1"}, vector=[0.1, 0.2, 0.3]),
            Node(data={"name": "node2"}, vector=[0.4, 0.5, 0.6])
        ]
        inserted_nodes = self.db.bulk_insert_nodes(nodes)
        self.assertEqual(len(inserted_nodes), 2)
        self.assertTrue(all(node.id is not None for node in inserted_nodes))

    def test_bulk_insert_edges(self):
        node1 = Node(data={"name": "node1"}, vector=[0.1, 0.2, 0.3])
        node2 = Node(data={"name": "node2"}, vector=[0.4, 0.5, 0.6])
        node3 = Node(data={"name": "node3"}, vector=[0.7, 0.8, 0.9])
        self.db.bulk_insert_nodes([node1, node2, node3])
        edges = [
            Edge(source_id=1, target_id=2, weight=0.5, relation="friendship"),
            Edge(source_id=2, target_id=3, weight=0.7, relation="colleague")
        ]
        self.db.bulk_insert_edges(edges)
        result = self.db.conn.execute("SELECT * FROM edges").fetchall()
        self.assertEqual(len(result), 2)

    def test_delete_node(self):
        node = Node(data={"name": "node1"}, vector=[0.1, 0.2, 0.3])
        node_id = self.db.insert_node(node)
        self.db.delete_node(node_id)
        result = self.db.conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        self.assertIsNone(result)

    def test_delete_non_existent_node(self):
        self.db.delete_node(999)
        result = self.db.conn.execute("SELECT * FROM nodes WHERE id = 999").fetchone()
        self.assertIsNone(result)

    def test_delete_edge(self):
        node1 = Node(data={"name": "node1"}, vector=[0.1, 0.2, 0.3])
        node2 = Node(data={"name": "node2"}, vector=[0.4, 0.5, 0.6])
        self.db.insert_node(node1)
        self.db.insert_node(node2)
        edge = Edge(source_id=1, target_id=2, weight=0.5, relation="friendship")
        self.db.insert_edge(edge)
        self.db.delete_edge(1, 2)
        result = self.db.conn.execute("SELECT * FROM edges WHERE source_id = 1 AND target_id = 2").fetchone()
        self.assertIsNone(result)

    def test_delete_non_existent_edge(self):
        self.db.delete_edge(999, 1000)
        result = self.db.conn.execute("SELECT * FROM edges WHERE source_id = 999 AND target_id = 1000").fetchone()
        self.assertIsNone(result)

    def test_transaction_handling(self):
        try:
            with self.db.transaction():
                self.db.conn.execute("INSERT INTO nodes (data, vector) VALUES ('{\"name\": \"node1\"}', [0.1, 0.2, 0.3]);")
                raise Exception("Force rollback")
        except:
            pass
        result = self.db.conn.execute("SELECT * FROM nodes WHERE data = '{\"name\": \"node1\"}'").fetchone()
        self.assertIsNone(result)

    def test_auto_increment_node_id(self):
        node1 = Node(data={"name": "node1"}, vector=[0.1, 0.2, 0.3])
        node2 = Node(data={"name": "node2"}, vector=[0.4, 0.5, 0.6])
        id1 = self.db.insert_node(node1)
        id2 = self.db.insert_node(node2)
        
        self.assertEqual(id1, 1)
        self.assertEqual(id2, 2)

    def test_auto_increment_edge_id(self):
        node1 = Node(data={"name": "node1"}, vector=[0.1, 0.2, 0.3])
        node2 = Node(data={"name": "node2"}, vector=[0.4, 0.5, 0.6])
        id1 = self.db.insert_node(node1)
        id2 = self.db.insert_node(node2)
        
        edge1 = Edge(source_id=id1, target_id=id2, weight=0.5, relation="friendship")
        edge2 = Edge(source_id=id2, target_id=id1, weight=0.7, relation="colleague")
        self.db.insert_edge(edge1)
        self.db.insert_edge(edge2)
        
        result1 = self.db.conn.execute("SELECT source_id, target_id FROM edges WHERE source_id = ? AND target_id = ?", (id1, id2)).fetchone()
        result2 = self.db.conn.execute("SELECT source_id, target_id FROM edges WHERE source_id = ? AND target_id = ?", (id2, id1)).fetchone()
        
        self.assertIsNotNone(result1)
        self.assertIsNotNone(result2)
        self.assertEqual(result1[0], id1)
        self.assertEqual(result1[1], id2)
        self.assertEqual(result2[0], id2)
        self.assertEqual(result2[1], id1)

    def tearDown(self):
        self.db.conn.close()
        # Ensure any test directories are cleaned up
        import shutil
        if os.path.exists('test_db_parquet'):
            shutil.rmtree('test_db_parquet')
        if os.path.exists('test_db_csv'):
            shutil.rmtree('test_db_csv')

class TestGraphRAGGetConnectedNodes(unittest.TestCase):

    def setUp(self):
        # Initialize the GraphRAG instance with an in-memory database for testing
        self.db = GraphRAG(database=':memory:', vector_length=3)
        self.db.create_tables()  # Ensure tables are created

        # Insert nodes and edges for testing
        self.node1 = Node(data={"name": "node1"}, vector=[0.1, 0.2, 0.3])
        self.node2 = Node(data={"name": "node2"}, vector=[0.4, 0.5, 0.6])
        self.node3 = Node(data={"name": "node3"}, vector=[0.7, 0.8, 0.9])
        self.node1_id = self.db.insert_node(self.node1)
        self.node2_id = self.db.insert_node(self.node2)
        self.node3_id = self.db.insert_node(self.node3)

        # Insert edges
        self.db.insert_edge(Edge(source_id=self.node1_id, target_id=self.node2_id, weight=0.5, relation="friendship"))
        self.db.insert_edge(Edge(source_id=self.node2_id, target_id=self.node3_id, weight=0.7, relation="colleague"))

    def test_get_connected_nodes(self):
        # Test for node1, should have node2 as connected
        connected_nodes = self.db.get_connected_nodes(self.node1_id)
        self.assertEqual(len(connected_nodes), 1)
        self.assertEqual(connected_nodes[0].id, self.node2_id)

        # Test for node2, should have node1 and node3 as connected
        connected_nodes = self.db.get_connected_nodes(self.node2_id)
        self.assertEqual(len(connected_nodes), 2)
        connected_ids = [node.id for node in connected_nodes]
        self.assertIn(self.node1_id, connected_ids)
        self.assertIn(self.node3_id, connected_ids)

        # Test for node3, should have node2 as connected
        connected_nodes = self.db.get_connected_nodes(self.node3_id)
        self.assertEqual(len(connected_nodes), 1)
        self.assertEqual(connected_nodes[0].id, self.node2_id)

    def tearDown(self):
        self.db.conn.close()

if __name__ == '__main__':
    unittest.main()

