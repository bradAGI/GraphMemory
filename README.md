# GraphMemory - GraphRAG Database

## Overview
This project provides an embedded graph database implementation with vector similarity search (VSS) using DuckDB. It includes a Python class `GraphRAG` for managing nodes and edges. 

Each node has a unique ID, a JSON data field (any arbitrary dictionary), and a vector of floating point values. 

Each edge has a unique ID, a source node ID, a target node ID, a relationship type, and a weight.

This database can be used for any graph-based RAG application or knowledge graph application.

Vector embeddings can be created using [sentence-transformers](https://www.sbert.net/) or other API based models.

## Requirements
- Python 3.x
- DuckDB
- Pydantic

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/bradAGI/GraphMemory
    ```
2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### GraphRAG Class
The `GraphRAG` class provides methods to manage nodes and edges, perform bulk inserts, create indexes, and find nearest neighbors using vector similarity search.

### Auto-Incrementing IDs
If you do not provide an ID for a node or edge, the database will automatically assign a unique ID.

### Example
```python
from graphrag.graphrag import GraphRAG
from graphrag.models import Node, Edge


# Initialize the database from disk (make sure to set vector_length correctly)
graph_db = GraphRAG(database='graph.db', vector_length=3)

# Insert nodes
node1 = Node(data={"name": "George Washington", "role": "President"}, vector=[0.1, 0.2, 0.3])
node1_id = graph_db.insert_node(node1)

node2 = Node(data={"name": "Thomas Jefferson", "role": "Secretary of State"}, vector=[0.4, 0.5, 0.6])
node2_id = graph_db.insert_node(node2)

# Insert edge
edge = Edge(source_id=node1_id, target_id=node2_id, relation="served_under", weight=0.5)
graph_db.insert_edge(edge)

# Print all nodes in the database
nodes = graph_db.nodes_to_json()
print("Nodes:", nodes)

# Print all edges in the database
edges = graph_db.edges_to_json()
print("Edges:", edges)

# Find connected nodes
connected_nodes = graph_db.connected_nodes(node1_id)
print("Connected Nodes:", connected_nodes)

# Find nearest neighbors
neighbors = graph_db.nearest_neighbors(vector=[0.1, 0.2, 0.3], limit=1)
print("Nearest Neighbors:", neighbors)

# Insert an edge between the two nodes with a relation
edge = Edge(source_id=node1_id, target_id=node2_id, relation="served_under", weight=0.5)
graph_db.insert_edge(edge)

# Define the additional nodes for bulk insert
nodes = [
    Node(data={"name": "Alexander Hamilton", "role": "Secretary of the Treasury", "term": "1789–1795"}, vector=[0.7, 0.8, 0.9]),
    Node(data={"name": "Oliver Wolcott Jr.", "role": "Secretary of the Treasury", "term": "1795–1797"}, vector=[1.6, 1.7, 1.8]),
]

# Bulk insert nodes
graph_db.bulk_insert_nodes(nodes)

# Define the additional edges for bulk insert
edges = [
    Edge(source_id=nodes[0].id, target_id=nodes[1].id, relation="succeeded_by", weight=0.7),
    Edge(source_id=nodes[1].id, target_id=nodes[2].id, relation="succeeded_by", weight=0.8)
]

# Bulk insert edges
graph_db.bulk_insert_edges(edges)

# Delete a node
graph_db.delete_node(nodes[-1].id)

# Delete an edge
graph_db.delete_edge(1, 2)

# Find nearest nodes to a given vector by distance
neighbors = graph_db.nearest_neighbors(vector=[0.1, 0.2, 0.3], limit=2)
print("Nearest Neighbors:", neighbors)

# Find connected nodes
connected_nodes = graph_db.connected_nodes(nodes[1].id)
print("Connected Nodes:", connected_nodes)
```

## GraphRAG Class Methods

The `GraphRAG` class provides the following public methods for interacting with the graph database:

1. `__init__(self, database=None, vector_length=3)`
   - Initializes the database connection and sets up the database schema if necessary.

2. `set_vector_length(self, vector_length)`
   - Sets the length of the vectors for the nodes in the database.

3. `create_tables(self)`
   - Creates the necessary database tables for nodes and edges if they do not exist.

4. `insert_node(self, node: Node) -> int`
   - Inserts a node into the database and returns the node ID.

5. `insert_edge(self, edge: Edge)`
   - Inserts an edge between two nodes in the database.

6. `bulk_insert_nodes(self, nodes: List[Node]) -> List[Node]`
   - Performs a bulk insert of multiple nodes into the database.

7. `bulk_insert_edges(self, edges: List[Edge])`
   - Performs a bulk insert of multiple edges into the database.

8. `delete_node(self, node_id: int)`
   - Deletes a node and its associated edges from the database.

9. `delete_edge(self, source_id: int, target_id: int)`
   - Deletes an edge from the database.

10. `create_index(self)`
    - Creates an index on the node vectors to improve search performance.

11. `nearest_neighbors(self, vector: List[float], limit: int) -> List[Neighbor]`
    - Finds and returns the nearest neighbor nodes based on vector similarity.

12. `connected_nodes(self, node_id: int) -> List[Node]`
    - Retrieves all nodes directly connected to the specified node.

13. `nodes_to_json(self)`
    - Returns a JSON representation of all nodes in the database.

14. `edges_to_json(self)`
    - Returns a JSON representation of all edges in the database.

15. `get_node(self, node_id: int)`
    - Retrieves a specific node by its ID.

16. `print_json(self)`
    - Prints the JSON representation of all nodes and edges in the database.

These methods facilitate the management and querying of the graph database, allowing for efficient data handling and retrieval.

## Testing
Unit tests are provided in `tests/tests.py`.

### Running Tests
To run the unit tests, use the following command:
```sh
python -m unittest discover -s tests
```

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.
