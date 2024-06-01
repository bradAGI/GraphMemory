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
connected_nodes = graph_db.get_connected_nodes(node1_id)
print("Connected Nodes:", connected_nodes)

# Find nearest neighbors
neighbors = graph_db.nearest_neighbors(vector=[0.1, 0.2, 0.3], limit=1)
print("Nearest Neighbors:", neighbors)

# Insert an edge between the two nodes with a relation
edge = Edge(source_id=node1_id, target_id=node2_id, relation="served_under", weight=0.5)
graph_db.insert_edge(edge)

# Define the additional nodes for bulk insert
nodes = [
    Node(data={"name": "Alexander Hamilton", "role": "Secretary of the Treasury"}, vector=[0.7, 0.8, 0.9]),
    Node(data={"name": "John Jay", "role": "Secretary of State", "term": "1789–1790"}, vector=[0.1, 0.2, 0.3]),
    Node(data={"name": "Edmund Randolph", "role": "Secretary of State", "term": "1794–1795"}, vector=[0.7, 0.8, 0.9]),
    Node(data={"name": "Timothy Pickering", "role": "Secretary of State", "term": "1795–1797"}, vector=[1.0, 1.1, 1.2]),
    Node(data={"name": "Oliver Wolcott Jr.", "role": "Secretary of the Treasury", "term": "1795–1797"}, vector=[1.6, 1.7, 1.8]),
    Node(data={"name": "Henry Knox", "role": "Secretary of War", "term": "1789–1794"}, vector=[1.9, 2.0, 2.1]),
    Node(data={"name": "James McHenry", "role": "Secretary of War", "term": "1796–1797"}, vector=[2.2, 2.3, 2.4]),
    Node(data={"name": "Edmund Randolph", "role": "Attorney General", "term": "1789–1794"}, vector=[2.5, 2.6, 2.7]),
    Node(data={"name": "William Bradford", "role": "Attorney General", "term": "1794–1795"}, vector=[2.8, 2.9, 3.0]),
    Node(data={"name": "Charles Lee", "role": "Attorney General", "term": "1795–1797"}, vector=[3.1, 3.2, 3.3])
]

# Bulk insert nodes
graph_db.bulk_insert_nodes(nodes)

# Bulk insert edges with relations
edges = [
    Edge(source_id=nodes[1].id, target_id=nodes[2].id, relation="succeeded_by", weight=0.7),
    Edge(source_id=nodes[2].id, target_id=nodes[3].id, relation="succeeded_by", weight=0.8)
]
graph_db.bulk_insert_edges(edges)

# Delete a node
graph_db.delete_node(nodes[-1].id)

# Delete an edge
graph_db.delete_edge(1, 2)

# Find nearest neighbors by vector distance
neighbors = graph_db.nearest_neighbors(vector=[0.1, 0.2, 0.3], limit=2)
print("Nearest Neighbors:", neighbors)

# Find connected nodes to John Jay
connected_nodes = graph_db.get_connected_nodes(nodes[1].id)
print("Connected Nodes:", connected_nodes)
```

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