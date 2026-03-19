[![](https://dcbadge.limes.pink/api/server/https://discord.gg/DSS3DmStV8)](https://discord.gg/DSS3DmStV8)

# GraphMemory - GraphRAG Database

![GraphMemory](https://github.com/bradAGI/GraphMemory/assets/46579244/9897dc2a-46c9-42e0-a8d3-2dcb1d93e6ae)

## Overview
An embedded graph database with vector similarity search (VSS), full-text search (BM25), and hybrid search using DuckDB. The `GraphMemory` class provides a complete API for managing nodes and edges, with support for Cypher queries, graph import/export, connection pooling, and automatic retry logic.

Each node has a unique ID, a JSON properties field (any arbitrary dictionary), a node type (ex: Person, Organization, etc.), and a vector of floating point values.

Each edge has a unique ID, a source node ID, a target node ID, a relationship type (ex: served_under, worked_with, etc.), and a weight.

This database can be used for any graph-based RAG application or knowledge graph application.

Vector embeddings can be created using [sentence-transformers](https://www.sbert.net/) or other API based models.

## Features

- **Vector Similarity Search** — HNSW-indexed nearest neighbor search with L2, cosine, and inner product distance metrics
- **Full-Text Search** — BM25-scored text search across node properties
- **Hybrid Search** — Combined vector + text search with configurable weights
- **Cypher Queries** — Query the graph using Cypher-like syntax
- **Graph Import/Export** — JSON, CSV, and GraphML formats
- **Connection Pooling** — Thread-safe operations with automatic retry on transient errors
- **Context Manager** — Use `with` statements for automatic resource cleanup

## Installation
```sh
pip install graphmemory
```

## Usage

### Initialization

```python
from graphmemory import GraphMemory, Node, Edge

# In-memory database
graph_db = GraphMemory(vector_length=1536)

# Persistent database with cosine distance
graph_db = GraphMemory(
    database='graph.db',
    vector_length=1536,
    distance_metric='cosine'  # 'l2', 'cosine', or 'inner_product'
)

# Using context manager
with GraphMemory(database='graph.db', vector_length=1536) as graph_db:
    # ... operations ...
    pass  # connection closed automatically
```

### Auto Generated UUID
IDs for nodes and edges are auto generated UUIDs.

### Support for Cypher Queries
The `GraphMemory` class supports Cypher queries via the `cypher` method.

Example: `MATCH (n:Person {name: 'George Washington', age: 57}) RETURN n`

### Example Usage
```python
from graphmemory import GraphMemory, Node, Edge

import json
from openai import OpenAI
import os

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Sample unstructured text
gw_text = "George Washington was the first President of the United States and served from 1789 to 1797."
tj_text = "Thomas Jefferson was the first Secretary of State of the United States and served from 1790 to 1793."
ah_text = "Alexander Hamilton was the first Secretary of the Treasury of the United States and served from 1789 to 1795."

# Extract structured data from unstructured text
def extract_attributes(text):
    return json.loads(client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Extract structured data from this text using the following attributes: \
             name, title, country, term_start, term_end"},
            {"role": "user", "content": text}
        ],
        seed=1
    ).choices[0].message.content)

# Calculate embedding for a given input
def calculate_embedding(input_json):
    return client.embeddings.create(
        input=input_json,
        model="text-embedding-3-small"
    ).data[0].embedding

gw_embedding = calculate_embedding(gw_text)
tj_embedding = calculate_embedding(tj_text)
ah_embedding = calculate_embedding(ah_text)

# Initialize the database from disk (make sure to set vector_length correctly)
graph_db = GraphMemory(database='graph.db', vector_length=len(gw_embedding))

# Extract structured data from unstructured text
gw_attributes = extract_attributes(gw_text)
tj_attributes = extract_attributes(tj_text)
ah_attributes = extract_attributes(ah_text)

print(gw_attributes)
print(tj_attributes)
print(ah_attributes)

# Output Example:
# {
#   'person': 'George Washington',
#   'title': 'President',
#   'country': 'United States',
#   'term_start': '1789',
#   'term_end': '1797'
# }
# {
#   'person': 'Thomas Jefferson',
#   'title': 'Secretary of State',
#   'country': 'United States',
#   'term_start': 1790,
#   'term_end': 1793
# }
# {
#   'person': 'Alexander Hamilton',
#   'title': 'Secretary of the Treasury',
#   'country': 'United States',
#   'term_start': 1789,
#   'term_end': 1795
# }


# Create nodes with UUIDs
gw_node = Node(properties=gw_attributes, vector=gw_embedding)
tj_node = Node(properties=tj_attributes, vector=tj_embedding)
ah_node = Node(properties=ah_attributes, vector=ah_embedding)

gw_node_id = graph_db.insert_node(gw_node)
if gw_node_id is None:
    raise ValueError("Failed to insert George Washington node")

tj_node_id = graph_db.insert_node(tj_node)
if tj_node_id is None:
    raise ValueError("Failed to insert Thomas Jefferson node")

ah_node_id = graph_db.insert_node(ah_node)
if ah_node_id is None:
    raise ValueError("Failed to insert Alexander Hamilton node")

# Insert edges
edge1 = Edge(source_id=gw_node_id, target_id=tj_node_id, relation="served_under", weight=0.5)
edge2 = Edge(source_id=gw_node_id, target_id=ah_node_id, relation="served_under", weight=0.5)
graph_db.insert_edge(edge1)
graph_db.insert_edge(edge2)

# Print edges
print(graph_db.edges_to_json())

# Find connected nodes
connected_nodes = graph_db.connected_nodes(gw_node_id)
for node in connected_nodes:
    print("Connected Node Data:", node.properties)

# Find nearest nodes by vector embedding
nearest_nodes = graph_db.nearest_nodes(calculate_embedding("George Washington"), limit=1)
print(nearest_nodes)
print("Nearest Node Data:", nearest_nodes[0].node.properties)
print("Nearest Node Distance:", nearest_nodes[0].distance)

# Full-text search across node properties
results = graph_db.search_nodes("President", limit=5)
for result in results:
    print(f"Found: {result.node.properties} (score: {result.score})")

# Hybrid search (combines vector similarity + text search)
results = graph_db.hybrid_search(
    query_text="President",
    query_vector=calculate_embedding("President of the United States"),
    limit=5,
    text_weight=0.5,
    vector_weight=0.5
)

# Get node/s by attribute (Who was the Secretary of State?)
nodes = graph_db.nodes_by_attribute("title", "Secretary of State")
if nodes:
    print("Node by attribute:", nodes[0].properties)
else:
    print("No nodes found with the attribute 'title' = 'Secretary of State'")

# What is the title of the people who served under George Washington?
for node in connected_nodes:
    print(f"{node.properties.get('name')} - {node.properties.get('title')}")

# Fetch a node by UUID
fetched_node = graph_db.get_node(gw_node_id)

# Update a node
graph_db.update_node(gw_node_id, properties={"name": "George Washington", "title": "President"})

# Delete an edge by source / target node id
graph_db.delete_edge(edge1.source_id, edge1.target_id)

# Export the graph
graph_json = graph_db.export_graph(format='json')
graph_csv = graph_db.export_graph(format='csv')
graph_graphml = graph_db.export_graph(format='graphml')
```

## Data Models

### Node
```python
Node(
    id: uuid.UUID,          # Auto-generated
    properties: dict | None, # Arbitrary key-value pairs
    type: str | None,        # Node type (e.g., "Person", "Organization")
    vector: list[float] | None  # Embedding vector
)
```

### Edge
```python
Edge(
    id: uuid.UUID,           # Auto-generated
    source_id: uuid.UUID,    # Source node ID
    target_id: uuid.UUID,    # Target node ID
    relation: str | None,    # Relationship type (e.g., "served_under")
    weight: float | None     # Edge weight
)
```

### NearestNode
```python
NearestNode(
    node: Node,       # The matched node
    distance: float   # Distance from query vector
)
```

### SearchResult
```python
SearchResult(
    node: Node,    # The matched node
    score: float   # Relevance score
)
```

## GraphMemory API Reference

### Initialization & Connection

| Method | Description |
|--------|-------------|
| `__init__(database=None, vector_length=3, distance_metric='l2', max_retries=3, retry_base_delay=0.1)` | Initialize the database. `database`: file path or `None` for in-memory. `distance_metric`: `'l2'`, `'cosine'`, or `'inner_product'`. |
| `close()` | Close the database connection (thread-safe). |
| `cursor()` | Return a new DuckDB cursor for individual operations. |
| `transaction()` | Context manager for database transactions. |

### Node Operations

| Method | Description |
|--------|-------------|
| `insert_node(node: Node) -> uuid.UUID` | Insert a node and return its ID. |
| `bulk_insert_nodes(nodes: list[Node]) -> list[Node]` | Bulk insert multiple nodes. |
| `get_node(node_id: uuid.UUID) -> Node` | Retrieve a node by ID. |
| `update_node(node_id: uuid.UUID, **kwargs) -> bool` | Update node fields (`type`, `properties`, `vector`). |
| `delete_node(node_id: uuid.UUID)` | Delete a node and its associated edges. |
| `bulk_delete_nodes(node_ids: list[uuid.UUID])` | Bulk delete multiple nodes and their edges. |
| `nodes_by_attribute(attribute, value, limit=None, offset=None) -> list[Node]` | Query nodes by a property key-value pair. |
| `get_nodes_vector(node_id: uuid.UUID) -> list[float]` | Retrieve the vector of a node. |
| `nodes_to_json(limit=None, offset=None) -> list[dict]` | Export all nodes as JSON. |

### Edge Operations

| Method | Description |
|--------|-------------|
| `insert_edge(edge: Edge)` | Insert an edge between two nodes. |
| `bulk_insert_edges(edges: list[Edge])` | Bulk insert multiple edges. |
| `get_edge(edge_id: uuid.UUID) -> Edge \| None` | Retrieve an edge by ID. |
| `get_edges_by_relation(relation: str) -> list[Edge]` | Get all edges with a given relation type. |
| `edges_by_attribute(attribute: str, value) -> list[Edge]` | Query edges by attribute. |
| `update_edge(edge_id: uuid.UUID, **kwargs) -> bool` | Update edge fields (`relation`, `weight`). |
| `delete_edge(source_id: uuid.UUID, target_id: uuid.UUID)` | Delete an edge by source and target node IDs. |
| `bulk_delete_edges(edge_ids: list[uuid.UUID])` | Bulk delete multiple edges. |
| `edges_to_json(limit=None, offset=None) -> list[dict]` | Export all edges as JSON. |

### Search & Similarity

| Method | Description |
|--------|-------------|
| `nearest_nodes(vector: list[float], limit: int) -> list[NearestNode]` | Find nearest neighbors by vector similarity. |
| `search_nodes(query_text: str, limit: int = 10) -> list[SearchResult]` | Full-text search across node properties (BM25). |
| `hybrid_search(query_text, query_vector, limit=10, text_weight=0.5, vector_weight=0.5) -> list[SearchResult]` | Combined text + vector search with configurable weights. |
| `create_index()` | Create an HNSW index on node vectors for faster search. |
| `set_vector_length(vector_length)` | Set the vector dimension for the database. |

### Graph Traversal

| Method | Description |
|--------|-------------|
| `connected_nodes(node_id: uuid.UUID) -> list[Node]` | Retrieve all nodes connected to a given node. |
| `cypher(cypher_query)` | Execute a Cypher-like query. |

### Import / Export

| Method | Description |
|--------|-------------|
| `export_graph(format='json')` | Export the graph. Formats: `'json'`, `'json_string'`, `'csv'`, `'graphml'`. |
| `import_graph(data, format='json')` | Import a graph from the given data and format. |

### Utility

| Method | Description |
|--------|-------------|
| `print_json()` | Print a JSON representation of all nodes and edges. |

## Examples

See the `examples/` directory for complete usage examples:

- **`openai_example.py`** — Uses OpenAI embeddings for node creation, similarity search, and attribute queries
- **`lexical_graph.py`** — Wikipedia text extraction with SentenceTransformer embeddings
- **`dspy_example_typed_pred.py`** — Knowledge graph extraction from unstructured text using DSPy

## Testing
Unit tests are provided in `tests/tests.py`.

### Running Tests
```sh
python3 -m unittest discover -s tests
```

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.
