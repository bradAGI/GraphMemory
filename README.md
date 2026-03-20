[![](https://dcbadge.limes.pink/api/server/https://discord.gg/DSS3DmStV8)](https://discord.gg/DSS3DmStV8)

# GraphMemory - GraphRAG Database

![GraphMemory](https://github.com/bradAGI/GraphMemory/assets/46579244/9897dc2a-46c9-42e0-a8d3-2dcb1d93e6ae)

An embedded graph database for RAG and knowledge graph applications, powered by [DuckDB](https://duckdb.org/). Vector similarity search, full-text search, hybrid search, merge/upsert, graph traversal, and a full GraphRAG retrieval pipeline — all in a single Python package.

## Features

- **Vector Search** — HNSW-indexed nearest neighbors (L2, cosine, inner product)
- **Full-Text Search** — BM25-scored search across node properties
- **Hybrid Search** — Combined vector + text with configurable weights
- **GraphRAG** — Retrieval pipeline: hybrid search → graph expansion → context assembly → LLM Q&A
- **Merge / Upsert** — Deduplicate nodes by property keys and edges by (source, target, relation)
- **Query Builder** — Fluent, parameterized API with multi-hop traversal
- **DSPy Extraction** — Entity/relationship extraction from text via DSPy (optional)
- **Graph Algorithms** — PageRank, centrality, components via NetworkX (optional)
- **Import / Export** — JSON, CSV, GraphML
- **Visualizer** — Interactive D3.js force-directed graph in the browser
- **Thread-Safe** — Connection pooling, transactions, automatic retry with exponential backoff

## Installation

```sh
pip install graphmemory

# Optional
pip install graphmemory[extraction]   # DSPy extraction
pip install graphmemory[algorithms]   # NetworkX algorithms
```

## Quick Start

```python
from graphmemory import GraphMemory, Node, Edge

graph = GraphMemory(database="graph.db", vector_length=3, distance_metric="cosine")

# Insert nodes
alice = Node(type="Person", properties={"name": "Alice", "role": "engineer"}, vector=[0.1, 0.8, 0.3])
bob = Node(type="Person", properties={"name": "Bob", "role": "manager"}, vector=[0.2, 0.7, 0.4])
graph.insert_node(alice)
graph.insert_node(bob)

# Insert edge
graph.insert_edge(Edge(source_id=alice.id, target_id=bob.id, relation="reports_to", weight=1.0))

# Vector search
nearest = graph.nearest_nodes(vector=[0.1, 0.8, 0.3], limit=5)

# Full-text search
results = graph.search_nodes("engineer", limit=10)

# Hybrid search
results = graph.hybrid_search("engineer", query_vector=[0.1, 0.8, 0.3], text_weight=0.5, vector_weight=0.5)

# Context manager
with GraphMemory(database="graph.db", vector_length=3) as graph:
    graph.insert_node(alice)
```

## Usage

### Query Builder

```python
# Filter by type and properties
results = graph.query().match(type="Person").where(role="engineer").execute()

# Multi-hop traversal
results = graph.query().traverse(source_id=alice.id, depth=2).execute()

# Paginate and order
results = graph.query().match(type="Person").order_by("name").limit(10).offset(0).execute()

# Query edges
edges = graph.query().match(type="Person").edges().execute()
```

### Merge / Upsert

Insert-or-update nodes matched by property keys. Edges deduplicate on `(source_id, target_id, relation)`.

```python
from graphmemory import MergeStrategy

# Insert if no match, update if "name" matches an existing Person node
result = graph.merge_node(alice, match_keys=["name"])
print(result.created)  # True = inserted, False = updated

# Bulk merge with strategy
results = graph.bulk_merge_nodes(nodes, match_keys=["name"], strategy=MergeStrategy.UPDATE)

# Edge merge
result = graph.merge_edge(edge)
results = graph.bulk_merge_edges(edges)
```

| Strategy | Behavior |
|----------|----------|
| `UPDATE` | Shallow merge — existing keys preserved, incoming keys added/overwritten (default) |
| `REPLACE` | Incoming properties fully replace existing |
| `KEEP` | Existing properties unchanged; only new nodes inserted |

### GraphRAG Retrieval

Full pipeline: hybrid search → multi-hop graph expansion → token-aware context assembly → LLM generation.

```python
# Retrieve context
result = graph.retrieve(query="Who leads ML?", query_vector=embedding, max_hops=2, max_tokens=4000)
print(result.context_text)      # Prompt-ready string
print(result.token_estimate)    # Token count estimate

# End-to-end Q&A
answer = graph.ask(query="Who leads ML?", query_vector=embedding, llm_callable=my_llm)
print(answer["answer"])
```

### DSPy Extraction

Requires `pip install graphmemory[extraction]`. Uses [DSPy](https://dspy.ai/) typed predictors to extract entities and relationships from text.

```python
from graphmemory.extraction import extract_and_store, extract_and_merge
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

text = """George Washington was the first President. Thomas Jefferson
served as Secretary of State under Washington."""

# Extract and insert (may create duplicates on repeated calls)
nodes, edges = extract_and_store(graph, text)

# Extract and merge (deduplicates against existing graph)
node_results, edge_results = extract_and_merge(graph, text, match_keys=["name"])
```

| Function | Description |
|----------|-------------|
| `extract_nodes(text)` | Extract entity nodes from text |
| `extract_edges(text, nodes)` | Extract relationships between known nodes |
| `extract(text)` | Extract both nodes and edges |
| `extract_and_store(graph, text)` | Extract and insert into graph |
| `extract_and_merge(graph, text, match_keys)` | Extract and merge (deduplicated) |

### Graph Algorithms

Requires `pip install graphmemory[algorithms]`. Powered by [NetworkX](https://networkx.org/).

```python
from graphmemory.algorithms import pagerank, betweenness_centrality, connected_components, to_networkx

scores = pagerank(graph)
centrality = betweenness_centrality(graph)
components = connected_components(graph)
G = to_networkx(graph)  # Export to NetworkX DiGraph
```

| Function | Description |
|----------|-------------|
| `pagerank(graph, alpha=0.85)` | PageRank scores for all nodes |
| `betweenness_centrality(graph)` | Betweenness centrality scores |
| `degree_distribution(graph)` | In/out/total degree per node |
| `connected_components(graph)` | Weakly connected components (largest first) |
| `to_networkx(graph)` | Export to `networkx.DiGraph` |

### Import / Export

```python
# Export
data = graph.export_graph(format="json")       # also: "csv", "graphml", "json_string"

# Import
graph.import_graph(data, format="json")
```

### Visualizer

Interactive D3.js force-directed graph visualization — opens in your browser with zero dependencies.

```python
# Open in browser
graph.visualize()

# Save to file
graph.visualize(output="my_graph.html", open_browser=False)
```

Features: drag nodes, zoom/pan, hover to highlight connections, click for detail panel, search bar, filter by node type.

## Data Models

| Model | Fields |
|-------|--------|
| `Node` | `id: UUID`, `type: str`, `properties: dict`, `vector: list[float]` |
| `Edge` | `id: UUID`, `source_id: UUID`, `target_id: UUID`, `relation: str`, `weight: float` |
| `NearestNode` | `node: Node`, `distance: float` |
| `SearchResult` | `node: Node`, `score: float` |
| `TraversalResult` | `node: Node`, `depth: int`, `path: list[UUID]` |
| `MergeResult` | `node: Node`, `created: bool` |
| `EdgeMergeResult` | `edge: Edge`, `created: bool` |
| `RetrievalResult` | `query: str`, `contexts: list`, `context_text: str`, `token_estimate: int` |

All IDs are auto-generated UUIDs. All models are [Pydantic](https://docs.pydantic.dev/) `BaseModel` instances.

## API Reference

### Connection

| Method | Description |
|--------|-------------|
| `GraphMemory(database=None, vector_length=3, distance_metric='l2', hnsw_ef_construction=128, hnsw_ef_search=64, hnsw_m=16, auto_index=True)` | Initialize. `None` = in-memory. HNSW index auto-created. |
| `close()` | Close connection (thread-safe, idempotent). |
| `transaction()` | Context manager for atomic operations. |

### Nodes

| Method | Description |
|--------|-------------|
| `insert_node(node) -> UUID` | Insert a node. |
| `bulk_insert_nodes(nodes) -> list[Node]` | Bulk insert. |
| `merge_node(node, match_keys, strategy=UPDATE) -> MergeResult` | Insert or update by property match. |
| `bulk_merge_nodes(nodes, match_keys, ...) -> list[MergeResult]` | Bulk merge. |
| `get_node(node_id) -> Node` | Get by ID. |
| `update_node(node_id, **kwargs) -> bool` | Update fields. |
| `delete_node(node_id)` | Delete node and its edges. |
| `bulk_delete_nodes(node_ids)` | Bulk delete. |
| `nodes_by_attribute(attr, value) -> list[Node]` | Query by property. |

### Edges

| Method | Description |
|--------|-------------|
| `insert_edge(edge)` | Insert an edge. |
| `bulk_insert_edges(edges)` | Bulk insert. |
| `merge_edge(edge) -> EdgeMergeResult` | Insert or update by (source, target, relation). |
| `bulk_merge_edges(edges) -> list[EdgeMergeResult]` | Bulk merge. |
| `get_edge(edge_id) -> Edge` | Get by ID. |
| `update_edge(edge_id, **kwargs) -> bool` | Update fields. |
| `delete_edge(source_id, target_id)` | Delete by endpoints. |
| `bulk_delete_edges(edge_ids)` | Bulk delete. |

### Search

| Method | Description |
|--------|-------------|
| `nearest_nodes(vector, limit) -> list[NearestNode]` | Vector similarity search. |
| `search_nodes(query_text, limit=10) -> list[SearchResult]` | Full-text BM25 search. |
| `hybrid_search(query_text, query_vector, ...) -> list[SearchResult]` | Combined text + vector search. |
| `create_index(ef_construction=None, ef_search=None, m=None)` | Create/recreate HNSW index with tunable params. Auto-called on init. |
| `compact_index()` | Compact HNSW index to reclaim space after deletions. |

### Retrieval

| Method | Description |
|--------|-------------|
| `retrieve(query, query_vector, ...) -> RetrievalResult` | Full GraphRAG retrieval pipeline. |
| `ask(query, query_vector, llm_callable, ...) -> dict` | Retrieval + LLM generation. |

### Traversal

| Method | Description |
|--------|-------------|
| `connected_nodes(node_id) -> list[Node]` | All nodes connected to a node. |
| `query() -> QueryBuilder` | Fluent query builder. |

### Import / Export

| Method | Description |
|--------|-------------|
| `export_graph(format='json')` | Export as JSON, CSV, GraphML, or JSON string. |
| `import_graph(data, format='json')` | Import from any supported format. |
| `visualize(output=None, open_browser=True) -> str` | Interactive D3.js graph visualization in the browser. |

## Examples

See `examples/` for complete usage:

- **`openai_example.py`** — OpenAI embeddings, similarity search, attribute queries
- **`lexical_graph.py`** — Wikipedia text with SentenceTransformer embeddings
- **`dspy_example_typed_pred.py`** — Knowledge graph extraction with DSPy

## Testing

296 tests covering all functionality.

```sh
python3 -m pytest tests/tests.py -v
```

## License

MIT License. See [LICENSE](LICENSE).

## Contributing

Contributions welcome — open an issue or submit a PR.
