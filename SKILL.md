---
name: graphmemory
description: Build and query embedded GraphRAG knowledge graphs with DuckDB-backed vector, full-text, and hybrid search. Use when the user wants to store entities and relations, run RAG over a graph, extract knowledge graphs from text with DSPy, run graph algorithms (PageRank, centrality, components), merge/upsert nodes and edges, fuzzy-dedupe an existing graph, or visualize interactively in a browser. Trigger phrases include "knowledge graph", "GraphRAG", "graph database", "hybrid search", "extract entities and relations", "DuckDB graph", "embedded graph store", "dedupe graph nodes".
---

# GraphMemory

Embedded GraphRAG database built on DuckDB. Single Python package — no server, no external services. Ships vector (HNSW), full-text (BM25), hybrid search, fluent query builder, multi-hop traversal, fuzzy dedup, DSPy extraction, NetworkX algorithms, and a zero-dep D3.js visualizer.

## When to reach for this

- Knowledge graph with semantic search (not just a vector DB, not just a graph DB).
- RAG where graph traversal is part of retrieval.
- Extract entities/relations from text and store them durably with dedup.
- Prototyping — file-backed or in-memory graph without Neo4j/Postgres/pgvector.

**Do not use** when: user already has Neo4j/Neptune/ArangoDB, or scale is hundreds of millions of nodes — GraphMemory is DuckDB-embedded, single-writer.

## Install

```sh
pip install graphmemory
pip install graphmemory[extraction]   # DSPy entity/relation extraction
pip install graphmemory[algorithms]   # NetworkX algorithms
```

## Decision table

| User intent | Method |
|---|---|
| Insert one node | `graph.insert_node(node)` |
| Bulk insert | `graph.bulk_insert_nodes(nodes)` |
| Insert-or-update by property | `graph.merge_node(node, match_keys=["name"])` |
| Fuzzy insert-or-update | `graph.merge_node(node, match_keys=["name"], similarity_threshold=0.9)` |
| Dedupe edges on `(src, tgt, relation)` | `graph.merge_edge(edge)` |
| Clean up existing duplicates | `graph.resolve_duplicates(match_keys=["name"], similarity_threshold=0.9)` |
| Pure vector kNN | `graph.nearest_nodes(vector, limit)` |
| Pure BM25 text | `graph.search_nodes(query, limit)` |
| Combined text + vector | `graph.hybrid_search(query, query_vector, text_weight, vector_weight)` |
| Lookup by property | `graph.nodes_by_attribute("name", "Alice")` |
| Direct neighbors | `graph.connected_nodes(node_id)` |
| Multi-hop traversal | `graph.query().traverse(source_id=id, depth=2).execute()` |
| Filtered query | `graph.query().match(type="Person").where(role="eng").execute()` |
| GraphRAG context assembly | `graph.retrieve(query, query_vector, max_hops, max_tokens)` |
| End-to-end Q&A | `graph.ask(query, query_vector, llm_callable=fn)` |
| Extract + store from text | `extract_and_merge(graph, text, match_keys=["name"])` |
| Extract in parallel across chunks | `extract_and_merge_parallel(graph, chunks, max_workers=8)` |
| PageRank / centrality | `pagerank(graph)`, `betweenness_centrality(graph)` |
| Atomic block | `with graph.transaction(): ...` |
| Browser visualization | `graph.visualize()` |

## Canonical snippets

### Init

```python
from graphmemory import GraphMemory, Node, Edge, MergeStrategy

# database=None is in-memory; pass a path for persistence.
# vector_length and distance_metric are fixed at init time.
graph = GraphMemory(
    database="graph.db",
    vector_length=1536,              # must match your embedding model
    distance_metric="cosine",        # "l2" | "cosine" | "inner_product"
    hnsw_ef_construction=128,
    hnsw_ef_search=64,
    hnsw_m=16,
    auto_index=True,                 # HNSW auto-built on init
    max_retries=3,                   # transient IO error retry
)
```

### Insert + merge

```python
alice = Node(type="Person", properties={"name": "Alice"}, vector=embed("Alice"))
bob = Node(type="Person", properties={"name": "Bob"}, vector=embed("Bob"))
graph.insert_node(alice)
graph.insert_node(bob)
graph.insert_edge(Edge(source_id=alice.id, target_id=bob.id, relation="reports_to"))

# Idempotent re-ingest on a natural key
graph.merge_node(alice, match_keys=["name"])

# Fuzzy merge — tolerates "Alice Smith" vs "alice smith"
graph.merge_node(
    alice,
    match_keys=["name"],
    similarity_threshold=0.9,        # Jaro-Winkler threshold (1.0 = exact)
    vector_threshold=0.2,            # optional cosine distance cap
    match_type=True,                 # also require same `type`
    strategy=MergeStrategy.UPDATE,   # UPDATE | REPLACE | KEEP
)
```

### Hybrid search

```python
results = graph.hybrid_search(
    query_text="who leads ML?",
    query_vector=embed("who leads ML?"),
    text_weight=0.5,
    vector_weight=0.5,
    limit=10,
)
for r in results:
    print(r.score, r.node.properties)
```

### GraphRAG

```python
# Context-only (own the prompt)
result = graph.retrieve(
    query=q, query_vector=qv,
    max_hops=2, max_tokens=4000, search_limit=10,
)
print(result.context_text, result.token_estimate, result.seed_node_count, result.total_node_count)

# End-to-end — llm_callable signature: (system_prompt, user_prompt) -> str
def my_llm(system, user):
    return openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    ).choices[0].message.content

answer = graph.ask(query=q, query_vector=qv, llm_callable=my_llm)
print(answer["answer"])
```

Pass `llm_callable=None` to get retrieval-only output — useful to inspect the context before wiring an LLM.

### Query builder

```python
# Filter by type + property
engineers = graph.query().match(type="Person").where(role="engineer").execute()

# Multi-hop traversal — returns TraversalResult with depth + path
two_hop = graph.query().traverse(source_id=alice.id, depth=2).execute()

# Paginate + order
page = graph.query().match(type="Person").order_by("name").limit(20).offset(40).execute()

# Return edges instead of nodes
edges = graph.query().match(type="Person").edges().execute()
```

### DSPy extraction

```python
import dspy
from graphmemory.extraction import extract_and_merge, extract_and_merge_parallel

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# Single pass
node_results, edge_results = extract_and_merge(
    graph, text, match_keys=["name"], similarity_threshold=0.88,
)

# Parallel across chunks — two phases: nodes first (all chunks), then edges
# with the full node context. Saturates your RPM.
node_results, edge_results = extract_and_merge_parallel(
    graph,
    chunks=paragraph_chunks,
    match_keys=["name"],
    similarity_threshold=0.88,
    max_workers=8,                   # match your provider's RPM headroom
    on_progress=lambda phase, done, total: print(f"{phase}: {done}/{total}"),
)
```

### Transactions

```python
with graph.transaction():
    graph.insert_node(a)
    graph.insert_node(b)
    graph.insert_edge(Edge(source_id=a.id, target_id=b.id, relation="x"))
# Exception inside the block → ROLLBACK. Clean exit → COMMIT.
```

## Advanced patterns

### Two-pass dedup (idiomatic)

Extract with a loose threshold, then clean up with a tighter one. This is the pattern in `examples/test_ingest.py`.

```python
# Pass 1 — during ingest, be permissive to avoid fragmenting entities
extract_and_merge_parallel(graph, chunks, similarity_threshold=0.88, max_workers=50)

# Pass 2 — after ingest, resolve residual duplicates more strictly
clusters = graph.resolve_duplicates(
    match_keys=["name"],
    match_type=True,
    similarity_threshold=0.9,
    vector_threshold=0.15,
)
for c in clusters:
    print(f"Kept {c.survivor.properties['name']}, merged {len(c.merged)} dups")
```

`resolve_duplicates` picks the first-seen node as survivor, reassigns all incoming/outgoing edges to it, and deletes the rest. Self-loops from the reassignment are dropped.

### Custom chunking + sequential linking

Pattern from `examples/lexical_graph.py`:

```python
prev = None
for chunk in chunks:
    node = Node(type="Chunk", properties={"text": chunk}, vector=embed(chunk))
    graph.insert_node(node)
    if prev is not None:
        graph.insert_edge(Edge(source_id=prev.id, target_id=node.id, relation="followed_by"))
    prev = node
```

### Inspect before asking

```python
result = graph.retrieve(query=q, query_vector=qv, max_hops=2, max_tokens=4000)
print(result.context_text)   # See exactly what the LLM would receive
# Tune max_hops / max_tokens / search_limit before wiring ask()
```

## Gotchas

- **`vector_length` and `distance_metric` are locked at init.** Swapping embedding models means a new database. Valid metrics: `"l2"`, `"cosine"`, `"inner_product"`.
- **Missing vectors are silently zero-filled** in `insert_node` — `bulk_insert_nodes` skips nodes whose vectors don't match `vector_length` and logs a warning. Validate upstream if correctness matters.
- **HNSW is auto-built on init** (`auto_index=True`). Tune via `hnsw_ef_construction`, `hnsw_ef_search`, `hnsw_m`. Call `graph.compact_index()` after heavy deletes to reclaim space (also called automatically by `delete_node`).
- **FTS index is lazy** — first `search_nodes`/`hybrid_search` call after writes rebuilds it. Expect first-search latency. Force a rebuild with `graph.reindex()` if you want it warm before traffic.
- **Edge dedup key is `(source_id, target_id, relation)`.** Relations are normalized (lowercased, underscored) before comparison — `"Reports To"` and `"reports_to"` collide. Edge properties are NOT part of the key.
- **`delete_node` cascades edges in both directions** (as source AND as target). No orphan-edge safety net.
- **`merge_node` strategies** — `UPDATE` shallow-merges dicts (incoming wins on collision), `REPLACE` overwrites wholesale, `KEEP` only inserts if new. Pick intentionally.
- **`similarity_threshold=1.0` is exact match** (the default). Lower it to enable Jaro-Winkler fuzzy matching on string properties. Non-string properties always use JSON equality.
- **`match_type=True` (default) requires same `type` for merge.** Set `False` to merge across types — rarely what you want.
- **`resolve_duplicates` is O(n²)-ish** in fuzzy mode. For large graphs, narrow with `match_type` and a tight `vector_threshold` first.
- **`extraction` and `algorithms` are optional extras.** Wrap imports in try/except or check `pip show` before recommending code that depends on them.
- **Single-writer DuckDB.** Connection pooling and `@with_retry` (exponential backoff on transient IO errors) are built in, but don't open the same file from multiple processes for concurrent writes.
- **`cursor()` returns independent cursors** for concurrent reads; the main connection is RLock-guarded for writes.
- **`ask()` with `llm_callable=None`** returns retrieval only — no generation. Always use this first to validate context before paying for LLM calls.

## Data models

| Model | Key fields |
|---|---|
| `Node` | `id: UUID`, `type: str \| None`, `properties: dict`, `vector: list[float]` |
| `Edge` | `id`, `source_id`, `target_id`, `relation: str`, `weight: float \| None` |
| `SearchResult` | `node`, `score` (higher = better for both BM25 and hybrid) |
| `NearestNode` | `node`, `distance` (lower = closer) |
| `TraversalResult` | `node`, `depth`, `path: list[UUID]` |
| `RetrievalContext` | `node`, `relationships: list[dict]`, `hop_distance: int` |
| `RetrievalResult` | `query`, `contexts`, `context_text`, `token_estimate`, `seed_node_count`, `total_node_count` |
| `MergeResult` | `node`, `created: bool` (True = inserted, False = updated) |
| `EdgeMergeResult` | `edge`, `created: bool` |
| `DuplicateCluster` | `survivor: Node`, `merged: list[Node]` |

All models are Pydantic. IDs auto-generate as UUIDs.

## Examples in the repo

- `examples/openai_example.py` — OpenAI embeddings, similarity search, attribute lookup
- `examples/lexical_graph.py` — chunked Wikipedia text with SentenceTransformer, sequential `followed_by` edges
- `examples/dspy_example_typed_pred.py` — DSPy typed-predictor extraction
- `examples/test_ingest.py` — parallel extraction (50 workers, 0.88 threshold) + post-pass `resolve_duplicates` at 0.90

Read `examples/test_ingest.py` before building a real ingest pipeline — it's the template.

## Testing

```sh
python3 -m pytest tests/tests.py -v
```

296 tests cover the public API. Run them when modifying the library.
