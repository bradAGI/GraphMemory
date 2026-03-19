# GraphMemory — Gap Analysis Report

**Date:** 2026-03-19
**Repo:** https://github.com/bradAGI/GraphMemory

---

## Executive Summary

GraphMemory is an embedded graph database with vector similarity search (VSS) built on DuckDB. The core functionality (CRUD for nodes/edges, vector search, basic Cypher support) works but has significant gaps in packaging, security, API completeness, testing, and production-readiness.

---

## 1. Critical Gaps

### 1.1 Security — SQL Injection in Cypher Parser
**File:** `graphmemory/database.py:291-408`

The `_cypher_to_sql` method builds SQL via string interpolation with user-supplied values. Property values from Cypher queries are embedded directly into the SQL string without parameterization:

```python
where_conditions.append(f"json_extract({alias}.properties, '$.{prop}') = json('{json.dumps(val)}')")
```

An attacker can craft a Cypher query with malicious property values to execute arbitrary SQL. This should use parameterized queries.

### 1.2 Bug — `delete_node` Deletes in Wrong Order
**File:** `graphmemory/database.py:150-158`

The method deletes the node first, then tries to delete associated edges. If foreign key constraints are enforced, this will fail. Edges should be deleted **before** the node:

```python
# Current (wrong order):
DELETE FROM nodes WHERE id = ?
DELETE FROM edges WHERE source_id = ? OR target_id = ?

# Correct order:
DELETE FROM edges WHERE source_id = ? OR target_id = ?
DELETE FROM nodes WHERE id = ?
```

### 1.3 Bug — `load_database` Double-Attach on Init
**File:** `graphmemory/database.py:25-26, 38-45`

When a database file path is provided and it exists, `__init__` first connects to it via `duckdb.connect(database=self.database)` and then calls `self.load_database(database)` which tries to `ATTACH DATABASE` the same file again. This will raise an error or produce unexpected behavior.

### 1.4 Type Hint Bug — `get_nodes_vector`
**File:** `graphmemory/database.py:265`

The parameter is typed as `node_id: int` but should be `node_id: uuid.UUID` to match the rest of the API.

---

## 2. Missing Project Infrastructure

### 2.1 No Package Configuration
There is no `setup.py`, `setup.cfg`, or `pyproject.toml`. The README says `pip install graphmemory` but there's no packaging config in the repo to build or publish the package. This is essential for distribution.

### 2.2 No `.gitignore`
No `.gitignore` exists. Database files (`.db`), `__pycache__`, `.egg-info`, virtual environments, etc. could accidentally be committed.

### 2.3 No Docker Support
No `Dockerfile` or `docker-compose.yml`. For a database library, having a containerized test/demo environment would be valuable.

### 2.4 No CI/CD Pipeline
No GitHub Actions, no `.github/workflows/`, no CI configuration of any kind. Tests are not automatically run on push or PR.

### 2.5 No Linting / Formatting Configuration
No `ruff.toml`, `pyproject.toml` lint config, `mypy.ini`, or similar. Code quality is not enforced.

---

## 3. API Completeness Gaps

### 3.1 No Update Operations
There is no `update_node()` or `update_edge()` method. Users cannot modify existing nodes or edges without deleting and re-inserting them.

### 3.2 No Edge Query Methods
- Cannot query edges by relation type (e.g., "find all 'friendship' edges")
- Cannot get edges for a specific node
- Cannot get an edge by its ID

### 3.3 No Multi-Hop Traversal
`connected_nodes()` only returns 1-hop neighbors. There is no support for:
- N-hop traversal
- Shortest path
- Path finding between two nodes

### 3.4 No Pagination
`nodes_to_json()` and `edges_to_json()` fetch **all** records with no limit/offset. This will fail or be very slow on large databases.

### 3.5 No Node Count / Edge Count
No methods to get the total number of nodes or edges in the database.

### 3.6 No Edge Properties
Edges only support `relation` (string) and `weight` (float). There is no generic `properties` field like nodes have. This limits expressiveness for many graph use cases.

### 3.7 Missing `close()` Method
The class supports context manager (`__enter__`/`__exit__`) but has no standalone `close()` method. Users who don't use `with` blocks have no clean way to close the connection.

### 3.8 Cypher Parser Very Limited
The Cypher-to-SQL translator only handles simple `MATCH (n:Label {props}) RETURN n` patterns. It does not support:
- `WHERE` clauses
- `CREATE`, `DELETE`, `SET` operations
- Variable-length paths `(a)-[*1..3]->(b)`
- `OPTIONAL MATCH`
- Aggregations (`COUNT`, `SUM`, etc.)
- `ORDER BY`, `LIMIT`, `SKIP`

### 3.9 No Return Type on `insert_node`
The method signature says `-> uuid.UUID` but it can return `None`. Should be `-> Optional[uuid.UUID]`.

---

## 4. Testing Gaps

### 4.1 Missing Test Coverage
The following methods/features have **no tests**:
- `nodes_by_attribute()` — no test
- `get_nodes_vector()` — no test
- `print_json()` — no test
- `load_database()` — no test
- `set_vector_length()` — no test
- `cypher()` with relationship patterns — no test
- Context manager (`__enter__` / `__exit__`) — no test
- `create_index()` effect on search performance — no test

### 4.2 Test File Issues
- `tests/tests.py:5` adds `../src` to sys.path but there is no `src/` directory — this is dead code
- Duplicate `import unittest` at line 239
- `print("*****", self.node_id)` debug statement left in test code (line 254)
- `logger.info(f"Inserted Node ID: {self.node_id}")` called before `self.node_id` is set (line 252)

### 4.3 No Integration Tests
All tests use `:memory:` databases. There are no tests for file-based database persistence, reopening databases, or concurrent access.

---

## 5. Code Quality Issues

### 5.1 Inconsistent Error Handling
- `insert_node()` silently returns `None` on error
- `insert_edge()` raises `ValueError` for missing nodes but silently logs `duckdb.Error`
- `delete_node()`, `delete_edge()` silently swallow errors
- No consistent pattern for error reporting to callers

### 5.2 Hardcoded Logging Configuration
`database.py:13-14` calls `logging.basicConfig(level=logging.INFO)` at module level. This overrides the application's logging configuration, which is an anti-pattern for a library.

### 5.3 Redundant Import
`database.py:293` has `import json` inside `_cypher_to_sql` even though `json` is already imported at module level (line 2).

### 5.4 Unused Import Alias
`database.py:8` uses `from typing import Dict as D` — this aliasing is unnecessary and harms readability.

### 5.5 Pinned Old Dependencies
`requirements.txt` pins `duckdb==1.0.0` and `pydantic==2.7.3`. These are older versions and may be missing bug fixes and features.

---

## 6. Documentation Gaps

### 6.1 No API Reference Docs
No auto-generated or hand-written API reference beyond the README method list. No docstrings on any methods.

### 6.2 No CHANGELOG
No changelog or release notes.

### 6.3 Examples Use Deprecated APIs
- `examples/dspy_example_typed_pred.py` uses `dspy.OpenAI` which is deprecated in newer dspy versions
- Examples reference `gpt-3.5-turbo` which may be sunset

### 6.4 No Contributing Guide
README says "Contributions are welcome!" but there's no CONTRIBUTING.md with guidelines.

---

## 7. Production-Readiness Gaps

### 7.1 No Thread Safety
No locking, no connection pooling. Concurrent access from multiple threads will cause data corruption or crashes.

### 7.2 No Connection Health Check
No way to verify the database connection is still alive.

### 7.3 No Backup / Export / Import
No methods for backing up the database, exporting to standard formats (GraphML, JSON-LD, CSV), or importing from them.

### 7.4 No Schema Migration Support
If the schema changes (e.g., adding a column to nodes), there's no migration path for existing databases.

---

## Priority Recommendations

| Priority | Gap | Effort |
|----------|-----|--------|
| **P0** | Fix SQL injection in Cypher parser | Small |
| **P0** | Fix `delete_node` ordering bug | Small |
| **P0** | Fix `load_database` double-attach bug | Small |
| **P0** | Add `pyproject.toml` for packaging | Small |
| **P1** | Add `.gitignore` | Small |
| **P1** | Add `update_node` / `update_edge` methods | Medium |
| **P1** | Fix return type hints | Small |
| **P1** | Remove hardcoded `logging.basicConfig` | Small |
| **P1** | Add CI/CD (GitHub Actions) | Medium |
| **P2** | Add pagination to query methods | Medium |
| **P2** | Add edge query methods | Medium |
| **P2** | Increase test coverage | Medium |
| **P2** | Add thread safety | Medium |
| **P3** | Multi-hop graph traversal | Large |
| **P3** | Expand Cypher support | Large |
| **P3** | Add export/import formats | Medium |
