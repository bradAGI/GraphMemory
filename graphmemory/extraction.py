"""DSPy-based node and relationship extraction for GraphMemory.

Extracts entities (nodes) and relationships (edges) from unstructured text
using DSPy typed predictors, then inserts them into a GraphMemory instance.

Requires the ``dspy`` optional dependency:
    pip install graphmemory[extraction]
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel, Field

from graphmemory.models import Edge, EdgeMergeResult, MergeResult, MergeStrategy, Node

if TYPE_CHECKING:
    from graphmemory.database import GraphMemory

logger = logging.getLogger(__name__)


def _require_dspy():
    """Import and return dspy, raising a clear error if unavailable."""
    try:
        import dspy
    except ImportError:
        raise ImportError(
            "dspy is required for extraction. "
            "Install it with: pip install graphmemory[extraction]"
        ) from None
    return dspy


# ---------------------------------------------------------------------------
# Pydantic output schemas for DSPy typed predictors
# ---------------------------------------------------------------------------


class ExtractedNode(BaseModel):
    """A node extracted from text."""

    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="Properties of the entity, e.g. name, age, location.",
    )
    type: str | None = Field(
        default=None,
        description="Category label for the node, e.g. Person, Organization, Location.",
    )
    proper_noun: bool = Field(
        default=False,
        description="Whether this entity is a proper noun (named entity).",
    )


class ExtractedEdge(BaseModel):
    """A relationship extracted from text."""

    source_id: str = Field(description="UUID string of the source node.")
    target_id: str = Field(description="UUID string of the target node.")
    relation: str = Field(
        description="Relation label, e.g. constructed_by, located_in, served_under.",
    )


# ---------------------------------------------------------------------------
# DSPy Signatures (lazily built so dspy import is deferred)
# ---------------------------------------------------------------------------

_NodeSignature = None
_EdgeSignature = None


def _get_signatures():
    """Build and cache DSPy Signature classes (deferred import)."""
    global _NodeSignature, _EdgeSignature
    if _NodeSignature is not None:
        return _NodeSignature, _EdgeSignature

    dspy = _require_dspy()

    class NodeExtractionSignature(dspy.Signature):
        """Extract named entities (proper nouns) from text as knowledge-graph nodes."""

        input_text: str = dspy.InputField(
            description="The unstructured text to extract entities from."
        )
        output_nodes: list[ExtractedNode] = dspy.OutputField(
            description="Deduplicated list of proper-noun entities found in the text."
        )

    class EdgeExtractionSignature(dspy.Signature):
        """Extract relationships between known nodes from text."""

        input_text: str = dspy.InputField(
            description="The unstructured text to extract relationships from."
        )
        input_nodes: list[Node] = dspy.InputField(
            description="The existing nodes in the graph that may be connected."
        )
        output_edges: list[ExtractedEdge] = dspy.OutputField(
            description="Relationships between the provided nodes found in the text."
        )

    _NodeSignature = NodeExtractionSignature
    _EdgeSignature = EdgeExtractionSignature
    return _NodeSignature, _EdgeSignature


# ---------------------------------------------------------------------------
# Core extraction API
# ---------------------------------------------------------------------------


def _make_predictor(dspy, signature):
    """Create a DSPy v3 predictor from a Signature."""
    return dspy.Predict(signature)


def extract_nodes(text: str, sentences: list[str] | None = None) -> list[Node]:
    """Extract entity nodes from text using a DSPy predictor.

    Args:
        text: Full text to extract from (used when *sentences* is ``None``).
        sentences: Optional pre-split sentences. If provided, each sentence is
            processed individually for finer-grained extraction. When ``None``,
            the text is split on ``'.'``.

    Returns:
        A list of :class:`~graphmemory.models.Node` instances (proper nouns only).
    """
    dspy = _require_dspy()
    NodeSig, _ = _get_signatures()
    predictor = _make_predictor(dspy, NodeSig)

    if sentences is None:
        sentences = [s.strip() for s in text.split(".") if s.strip()]

    nodes: list[Node] = []
    for sentence in sentences:
        try:
            result = predictor(input_text=sentence)
            for extracted in result.output_nodes:
                if extracted.properties and extracted.proper_noun:
                    nodes.append(
                        Node(properties=extracted.properties, type=extracted.type)
                    )
        except Exception as e:
            logger.warning("Node extraction failed for sentence: %s", e)

    return nodes


def extract_edges(
    text: str,
    nodes: list[Node],
    sentences: list[str] | None = None,
) -> list[Edge]:
    """Extract relationship edges from text given a set of known nodes.

    Args:
        text: Full text to extract from (used when *sentences* is ``None``).
        nodes: Existing nodes whose IDs may appear as source/target.
        sentences: Optional pre-split sentences.

    Returns:
        A list of :class:`~graphmemory.models.Edge` instances.
    """
    dspy = _require_dspy()
    _, EdgeSig = _get_signatures()
    predictor = _make_predictor(dspy, EdgeSig)

    if sentences is None:
        sentences = [s.strip() for s in text.split(".") if s.strip()]

    node_id_set = {str(n.id) for n in nodes}
    edges: list[Edge] = []
    for sentence in sentences:
        try:
            result = predictor(input_text=sentence, input_nodes=nodes)
            for extracted in result.output_edges:
                if (
                    extracted.source_id in node_id_set
                    and extracted.target_id in node_id_set
                    and extracted.relation
                ):
                    edges.append(
                        Edge(
                            source_id=extracted.source_id,
                            target_id=extracted.target_id,
                            relation=extracted.relation,
                        )
                    )
        except Exception as e:
            logger.warning("Edge extraction failed for sentence: %s", e)

    return edges


def extract(
    text: str,
    sentences: list[str] | None = None,
) -> tuple[list[Node], list[Edge]]:
    """Extract both nodes and edges from unstructured text.

    This is a convenience wrapper that calls :func:`extract_nodes` then
    :func:`extract_edges` in sequence.

    Args:
        text: The unstructured text to process.
        sentences: Optional pre-split sentences.

    Returns:
        A ``(nodes, edges)`` tuple.
    """
    if sentences is None:
        sentences = [s.strip() for s in text.split(".") if s.strip()]

    nodes = extract_nodes(text, sentences=sentences)
    edges = extract_edges(text, nodes, sentences=sentences)
    return nodes, edges


def extract_and_store(
    graph: GraphMemory,
    text: str,
    sentences: list[str] | None = None,
) -> tuple[list[Node], list[Edge]]:
    """Extract nodes and edges from text and insert them into a GraphMemory instance.

    Args:
        graph: A :class:`~graphmemory.database.GraphMemory` instance.
        text: The unstructured text to process.
        sentences: Optional pre-split sentences.

    Returns:
        A ``(inserted_nodes, edges)`` tuple.
    """
    nodes, edges = extract(text, sentences=sentences)

    inserted_nodes = graph.bulk_insert_nodes(nodes) if nodes else []
    if edges:
        graph.bulk_insert_edges(edges)

    logger.info(
        "Extracted and stored %d nodes and %d edges.", len(inserted_nodes), len(edges)
    )
    return inserted_nodes, edges


def extract_and_merge(
    graph: GraphMemory,
    text: str,
    match_keys: list[str] | None = None,
    match_type: bool = True,
    strategy: MergeStrategy = MergeStrategy.UPDATE,
    sentences: list[str] | None = None,
    similarity_threshold: float = 1.0,
    vector_threshold: float | None = None,
) -> tuple[list[MergeResult], list[EdgeMergeResult]]:
    """Extract nodes and edges from text, merging with existing graph data.

    Unlike :func:`extract_and_store`, this deduplicates against existing nodes
    by matching on the specified property keys, and deduplicates edges on
    ``(source_id, target_id, relation)``.

    Args:
        graph: A :class:`~graphmemory.database.GraphMemory` instance.
        text: The unstructured text to process.
        match_keys: Property names to match nodes on (default ``["name"]``).
        match_type: Also require ``node.type`` to match (default ``True``).
        strategy: How to merge properties on match.
        sentences: Optional pre-split sentences.

    Returns:
        A ``(node_results, edge_results)`` tuple of merge results.
    """
    if match_keys is None:
        match_keys = ["name"]

    nodes, edges = extract(text, sentences=sentences)

    node_results = graph.bulk_merge_nodes(
        nodes, match_keys=match_keys, match_type=match_type, strategy=strategy,
        similarity_threshold=similarity_threshold, vector_threshold=vector_threshold,
    ) if nodes else []

    edge_results = graph.bulk_merge_edges(edges) if edges else []

    logger.info(
        "Extracted and merged %d nodes and %d edges.",
        len(node_results),
        len(edge_results),
    )
    return node_results, edge_results


# ---------------------------------------------------------------------------
# Parallel extraction
# ---------------------------------------------------------------------------


def _extract_nodes_chunk(chunk: str) -> list[Node]:
    """Extract nodes from a single chunk (thread-safe, no DB access)."""
    return extract_nodes(chunk, sentences=[chunk])


def _extract_edges_chunk(chunk: str, nodes: list[Node]) -> list[Edge]:
    """Extract edges from a single chunk given known nodes (thread-safe)."""
    return extract_edges(chunk, nodes, sentences=[chunk])


def extract_and_merge_parallel(
    graph: GraphMemory,
    chunks: list[str],
    match_keys: list[str] | None = None,
    match_type: bool = True,
    strategy: MergeStrategy = MergeStrategy.UPDATE,
    similarity_threshold: float = 1.0,
    vector_threshold: float | None = None,
    max_workers: int = 8,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> tuple[list[MergeResult], list[EdgeMergeResult]]:
    """Extract from multiple text chunks in parallel, then merge sequentially.

    Runs in two parallel phases to maximize LLM throughput:
      1. Node extraction — all chunks concurrently (saturate RPM)
      2. Edge extraction — all chunks concurrently (with all extracted nodes as context)
    Then merges into DB sequentially.

    Args:
        graph: A :class:`~graphmemory.database.GraphMemory` instance.
        chunks: List of text chunks to process.
        match_keys: Property names to match nodes on (default ``["name"]``).
        match_type: Also require ``node.type`` to match.
        strategy: How to merge properties on match.
        similarity_threshold: Jaro-Winkler threshold for fuzzy matching.
        vector_threshold: Max cosine distance for vector similarity.
        max_workers: Max concurrent LLM calls (match your RPM headroom).
        on_progress: Optional callback ``(phase, completed, total)``.

    Returns:
        Aggregated ``(node_results, edge_results)`` across all chunks.
    """
    if match_keys is None:
        match_keys = ["name"]

    total = len(chunks)

    # Phase 1: Extract nodes from ALL chunks in parallel
    chunk_nodes: dict[int, list[Node]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {
            pool.submit(_extract_nodes_chunk, chunk): i
            for i, chunk in enumerate(chunks)
        }
        done = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                chunk_nodes[idx] = future.result()
            except Exception as e:
                logger.warning("Node extraction failed for chunk %d: %s", idx + 1, e)
                chunk_nodes[idx] = []
            done += 1
            if on_progress:
                on_progress("nodes", done, total)

    # Merge all nodes into DB sequentially to build the full node set
    all_node_results: list[MergeResult] = []
    for idx in range(total):
        nodes = chunk_nodes.get(idx, [])
        if nodes:
            results = graph.bulk_merge_nodes(
                nodes, match_keys=match_keys, match_type=match_type,
                strategy=strategy, similarity_threshold=similarity_threshold,
                vector_threshold=vector_threshold,
            )
            all_node_results.extend(results)

    # Build complete node list for edge extraction context
    all_nodes = [r.node for r in all_node_results]
    logger.info("Phase 1 complete: %d nodes extracted and merged.", len(all_nodes))

    # Phase 2: Extract edges from ALL chunks in parallel (with full node context)
    chunk_edges: dict[int, list[Edge]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {
            pool.submit(_extract_edges_chunk, chunk, all_nodes): i
            for i, chunk in enumerate(chunks)
        }
        done = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                chunk_edges[idx] = future.result()
            except Exception as e:
                logger.warning("Edge extraction failed for chunk %d: %s", idx + 1, e)
                chunk_edges[idx] = []
            done += 1
            if on_progress:
                on_progress("edges", done, total)

    # Merge all edges into DB sequentially
    all_edge_results: list[EdgeMergeResult] = []
    for idx in range(total):
        edges = chunk_edges.get(idx, [])
        if edges:
            results = graph.bulk_merge_edges(edges)
            all_edge_results.extend(results)

    logger.info(
        "Parallel extraction complete: %d chunks, %d nodes, %d edges.",
        total, len(all_node_results), len(all_edge_results),
    )
    return all_node_results, all_edge_results
