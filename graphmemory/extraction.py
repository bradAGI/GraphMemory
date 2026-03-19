"""DSPy-based node and relationship extraction for GraphMemory.

Extracts entities (nodes) and relationships (edges) from unstructured text
using DSPy typed predictors, then inserts them into a GraphMemory instance.

Requires the ``dspy`` optional dependency:
    pip install graphmemory[extraction]
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from graphmemory.models import Edge, Node

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


def extract_nodes(text: str, sentences: list[str] | None = None) -> list[Node]:
    """Extract entity nodes from text using a DSPy typed predictor.

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
    predictor = dspy.TypedPredictor(NodeSig)

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
    predictor = dspy.TypedPredictor(EdgeSig)

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
