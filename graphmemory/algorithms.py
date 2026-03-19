"""Graph algorithms module for GraphMemory.

Provides PageRank, betweenness centrality, degree distribution, and connected
components via NetworkX as an optional dependency.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graphmemory.database import GraphMemory

logger = logging.getLogger(__name__)


def _require_networkx():
    """Import and return networkx, raising a clear error if unavailable."""
    try:
        import networkx as nx
    except ImportError:
        raise ImportError(
            "networkx is required for graph algorithms. "
            "Install it with: pip install graphmemory[algorithms]"
        ) from None
    return nx


def to_networkx(graph: GraphMemory) -> "nx.DiGraph":
    """Export a GraphMemory instance to a NetworkX DiGraph.

    Nodes are keyed by their UUID string. Node attributes include ``type``,
    ``properties``, and ``vector``. Edge attributes include ``relation``,
    ``weight``, and the original edge ``id``.

    Args:
        graph: A :class:`GraphMemory` instance.

    Returns:
        A ``networkx.DiGraph`` representing the full graph.
    """
    nx = _require_networkx()
    G = nx.DiGraph()

    for n in graph.nodes_to_json():
        G.add_node(
            n["id"],
            type=n["type"],
            properties=n["properties"],
            vector=n["vector"],
        )

    for e in graph.edges_to_json():
        G.add_edge(
            e["source_id"],
            e["target_id"],
            id=e["id"],
            relation=e["relation"],
            weight=e["weight"] if e["weight"] is not None else 1.0,
        )

    return G


def pagerank(
    graph: GraphMemory,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1.0e-6,
) -> dict[str, float]:
    """Compute PageRank for every node in the graph.

    Args:
        graph: A :class:`GraphMemory` instance.
        alpha: Damping factor (default 0.85).
        max_iter: Maximum iterations for the power-method solver.
        tol: Convergence tolerance.

    Returns:
        A dict mapping node-id strings to their PageRank score.
    """
    nx = _require_networkx()
    G = to_networkx(graph)
    return nx.pagerank(G, alpha=alpha, max_iter=max_iter, tol=tol)


def betweenness_centrality(
    graph: GraphMemory,
    normalized: bool = True,
    endpoints: bool = False,
) -> dict[str, float]:
    """Compute betweenness centrality for every node.

    Args:
        graph: A :class:`GraphMemory` instance.
        normalized: If ``True``, scores are normalized to ``[0, 1]``.
        endpoints: If ``True``, include the endpoints in shortest-path counts.

    Returns:
        A dict mapping node-id strings to their betweenness centrality score.
    """
    nx = _require_networkx()
    G = to_networkx(graph)
    return nx.betweenness_centrality(G, normalized=normalized, endpoints=endpoints)


def degree_distribution(graph: GraphMemory) -> dict[str, dict[str, int]]:
    """Compute degree statistics for every node.

    Args:
        graph: A :class:`GraphMemory` instance.

    Returns:
        A dict mapping node-id strings to a dict with keys
        ``in_degree``, ``out_degree``, and ``total_degree``.
    """
    nx = _require_networkx()
    G = to_networkx(graph)
    result: dict[str, dict[str, int]] = {}
    for node_id in G.nodes:
        result[node_id] = {
            "in_degree": G.in_degree(node_id),
            "out_degree": G.out_degree(node_id),
            "total_degree": G.in_degree(node_id) + G.out_degree(node_id),
        }
    return result


def connected_components(graph: GraphMemory) -> list[set[str]]:
    """Find weakly connected components of the graph.

    Uses the *weakly* connected components algorithm (appropriate for
    directed graphs) so that edge direction is ignored when deciding
    reachability.

    Args:
        graph: A :class:`GraphMemory` instance.

    Returns:
        A list of sets, each containing the node-id strings of one
        connected component, sorted largest-first.
    """
    nx = _require_networkx()
    G = to_networkx(graph)
    components = list(nx.weakly_connected_components(G))
    components.sort(key=len, reverse=True)
    return components
