import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class GraphEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: uuid.UUID = Field(default_factory=uuid.uuid4, description="The unique identifier for the entity.")


class Node(GraphEntity):
    properties: dict[str, Any] | None = Field(default_factory=dict, description="Properties of the entity.")
    type: str | None = Field(default=None, description="Optional label for the node to categorize it, ex: Person")
    vector: list[float] | None = Field(default_factory=list, description="Vector representation of the node.")


class Edge(GraphEntity):
    source_id: uuid.UUID
    target_id: uuid.UUID
    relation: str = Field(default=None, description="Relation between the source and target nodes")
    weight: float | None = Field(default=None, description="Weight of the edge")


class NearestNode(BaseModel):
    node: Node
    distance: float


class SearchResult(BaseModel):
    node: Node
    score: float


class TraversalResult(BaseModel):
    node: Node
    depth: int
    path: list[uuid.UUID] = Field(default_factory=list, description="Node IDs in the path from source to this node.")


class RetrievalContext(BaseModel):
    """A single piece of context from a retrieved node with its relationship path."""
    node: Node
    relationships: list[dict[str, Any]] = Field(default_factory=list, description="Edges connecting this node to the query results")
    hop_distance: int = Field(default=0, description="Number of hops from the nearest seed node")


class RetrievalResult(BaseModel):
    """Result of a full GraphRAG retrieval pipeline."""
    query: str
    contexts: list[RetrievalContext] = Field(default_factory=list, description="Retrieved context items ordered by relevance")
    context_text: str = Field(default="", description="Assembled prompt-ready context string")
    token_estimate: int = Field(default=0, description="Estimated token count of context_text")
    seed_node_count: int = Field(default=0, description="Number of initial nodes found by search")
    total_node_count: int = Field(default=0, description="Total nodes after graph expansion")
