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
