from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uuid

class GraphEntity(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, description="The unique identifier for the entity.")

    class Config:
        extra = 'forbid'

class Node(GraphEntity):
    properties: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Properties of the entity.")
    type: Optional[str] = Field(default=None, description="Optional label for the node to categorize it, ex: Person")
    vector: Optional[List[float]] = Field(default_factory=list, description="Vector representation of the node.")

    class Config:
        extra = 'forbid'

class Edge(GraphEntity):
    source_id: uuid.UUID
    target_id: uuid.UUID
    relation: str = Field(default=None, description="Relation between the source and target nodes")
    weight: Optional[float] = Field(default=None, description="Weight of the edge")

    class Config:
        extra = 'forbid'

class NearestNode(BaseModel):
    node: Node
    distance: float