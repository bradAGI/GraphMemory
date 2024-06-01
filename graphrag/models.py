from pydantic import BaseModel
from typing import List, Dict, Any

class Node(BaseModel):
    id: int = None
    data: Dict[str, Any]
    vector: List[float]

class Edge(BaseModel):
    source_id: int
    target_id: int
    relation: str
    weight: float

class Neighbor(BaseModel):
    node: Node
    distance: float
