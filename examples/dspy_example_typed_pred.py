import dspy
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from graphmemory import GraphMemory, Node, Edge

lm = dspy.OpenAI(
    model="gpt-3.5-turbo",
)
dspy.settings.configure(lm=lm)


class NodeOutput(BaseModel):
    """
    A node in the knowledge graph.
    """
    properties: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Properties of the entity. ex: name, age, gender, etc.")
    type: Optional[str] = Field(
        default=None, description="Optional label for the node to categorize it, ex: Person")
    proper_noun: bool = Field(
        default=False, description="Whether the node is a proper noun.")


class EdgeOutput(BaseModel):
    source_id: str = Field(description="The source node id")
    target_id: str = Field(description="The target node id")
    relation: str = Field(
        default=None, description="Relation between the source and target nodes. ex: constructed_by, served_under, etc.")

# Define the signature for the nodes and edges


class NodesSignature(dspy.Signature):
    """
    A signature for updating a knowledge graph nodes with new information.
    """
    input_text: str = dspy.InputField(
        description="The input text to extract nodes from.")
    output_nodes: List[NodeOutput] = dspy.OutputField(
        description="The proper noun node list updated from input text without any duplicates.")


class EdgesSignature(dspy.Signature):
    """
    A signature for updating a knowledge graph edges with new information.
    """
    input_text: str = dspy.InputField(
        description="The unstructured input text to extract edges from.")
    input_nodes: List[Node] = dspy.InputField(
        description="The nodes in the graph to connect.")
    output_edges: List[EdgeOutput] = dspy.OutputField(
        description="The edge list connecting the nodes in the graph.")


unstructured_text = '''
Hoover Dam is a concrete arch-gravity dam in the Black Canyon of the Colorado River,
on the border between the U.S. states of Nevada and Arizona. Constructed between
1931 and 1936, during the Great Depression, it was dedicated on September 30, 1935,
by President Franklin D. Roosevelt. Its construction was the result of a massive
effort involving thousands of workers, and cost over 100 lives. In bills passed by
Congress during its construction, it was referred to as the Hoover Dam, after
President Herbert Hoover, but was named the Boulder Dam by the Roosevelt
administration. In 1947, the name Hoover Dam was restored by Congress.

Since about 1900, the Black Canyon and nearby Boulder Canyon had been investigated
for their potential to support a dam that would control floods, provide irrigation
water and produce hydroelectric power. In 1928, Congress authorized the project.
The winning bid to build the dam was submitted by a consortium named Six Companies,
Inc., which began construction in early 1931. Such a large concrete structure had
never been built before, and some of the techniques used were unproven. The torrid
summer weather and lack of facilities near the site also presented difficulties.
Nevertheless, Six Companies turned the dam over to the federal government on March
1, 1936, more than two years ahead of schedule.
'''

sentences = unstructured_text.split(".")
nodes_predictor = dspy.TypedPredictor(NodesSignature)
edges_predictor = dspy.TypedPredictor(EdgesSignature)

nodes = []
edges = []

for sentence in sentences:
    try:
        new_nodes_dicts = nodes_predictor(input_text=sentence).output_nodes
        new_nodes = [Node(properties=node_dict.properties, type=node_dict.type)
                     for node_dict in new_nodes_dicts if node_dict.properties and node_dict.proper_noun]
        for node in new_nodes:
            nodes.append(node)
            print(f"Added new node: {node.properties}")
    except Exception as e:
        pass

for sentence in sentences:
    try:
        new_edges_dict = edges_predictor(
            input_text=sentence, input_nodes=nodes).output_edges
        new_edges = [Edge(source_id=edge.source_id, target_id=edge.target_id,
                          relation=edge.relation) for edge in new_edges_dict]
        for edge in new_edges:
            if edge.source_id and edge.target_id and edge.relation:
                edges.append(edge)
                print(f"Added new edge: {edge.source_id} - {edge.relation} - {edge.target_id}")
    except Exception as e:
        pass


# Create an instance of GraphMemory
graph_memory = GraphMemory(database="hoover.db")
# Bulk insert nodes
inserted_nodes = graph_memory.bulk_insert_nodes(nodes)
print(f"Inserted {len(inserted_nodes)} nodes.")
# Bulk insert edges
graph_memory.bulk_insert_edges(edges)
print(f"Inserted {len(edges)} edges.")

# Print the graph
graph_memory.print_json()
