from graphrag import GraphRAG
from graphrag.models import Node, Edge

import json
from openai import OpenAI
import os

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Use a LLM to extract structured data from a unstructured text (there are a variety of ways to do this)
text = "George Washington was the first President of the United States and served from 1789 to 1797."

def extract_dict(text):
    return json.loads(client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Extract structured data from this text: " + text}
        ]
        ).choices[0].message.content)


gw_dict = extract_dict(text)
print(gw_dict)

# Output:
# {
#   "President": "George Washington",
#   "Position": "First President",
#   "Country": "United States",
#   "Term": "1789-1797"
# }

def calculate_embedding(input_json):
    return client.embeddings.create(
        input=input_json,
        model="text-embedding-3-small"
    ).data[0].embedding


embedding = calculate_embedding(gw_dict)
print(embedding)

# Output:
# [-0.006929283495992422, -0.005336422007530928, ... (omitted for spacing),  0.04664124920964241, -0.024047505110502243]


# Initialize the database from disk (make sure to set vector_length correctly)
graph_db = GraphRAG(database='graph.db', vector_length=len(embedding))

# Insert nodes
node1 = Node(data=gw_dict, vector=embedding)
node1_id = graph_db.insert_node(node1)

text2 = "Thomas Jefferson was the first Secretary of State of the United States and served from 1797 to 1801."
tj_dict = extract_dict(text2)

node2 = Node(data=tj_dict, vector=calculate_embedding(tj_dict))
node2_id = graph_db.insert_node(node2)

# Insert edge
edge = Edge(source_id=node1_id, target_id=node2_id, relation="served_under", weight=0.5)
graph_db.insert_edge(edge)

# Print all nodes in the database
nodes = graph_db.nodes_to_json()
print("Nodes:", nodes)

# Print all edges in the database
edges = graph_db.edges_to_json()
print("Edges:", edges)

# Find connected nodes
connected_nodes = graph_db.connected_nodes(node1_id)
print("Connected Nodes:", connected_nodes)

# Find nearest neighbors
neighbors = graph_db.nearest_neighbors(calculate_embedding({"President": "George Washington"}), limit=1)
print("Nearest Neighbors:", neighbors)

