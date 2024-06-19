from graphmemory import GraphMemory, Node
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import requests

# Get wikipedia page on Hoover Dam
r = requests.get(
    'https://en.wikipedia.org/w/api.php?action=query&format=json&titles=Hoover%20Dam&prop=extracts&explaintext')
text = r.json()['query']['pages']['14308']['extract']

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(text)

model = SentenceTransformer("all-MiniLM-L6-v2")
client = OpenAI()
graph_memory = GraphMemory(
    database="hoover.db",
    vector_length=model.get_sentence_embedding_dimension()
)

for text in texts:
    embedding = model.encode(text)
    embedding = [float(e) for e in embedding]
    node = Node(type="text", properties={"content": text}, vector=embedding)
    graph_memory.insert_node(node)

query = "How many tourists per year visit the Hoover Dam?"
query_embedding = model.encode(query)
query_embedding = [float(e) for e in query_embedding]
results = graph_memory.nearest_nodes(vector=query_embedding, limit=1)

if results:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": f"You are a helpful assistant. \
                   Answer the following question {query} based on provided context."},
                  {"role": "user", "content": "Context is: " + results[0].node.properties["content"]}]
    )
    print(response.choices[0].message.content)
else:
    print("No results found.")
