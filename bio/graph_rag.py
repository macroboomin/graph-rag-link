import networkx as nx
import pickle
import torch
import json
import numpy as np
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
import warnings
import ast
from tqdm import tqdm
import pandas as pd
warnings.filterwarnings("ignore")

import os
import sys
# Load the knowledge graph
with open('knowledge_graph.gpickle', 'rb') as f:
    G_loaded = pickle.load(f)

# Initialize the model and tokenizer for Phi-3
model = AutoModelForCausalLM.from_pretrained(
    "EmergentMethods/Phi-3-mini-4k-instruct-graph",  
    device_map="cuda",  
    torch_dtype="auto",  
    trust_remote_code=True,  
)

tokenizer = AutoTokenizer.from_pretrained("EmergentMethods/Phi-3-mini-4k-instruct-graph")

pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer,
)

generation_args = { 
    "max_new_tokens": 3000, 
    "return_full_text": False, 
    "temperature": 0.0, 
    "do_sample": False, 
}

model_name = "intfloat/multilingual-e5-large-instruct"
embeddings_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"device": "cuda"}, 
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = FAISS.load_local('./faiss_paper', embeddings_model, allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={'k': 5, 'fetch_k': 50}
)

def extract_entities_and_relationships(query_text):
    messages = [
        {"role": "system", "content": """
        A chat between a curious user and an artificial intelligence Assistant. The Assistant is an expert at identifying entities and relationships in text. The Assistant responds in JSON output only.

        The User provides text in the format:

        -------Text begin-------
        <User provided text>
        -------Text end-------

        The Assistant follows the following steps before replying to the User:

        1. **identify the most important entities** The Assistant identifies the most important entities in the text. These entities are listed in the JSON output under the key "nodes", they follow the structure of a list of dictionaries where each dict is:

        "nodes":[{"id": <entity N>, "type": <type>, "detailed_type": <detailed type>}, ...]

        where "type": <type> is a broad categorization of the entity. "detailed type": <detailed_type>  is a very descriptive categorization of the entity.

        2. **determine relationships** The Assistant uses the text between -------Text begin------- and -------Text end------- to determine the relationships between the entities identified in the "nodes" list defined above. These relationships are called "edges" and they follow the structure of:

        "edges":[{"from": <entity 1>, "to": <entity 2>, "label": <relationship>}, ...]

        The <entity N> must correspond to the "id" of an entity in the "nodes" list.

        The Assistant never repeats the same node twice. The Assistant never repeats the same edge twice.
        The Assistant responds to the User in JSON only, according to the following JSON schema:

        {"type":"object","properties":{"nodes":{"type":"array","items":{"type":"object","properties":{"id":{"type":"string"},"type":{"type":"string"},"detailed_type":{"type":"string"}},"required":["id","type","detailed_type"],"additionalProperties":false}},"edges":{"type":"array","items":{"type":"object","properties":{"from":{"type":"string"},"to":{"type":"string"},"label":{"type":"string"}},"required":["from","to","label"],"additionalProperties":false}}},"required":["nodes","edges"],"additionalProperties":false}
        """}, 
        {"role": "user", "content": f"""
        -------Text begin-------
        {query_text}
        -------Text end-------
        """}
    ]
    output = pipe(messages, **generation_args)
    generated_text = output[0]['generated_text']
    
    try:
        result = eval(generated_text)
        return result.get('nodes', []), result.get('edges', [])
    except Exception as e:
        print(f"Failed to parse output: {e}")
        return [], []

def find_similar_and_adjacent_nodes(G, query_nodes):
    similar_nodes = set()
    for query_node in query_nodes:
        node_id = query_node.get('id')
        if not node_id:
            print(f"Warning: Missing 'id' in node: {query_node}")
            continue  # Skip this node if 'id' is missing
        
        for node, attrs in G.nodes(data=True):
            if node == node_id:
                similar_nodes.add(node)
                
                # Add all directly adjacent nodes
                neighbors = set(G.predecessors(node)).union(set(G.successors(node)))
                similar_nodes.update(neighbors)
                
    return list(similar_nodes)

def get_subgraph(G, nodes):
    subgraph = G.subgraph(nodes)
    return subgraph

def generate_summary(subgraph):
    nodes_data = []
    for node in subgraph.nodes:
        node_data = {
            'id': node,
            'type': G_loaded.nodes[node].get('type', 'Unknown'),
            'detailed_type': G_loaded.nodes[node].get('detailed_type', 'Unknown')
        }
        nodes_data.append(node_data)
    
    edges_data = [{'from': u, 'to': v, 'label': G_loaded.edges[u, v].get('label', 'Unknown')} for u, v in subgraph.edges]
    
    summary_query = f"""
    Based on the following nodes and edges, please generate a summary:
    Nodes: {json.dumps(nodes_data, indent=4)}
    Edges: {json.dumps(edges_data, indent=4)}
    """
    
    messages = [
        {"role": "system", "content": """
        A chat between a curious user and an artificial intelligence Assistant. The Assistant generates summaries of the given data.
        """}, 
        {"role": "user", "content": summary_query}
    ]
    
    output = pipe(messages, **generation_args)
    return output[0]['generated_text']

def retrieve_graph(query_text):
    query_nodes, query_edges = extract_entities_and_relationships(query_text)
    if not query_nodes:
        return ""
    similar_and_adjacent_nodes = find_similar_and_adjacent_nodes(G_loaded, query_nodes)
    subgraph = get_subgraph(G_loaded, similar_and_adjacent_nodes)
    summary = generate_summary(subgraph)
    return summary


# Create a mapping from entity names to unique indices
entity_to_index = {entity: i for i, entity in enumerate(G_loaded.nodes())}
index_to_entity = {i: entity for entity, i in entity_to_index.items()}

# Create a mapping from relation names to unique indices
relations = set(edge_data['label'] for _, _, edge_data in G_loaded.edges(data=True))
relation_to_index = {relation: i for i, relation in enumerate(relations)}
index_to_relation = {i: relation for relation, i in relation_to_index.items()}


# Query
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)

from utils.phi3_long import Phi3ChatCompletion
prompt = {
    "role": "system",
    "content": (
        "Read the question, provide your answer and your confidence in this answer. "
        "Note: The confidence indicates how likely you think your answer is true.\n"
    )
}

question = "There are several ways to construct estimate for the probability that the model’s answer is correct. Explain me about Probe and LoRA in that term."
context = retrieve_graph(question)

messages = [
    prompt,
    {"role": "user", "content": (
        f"Answer the question based solely only on the following context: {context}\n\n"
        "Note that if you cannot find appropriate answer from the context, the confidence must be low."
    )},
    {"role": "user", "content": (
        f"Question: {question}\n"
        )}
]

generated = Phi3ChatCompletion(messages)
print(generated)




# Confidence 
class DistMult(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(DistMult, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)

    def forward(self, head, relation, tail):
        # Get embeddings
        head_emb = self.entity_embeddings(head)
        relation_emb = self.relation_embeddings(relation)
        tail_emb = self.entity_embeddings(tail)
        
        # Score function
        score = torch.sum(head_emb * relation_emb * tail_emb, dim=1)
        return score
    
def generate_knowledge_graph_from_text(text, pipe, generation_args):
    G_small = nx.DiGraph()

    messages = [
        {"role": "system", "content": """
        A chat between a curious user and an artificial intelligence Assistant. The Assistant is an expert at identifying entities and relationships in text. The Assistant responds in JSON output only.

        The User provides text in the format:

        -------Text begin-------
        <User provided text>
        -------Text end-------

        The Assistant follows the following steps before replying to the User:

        1. **identify the most important entities** The Assistant identifies the most important entities in the text. These entities are listed in the JSON output under the key "nodes", they follow the structure of a list of dictionaries where each dict is:

        "nodes":[{"id": <entity N>, "type": <type>, "detailed_type": <detailed type>}, ...]

        where "type": <type> is a broad categorization of the entity. "detailed type": <detailed_type>  is a very descriptive categorization of the entity.

        2. **determine relationships** The Assistant uses the text between -------Text begin------- and -------Text end------- to determine the relationships between the entities identified in the "nodes" list defined above. These relationships are called "edges" and they follow the structure of:

        "edges":[{"from": <entity 1>, "to": <entity 2>, "label": <relationship>}, ...]

        The <entity N> must correspond to the "id" of an entity in the "nodes" list.

        The Assistant never repeats the same node twice. The Assistant never repeats the same edge twice.
        The Assistant responds to the User in JSON only, according to the following JSON schema:

        {"type":"object","properties":{"nodes":{"type":"array","items":{"type":"object","properties":{"id":{"type":"string"},"type":{"type":"string"},"detailed_type":{"type":"string"}},"required":["id","type","detailed_type"],"additionalProperties":false}},"edges":{"type":"array","items":{"type":"object","properties":{"from":{"type":"string"},"to":{"type":"string"},"label":{"type":"string"}},"required":["from","to","label"],"additionalProperties":false}}},"required":["nodes","edges"],"additionalProperties":false}
        """}, 
        {"role": "user", "content": f"""
        -------Text begin-------
        {text}
        -------Text end-------
        """}
    ]

    output = pipe(messages, **generation_args)
    generated_text = output[0]['generated_text']

    try:
        result = eval(generated_text)
        nodes = result.get('nodes', [])
        edges = result.get('edges', [])
        print(nodes)
        print(edges)

        # Add nodes to the graph
        for node in nodes:
            G_small.add_node(node['id'], type=node['type'], detailed_type=node['detailed_type'])

        # Add edges to the graph
        for edge in edges:
            G_small.add_edge(edge['from'], edge['to'], label=edge['label'])

    except Exception as e:
        print(f"Failed to parse output: {e}")

    return G_small

def predict_links(G_small, model, entity_to_index, relation_to_index):
    scores = {}
    for u, v, edge_data in G_small.edges(data=True):
        if u in entity_to_index and v in entity_to_index:
            head = torch.tensor([entity_to_index[u]], dtype=torch.long)
            tail = torch.tensor([entity_to_index[v]], dtype=torch.long)
            
            # Dynamically add new relation to the relation_to_index if it's missing
            relation_label = edge_data['label']
            if relation_label not in relation_to_index:
                print(f"New relation '{relation_label}' detected. Adding to relation_to_index.")
                new_relation_idx = len(relation_to_index)
                relation_to_index[relation_label] = new_relation_idx
                model.relation_embeddings = nn.Embedding(len(relation_to_index), model.entity_embeddings.embedding_dim).to(model.relation_embeddings.weight.device)
                # Initialize new relation embeddings
                nn.init.xavier_uniform_(model.relation_embeddings.weight.data)
            
            relation = torch.tensor([relation_to_index[relation_label]], dtype=torch.long)
            
            # Get the score for the triplet
            score = torch.sigmoid(model(head, relation, tail)).item()
            scores[(u, v, relation_label)] = score
        else:
            # If either entity is not found, assign a low score
            score = 0.5
            if u not in entity_to_index:
                print(f"Warning: Entity '{u}' not found in the large graph.")
            if v not in entity_to_index:
                print(f"Warning: Entity '{v}' not found in the large graph.")
            scores[(u, v, edge_data['label'])] = score
    return scores

### Workflow
# Load DistMult model
embedding_dim = 100 
num_entities = len(entity_to_index)
num_relations = len(relation_to_index)

# Initialize the DistMult model
distmult_model = DistMult(num_entities, num_relations, embedding_dim)

model_load_path = "distmult_model.pth"
distmult_model.load_state_dict(torch.load(model_load_path))
distmult_model.eval()
print(f"DistMult model loaded from {model_load_path}")

G_generated = generate_knowledge_graph_from_text(generated, pipe, generation_args)

predicted_scores = predict_links(G_generated, distmult_model, entity_to_index, relation_to_index)

for (head, tail, relation), score in predicted_scores.items():
    print(f"Edge ({head}, {relation}, {tail}) score: {score}")
    
scores = [score for _, score in predicted_scores.items()]
mean_score = np.mean(scores) * 100
print(f"Confidence: {mean_score:.2f}%")
