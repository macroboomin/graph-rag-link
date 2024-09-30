import pickle
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import networkx as nx
import matplotlib.pyplot as plt
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

# Load split documents
with open("paper_split_documents.pickle", 'rb') as fw:
    split_documents = pickle.load(fw)

# Initialize the model and tokenizer
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
    "max_new_tokens": 2500, 
    "return_full_text": False, 
    "temperature": 0.0, 
    "do_sample": False, 
}

# Initialize a graph
G = nx.DiGraph()

# Extract entities and relationships
for doc in tqdm(split_documents, desc="Extracting Entities and Relationships"):
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
        {doc.page_content}
        -------Text end-------
        """}
    ]
    
    output = pipe(messages, **generation_args)
    generated_text = output[0]['generated_text']
    #print(generated_text)
    
    # Parse JSON output
    try:
        result = eval(generated_text)
        nodes = result.get('nodes', [])
        edges = result.get('edges', [])
        
        # Add nodes to the graph
        for node in nodes:
            G.add_node(node['id'], type=node['type'], detailed_type=node['detailed_type'])
        
        # Add edges to the graph
        for edge in edges:
            G.add_edge(edge['from'], edge['to'], label=edge['label'])
    
    except Exception as e:
        print(f"Failed to parse output: {e}")
        continue

# Save the graph for later use
with open('knowledge_graph.gpickle', 'wb') as f:
    pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)

# Visualize the graph (simple visualization)
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, k=0.5)
nx.draw(G, pos, with_labels=True, node_size=50, font_size=8)
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
plt.savefig('paperasq.png')
