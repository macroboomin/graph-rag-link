import networkx as nx
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import warnings
warnings.filterwarnings("ignore")

import os

# Load the pre-saved knowledge graph
with open('knowledge_graph.gpickle', 'rb') as f:
    G_large = pickle.load(f)

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
    "max_new_tokens": 2500, 
    "return_full_text": False, 
    "temperature": 0.0, 
    "do_sample": False, 
}

# Create a mapping from entity names to unique indices
entity_to_index = {entity: i for i, entity in enumerate(G_large.nodes())}
index_to_entity = {i: entity for entity, i in entity_to_index.items()}

# Create a mapping from relation names to unique indices
relations = set(edge_data['label'] for _, _, edge_data in G_large.edges(data=True))
relation_to_index = {relation: i for i, relation in enumerate(relations)}
index_to_relation = {i: relation for relation, i in relation_to_index.items()}

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

        # Add nodes to the graph
        for node in nodes:
            G_small.add_node(node['id'], type=node['type'], detailed_type=node['detailed_type'])

        # Add edges to the graph
        for edge in edges:
            G_small.add_edge(edge['from'], edge['to'], label=edge['label'])

    except Exception as e:
        print(f"Failed to parse output: {e}")

    return G_small


def generate_training_data(G, entity_to_index, relation_to_index):
    data = []
    for u, v, edge_data in G.edges(data=True):
        head = entity_to_index[u]
        tail = entity_to_index[v]
        relation = relation_to_index[edge_data['label']]
        data.append((head, relation, tail))
    return data

# Convert the large graph's edges into training data
train_data = generate_training_data(G_large, entity_to_index, relation_to_index)

embedding_dim = 100
num_entities = len(entity_to_index)
num_relations = len(relation_to_index)

# Initialize the DistMult model
model = DistMult(num_entities, num_relations, embedding_dim)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 12  # Adjust as needed
for epoch in range(num_epochs):
    total_loss = 0
    for head, relation, tail in tqdm(train_data):
        # Prepare inputs
        head = torch.tensor([head], dtype=torch.long)
        relation = torch.tensor([relation], dtype=torch.long)
        tail = torch.tensor([tail], dtype=torch.long)
        
        # Forward pass
        score = model(head, relation, tail)
        
        # Prepare target (positive sample = 1)
        target = torch.tensor([1], dtype=torch.float)
        
        # Compute loss
        loss = criterion(score, target)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_data)}")

# Save the trained model
model_save_path = "distmult_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Initialize the model
model = DistMult(num_entities, num_relations, embedding_dim)

# Load the model's saved state dictionary
model_load_path = "distmult_model.pth"
model.load_state_dict(torch.load(model_load_path))
model.eval()  # Set the model to evaluation mode
print(f"Model loaded from {model_load_path}")

'''
def predict_links(G_small, model, entity_to_index, relation_to_index):
    scores = {}
    for u, v, edge_data in G_small.edges(data=True):
        if u in entity_to_index and v in entity_to_index:
            head = torch.tensor([entity_to_index[u]], dtype=torch.long)
            tail = torch.tensor([entity_to_index[v]], dtype=torch.long)
            relation = torch.tensor([relation_to_index[edge_data['label']]], dtype=torch.long)
            
            # Get the score for the triplet
            score = torch.sigmoid(model(head, relation, tail)).item()
        else:
            # If either entity is not found, assign a low score
            score = 0.0
            if u not in entity_to_index:
                print(f"Warning: Entity '{u}' not found in the large graph.")
            if v not in entity_to_index:
                print(f"Warning: Entity '{v}' not found in the large graph.")
        
        scores[(u, v, edge_data['label'])] = score
    return scores

# Example usage:
new_text = "The deletion in this chromosomal region is responsible for the symptoms of AS."
G_small = generate_knowledge_graph_from_text(new_text, pipe, generation_args)

nodes_data = [{'id': node, 'attributes': G_small.nodes[node]} for node in G_small.nodes()]
edges_data = [{'from': u, 'to': v, 'attributes': G_small.edges[u, v]} for u, v in G_small.edges()]

predicted_scores = predict_links(G_small, model, entity_to_index, relation_to_index)

for (head, tail, relation), score in predicted_scores.items():
    print(f"Edge ({head}, {relation}, {tail}) score: {score}")
'''