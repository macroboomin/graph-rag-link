import pandas as pd
import torch
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
import ast
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

import warnings
warnings.filterwarnings('ignore')

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

### Base RAG ###
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

question = "What is Probe?"
        
torch.cuda.empty_cache()
context = retriever.get_relevant_documents(question)
for index, value in enumerate(context, start=1):
    print(f"{index}. {value}")

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

print(Phi3ChatCompletion(messages))
