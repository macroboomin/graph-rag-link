import pickle
import pandas as pd
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader

import warnings
warnings.filterwarnings('ignore')

import os
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

# Load the dataset
loader = PyPDFLoader("2406.08391v1.pdf")
documents = loader.load()

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300,
    chunk_overlap=50,
    encoding_name='cl100k_base'
)

# Initialize a list to store the split documents
split_documents = []

# Iterate through each passage in the dataframe and split them individually
for doc in tqdm(documents):
    split_docs = text_splitter.split_documents([doc])
    split_documents.extend(split_docs)

# Save the split documents to a pickle file
with open("paper_split_documents.pickle", 'wb') as fw:
    pickle.dump(split_documents, fw)

# Load and check the saved split documents
with open("paper_split_documents.pickle", 'rb') as fw:
    split_documents = pickle.load(fw)

# Output the number of chunks
print(f"Number of split documents: {len(split_documents)}")

# Step 4: Generate Embeddings
model_name = "intfloat/multilingual-e5-large-instruct"
embeddings_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"device": "cuda"}, 
    encode_kwargs={"normalize_embeddings": True},
)
print("Embedding models imported")

# Step 5: Create FAISS Vector Store
vectorstore = FAISS.from_documents(
    documents=split_documents,
    embedding=embeddings_model,
    distance_strategy=DistanceStrategy.COSINE
)
vectorstore.save_local("./faiss_paper")
print("FAISS index saved to disk")

# Load the FAISS index
vectorstore = FAISS.load_local('./faiss_paper', embeddings_model, allow_dangerous_deserialization=True)
print("FAISS index loaded from disk")
