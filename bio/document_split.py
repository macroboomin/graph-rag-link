import pickle
import pandas as pd
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain.docstore.document import Document

import warnings
warnings.filterwarnings('ignore')

import os
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

# Load the dataset
df = pd.read_parquet("hf://datasets/enelpol/rag-mini-bioasq/text-corpus/test-00000-of-00001.parquet")

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=600,
    chunk_overlap=100,
    encoding_name='cl100k_base'
)

# Initialize a list to store the split documents
split_documents = []

# Iterate through each passage in the dataframe and split them individually
for passage in tqdm(df['passage'].tolist()):
    document = Document(page_content=passage)
    split_docs = text_splitter.split_documents([document])
    split_documents.extend(split_docs)

# Save the split documents to a pickle file
with open("bio_split_documents.pickle", 'wb') as fw:
    pickle.dump(split_documents, fw)

# Load and check the saved split documents
with open("bio_split_documents.pickle", 'rb') as fw:
    split_documents = pickle.load(fw)

# Output the number of chunks
print(f"Number of split documents: {len(split_documents)}")
