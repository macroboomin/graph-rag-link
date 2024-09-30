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

vectorstore = FAISS.load_local('./faiss_bio', embeddings_model, allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={'k': 5, 'fetch_k': 50}
)

### Load Data ###
### Process Data ###
splits = {'test': 'college_biology/test-00000-of-00001.parquet', 'validation': 'college_biology/validation-00000-of-00001.parquet', 'dev': 'college_biology/dev-00000-of-00001.parquet'}
Col_Bio = pd.read_parquet("hf://datasets/cais/mmlu/" + splits["test"])

splits = {'test': 'high_school_biology/test-00000-of-00001.parquet', 'validation': 'high_school_biology/validation-00000-of-00001.parquet', 'dev': 'high_school_biology/dev-00000-of-00001.parquet'}
High_Bio = pd.read_parquet("hf://datasets/cais/mmlu/" + splits["test"])

splits = {'test': 'medical_genetics/test-00000-of-00001.parquet', 'validation': 'medical_genetics/validation-00000-of-00001.parquet', 'dev': 'medical_genetics/dev-00000-of-00001.parquet'}
Med_Gen = pd.read_parquet("hf://datasets/cais/mmlu/" + splits["test"])

splits = {'test': 'professional_medicine/test-00000-of-00001.parquet', 'validation': 'professional_medicine/validation-00000-of-00001.parquet', 'dev': 'professional_medicine/dev-00000-of-00001.parquet'}
Pro_Med = pd.read_parquet("hf://datasets/cais/mmlu/" + splits["test"])

splits = {'test': 'virology/test-00000-of-00001.parquet', 'validation': 'virology/validation-00000-of-00001.parquet', 'dev': 'virology/dev-00000-of-00001.parquet'}
Virology = pd.read_parquet("hf://datasets/cais/mmlu/" + splits["test"])

def preprocess_data(df):
    df = df.drop(columns=['subject'])
    df['answer'] = df['answer'] + 1
    df['gen_answer'] = None
    df['confidence'] = None
    df['correct'] = None
    return df

Col_Bio = preprocess_data(Col_Bio)
High_Bio = preprocess_data(High_Bio)
Med_Gen = preprocess_data(Med_Gen)
Pro_Med = preprocess_data(Pro_Med)
Virology = preprocess_data(Virology)

### Base RAG ###
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)

from utils.phi3 import Phi3ChatCompletion
from utils.func import *
prompt = {
    "role": "system",
    "content": (
        "Read the question, provide your answer and your confidence in this answer. "
        "Note: The confidence indicates how likely you think your answer is true.\n"
        "Use the following format to answer:\n"
        "```Answer and Confidence (0-100): [ONLY the {answer_number}; not a complete sentence or symbols], "
        "[Your confidence level, please only include the numerical number in the range of 0-100]%```\n"
        "Only the answer and confidence, don't give me the explanations."
        "For example, '''Answer and Confidence(0-100): 3, 80%'''"
    )
}

def process_dataset(df, name):
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Processing {name} rows"):
        torch.cuda.empty_cache()
        context = retriever.get_relevant_documents(df.loc[index]['question'])
        
        messages = [
            prompt,
            {"role": "user", "content": (
                f"Answer the question based only on the following context: {context}\n\n"
                "Note that if you cannot find appropriate answer from the context, the confidence must be low."
            )},
            {"role": "user", "content": (
                f"Question: {df.loc[index]['question']}\n"
                f"Options: 1. {df.loc[index]['choices'][0]}\n"
                f"2. {df.loc[index]['choices'][1]}\n"
                f"3. {df.loc[index]['choices'][2]}\n"
                f"4. {df.loc[index]['choices'][3]}\n"
                "Remember that you must have a format like '''Answer and Confidence (0-100): 3, 85%'''"
            )}
        ]

        gen_answer, confidence = extract_answer_and_confidence(Phi3ChatCompletion(messages))
        if gen_answer == 0 and confidence == 0:
            continue

        df.at[index, 'gen_answer'] = gen_answer
        df.at[index, 'confidence'] = confidence
        df.at[index, 'correct'] = 1 if gen_answer == row['answer'] else 0

    df.dropna(subset=['gen_answer'], inplace=True)
    print(df)

    # Save results to CSV
    df.to_csv(f'./base_results/{name}_base.csv', index=False)
    print(f"{name} dataset saved to {name}_base.csv")

process_dataset(Col_Bio, "Col_Bio")
process_dataset(High_Bio, "High_Bio")
process_dataset(Med_Gen, "Med_Gen")
process_dataset(Pro_Med, "Pro_Med")
process_dataset(Virology, "Virology")
