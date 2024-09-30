import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 

import os
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

torch.random.manual_seed(202407) 
model = AutoModelForCausalLM.from_pretrained( 
    "microsoft/Phi-3-mini-4k-instruct",  
    device_map="cuda",  
    torch_dtype="auto",  
    trust_remote_code=True,  
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct") 

#messages = [ 
#    {"role": "system", "content": "Read the question, provide your answer and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.\nUse the following format to answer:\n```Answer and Confidence (0-100): [ONLY the {answer_type}; not a complete sentence], [Your confidence level, please only include the numerical number in the range of 0-100]%```\nOnly the answer and confidence, don't give me the explanation."}, 
#    {"role": "user", "content": "Question: Typical advertising regulatory bodies suggest, for example that adverts must not: encourage _________, cause unnecessary ________ or _____, and must not cause _______ offence.\nOptions: 1. Unsafe practices, Wants, Fear, Trivial\n2. Unsafe practices, Distress, Fear, Serious\n3. Safe practices, Wants, Jealousy, Trivial\n4. Safe practices, Distress, Jealousy, Serious"}, 
#    {"role": "assistant", "content": "1, 60%"}, 
#    {"role": "user", "content": "Question: ______ are the obligations of workers towards their employer, based on individual contracts and wider employment laws.\nOptions: 1. Employee rights\n2. Employee rights\n3. Employer duties\n4. Employee duties"},
#]

pipe = pipeline( 
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
) 

generation_args = { 
    "max_new_tokens": 1000, 
    "return_full_text": False, 
    "temperature": 0.0, 
    "do_sample": False, 
} 

def Phi3ChatCompletion(messages):
    output = pipe(messages, **generation_args) 
    return output[0]['generated_text']