import os
# gpu set
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # 2 gpus for Llama-3-8B-Intruct, 2 gpus for GLM-4-9B-0414

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

model_ = None
tokenizer = None
terminators = None

def model_get(model_name):
    global model_, tokenizer, terminators
    if model_ is None:
        print('load model: '+ model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_ = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        terminators = [
            tokenizer.eos_token_id,
            # tokenizer.convert_tokens_to_ids("<|eot_id|>") # Llama-3-8B-Instruct
            tokenizer.convert_tokens_to_ids("<|endoftext|>") # GLM-4-9B-0414
        ]
    else:
        print(model_name + 'is already loaded')
    
    return model_, tokenizer, terminators