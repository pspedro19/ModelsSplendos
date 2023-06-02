import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

generate_text = pipeline(model="aisquared/dlite-v2-1_5b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")

res = generate_text("Who was George Washington?")
print(res)
