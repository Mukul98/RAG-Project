from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM

import os
# os.environ["HF_TOKEN"]=sec_key

model_id = "gpt2"
model=AutoModelForCausalLM.from_pretrained(model_id)
tokenizer=AutoTokenizer.from_pretrained(model_id)

pipe=pipeline("text-generation",model=model,tokenizer=tokenizer,max_new_tokens=100)
hf=HuggingFacePipeline(pipeline=pipe)

answer = hf.invoke("Tesla is a company")
print(answer)