from transformers import AutoTokenizer
import transformers
import torch


code_generation_model = "codellama/CodeLlama-7b-hf"
code_generation_model_tokenizer = AutoTokenizer.from_pretrained(code_generation_model)
code_generation_model_pipeline = transformers.pipeline("text-generation", model="codellama/CodeLlama-7b-hf", torch_dtype=torch.float16)

messages = f"Cretae a fibonnaci function"
outputs = code_generation_model_pipeline(messages, max_new_tokens=1024)
print(outputs)
print(outputs[0]["generated_text"])