import os
import json
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import traceback
import re
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Image, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from datetime import datetime
from reportlab.platypus import Image, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
# from prompts import Prompts
from prompts_v2 import Prompts
import logging
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

base_model = "llama3_3b_instruct"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(base_model)
tokenizer = AutoTokenizer.from_pretrained(base_model)
base_model_pipeline = transformers.pipeline("text-generation", model=base_model, model_kwargs={"torch_dtype": torch.float16}, device=device)

def data_to_visualization_prompt(data): 
    system_prompt = (
        "You are a world-class expert in data visualization code generation."
        "Make the visualization as simple as possible"
        "Use a python visualization packaage, such as MATPLOTLIB"
    )
    user_prompt = (
        f"Here is the visual description task you need to interpret and convert into Python code:{data}"
        "IMPORTANT INSTRUCTIONS:\n"
        "- Only output Python code. No explanations or comments.\n"
        "- The code MUST be directly executable without errors using Python's `exec()`.\n"
        "- Do not include any additional text. Only return the raw, complete Python code."
    )
    return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

data = "Draw 2 parallel lines"
messages = data_to_visualization_prompt(data)
outputs = base_model_pipeline(messages, max_new_tokens=1024)
response = outputs[0]["generated_text"][-1]['content']

cleaned_code = response.strip().replace('```python', '').replace('```', '').strip()
print(cleaned_code)
local_vars = {}
exec(cleaned_code, globals(), local_vars)




# os.load("OPENAIKEY")