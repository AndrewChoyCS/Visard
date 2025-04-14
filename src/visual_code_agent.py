import os
import json
from openai import OpenAI
import os
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
# from refined_prompts import Prompts
import logging
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# base_model = "llama3_3b_instruct"
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# model = AutoModelForCausalLM.from_pretrained(base_model)
# tokenizer = AutoTokenizer.from_pretrained(base_model)
# base_model_pipeline = transformers.pipeline("text-generation", model=base_model, model_kwargs={"torch_dtype": torch.float16}, device=device)

# def data_to_visualization_prompt(data): 
#     system_prompt = (
#         "You are a world-class expert in data visualization code generation."
#         "Make the visualization as simple as possible"
#         "Use a python visualization packaage, such as MATPLOTLIB"
#     )
#     user_prompt = (
#         f"Here is the visual description task you need to interpret and convert into Python code:{data}"
#         "IMPORTANT INSTRUCTIONS:\n"
#         "- Only output Python code. No explanations or comments.\n"
#         "- The code MUST be directly executable without errors using Python's `exec()`.\n"
#         "- Do not include any additional text. Only return the raw, complete Python code."
#     )
#     return [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt}
#         ]

# data = "Draw 2 parallel lines"
# messages = data_to_visualization_prompt(data)
# outputs = base_model_pipeline(messages, max_new_tokens=1024)
# response = outputs[0]["generated_text"][-1]['content']

# cleaned_code = response.strip().replace('```python', '').replace('```', '').strip()
# print(cleaned_code)
# local_vars = {}
# exec(cleaned_code, globals(), local_vars)

#GPT4 

# def data_to_visualization_prompt(data): 
#     system_prompt = (
#         "You are a world-class expert in data visualization code generation."
#         "Make the visualization as simple as possible"
#         "Use a python visualization package, such as MATPLOTLIB"
#     )
#     user_prompt = (
#         f"Here is the visual description task you need to interpret and convert into Python code:{data}"
#         "IMPORTANT INSTRUCTIONS:\n"
#         "- Only output Python code. No explanations or comments.\n"
#         "- The code MUST be directly executable without errors using Python's `exec()`.\n"
#         "- Do not include any additional text. Only return the raw, complete Python code."
#     )
#     return [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt}
#         ]
    


# """

# The way gradient descent manages to find the minima of functions is easiest to imagine in 

# three dimensions.

# \[f(x, y)\]   defines some hilly terrain when graphed as a height map


# We learned that the gradient evaluated at any point represents the direction of steepest ascent up this hilly terrain. 
#  start at a random input, and as many times as we can, take a small step in the direction of the gradient to move uphill. 

# To minimize the function, we can instead follow the negative of the gradient, and thus go in the direction of steepest descent. 
# if we start at a point \[x_0\]  and move a positive distance \[\alpha\] in the direction of the negative gradient, 
# then our new and improved  \[x_1\]  will look like this: \[x_1 = x_0 - \alpha \nabla f(x_0)\] 
# More generally, we can write a formula for turning  \[x_n\] into \[x_{n + 1}\]:\[x_{n + 1} = x_n - \alpha \nabla f(x_n)\]
# """





def visualization_code_prompt(general_description):
        system_prompt = (
            "You're an expert in educational data visualization and you know everything about best data visualization practices."
            "Your code must be production-quality: well-structured, efficiently implemented, and fully executable without errors. "
            "You must adhere to these principles:\n"
            "1. Use matplotlib as the primary library with appropriate specialized libraries as needed\n"
            "2. Implement EVERY detail specified in the visual description exactly as specified\n"
            "3. Create robust code that gracefully handles edge cases\n"
            "4. Add minimal explanatory comments at section boundaries only\n"
            "5. Emphasize visual clarity through appropriate font sizes, line weights, and color contrast"
        )
        
        user_prompt = (
            f"Create executable Python code implementing this specification exactly:\n\n{general_description}\n\n"
            "Requirements:\n"
            "1. Generate COMPLETE, STANDALONE code that requires no modifications to run\n"
            "2. Include all necessary imports at the beginning\n"
            "3. Create intermediary variables for complex calculations to improve readability\n"
            "4. Set figure dimensions and DPI for high-quality output\n"
            "5. Use plt.tight_layout() or equivalent to avoid element crowding\n"
            "6. Include plt.show() at the end\n\n"
            "ONLY PROVIDE THE PYTHON CODE - NO EXPLANATIONS OR COMMENTARY."
        )
        
        return [
            system_prompt,
            user_prompt
        ]

def simple_query_agent(data):
        system_prompt = (
            f"You are an expert in Gradient Descent and creating ideas and descriptions for educational visualizations."
            f"Your task is to create a structured description for a visualization that will explain {data}"
            "Your response MUST be clear, concise and valuable for creating a visual."
            f"Your response MUST prioritize the explanation of Gradient Descent"
            "Your response MUST be clear and detailed to guide human implementation"
            f"Your instruction is to create a detailed description of the visualization with the following format"
            
            # f"Your are provided the following context: {data}. "
        )
        
        user_prompt = (
            "Your response MUST be formatted as A VALID JSON object with exactly the following format:\n\n"
            
            "{\n"
            '  "Concept": "{A clear and concrete expression of the core concept in that is being explained(1-4 words)}",\n'
            '  "Title": "{A short, descriptive title of the visualization (1-4 words)}",\n'
            '  "Objective": "{A clear, concrete and measurable learning outcome of the visualization without any ambiguity. For example, \"After seeing this visualizations, learners will be able to explain why the derivative of a function at turning points is zero. \"}",\n'
            '  "Description": "{A clear description of the visualization in 7-8 sentences that is fully enough to reproduce the visualization.",\n'
            '  "Emphasis": "{A list of 3-5 key conceptual points the visualization must highlight in a clear and reproducible way.",\n'
            '  "Outline": "{A clear description of the visual flow of figures in 2-3 sentences, For example, \" The visualization shows a function on 2D axes with maximas and minimas, the gradient tangent is drawn at some points, and clearly drawn at turning points to be horizontal. \"}",\n'
            '  "Type": "{A category for the content the visualization is explaining. It muse be either Definition or Process Explanation or Problem Explanation or Example Explanation}",\n'
            '  "Student Background": "{A clear specification of the expected student background level needed to understand the visualization in 2-6 words(e.g., introductory calculus)}",\n'
            '  "Related Topics": "{A list of 2-5 relevant topics to the visualization}",\n'
            "}\n\n"
            "THE OUTPUT MUST ONLY USE THE JSON FORMAT ABOVE and be a STRING"
            
        
        )
        
        return [
            system_prompt,
            user_prompt
        ]

data= """Gradient descent is an algorithm that numerically estimates where a function outputs its lowest values. That means it finds local minima, but not by setting \[\nabla f = 0\] like we've seen before. Instead of finding minima by manipulating symbols, gradient descent approximates the solution with numbers. Furthermore, all it needs in order to run is a function's numerical output, no formula required. The way gradient descent manages to find the minima of functions is easiest to imagine in three dimensions.
Think of a function \[f(x, y)\]  that defines some hilly terrain when graphed as a height map. We learned that the gradient evaluated at any point represents the direction of steepest ascent up this hilly terrain. That might spark an idea for how we could maximize the function: start at a random input, and as many times as we can, take a small step in the direction of the gradient to move uphill. In other words, walk up the hill.
To minimize the function, we can instead follow the negative of the gradient, and thus go in the direction of steepest descent. This is gradient descent. Formally, if we start at a point \[x_0\]  and move a positive distance \[\alpha\] in the direction of the negative gradient, then our new and improved  \[x_1\]  will look like this: \[x_1 = x_0 - \alpha \nabla f(x_0)\] More generally, we can write a formula for turning  \[x_n\] into \[x_{n + 1}\]:\[x_{n + 1} = x_n - \alpha \nabla f(x_n)\]
"""
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)
response_simple_query = client.responses.create(
    model="gpt-4o",
    instructions = general_description_prompt(data)[0],
    input=general_description_prompt(data)[1], 
)

response_code = client.responses.create(
    model="gpt-4o",
    instructions = response_simple_query.output_text[0],
    input=response_simple_query.output_text[1])




print(response_visual_instructions.output_text)
print(response_code.output_text)
