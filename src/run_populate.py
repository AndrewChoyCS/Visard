from populate import PopulatePipeline
import json
import pandas as pd 

# data = pd.read_csv("data/trueData.csv")

with open('data/subcategory_to_questions.json', 'r') as f:
    data = json.load(f)

for category, questions in data.items():
    s = pd.Series(questions)
    pipe = PopulatePipeline(s, category) 
    print(f"Processed category: {category} with {len(questions)} questions.")