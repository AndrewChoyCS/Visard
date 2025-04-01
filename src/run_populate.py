from populate import PopulatePipeline
import json
import pandas as pd 

data = pd.read_csv("data/trueData.csv")

pipe = PopulatePipeline(data)