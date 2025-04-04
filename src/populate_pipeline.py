import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import transformers
import torch
import traceback
import re
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Image, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from datetime import datetime
from reportlab.platypus import Image, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from prompts import Prompts  
from pipeline import Pipeline
import logging

class PopulatePipeline(Pipeline): 
    def __init__(self, data, category, output_dir='logger'):
        super().__init__(data, output_dir) 
        self.category = category
        self.directory = f"data/{self.category}"
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        self.logger.info(f"Directory created: {self.directory}")
        
        number_data_points = data.shape[0]

        for i in range(number_data_points):
            self.data_index = i + 1
            self.run(data.iloc[i])

    def check_file_exsists(self, file_name):
        if os.path.isfile(file_name):
            self.logger.warning(f"Attempting to duplicate datapoint: {file_name} --> Terminating process")
            return True
        return False

    def run(self, data):
        fileName = os.path.join(self.directory, f"data_entry_{self.category}_{self.data_index}.json")
        if self.check_file_exsists(fileName):
            return
        
        self.logger.info(f"Processing data entry {self.data_index} for category {self.category}")
        
        try:
            goalRet = self.goal_explorer_agent(data)
            self.logger.info(f"Goal Explorer Output for entry {self.data_index}: {goalRet}")
            
            generalDescription = self.goal_to_general_description_agent(data, goalRet)
            self.logger.info(f"General Description for entry {self.data_index}: {generalDescription}")
            
            visualDescription = self.general_description_to_visual_description_agent(data, goalRet, generalDescription)
            self.logger.info(f"Visual Description for entry {self.data_index}: {visualDescription}")
            
            code = self.visual_description_to_visualization_code_agent(goalRet, visualDescription)
            self.logger.info(f"Visualization Code for entry {self.data_index}: {code}")
            
            finalCode = self.run_code(code)
            self.logger.info(f"Code execution completed for entry {self.data_index}.")
            
            self.save_data_entry(data, goalRet, generalDescription, visualDescription, finalCode, fileName)
            self.logger.info(f"Data entry {self.data_index} saved successfully as {fileName}")
        except Exception as e:
            self.logger.error(f"Error processing data entry {self.data_index}: {str(e)}")
            traceback.print_exc()

    def save_data_entry(self, data, goal, generalDescription, visualDescription, code, fileName):
        entry = {
            "data": data,
            "goal": goal,
            "general_description": generalDescription,
            "visual_description": visualDescription,
            "code": code
        }
        
        try:
            with open(fileName, "w") as currFile:
                json.dump(entry, currFile, indent=4)
            self.logger.info(f"Data entry saved successfully to {fileName}")
        except Exception as e:
            self.logger.error(f"Failed to save data entry to {fileName}: {str(e)}")
            traceback.print_exc()
