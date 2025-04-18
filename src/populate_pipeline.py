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
from prompts_v3 import Prompts  
from pipeline import Pipeline
import logging

class PopulatePipeline(Pipeline): 
    #try1.txt logger
    def __init__(self, data, topic, output_dir='logger'):
        super().__init__(data, output_dir) 
        self.topic=topic
        self.directory = f"data/{self.topic}"
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        self.logger.info(f"Directory created: {self.directory}")
        
        # number_data_points = data.shape[0]

        for i in range(len(data)):
            sample_dir = os.path.join(self.directory, f"data_sample{i+1}")
            os.makedirs(sample_dir, exist_ok=True)
            self.logger.output_dir=sample_dir
            self.logger.info(f"Data sample directory created: {sample_dir}")
            self.data_index = i + 1
            for j in range(1):
                file_name = os.path.join(sample_dir, f"try{j+1}.json")
                img_filename  = os.path.join(sample_dir, f"try{j+1}.png")
                log_filename = os.path.join(sample_dir, f"try{j+1}.txt")
                self.logger = logging.getLogger('PipelineLogger')
                self.logger.setLevel(logging.INFO)
        
        
                console_handler = logging.StreamHandler()
                file_handler = logging.FileHandler(log_filename)
        
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                console_handler.setFormatter(formatter)
                file_handler.setFormatter(formatter)
        
                self.logger.addHandler(console_handler)
                self.logger.addHandler(file_handler)                    
                self.logger.info(f"Starting try #{j+1}")
                goal, code = self.run(data[i], self.topic,img_filename)
                self.logger.info(f"Finished try #{j+1}")
                if not self.check_file_exists(file_name):
                    self.save_data_entry(data[i], goal, code, file_name)


    def check_file_exists(self, file_name):
        if os.path.isfile(file_name):
            self.logger.warning(f"Attempting to duplicate datapoint: {file_name} --> Terminating process")
            return True
        return False
    def save_data_entry(self, data, goal, code, fileName):
        entry = {
            "data": data,
            "goal": goal,
            "code": code
        }
        
        try:
            with open(fileName, "w") as currFile:
                json.dump(entry, currFile, indent=4)
            self.logger.info(f"Data entry saved successfully to {fileName}")
        except Exception as e:
            self.logger.error(f"Failed to save data entry to {fileName}: {str(e)}")
            traceback.print_exc()
            
