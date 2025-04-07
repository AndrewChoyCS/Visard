import os
import json
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
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
# from prompts import Prompts
from refined_prompts import Prompts
import logging

MAX_ATTEMPTS = 8

class Pipeline():
    def __init__(self, data, output_dir='logger'):

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.setup_logger()
        self.logger.info("Pipeline initialized.")
        self.load_models()

        self.TOPIC = "Gradient Descent"
        self.prompts = Prompts(self.TOPIC)
        # Comment the line below when running run_populate
        self.run(data)

        
    def run(self, data):
        try:
            goalRet = self.goal_explorer_agent(data)
            self.logger.info(f"Goal Explorer Output: {goalRet}")
            
            generalDescription = self.goal_to_general_description_agent(data, goalRet)
            self.logger.info(f"General Description: {generalDescription}")
            
            # visualDescription = self.general_description_to_visual_description_agent(data, goalRet, generalDescription)
            # self.logger.info(f"Visual Description: {visualDescription}")
            visualDescription = generalDescription
            
            # This is the base code 
            codeGenerated = self.visual_description_to_visualization_code_agent(goalRet, visualDescription)
            self.logger.info(f"Visualization Code: {codeGenerated}")

            # Run the judge, on the visualized code, and only exectue it when you above threshold
            score = 0
            while int(score) < 70:
                styledVisualizationCode = self.visual_refinement_agent(codeGenerated)
                self.logger.info(f"Styled Visualization Code: {styledVisualizationCode}")
                executableCode = self.run_code(styledVisualizationCode)
                if executableCode == "NO CODE GENERATED": 
                    raise ValueError("No code was generated during visualization.")
                
                score = self.visualization_judge_agent(goalRet, generalDescription, styledVisualizationCode)
                self.logger.info(f"This Visualization scored: {score}")

            self.logger.info("Code execution completed.")
            
            # visualization_path = self.save_visualization(finalCode)
            # self.logger.info(f"Visualization saved at: {visualization_path}")
            
            # learning_blurb = self.generate_learning_blurb_agent(data, goalRet, generalDescription, visualDescription, finalCode)
            # self.logger.info(f"Learning Blurb: {learning_blurb}")
            
            # self.create_pdf(visualization_path, learning_blurb)
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            traceback.print_exc()
    
    def setup_logger(self):
        """Set up logger for pipeline."""
        self.logger = logging.getLogger('PipelineLogger')
        self.logger.setLevel(logging.INFO)
        
        # Create handlers for logging to both console and file
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(os.path.join(self.output_dir, 'pipeline_log.txt'))
        
        # Set log format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def load_models(self):
        self.models = ["llama3_3b_instruct"]
        self.base_model = "llama3_3b_instruct"
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(self.base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.base_model_pipeline = transformers.pipeline("text-generation", model="meta-llama/Llama-3.2-3B-Instruct", model_kwargs={"torch_dtype": torch.float16})

        self.code_generation_model = "codellama/CodeLlama-7b-hf"
        self.code_generation_model_tokenizer = AutoTokenizer.from_pretrained(self.code_generation_model)
        self.code_generation_model_pipeline = transformers.pipeline("text-generation", model="codellama/CodeLlama-7b-hf", torch_dtype=torch.float16)

    def clean_data(self):
        pass

    def goal_explorer_agent(self, data):
        self.logger.info("Executing Goal Explorer Agent")
        messages = self.prompts.goal_explorer_prompt(data)
        outputs = self.base_model_pipeline(messages, max_new_tokens=512)
        response = outputs[0]["generated_text"][-1]['content']
        return response

    def goal_to_general_description_agent(self, data: str, goal: str):
        self.logger.info("Executing Goal to General Description Agent")
        messages = self.prompts.general_description_prompt(data, goal)
        outputs = self.base_model_pipeline(messages, max_new_tokens=1024)
        response = outputs[0]["generated_text"][-1]['content']
        return response

    def general_description_to_visual_description_agent(self, data, goal, generalDescription):
        self.logger.info("Executing General Description to Visual Description Agent")
        messages = self.prompts.visual_description_prompt(data, goal, generalDescription)
        outputs = self.base_model_pipeline(messages, max_new_tokens=1024)
        response = outputs[0]["generated_text"][-1]['content']
        return response

    def visual_description_to_visualization_code_agent(self, goal, visualDescription):
        self.logger.info("Executing Visual Description to Visualization Code Agent")
        messages = self.prompts.visualization_code_prompt(goal, visualDescription)
        outputs = self.base_model_pipeline(messages, max_new_tokens=1024)
        response = outputs[0]["generated_text"][-1]['content']
        return response
    
    # def code_evaluation_agent(self, original_code, generalDescription):
    #     messages = self.prompts.code_error_correction_prompt(original_code, error_message)
    #     outputs = self.pipeline(messages, max_new_tokens=1024)
    #     response = outputs[0]["generated_text"][-1]['content']
    #     return response

    def visual_refinement_agent(self, code):
        self.logger.info("Executing Visual Description to Visualization Code Agent")
        messages = self.prompts.visual_refinement_prompt(code)
        outputs = self.base_model_pipeline(messages, max_new_tokens=1024)
        response = outputs[0]["generated_text"][-1]['content']
        return response
    
    def visualization_judge_agent(self, goal, general_description, code ):
        self.logger.info("Executing Jude Agent")
        messages = self.prompts.visualization_judge_prompt(goal, general_description, code)
        outputs = self.base_model_pipeline(messages, max_new_tokens=1024)
        response = outputs[0]["generated_text"][-1]['content']
        return response

    def code_error_correction_agent(self, original_code, error_message):
        self.logger.info("Executing Code Error Correction Agent")
        messages = self.prompts.code_error_correction_prompt(original_code, error_message)
        outputs = self.base_model_pipeline(messages, max_new_tokens=2048)
        # outputs = self.code_generation_model_pipeline(messages, max_new_tokens=1024)
        response = outputs[0]["generated_text"][-1]['content']
        return response


    """
    This run_code function has the functionality to attempt multiple models, but I have taken it out for now. It has a bug that the next model will not load.
    """
    # def run_code(self, code):
    #     self.logger.info("Executing Code")
    #     code_generation_model = self.base_model
    #     additional_models = ["llama3_3b_instruct", "Qwen/Qwen2.5-7B-Instruct"]
    #     max_attempts = 8
    #     attempt = 0
    #     model_loaded = False
    #     current_code = code
        
    #     while attempt < max_attempts and not model_loaded:
    #         try:
    #             cleaned_code = current_code.strip().replace('```python', '').replace('```', '').strip()
    #             local_vars = {}
    #             exec(cleaned_code, globals(), local_vars)
    #             plt.close()
    #             model_loaded = True
    #             self.logger.info(f"Code executed successfully on attempt {attempt + 1}")
    #         except Exception as e:
    #             self.logger.warning(f"Error on attempt {attempt + 1}: {str(e)}")
    #             try:
    #                 corrected_code = self.code_error_correction_agent(current_code, str(e))
    #                 self.logger.info(f"Corrected Code: {corrected_code}")
    #                 current_code = corrected_code
    #                 self.logger.info("Attempting to run corrected code...")
    #             except Exception as correction_error:
    #                 self.logger.error(f"Error during code correction: {correction_error}")
                    
    #             if attempt + 1 == max_attempts:
    #                 self.logger.error("Maximum attempts reached for the current model. Switching to a new model.")
    #                 current_model_index = additional_models.index(self.base_model)
    #                 next_model_index = current_model_index + 1
    #                 if next_model_index == len(additional_models): 
    #                     break
    #                 self.base_model = additional_models[next_model_index]
    #                 self.base_model_pipeline = transformers.pipeline("text-generation", model=self.base_model, trust_remote_code=True)
    #                 attempt = 0 
    #             else:
    #                 attempt += 1

    #     if not model_loaded:
    #         self.logger.error("Failed to execute code after maximum attempts with all models.")
    #         raise RuntimeError("Could not execute the code after max attempts with all models.")
        
    #     return current_code  # Return the final executed code
    

    def run_code(self, code):
        self.logger.info("Executing Code")
        attempt = 1
        while attempt < MAX_ATTEMPTS:
            try: 
                cleaned_code = code.strip().replace('```python', '').replace('```', '').strip()
                local_vars = {}
                exec(cleaned_code, globals(), local_vars)
                plt.close()
                self.logger.info(f"Code executed successfully on attempt {attempt}")
                return cleaned_code
            except Exception as e: 
                self.logger.warning(f"Error on attempt {attempt}: {str(e)}")
                try:
                    corrected_code = self.code_error_correction_agent(code, str(e))
                    self.logger.info(f"Corrected Code: {corrected_code}")
                    code = corrected_code
                    self.logger.info("Attempting to run corrected code...")
                except Exception as correction_error:
                    self.logger.error(f"Error during code correction: {correction_error}")
            attempt += 1
        self.logger.error("Failed to execute code after maximum attempts")
        return "NO CODE GENERATED"
        
    def generate_learning_blurb_agent(self, data, goal, generalDescription, visualDescription, code):
        self.logger.info("Executing Learning Blurb Agent")
        messages = self.prompts.learning_blurb_prompt(data, goal, generalDescription, visualDescription, code)
        outputs = self.base_model_pipeline(messages, max_new_tokens=256)
        response = outputs[0]["generated_text"][-1]['content']
        return response
