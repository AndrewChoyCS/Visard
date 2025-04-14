import os
import json
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import traceback
import re
import numpy as np
from datetime import datetime
from prompts_v3 import Prompts
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from logger import Logger
from openai import OpenAI


OPENAI = True
ANTHROPIC = False
OPEN_SOURCE = False

class Pipeline():
    def __init__(self, data, topic):
        self.logger = Logger()     
        self.logger.info("Pipeline initialized.")
        self.logger.info(f"Initial Data: {data}")
        self.load_models()
        self.prompts = Prompts(topic)
        self.run(data, topic)

    def run(self, data, topic):
        try:
            self.logger.info(f"Starting pipeline run for topic: {topic}")
            simple_goal = self.simple_query_agent(data)
            self.logger.info(f"Simple goal generated: {simple_goal}")
            code = self.visualization_code_generator_agent(simple_goal)
            self.logger.info(f"Visualization code generated: {code}")
            corrected_code = self.run_code(code)
            self.logger.info(f"Code after execution: {corrected_code}")
            judge_score = self.first_judge(simple_goal, corrected_code)
            self.logger.info(f"First Judge Score: {judge_score}")
            score = 0
            while score < 70: 
                break
                score1 = self.judge_score1()
                score2 = self.judge_score2()
                score3 = self.judge_score3()
                score4 = self.judge_score4()
                score5 = self.judge_score5()
            self.logger.info("Completed Pipeline ✅")
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            traceback.print_exc()

    def load_models(self):
        self.logger.info("Loading models...")
        if OPENAI: 
            self.openai_client = OpenAI()
            self.openai_client.api_key = os.environ.get("OPENAI_API_KEY")
            self.logger.info("OpenAI client loaded.")
        else:
            self.logger.warning("OPENAI is not enabled in configuration.")

    def run_inference(self, prompt, model, max_tokens):
        self.logger.info(f"Running inference with model: {model}")
        if OPENAI: 
            try: 
                model_instructions, model_input = prompt
                response = self.openai_client.responses.create(
                    model=model,
                    instructions=model_instructions,
                    input=model_input,
                    # max_tokens=max_tokens
                    )
                self.logger.info(f"Inference successful for model {model}.")
                return response.output[0].content[0].text
            except Exception as e:
                self.logger.error(f"Error during OpenAI API call: {e}")
                return None
        else:
            self.logger.warning("OPENAI is not enabled, cannot run inference.")
            return None

    def execute_agent(self, pipeline, max_new_tokens, prompt_method, *args):
        self.logger.info(f"Executing agent with pipeline: {pipeline}")
        messages = prompt_method(*args)
        # print(messages)
        if pipeline == "base_model":
            outputs = self.run_inference(messages, 'gpt-4o-mini', max_tokens=max_new_tokens)
        elif pipeline == "code_generation_model":
            outputs = self.run_inference(messages, 'gpt-4o-mini', max_tokens=max_new_tokens)
        else:
            raise ValueError(f"Unknown pipeline: {pipeline}")
        # response = outputs[0]["generated_text"][-1]['content']
        response = outputs
        self.logger.info(f"Agent response: {response}")
        return response

    def simple_query_agent(self, data):
        self.logger.info("Executing Simple Query Agent")
        return self.execute_agent('base_model', 512, self.prompts.simple_query_prompt, data)

    def visualization_code_generator_agent(self, simple_goal):
        self.logger.info("Executing Visualization Code Generator Agent")
        return self.execute_agent('code_generation_model', 1024, self.prompts.visualization_code_generator_prompt, simple_goal)

    def run_code(self, code):
        MAX_ATTEMPTS = 8
        self.logger.info("Executing Code")
        attempt = 1
        while attempt < MAX_ATTEMPTS:
            try: 
                cleaned_code = code.strip().replace('```python', '').replace('```', '').strip()
                self.logger.info(f"Attempting to execute cleaned code: {cleaned_code}")
                local_vars = {}
                exec(cleaned_code, globals(), local_vars)
                plt.close()
                self.logger.info(f"Code executed successfully on attempt {attempt}")
                return cleaned_code
            except Exception as e: 
                self.logger.warning(f"Error on attempt {attempt}: {str(e)}")
                try:
                    erorr_explantion = self.code_error_identifier_agent(code, str(e))
                    self.logger.info(f"The Error Explanation: {erorr_explantion}")
                    corrected_code = self.code_error_correction_agent(code, str(e), erorr_explantion)
                    self.logger.info(f"Corrected Code: {corrected_code}")
                    code = corrected_code
                    self.logger.info("Attempting to run corrected code...")
                except Exception as correction_error:
                    self.logger.error(f"Error during code correction: {correction_error}")
            attempt += 1
        self.logger.error("Failed to execute code after maximum attempts")
        return "NO CODE GENERATED"  
    
    def code_error_identifier_agent(self, code, error_message):
        self.logger.info("Executing Code Error Identifier Agent")
        return self.execute_agent('base_model', 2048, self.prompts.code_error_identifier_prompt, code, error_message)

    def code_error_correction_agent(self, original_code, error_message, explanation):
        self.logger.info("Executing Code Error Correction Agent")
        return self.execute_agent('code_generation_model', 2048, self.prompts.code_error_correction_prompt, original_code, error_message, explanation)

    def first_judge(self, goal, corrected_code):
        self.logger.info("Executing Visual Judge Agent")
        return self.execute_agent('base_model', 2048, self.prompts.first_judge_prompt, goal, corrected_code)

        
