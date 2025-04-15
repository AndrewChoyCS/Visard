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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from logger import Logger
from openai import OpenAI

OPENAI = True
ANTHROPIC = False
OPEN_SOURCE = False

class Pipeline():
    def __init__(self, data, topic):
        self.output_dir = "research_results"
        self.logger = Logger()     
        self.logger.info("Pipeline initialized.")
        self.logger.info(f"Initial Data: {data}")
        self.load_models()
        self.prompts = Prompts(topic)

    def run(self, data, topic, img_filename=None):
        try:
            self.logger.info(f"Starting pipeline run for topic: {topic}")
            simple_goal = self.simple_query_agent(data)
            self.logger.info(f"Simple goal generated: {simple_goal}")
            code = self.visualization_code_generator_agent(simple_goal, self.output_dir)
            self.logger.info(f"Visualization code generated: {code}")
            corrected_code = self.run_code(code)
            self.logger.info(f"Code after execution: {corrected_code}")
            final_code = self.run_sequence_of_judges(simple_goal, code)
            self.logger.info(f"Final code after all judges: {final_code}")
            self.run_final_code(final_code, img_filename)
            self.logger.info("Completed Pipeline ✅")
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            traceback.print_exc()
        return simple_goal, final_code

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
    
    def run_final_code(self, code, img_filename): 
        self.logger.info("Running final code and saving visualization.")
        cleaned_code = code.strip().replace('```python', '').replace('```', '').strip()
        if img_filename is None:
            img_filename = os.path.join(self.output_dir, "bello.png")
        else:
            img_dir = os.path.dirname(img_filename)
            os.makedirs(img_dir, exist_ok=True)
        # os.makedirs(self.output_dir, exist_ok=True)
        cleaned_code += "\n\n"
        cleaned_code += f"plt.savefig(\"{img_filename}\")"
        # cleaned_code += f"plt.savefig(\"{self.output_dir}/bello.png\")"
        
        local_vars = {}
        exec(cleaned_code, globals(), local_vars)
        self.logger.info("Final visualization saved.")
        return

    def parse_judge_response(self, response):
        lines = response.strip().split('\n', 1)
        result = lines[0].strip().lower()
        feedback = lines[1].strip() if len(lines) > 1 else "No feeback provided!"
        
        return result, feedback
            
    def run_sequence_of_judges(self, goal, code):
        self.logger.info("Executing Sequence of Judges")
        response = self.goal_alignment_judge_agent(goal, code) 
        self.logger.info(f"Goal Alignment Judge response: {response}")
        result, feedback = self.parse_judge_response(response)
        if not result:
            self.logger.info("Goal Alignment Judge failed. Regenerating code from feedback...")
            generated_code_from_feedback = self.code_generator_from_judge_feedback_agent(code, feedback)
            self.logger.info(f"Generated code from feedback: {generated_code_from_feedback}")
            corrected_code = self.run_code(generated_code_from_feedback)
            return self.run_sequence_of_judges(goal, corrected_code)
        self.logger.info("Passed Goal Alignment Judge ✅")
        
        response = self.visual_clarity_judge_agent(code) 
        self.logger.info(f"Visual Clarity Judge response: {response}")
        result, feedback = self.parse_judge_response(response)
        if not result:
            self.logger.info("Visual Clarity Judge failed. Regenerating code from feedback...")
            generated_code_from_feedback = self.code_generator_from_judge_feedback_agent(code, feedback)
            self.logger.info(f"Generated code from feedback: {generated_code_from_feedback}")
            corrected_code = self.run_code(generated_code_from_feedback)
            return self.run_sequence_of_judges(goal, corrected_code)
        self.logger.info("Passed Visual Clarity Judge ✅")
        return code
    
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

    def visualization_code_generator_agent(self, simple_goal, output_dir):
        self.logger.info("Executing Visualization Code Generator Agent")
        return self.execute_agent('code_generation_model', 1024, self.prompts.visualization_code_generator_prompt, simple_goal, output_dir)
    
    def code_error_identifier_agent(self, code, error_message):
        self.logger.info("Executing Code Error Identifier Agent")
        return self.execute_agent('base_model', 2048, self.prompts.code_error_identifier_prompt, code, error_message)

    def code_error_correction_agent(self, original_code, error_message, explanation):
        self.logger.info("Executing Code Error Correction Agent")
        return self.execute_agent('code_generation_model', 2048, self.prompts.code_error_correction_prompt, original_code, error_message, explanation)

    def goal_alignment_judge_agent(self, goal, code): 
        self.logger.info("Executing Goal Alignment Judge")
        return self.execute_agent('base_model', 10, self.prompts.goal_alignment_judge_prompt, goal, code)
    
    def visual_clarity_judge_agent(self, code):
        self.logger.info("Executing Visual Clarity Judge")
        return self.execute_agent('base_model', 10, self.prompts.visual_clarity_judge_prompt, code)
    
    def code_generator_from_judge_feedback_agent(self, code, feedback):
        self.logger.info("Executing Code Generator From Judge Feedback Agent")
        return self.execute_agent('code_generation_model', 2048, self.prompts.code_generator_from_judge_feedback_prompt, code, feedback)

        
