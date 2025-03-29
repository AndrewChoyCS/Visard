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
from prompts import Prompts  # Import the Prompts class

class Pipeline(): 
    def __init__(self, data, output_dir='research_results'):
        # Create output directory if it doesn't exist
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.models = ["llama3_3b_instruct", "deepseek-ai/DeepSeek-V3-0324"]
        self.base_model = "llama3_3b_instruct"
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(self.base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.pipeline = transformers.pipeline("text-generation", model="meta-llama/Llama-3.2-3B-Instruct", model_kwargs={"torch_dtype": torch.float16})
        self.TOPIC = "Gradient Descent"
        
        # Initialize the prompts
        self.prompts = Prompts(self.TOPIC)

        # Run the pipeline
        goalRet = self.goal_explorer_agent(data)
        print(goalRet)
        generalDescription = self.goal_to_general_description_agent(data, goalRet)
        print(generalDescription)
        visualDescription = self.general_description_to_visual_description_agent(data, goalRet, generalDescription)
        print(visualDescription)
        code = self.visual_description_to_visualization_code_agent(goalRet, visualDescription)
        print(code)
        finalCode = self.run_code(code)
        visualization_path = self.save_visualization(finalCode)
        learning_blurb = self.generate_learning_blurb_agent(data, goalRet, generalDescription, visualDescription, finalCode)
        print("Learning Blurb:", learning_blurb)
        self.create_pdf(visualization_path, learning_blurb)

    def clean_data():
        pass

    def save_visualization(self, code):
        try:
            cleaned_code = code.strip().replace('```python', '').replace('```', '').strip()
            
            local_namespace = {}
            exec(cleaned_code, globals(), local_namespace)
            plt.tight_layout()
            figure_path = os.path.join(self.output_dir, 'gradient_descent_visualization.png')
            if plt.gcf().get_axes():
                plt.savefig(figure_path, dpi=300, bbox_inches='tight')
                plt.close() 
                print(f"Visualization saved to {figure_path}")
                return figure_path
            else:
                print("No figure was generated to save.")
                return None
        
        except Exception as e:
            print(f"Error saving visualization: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_pdf(self, image_path, blurb):
        try:
            pdf_path = os.path.join(self.output_dir, f'{self.TOPIC}_visualization_report.pdf')
            
            doc = SimpleDocTemplate(pdf_path, pagesize=letter, 
                                    rightMargin=72, leftMargin=72, 
                                    topMargin=72, bottomMargin=18)
            
            story = []
            
            styles = getSampleStyleSheet()
            title_style = styles['Title']
            normal_style = styles['Normal']
            
            story.append(Paragraph(f"{self.TOPIC} Visualization", title_style))
            
            if image_path and os.path.exists(image_path):
                img = Image(image_path, width=6*inch, height=4*inch)
                img.hAlign = 'CENTER'
                story.append(img)
            
            story.append(Paragraph("<br/><br/>Learning Insights:", styles['Heading3']))
            story.append(Paragraph(blurb, normal_style))
            # Build PDF
            doc.build(story)
            
            print(f"PDF report created at {pdf_path}")
        except Exception as e:
            print(f"Error creating PDF: {e}")

    def goal_explorer_agent(self, data):
        print("Executing Goal Explorer Agent")
        messages = self.prompts.goal_explorer_prompt(data)
        outputs = self.pipeline(messages, max_new_tokens=512)
        response = outputs[0]["generated_text"][-1]['content']
        return response

    def goal_to_general_description_agent(self, data: str, goal: str):
        messages = self.prompts.general_description_prompt(data, goal)
        outputs = self.pipeline(messages, max_new_tokens=1024)
        response = outputs[0]["generated_text"][-1]['content']
        return response

    def general_description_to_visual_description_agent(self, data, goal, generalDescription):
        messages = self.prompts.visual_description_prompt(data, goal, generalDescription)
        outputs = self.pipeline(messages, max_new_tokens=1024)
        response = outputs[0]["generated_text"][-1]['content']
        return response

    def visual_description_to_visualization_code_agent(self, goal, visualDescription):
        messages = self.prompts.visualization_code_prompt(goal, visualDescription)
        outputs = self.pipeline(messages, max_new_tokens=1024)
        response = outputs[0]["generated_text"][-1]['content']
        return response
    
    def code_error_correction_agent(self, original_code, error_message):
        messages = self.prompts.code_error_correction_prompt(original_code, error_message)
        outputs = self.pipeline(messages, max_new_tokens=1024)
        response = outputs[0]["generated_text"][-1]['content']
        return response

    def run_code(self, code):
        additional_models = ["llama3_3b_instruct", "Qwen/Qwen2.5-7B-Instruct"]
        max_attempts = 8
        attempt = 0
        model_loaded = False
        current_code = code
        
        while attempt < max_attempts and not model_loaded:
            try:
                cleaned_code = current_code.strip().replace('```python', '').replace('```', '').strip()
                local_vars = {}
                exec(cleaned_code, globals(), local_vars)
                # exec(cleaned_code)
                model_loaded = True
                print(f"Code executed successfully on attempt {attempt + 1}")
            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {str(e)}")
                try:
                    corrected_code = self.code_error_correction_agent(current_code, str(e))
                    print("Corrected Code:", corrected_code)
                    current_code = corrected_code
                    print("Attempting to run corrected code...")
                except Exception as correction_error:
                    print(f"Error during code correction: {correction_error}")
                    
                if attempt + 1 == max_attempts:
                    print("Maximum attempts reached for the current model. Switching to a new model.")
                    current_model_index = additional_models.index(self.base_model)
                    next_model_index = current_model_index + 1
                    if next_model_index == len(additional_models): 
                        break
                    self.base_model = additional_models[next_model_index]
                    self.pipeline = transformers.pipeline("text-generation", model=self.base_model, trust_remote_code=True)
                    attempt = 0 
                else:
                    attempt += 1

        if not model_loaded:
            print("Failed to execute code after maximum attempts with all models.")
            raise RuntimeError("Could not execute the code after max attempts with all models.")
        
        return current_code  # Return the final executed code
    
    def generate_learning_blurb_agent(self, data, goal, generalDescription, visualDescription, code):
        messages = self.prompts.learning_blurb_prompt(data, goal, generalDescription, visualDescription, code)
        outputs = self.pipeline(messages, max_new_tokens=256)
        response = outputs[0]["generated_text"][-1]['content']
        return response