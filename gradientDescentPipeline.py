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

        # Run the pipeline
        goalRet = self.goal_explorer_agent(data)
        print(goalRet)
        generalDescription = self.goal_to_general_description_agent(data, goalRet)
        print(generalDescription)
        visualDescription = self.general_description_to_visual_description_agent(data, goalRet, generalDescription)
        print(visualDescription)
        code = self.visual_description_to_visualization_code_agent(visualDescription)
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
            # Ensure the code is a clean, executable string
            cleaned_code = code.strip().replace('```python', '').replace('```', '').strip()
            
            # Create a local namespace to execute the code
            local_namespace = {}
            
            # Execute the code
            exec(cleaned_code, globals(), local_namespace)
            
            # Try to save the current figure
            plt.tight_layout()
            figure_path = os.path.join(self.output_dir, 'gradient_descent_visualization.png')
            
            # Check if a figure exists before saving
            if plt.gcf().get_axes():
                plt.savefig(figure_path, dpi=300, bbox_inches='tight')
                plt.close()  # Close the figure to free up memory
                
                print(f"Visualization saved to {figure_path}")
                return figure_path
            else:
                print("No figure was generated to save.")
                return None
        
        except Exception as e:
            print(f"Error saving visualization: {e}")
            # Print the traceback for more detailed error information
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
        
        messages = [
            {"role": "system", "content": f"You are a expert professor in topic of {self.TOPIC}. A student comes with content and needs more help understanding the content. This is the content: {data}. Your goal is to create a visualization to aid the student in better understanding the content."},
            # {"role": "user", "content": f'Given this data: {data}. I want to create a visualization for this image. What are the most important aspects to this image to teach the student about the topic. Provide one main idea on how I can build a visualization based on this data, this idea must be feasible from a coding package standpoint, meaning I should be able to code up your idea. Only give me the idea nothing more. Format your reponse in the following. Visualization Idea: (put your idea here). This idea should NOT be code it should be a generic idea returned in text form.'}
             {"role": "user", "content":'If you were to build a visualization what would it ential. Return your idea in text not code. There should be no code at all. Provide your ideas on how you would build this visualization to aid the student to better understand the content given. This visualization mut be static, and will go in a texbook, so make sure it is pedigogically aligned. '}
        ]
        outputs = self.pipeline(
                messages,
                max_new_tokens=512)
        response = outputs[0]["generated_text"][-1]['content']
        return response

    def goal_to_general_description_agent(self, data: str, goal: str):
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are an expert professor in the topic of {self.TOPIC}. "
                    f"The student provides the following context: {data}"
                    "A student is seeking guidance related to this topic and aims to create a visualization. "
                    "Your task is to assist them by structuring the necessary information in a JSON format."
                    
                ),
            },
            {
                "role": "user",
                "content": (
                    f"A student presents the following question goal: '{goal}'."
                    "Before proceeding with visualization, the student must first understand key foundational aspects. "
                    "Your response should be formatted as a JSON object with the following fields:\n\n"
                    "{\n"
                    '  "Title": "{Title of the Visualization}",\n'
                    '  "Type": "{Definition | Problem Explanation | Example Explanation}",\n'
                    '  "Concept": "{Core concept being explained (e.g., double integral, limit, derivative)}",\n'
                    '  "Definition": "{A clear, concise definition of the concept, if applicable}",\n'
                    '  "Student Background": "{The expected background level (e.g., introductory calculus)}",\n'
                    '  "Objective": "{What the student should understand after viewing the visualization}",\n'
                    '  "Emphasis": "{Key points to highlight or common misconceptions to address}",\n'
                    '  "Related Topics": "{Other relevant topics or extensions (e.g., single integrals, triple integrals, volume computation)}",\n'
                    '  "Conclusion": "{A brief summary of the key takeaway from the visualization}"\n'
                    "}\n\n"
                    "Ensure that the response is well-structured, concise, and clear to assist the student effectively. Do not include anything other than the JSON in your response"
                ),
            }
        ]
        outputs = self.pipeline(
                        messages,
                        max_new_tokens=1024)
        response  = outputs[0]["generated_text"][-1]['content']
        return response

    def general_description_to_visual_description_agent(self, data, goal, generalDescription):
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are an expert professor in the topic of {self.TOPIC}. "
                    f"The student provides the following context: {data} "
                    f"The student provides the following general description: {generalDescription}. "
                    "A student is seeking guidance related to this topic and aims to create a visualization. "
                    "Your task is to assist them by structuring the necessary information in a JSON format. "
                    "Use the following JSON structure for the response:"
                    "\n\n"
                    "{\n"
                    '  "Title": "{A concise title for the visual}",\n'
                    '  "Overview": "{A brief summary describing what the visual represents.}",\n'
                    '  "Elements": {\n'
                    '    "Element1": "{Description of the first major visual component (type, color, shape, size, position)}",\n'
                    '    "Element2": "{Description of the second major component}",\n'
                    '    "...": "{Additional visual elements as needed}"\n'
                    '  },\n'
                    '  "Layout": "{Details on the spatial relationships and arrangement of elements (e.g., \'Element1 is centered; Element2 is to the left of Element1\')}",\n'
                    '  "Annotations": {\n'
                    '    "Annotation1": "{Text or labels added to the visual, including style, placement, and any pointer or arrow details}",\n'
                    '    "Annotation2": "{Additional annotation details}",\n'
                    '    "...": "{Other annotations if applicable}"\n'
                    '  },\n'
                    '  "Axes/Scale/Legends": "{Description of axes (if present), scales, legends, grid lines, or any reference markers}",\n'
                    '  "Styling": "{Stylistic details such as color schemes, fonts, transparency, and overall design choices}",\n'
                    '  "Conclusion": "{A short statement summarizing the overall message or insight conveyed by the visual}"\n'
                    "}\n\n"
                    "Ensure that the response is concise, clear, and well-structured to assist the student in the best possible way. Do not include anything other than the JSON in your response."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"A student presents the following question goal: '{goal}'. "
                    "Before proceeding with coding the visualization, the student must first understand key foundational aspects. "
                    "Your response should be formatted as a JSON object using the structure above."
                ),
            }
        ]
        
        outputs = self.pipeline(messages, max_new_tokens=1024)
        response = outputs[0]["generated_text"][-1]['content']
        return response


    def visual_description_to_visualization_code_agent(self, visual_description):
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are an expert data visualization assistant. "
                    "Your task is to interpret the provided visual description and generate Python code for the visualization. "
                    "The code should be generated using Python packages such as matplotlib, seaborn, and etc (whatever best suits the visualization) to plot 3D or 2D surfaces, handle annotations, and apply custom styling as described in the visual description.\n\n"
                    "The visual description includes details like the title, overview, elements, layout, annotations, axes, and styling, all of which should guide the Python code creation."
                    "This visualization should be created with the aim to teach students, so it must be pedigogically aligned."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Here is the visual description you need to interpret and create the visualization code for:{visual_description}\n"
                    "Using the details provided, generate Python code using matplotlib to create the described visualization. "
                    "The code should include the necessary plots, surface creation, axis labels, annotations, and any required styling and transparency effects."
                    "Only output code. No comments within the code. Just executable code, that if you pass it through the exec function in python it will not error. Again it is very important that you only return executable python code"
                ),
            }
        ]
        outputs = self.pipeline(messages, max_new_tokens=1024)
        response = outputs[0]["generated_text"][-1]['content']
        return response
    

    def code_error_correction_agent(self, original_code, error_message):
        messages = [
            {
                "role": "system", 
                "content": (
                    "You are an expert Python programmer and debugging assistant. "
                    "Your task is to take the original code and the specific error message, "
                    "and generate a corrected version of the code that resolves the error. "
                    "Pay special attention to type conversions, dictionary creation, and "
                    "matplotlib-specific plotting issues."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Original Code:\n{original_code}\n\n"
                    f"Error Message:\n{error_message}\n\n"
                    # "The error suggests a mapping (dictionary) issue."
                    "Carefully review the code, especially around method calls and argument passing"
                    "Return only the corrected Python code that can be directly executed. "
                    "Do not include any markdown formatting or code block markers."
                    "Make sure you change the original code. Do not repeat the same eror message"
                    "Change the original code completely. Make sure you are using matplotlib."
                    "Do not return any extra text about yout changes, just the code itself."
                )
            }
        ]
        
        outputs = self.pipeline(
            messages,
            max_new_tokens=1024
        )
        
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
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are an expert educational content creator specializing in explaining complex topics like {self.TOPIC}. "
                    "Your task is to create a concise, engaging, and pedagogically effective blurb that reinforces the key learning points "
                    "of a visualization, making the technical concept more accessible to students."
                )
            },
            {
                "role": "user", 
                "content": (
                    f"Given the following context:\n"
                    f"Original Data: {data}\n"
                    f"Learning Goal: {goal}\n"
                    f"General Description: {generalDescription}\n"
                    f"Visual Description: {visualDescription}\n"
                    f"Visualizarion Code: {code}]\n\n"
                    "Create a learning blurb that:\n"
                    "1. Explains the core concept in simple language\n"
                    "2. Highlights the key insights from the visualization\n"
                    "3. Provides a memorable takeaway for students\n"
                    "4. Is no more than 150-200 words\n"
                    "5. Uses an engaging, conversational tone appropriate for students\n"
                    "Format your response as a single paragraph."
                )
            }
        ]
        
        outputs = self.pipeline(
            messages,
            max_new_tokens=256
        )
        
        response = outputs[0]["generated_text"][-1]['content']
        return response

data = """
        In our analysis above, we focused our attention on the global minimum of the loss function. You may be wondering: what about
the local minimum thatʼs just to the left?
If we had chosen a different starting guess for , or a different value for the learning rate , our algorithm may have gotten
“stuck” and converged on the local minimum, rather than on the true optimum value of loss.
If the loss function is convex, gradient descent is guaranteed to converge and find the global minimum of the objective function.
Formally, a function is convex if:
for all in the domain of and .
To put this into words: if you drew a line between any two points on the curve, all values on the curve must be
on or below the
line. Importantly, any local minimum of a convex function is also its global minimum so we avoid the situation where the
algorithm converges on some critical point that is not the minimum of the function.
        """

pipe = Pipeline(data)