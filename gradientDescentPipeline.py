# TODO
# input = paragraph about graident descent, output = General description
 
# input = General Description,output = Visual Description

# input = Visual Description, output = Code for visualization

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

class Pipeline(): 
    def __init__(self, data):
        self.model = AutoModelForCausalLM.from_pretrained("llama3_3b_instruct")
        self.tokenizer = AutoTokenizer.from_pretrained("llama3_3b_instruct")
        self.pipeline = transformers.pipeline(
                            "text-generation",
                            model="meta-llama/Llama-3.2-3B-Instruct",
                            model_kwargs={"torch_dtype": torch.bfloat16},
                        )
        self.TOPIC = "Graident Descent"


        goalRet = self.goal_explorer_agent(data)
        print(goalRet)
        generalDescription = self.goal_to_general_description_agent(data, goalRet)
        print(generalDescription)
        visualDescription = self.general_description_to_visual_description_agent(data, goalRet, generalDescription)
        print(visualDescription)
        code = self.visual_description_to_visualization_code_agent(visualDescription)
        print(code)

        
    def chunking_agent():
        pass 
    
    def clean_data():
        pass

    def goal_explorer_agent(self, data):
        print("Executing Goal Explorer Agent")
        
        messages = [
            {"role": "system", "content": f"You are a expert professor in topic of {self.TOPIC}"},
            {"role": "user", "content": f'Given this data: {data}. I want to create a visualization for this image. Provide one idea on how I can build a visualization based on this data, this idea must be feasible from a coding package standpoint, meaning I should be able to code up your idea. Only give me the idea nothing more. Format your reponse in the following. Visualization Idea: (put your idea here)'}
        ]
        outputs = self.pipeline(
                messages,
                max_new_tokens=256)
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
        
        # Call the pipeline with the formatted messages and the appropriate max_new_tokens
        outputs = self.pipeline(messages, max_new_tokens=1024)
        
        # Extract the content from the response and return it
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
                    "Using the details provided, generate Python code using packages to create the described visualization. "
                    "The code should include the necessary plots, surface creation, axis labels, annotations, and any required styling and transparency effects."
                    "Only output code. No comments within the code. Just executable code"
                ),
            }
        ]

        # Use the model's pipeline to generate code based on the description
        outputs = self.pipeline(messages, max_new_tokens=1024)

        # Extract and return the generated Python code
        response = outputs[0]["generated_text"][-1]['content']
        print(type(response))
        return response

data = """
        Looking at the function across this domain, it is clear that the function’s minimum value occurs around theta = 5.3.Let’s pretend for a moment that we couldn’t see the full view of the cost function. How would we guess the value of thetathat minimizes the function?

        It turns out that the first derivative of the function can give us a clue. In the graph below, the function and its derivative are plotted, with points where the derivative is equal to 0 plotted in light green.

        In the plots below, the line indicates the value of the derivative of each value of. The derivative is negative where it is red and positive where it is green.
        """

Pipeline(data)
