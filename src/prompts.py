class Prompts:
    def __init__(self, topic):
        self.topic = topic
    
    def goal_explorer_prompt(self, data):
        system_prompt = (
            f"You are a expert professor in topic of {self.topic}. "
            f"A student comes with content and needs more help understanding the content. "
            f"The student provides the following context: {data}. "
            f"Your goal is to create a visualization to aid the student in better understanding the content."
        )
        
        user_prompt = (
            "If you were to build a visualization what would it ential. "
            "Return your idea in text not code. There should be no code at all. "
            "Provide your ideas on how you would build this visualization to aid the student to better understand the content given. "
            "This visualization mut be static, and will go in a texbook, so make sure it is pedigogically aligned."
        )
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def general_description_prompt(self, data, goal):
        system_prompt = (
            f"You are an expert professor in the topic of {self.topic}. "
            f"The student provides the following context: {data}"
            "A student is seeking guidance related to this topic and aims to create a visualization. "
            "Your task is to assist them by structuring the necessary information in a JSON format."
        )
        
        user_prompt = (
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
            "Ensure that the response is well-structured, concise, and clear to assist the student effectively. "
            "Do not include anything other than the JSON in your response"
        )
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def visual_description_prompt(self, data, goal, general_description):
        system_prompt = (
            f"You are an expert professor in the topic of {self.topic}. "
            f"The student provides the following context: {data} "
            f"The student provides the following general description: {general_description}. "
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
            "Ensure that the response is concise, clear, and well-structured to assist the student in the best possible way. "
            "Do not include anything other than the JSON in your response."
        )
        
        user_prompt = (
            f"A student presents the following question goal: '{goal}'. "
            "Before proceeding with coding the visualization, the student must first understand key foundational aspects. "
            "Your response should be formatted as a JSON object using the structure above."
        )
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def visualization_code_prompt(self, goal, visual_description):
        system_prompt = (
            f"You are an expert data visualization assistant. "
            f"You are given this goal: {goal}"
            "Your task is to interpret the provided visual description and generate Python code for the visualization. "
            "The code should be generated using Python packages such as matplotlib, seaborn, and etc (whatever best suits the visualization) to plot 3D or 2D surfaces, handle annotations, and apply custom styling as described in the visual description.\n\n"
            "The visual description includes details like the title, overview, elements, layout, annotations, axes, and styling, all of which should guide the Python code creation."
            "This visualization should be created with the aim to teach students, so it must be pedigogically aligned."
        )
        
        user_prompt = (
            f"Here is the visual description you need to interpret and create the visualization code for:{visual_description}\n"
            "Using the details provided, generate Python code using matplotlib to create the described visualization. "
            "The code should include the necessary plots, surface creation, axis labels, annotations, and any required styling and transparency effects."
            "Only output code. No comments within the code. Just executable code, that if you pass it through the exec function in python it will not error. "
            "Again it is very important that you only return executable python code"
        )
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def code_error_correction_prompt(self, original_code, error_message):
        system_prompt = (
            "You are an expert Python programmer and debugging assistant. "
            "Your task is to take the original code and the specific error message, "
            "and generate a corrected version of the code that resolves the error. "
            "Pay special attention to type conversions, dictionary creation, and "
            "matplotlib-specific plotting issues."
        )
        
        user_prompt = (
            f"Original Code:\n{original_code}\n\n"
            f"Error Message:\n{error_message}\n\n"
            "Carefully review the code, especially around method calls and argument passing"
            "Return only the corrected Python code that can be directly executed. "
            "Do not include any markdown formatting or code block markers."
            "Make sure you change the original code. Do not repeat the same eror message"
            "Change the original code completely. Make sure you are using matplotlib."
            "Do not return any extra text about yout changes, just the code itself."
        )
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def learning_blurb_prompt(self, data, goal, general_description, visual_description, code):
        system_prompt = (
            f"You are an expert educational content creator specializing in explaining complex topics like {self.topic}. "
            "Your task is to create a concise, engaging, and pedagogically effective blurb that reinforces the key learning points "
            "of a visualization, making the technical concept more accessible to students."
        )
        
        user_prompt = (
            f"Given the following context:\n"
            f"Original Data: {data}\n"
            f"Learning Goal: {goal}\n"
            f"General Description: {general_description}\n"
            f"Visual Description: {visual_description}\n"
            f"Visualizarion Code: {code}]\n\n"
            "Create a learning blurb that:\n"
            "1. Explains the core concept in simple language\n"
            "2. Highlights the key insights from the visualization\n"
            "3. Provides a memorable takeaway for students\n"
            "4. Is no more than 150-200 words\n"
            "5. Uses an engaging, conversational tone appropriate for students\n"
            "Format your response as a single paragraph."
        )
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]