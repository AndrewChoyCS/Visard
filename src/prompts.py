class Prompts:
    def __init__(self, topic):
        self.topic = topic
    
    def goal_explorer_prompt(self, data):
        system_prompt = (
            f"You are an expert in {self.topic}, instructional visualization, and visualization creation."
            f"You need to create a visualization to explain {self.topic} clearly and simply. "
            f"Your are provided the following context: {data}. "
            f"Extract and summarize the key entities provided in {data} in 6-10 concise sentences." 
            f"Clearly list the most important entities, concepts, relationships, and equations from the data that must be visually represented to effectively explain {self.topic}."       
        )
        
        user_prompt = (
            f"Given your summarization above generate a complex and insightful goal in 4-5 concise sentences about visualizing {self.topic} based on data"
            "The goal you generate MUST only be in text with no code at all."
            f"The goal must clearly specify what the planned visualization aims to explain."
            f"The goal must clearly specify the relationships and main figures of the planned visualization."
            f"The goal must specify how the planned visualization would aid student understanding of {self.topic}"
            
        )
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def general_description_prompt(self, data, goal):
        system_prompt = (
            f"You are an expert in {self.topic} and creating ideas and descriptions for educational visualizations."
            f"Your task is to create a structured and clear idea for a visualization with the following goal: '{goal}'."
            "Your response MUST be clear, concise and valuable for creating a visual."
            f"Your response MUST prioritize the explanation of {self.topic}"
            "Your response MUST be clear and detailed to guide human implementation"
            f"Your instruction is to create a detailed description of the visualization with the following format"
            
            # f"Your are provided the following context: {data}. "
        )
        
        user_prompt = (
            "Your response MUST be formatted as A VALID JSON object with exactly the following format:\n\n"
            
            "{\n"
            '  "Concept": "{A clear and concrete expression of the core concept in that is being explained(1-4 words)}",\n'
            '  "Title": "{A short, descriptive title of the visualization (1-4 words)}",\n'
            '  "Description": "{A clear description of the visualization in 7-8 sentences that is fully enough to reproduce the visualization.",\n'
            '  "Objective": "{A clear, concrete and measurable learning outcome of the visualization without any ambiguity. For example, \"After seeing this visualizations, learners will be able to explain why the derivative of a function at turning points is zero. \"}",\n'
            '  "Type": "{A category for the content the visualization is explaining. It muse be either Definition or Process Explanation or Problem Explanation or Example Explanation}",\n'
            '  "Student Background": "{A clear specification of the expected student background level needed to understand the visualization in 2-6 words(e.g., introductory calculus)}",\n'
            '  "Emphasis": "{A list of 3-5 key points the visualization must highlight in a clear and reproducible way.",\n'
            '  "Related Topics": "{A list of 2-5 relevant topics to the visualization}",\n'
            "}\n\n"
            "THE OUTPUT MUST ONLY USE THE JSON FORMAT ABOVE."
            
        
        )
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def visual_description_prompt(self, data, goal, general_description):
        system_prompt = (
                f"You are a world-class expert in data visualization and technical communication. "
                f"Your job is to design a clear, fully specified, and technically sound visualization based on the following: \n\n"
                f"Goal: '{goal}'\n"
                f"Description: '{general_description}'\n\n"
                f"The visualization plan you produce MUST follow data visualization best practices "
                f"(e.g., use bar charts instead of pie charts for comparing quantities, label axes clearly, avoid unnecessary chartjunk). "
                f"It MUST be meaningful, pedagogically sound, and appropriate for the stated goal and topic. "
                "The visualization must be pedagogically effective — using simple and clear visual elements, strong color contrast, legible fonts, and clearly placed annotations. "
                "It should be optimized for teaching and easy comprehension by students."
                "The visualization must be SIMPLE, INTERPRETABLE, and DIRECTLY aligned with the description provided."
            
        )
        
        user_prompt = (
                "Create a visualization specification that is COMPLETE and DETAILED enough to allow a developer to generate the plot exactly as intended, "
                "without needing any further clarification or assumptions. \n\n"
                "Use minimal explanatory text — focus on specific instructions about layout, axes, colors, chart type, labels, annotations, legends, and interactions. "
                "Do not assume the user understands your intent — be explicit in every visual detail. \n\n"
                "Your output MUST be a single VALID JSON object with the following format:\n\n"
            
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
            
            "Your response MUST ONLY BE the JSON object without any explanation or additional text."
        )
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def visualization_code_prompt(self, goal, visual_description):
        system_prompt = (
            "You are a world-class expert in data visualization code generation. "
            f"You are highly skilled in using Python libraries such as matplotlib, seaborn, and others to create high-quality, aesthetically pleasing, and educational visualizations.\n\n"
            f"Your task is to generate Python code that fulfills the following visualization goal:\n'{goal}'\n\n"
            "You will receive a detailed visual description that includes elements such as the title, overview, layout, axes, annotations, and styling preferences. "
            "Your job is to interpret this description and generate the corresponding Python code.\n\n"
        )
        
        user_prompt = (
            f"Here is the visual description you need to interpret and convert into Python code:\n\n{visual_description}\n\n"
            "Using the information above, generate clean, executable Python code using matplotlib (or another appropriate library). "
            "The code must implement the described visualization exactly, including layout, axes, labels, annotations, and any required styling or transparency effects.\n\n"
            "IMPORTANT INSTRUCTIONS:\n"
            "- Only output Python code. No explanations or comments.\n"
            "- The code MUST be directly executable without errors using Python's `exec()`.\n"
            "- Do not include any additional text. Only return the raw, complete Python code."
    )
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def visualization_evaluator_prompt(self, original_code, visual_description, general_description):
        system_prompt = (
            f"You are an expert in data visualization and the subject of '{self.topic}'.\n\n"
            "Your task is to evaluate whether the visual elements in a given Python visualization code—such as object coordinates, positions, and annotations—"
            "match the intended implementation plan and support the educational purpose.\n\n"
            "The plan is your reference, but your primary goal is to ensure that the rendered visualization effectively communicates the intended concept.\n\n"
            "If the current code meets all visual and pedagogical requirements, return the code as is.\n"
            "If any adjustments are needed, return the complete corrected version of the code."
            
        )
        
        user_prompt = (
                "Here is the original implementation plan and the corresponding Python visualization code:\n\n"
                "IDEA:\n{general_description}\n\n"
                "PLAN:\n{visual_description}\n\n"
                "CODE:\n{original_code}\n\n"
                "Evaluate the visualization based on the plan and educational goals. If everything is correct, return the exact same code. "
            
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
    
    # def learning_blurb_prompt(self, data, goal, general_description, visual_description, code):
    #     system_prompt = (
    #         f"You are an expert educational content creator specializing in explaining complex topics like {self.topic}. "
    #         "Your task is to create a concise, engaging, and pedagogically effective blurb that reinforces the key learning points "
    #         "of a visualization, making the technical concept more accessible to students."
    #     )
        
    #     user_prompt = (
    #         f"Given the following context:\n"
    #         f"Original Data: {data}\n"
    #         f"Learning Goal: {goal}\n"
    #         f"General Description: {general_description}\n"
    #         f"Visual Description: {visual_description}\n"
    #         f"Visualizarion Code: {code}]\n\n"
    #         "Create a learning blurb that:\n"
    #         "1. Explains the core concept in simple language\n"
    #         "2. Highlights the key insights from the visualization\n"
    #         "3. Provides a memorable takeaway for students\n"
    #         "4. Is no more than 150-200 words\n"
    #         "5. Uses an engaging, conversational tone appropriate for students\n"
    #         "Format your response as a single paragraph."
    #     )
        
    #     return [
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": user_prompt}
    #     ]