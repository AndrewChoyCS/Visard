class Prompts:
    def __init__(self, topic):
        self.topic = topic
    
    def simple_query_prompt(self, data):
        system_prompt = (
            f"You are an expert instructor in {self.topic} and visual learning theory."
            f"You're given the following paragraph from a textbook: {data}"
            f"Your task is to transform the context that you get from {data} into a clear and structured query for a visualization expert to create an appropriate visualization to aid the understanding of that paragraph."
            f"The goal of your query should be to illustrate an important idea, concept, or explanation from the data in a clear and intuitive way with a visual representation."
            f"The query should guide the visualization expert to create a visual that is useful for instruction, is creative, clear, and concisely labeled."
            # "The goal is to make an abstract idea more accessible."
            "Ensure that the query inspires a visualization that not only explains the concept but also reveals the underlying relationships, hierarchies, or progressions inherent in the material."
            "Specify that the visualization must use visual representation of the concept and NOT A MIND MAP (feel free to use any visual objects and include multiple graphs)"
            "Describe how the visualization could integrate best practices in instructional design—such as highlighting cause-effect relationships, sequential processes, or comparative analysis—to deepen the learner’s insight."
            # "Specify that the visualization should include clear icons, strategic use of color to differentiate concepts, and annotated components that guide the viewer through the key ideas step-by-step."
            # "Strive for a balance between creative visualization and academic rigor, ensuring that the depiction not only attracts the audience but also withstands scholarly examination."

        )
        user_prompt = (
            f"Extract a core principle from the following text that would be best explained with a visualization: {data} "
            f"Based on the extracted principle/concept, create a clear query starting with 'Create a visualization to explain' and follow it with 2-5 sentences describing the visualization’s goal, the specific concept it is intended to clarify, and the key educational outcomes expected. "
            "Specify which aspect of the principle should be visualized (such as relationships, sequences, or hierarchies), "
            "suggest appropriate visualization styles (e.g., infographic, flowchart, diagram), and explain how the visual aids in understanding the concept deeply and clearly."
            "Create an objective that focuses on deep and crucial understanding of the concept."

        )

        return system_prompt, user_prompt
        
    def visualization_code_generator_prompt(self, goal):
        system_prompt = (
            "You are a senior visualization developer specializing in educational graphics with Python."
            "You are an expert in transforming abstract concepts into amazing visualizations that make the concepts clear, accessible and appealing to understand."
            "The created visualization must follow visualization best practices."
            f"The created visualization must be very clear, include well-placed explanatory labels and be consistent with the {goal}."
            "If the same goal can be achieved with simpler visual elements prefer that."
            "Use minimal explanatory text, convey the idea with visual elements"
            "When possible, use simpler visual elements that achieve the same objective, avoiding extra complexity."
            "Ensure that visual elements are clear to interpret."
            "Confirm that every labeled component is explicitly defined and mathematically validated, preventing any undefined or empty labels in the visualization."
            
        )
        user_prompt = (
            f"Your task is: {goal}"
            "Create executable Python code using the Matplotlib package"
            "Implement best practices in Matplotlib, ensuring that each plot is comprehensive, the visual hierarchy is maintained, and key information is presented prominently."
            "Structure the code so that it can be easily customized or extended for future enhancements, keeping it flexible for integrating additional data or visualization elements if needed."
            "Include plt.show() at the end"
            "ONLY PROVIDE THE PYTHON CODE - NO EXPLANATIONS OR COMMENTARY."
        )

        return system_prompt, user_prompt
        
    def code_error_identifier_prompt(self, original_code, error_message):
        system_prompt = (
            "You are a Python debugging specialist with expertise in Matplotlib and visualization libraries. "
            "Your task is to diagnose and fix execution errors in visualization code, producing working solutions "
            "that maintain the original code's intent while resolving all technical issues. "
            "You have exceptional ability to interpret error traces, identify root causes, and implement proper fixes."
        )

        user_prompt = (
            f"CODE WITH ERROR: {original_code}"
            f"ERROR MESSAGE: {error_message}"
            "Given this code with the error message and the code, explain how the error message and how it occured."
            "Do a complete error analysis, provide lines where possible."
            "Preserve all code that is not causing the current error."
            "In addition, also explain in great detail how you would fix this error. Clearly indicate what you need to change"
            "in order to change this code and have it be executable with no errors"
            "Do not return any code. Just return an explanation of how you would fix the errors, so they a no longer produced. "
        )

        return system_prompt, user_prompt

    
    def code_error_correction_prompt(self, original_code, error_message, explanation):
        system_prompt = (
            "You are a Python debugging specialist with expertise in Matplotlib and visualization libraries. "
            "Your task is to diagnose and fix execution errors in visualization code, producing working solutions "
            "that maintain the original code's intent while resolving all technical issues. "
            "You have exceptional ability to interpret error traces, identify root causes, and implement proper fixes."
        )
        
        user_prompt = (
            f"Fix this visualization code that has generated an error:\n\n"
            f"CODE WITH ERROR:\n{original_code}\n\n"
            f"ERROR MESSAGE:\n{error_message}\n\n"
            "INSTRUCTIONS:\n"
            "1. Analyze the error message to identify the precise issue\n"
            "2. Review the surrounding context to understand dependent code\n"
            "3. Fix ALL potential issues, not just the immediate error\n"
            "4. Maintain the original visualization's appearance and functionality\n"
            "5. Return ONLY the complete, corrected code ready for execution\n\n"

            f"Here is a guide on how to fix it: {explanation}"
            "Your output must be ONLY the fixed code with no explanations, comments about changes, or markdown formatting."
        )

        return system_prompt, user_prompt


    def first_judge_prompt(self, goal, corrected_code):

        system_prompt = (
            "You are tasked to score an instructional visualization having its goal and the code for the visual."
            f"GOAL: {goal}"
            f"CODE: {corrected_code}"
        )

        user_prompt = (
            "The instructions for scoring each criteria are as follows:"
            "1.Assign a score from **1 to 5** for each dimension:"
            "- **1**: Very poor quality, completely fails to meet the criteria."
            "- **2**: Below average, significant issues present."
            "- **3**: Acceptable, meets the basic criteria with minor issues."
            "- **4**: Good, performs well with no major issues."
            "- **5**: Excellent, fully meets or exceeds expectations."
            #adapted form TheoremExplainerAgent
            
            "Rank the visual based on the following criteria"
            "You MUST assign a score for each rubric item and return the SUM of the scores "
            f"1. GOAL ALIGNMENT (0-5 points)\n"
            "   - How well does the visualization align with the stated learning goal: {goal}?\n"
            "   - Does it accurately represent the core concepts described in the general description?{general_description}\n"
            "   - Does it emphasize the key points mentioned in the 'Emphasis' section?\n\n"
            "   - Does it have a clear topic that it's trying to explain?\n\n"
            "   - Are key insights provided clearly, with appropriate context and conclusions??\n\n"

            
            "2. TECHNICAL CORRECTNESS (0-5 points)\n"
            "   - Is the visualization mathematically/scientifically accurate?\n"
            "   - Are axes, labels, scales, and units appropriate and accurate?\n"
            "   - Are relationships between elements correctly depicted?\n\n"
            "   - Does it apply 'appropriate'graphic' variable'types'for'the'data' type'and'scale?\n\n"
            
            
            "3. VISUAL CLARITY (0-5 points)\n"
            "   - Is the visualization immediately interpretable without excessive cognitive load?\n"
            "   - Are colors, contrasts, and visual hierarchies effectively used?\n"
            "   - Are annotations clear, well-placed, and helpful?\n\n"
            "   -    Do the visuals communicate the data effectively??\n\n"
            
            "4. PEDAGOGICAL EFFECTIVENESS (0-5 points)\n"
            "   - Does the visualization facilitate understanding of the concept?\n"
            "   - Are complexity and detail appropriate for the stated student background?\n"
            "   - Does it provide insight beyond what text alone could convey?\n\n"
            "   - Does everything in'the' visualization'conveys some' information'to'the'viewer.'\n\n"
            "   - Is the visualization to the intended audience. Is the audience properly considered in terms of visual design, conveyed?"
            "   - Do the Legends should describe and explain every graphic variable type employed."
            
            "IMPORTANT: Return ONLY a single numerical score between 0-20. Do not include any explanation, "
            
            "ONLY RETURN A SINGLE INTEGER WITH NO EXPLANATIONS OR COMMENTS "
        )

        return system_prompt, user_prompt