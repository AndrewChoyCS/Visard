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
            # f"Ensure that the query is aligned with {self.topic}"
            f"Ensure that the learning objectives are is aligned with {self.topic}"
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
        
    def visualization_code_generator_prompt(self, goal, output_dir):
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
            "Ensure that all visual elements contribute to some learning outcome."
            "Ensure that all visual elements contribute are clearly distinguishable."

        )
        user_prompt = (
            f"Your task is: {goal}"
            "Create executable Python code using the Matplotlib package"
            "Implement best practices in Matplotlib, ensuring that each plot is comprehensive, the visual hierarchy is maintained, and key information is presented prominently."
            "Do not create any functions within the code."
            "Write the code so i can pass it through python exec() with no erros"
            "DO NOT include plt.show() at the end"
            # f"Come up with a title and save the figure you generated using plt.savefig({output_dir} + str(TITLE)) at the end"
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
            "DO NOT INCLUDE plt.show() at the end"
        )

        return system_prompt, user_prompt

    
    # def goal_alignment_judge_prompt(self, goal, corrected_code): 
    #     system_prompt = (
    #         "You are an expert evaluator reviewing a data visualization based on a stated learning goal.\n"
    #         "Assess how well the provided Python code achieves this goal in terms of clarity, alignment, and insight delivery.\n\n"
    #         f"GOAL:\n{goal}\n\nCODE:\n{corrected_code}\n"
    #     )

    #     user_prompt = (
    #         "Evaluate the visualization using these criteria:\n"
    #         "1. Does the visualization effectively align with the learning goal?\n"
    #         "2. Is the topic clear and understandable?\n"
    #         "3. Are key insights presented clearly with proper context and conclusions?\n\n"
    #         "YOUR RESPONSE MUST FOLLOW THIS EXACT FORMAT (no deviations):\n"
    #         "[true|false]\n\n"
    #         "Your feedback here, written as a paragraph.\n\n"
    #         "**DO NOT** list bullets. **DO NOT** restate the evaluation criteria.\n"
    #         "**ONLY** return a single word on the first line ('true' or 'false'), followed by improvement feedback in paragraph form."
    #     )
    #     return system_prompt, user_prompt

    def goal_alignment_judge_prompt(self, goal, corrected_code): 
        system_prompt = (
            "You are an expert evaluator reviewing a data visualization based on a stated learning goal.\n"
            "Assess how well the provided Python code achieves this goal in terms of clarity, alignment, and insight delivery.\n\n"
            f"GOAL:\n{goal}\n\nCODE:\n{corrected_code}\n"
        )

        user_prompt = (
            "Evaluate the visualization using the following criteria and provide a score on a scale of 1 to 5 for each:\n"
            "1. Does the visualization effectively align with the learning goal?\n"
            "2. Is the topic clear and understandable?\n"
            "3. Are key insights presented clearly with proper context and conclusions?\n\n"
            "Use the following rubric to score each aspect on a scale of 1 to 5:\n"
            "5 - Excellent: The visualization excels in this criterion.\n"
            "4 - Good: The visualization is strong in this criterion, with minor improvements needed.\n"
            "3 - Fair: The visualization meets basic requirements but lacks depth or clarity.\n"
            "2 - Poor: The visualization doesn't meet the criterion adequately.\n"
            "1 - Very Poor: The visualization fails to meet this criterion.\n\n"
            "YOUR RESPONSE MUST FOLLOW THIS EXACT FORMAT (no deviations):\n"
            "[1-5]\n"
            "Feedback: Your evaluation here, written as a paragraph, including suggestions for improvement.\n\n"
            "**DO NOT** list bullets. **DO NOT** restate the evaluation criteria.\n"
            "**ONLY** return the overall score followed by actionable feedback."
        )
        return system_prompt, user_prompt

    def visual_clarity_judge_prompt(self, corrected_code): 
        system_prompt = (
            "You are an expert in data visualization reviewing the visual output generated from the following Python code.\n"
            "Your task is to assess how clear, interpretable, and visually effective the resulting chart would be.\n\n"
            f"CODE:\n{corrected_code}\n"
        )

        user_prompt = (
            "Evaluate the visualization using the following criteria and provide a score on a scale of 1 to 5 for each:\n"
            "1. Is the visualization easy to interpret at a glance?\n"
            "2. Are colors, contrast, and visual hierarchy used effectively?\n"
            "3. Are labels, titles, and annotations clear and helpful?\n"
            "4. Does the design effectively communicate the intended data insights?\n\n"
            "Use the following rubric to score each aspect on a scale of 1 to 5:\n"
            "5 - Excellent: The visualization excels in this criterion.\n"
            "4 - Good: The visualization is strong in this criterion, with minor improvements needed.\n"
            "3 - Fair: The visualization meets basic requirements but lacks depth or clarity.\n"
            "2 - Poor: The visualization doesn't meet the criterion adequately.\n"
            "1 - Very Poor: The visualization fails to meet this criterion.\n\n"
            "YOUR RESPONSE MUST FOLLOW THIS EXACT FORMAT (no deviations):\n"
            "[1-5]\n"
            "Feedback: Your evaluation here, written as a paragraph, including suggestions for improvement.\n\n"
            "**DO NOT** use bullet points.\n"
            "**DO NOT** repeat the evaluation questions.\n"
            "**ONLY** return the overall score followed by actionable feedback."
        )
        return system_prompt, user_prompt
    
    def code_generator_from_judge_feedback_prompt(self, code, feedback): 
        system_prompt = (
            "You are a skilled developer tasked with improving a data visualization script based on expert feedback.\n"
            "Below is the original code and the feedback that highlights what needs to be improved."
            f"\n\nCODE:\n{code}\n\n"
            f"FEEDBACK:\n{feedback}\n"
        )
        user_prompt = (
            "Revise the code to address the feedback.\n"
            "Your output must be ONLY the updated code — no explanations, comments, or markdown formatting."
            "DO NOT INCLUDE plt.show() at the end"
        )
        return system_prompt, user_prompt
