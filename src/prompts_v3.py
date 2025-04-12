class Prompts:
    def __init__(self, topic):
        self.topic = topic
    
    def simple_query_prompt(self, data):
        system_prompt = (
            f"You are an expert in {self.topic}, instructional visualization, and visualization creation."
        )
        user_prompt = (
            f"Extract the core principles from the following text that would be useful to create a visualization: {data}"
            "Return in simple terms what you would include in the visulization"
        )

        # return [
        #     {"role": "system", "content": system_prompt},
        #     {"role": "user", "content": user_prompt}
        # ]
        return system_prompt, user_prompt
        
    def visualization_code_generator_prompt(self, goal):
        system_prompt = (
            "You are a senior visualization developer specializing in educational graphics with Python. "
        )
        user_prompt = (
            f"You are given this goal: {goal}"
            "Create executable Python code using the matplot lib package"
            "Include plt.show() at the end"
            "ONLY PROVIDE THE PYTHON CODE - NO EXPLANATIONS OR COMMENTARY."
        )
        # return [
        #     {"role": "system", "content": system_prompt},
        #     {"role": "user", "content": user_prompt}
        # ] 
        return system_prompt, user_prompt
        
    def code_error_identifier_prompt(self, original_code, error_message):
        system_prompt = (
            "You are a Python debugging specialist with expertise in matplotlib and visualization libraries. "
            "Your task is to diagnose and fix execution errors in visualization code, producing working solutions "
            "that maintain the original code's intent while resolving all technical issues. "
            "You have exceptional ability to interpret error traces, identify root causes, and implement proper fixes."
        )

        user_prompt = (
            f"CODE WITH ERROR: {original_code}"
            f"ERROR MESSEAGE: {error_message}"
            "Given this code with the erorr message and the code, explain how the error message and how it occured."
            "In addition, also explain in great detail how you would fix this error. Clearly indicate what you need to change"
            "in order to change this code and have it be executable with no errors"
            "Do not return any code. Just return an explanation of how you would fix the errors, so they a no longer produced. "
        )
        # return [
        #     {"role": "system", "content": system_prompt},
        #     {"role": "user", "content": user_prompt}
        # ]
        return system_prompt, user_prompt

    
    def code_error_correction_prompt(self, original_code, error_message, explanation):
        system_prompt = (
            "You are a Python debugging specialist with expertise in matplotlib and visualization libraries. "
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
        # return [
        #     {"role": "system", "content": system_prompt},
        #     {"role": "user", "content": user_prompt}
        # ]
        return system_prompt, user_prompt
