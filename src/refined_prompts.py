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
            f"You are a visualization design specialist with expertise in {self.topic} and cognitive learning theory. "
            f"You will create a detailed, implementation-ready visualization concept that achieves this goal: '{goal}'. "
            f"Your design must transform abstract concepts into concrete visual elements that create immediate cognitive understanding. "
            f"Focus on creating a visualization that is memorable, reduces cognitive load, and creates clear mental models of {self.topic}. "
            f"Your description must be precise enough that multiple designers would create nearly identical visualizations from it."
            f"Your specifications must follow established visualization best practices including:\n"
            f"- Using appropriate chart types for the data relationships involved\n"
            f"- Employing effective color schemes with strong contrast for instructional clarity\n"
            f"- Minimizing chart junk while maximizing data-ink ratio\n"
            f"- Creating clear visual hierarchies through size, position, and color\n"
            f"- Positioning annotations to minimize overlaps and maximize readability\n"
            f"Every visual element must directly support student understanding of {self.topic}."
        )
        
        user_prompt = (
            "Structure your response as a VALID JSON object with exactly the following format:\n\n"
            
            "{\n"
            '  "Concept": "{The single core principle being visualized (1-4 words)}",\n'
            '  "Title": "{A precise, informative title (1-4 words)}",\n'
            '  "Objective": "{A concrete, measurable learning outcome using Bloom\'s taxonomy verbs (e.g., identify, explain, predict, calculate)}",\n'
            '  "Description": "{A detailed visual scenario in 7-8 sentences specifying exactly what will be shown, how elements interact, and what changes or transformations occur}",\n'
            '  "Emphasis": "{3-5 critical insights the visualization must make visually obvious, listed as discrete points}",\n'
            '  "Outline": "{A structural blueprint of the visualization specifying spatial relationships, movement, sequencing, and focal points in 2-3 sentences}",\n'
            '  "Elements": "{A clear list of the essential components of the visualization in 4-8 words}",\n'
            '  "Type": "{ONE category from: Definition, Process Explanation, Problem Solving, or Conceptual Relationship}",\n'
            '  "Student Background": "{Precise prerequisite knowledge required (2-6 words)}",\n'
            '  "Related Topics": "{2-5 closely connected concepts that could be explored next}"\n'
            "}\n\n"
            "RESPOND ONLY WITH THE JSON OBJECT - NO EXPLANATIONS OR ADDITIONAL TEXT."
        )
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    # def visual_description_prompt(self, data, goal, general_description):
    #     system_prompt = (
    #         f"You are a technical visualization engineer with expertise in perceptual psychology and {self.topic}. "
    #         f"You will translate this visualization goal '{goal}' and description '{general_description}' into precise technical specifications. "
    #         f"Your specifications must follow established visualization best practices including:\n"
    #         f"- Using appropriate chart types for the data relationships involved\n"
    #         f"- Employing effective color schemes with strong contrast for instructional clarity\n"
    #         f"- Minimizing chart junk while maximizing data-ink ratio\n"
    #         f"- Creating clear visual hierarchies through size, position, and color\n"
    #         f"- Positioning annotations to minimize overlaps and maximize readability\n"
    #         f"Every visual element must directly support student understanding of {self.topic}."
    #     )
        
    #     user_prompt = (
    #         "Create a comprehensive implementation specification structured as a VALID JSON object with the following elements:\n\n"
            
    #         "{\n"
    #         '  "Title": "{Concise title for the visualization (1-4 words)}",\n'
    #         '  "Overview": "{Brief technical summary of visualization purpose and structure}",\n'
    #         '  "Elements": {\n'
    #         '    "Element1": [\n'
    #         '      {\n'
    #         '        "Type": "curve|line|point|vline|hline|area|shape",\n'
    #         '        "Expression": "{Mathematical expression for parametric elements}",\n'
    #         '        "Coordinates": [x, y],\n'
    #         '        "Color": "{CSS color name or hex code}",\n'
    #         '        "Width": "{Line width or stroke in pixels}",\n'
    #         '        "Style": "solid|dashed|dotted",\n'
    #         '        "Size": "{Point size in pixels}",\n'
    #         '        "Label": "{Element label text}"\n'
    #         '      }\n'
    #         '    ],\n'
    #         '    "Element2": [{...}],\n'
    #         '    "...": "[{Additional elements}]"\n'
    #         '  },\n'
    #         '  "Layout": "{Precise positioning and spatial relationships between elements}",\n'
    #         '  "Annotations": {\n'
    #         '    "Annotation1": [\n'
    #         '      {\n'
    #         '        "Text": "{Annotation content}",\n'
    #         '        "Position": "above_point|below_point|above_line|custom",\n'
    #         '        "ReferencePoint": [x, y],\n'
    #         '        "FontSize": "{Font size in points}",\n'
    #         '        "FontWeight": "normal|bold",\n'
    #         '        "FontStyle": "normal|italic",\n'
    #         '        "Color": "{Text color}",\n'
    #         '        "Arrow": true|false,\n'
    #         '        "ArrowColor": "{Arrow color}"\n'
    #         '      }\n'
    #         '    ],\n'
    #         '    "Annotation2": [{...}]\n'
    #         '  },\n'
    #         '  "Axes": {\n'
    #         '    "X": {\n'
    #         '      "Range": [minX, maxX],\n'
    #         '      "Ticks": "{Step size}",\n'
    #         '      "Label": "{X-axis label}",\n'
    #         '      "Arrow": true|false\n'
    #         '    },\n'
    #         '    "Y": {\n'
    #         '      "Range": [minY, maxY],\n'
    #         '      "Ticks": "{Step size}",\n'
    #         '      "Label": "{Y-axis label}",\n'
    #         '      "Arrow": true|false\n'
    #         '    },\n'
    #         '    "Grid": {\n'
    #         '      "Enabled": true|false,\n'
    #         '      "Style": "dashed|solid",\n'
    #         '      "Color": "{Grid color}"\n'
    #         '    }\n'
    #         '  },\n'
    #         '  "Styling": {\n'
    #         '    "Font": "{Font family name}",\n'
    #         '    "Background": "{Background color}",\n'
    #         '    "Layout": "centered|grid|split|overlay",\n'
    #         '    "TightLayout": true|false\n'
    #         '  },\n'
    #         '  "Conclusion": "{Core insight the visualization conveys}"\n'
    #         "}\n\n"
            
    #         "IMPORTANT: Provide complete and explicit specifications for every value - avoid placeholders. Your JSON must be valid and directly implementable without further clarification."
    #     )
        
    #     return [
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": user_prompt}
    #     ]
    

    def visualization_code_prompt(self, goal, general_description):
        system_prompt = (
            "You are a senior visualization developer specializing in educational graphics with Python. "
            f"You will translate this visual specification into executable Python code that creates a visualization meeting the goal: '{goal}'. "
            "Your code must be production-quality: well-structured, efficiently implemented, and fully executable without errors. "
            "You must adhere to these principles:\n"
            "1. Use matplotlib as the primary library with appropriate specialized libraries as needed\n"
            "2. Implement EVERY detail specified in the visual description exactly as specified\n"
            "3. Create robust code that gracefully handles edge cases\n"
            "4. Add minimal explanatory comments at section boundaries only\n"
            "5. Emphasize visual clarity through appropriate font sizes, line weights, and color contrast"
        )
        
        user_prompt = (
            f"Create executable Python code implementing this specification exactly:\n\n{general_description}\n\n"
            "Requirements:\n"
            "1. Generate COMPLETE, STANDALONE code that requires no modifications to run\n"
            "2. Include all necessary imports at the beginning\n"
            "3. Create intermediary variables for complex calculations to improve readability\n"
            "4. Set figure dimensions and DPI for high-quality output\n"
            "5. Use plt.tight_layout() or equivalent to avoid element crowding\n"
            "6. Include plt.show() at the end\n\n"
            "ONLY PROVIDE THE PYTHON CODE - NO EXPLANATIONS OR COMMENTARY."
        )
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
    def visualization_evaluator_prompt(self, original_code, visual_description, general_description):
        system_prompt = (
            f"You are a quality assurance specialist for educational visualizations with expertise in {self.topic}. "
            "Your role is to critically evaluate visualization code against specifications and educational goals, "
            "identifying and correcting any issues that would compromise instructional effectiveness. "
            "Approach your review with extreme attention to detail, checking every parameter, coordinate, and annotation "
            "against the specifications while also considering the broader educational purpose."
        )
        
        user_prompt = (
            "Review this visualization implementation against its specifications:\n\n"
            f"EDUCATIONAL CONCEPT:\n{general_description}\n\n"
            f"TECHNICAL SPECIFICATION:\n{visual_description}\n\n"
            f"IMPLEMENTATION:\n{original_code}\n\n"
            "Perform a comprehensive review checking for:\n"
            "1. ACCURACY: Mathematical correctness and accurate representation of concepts\n"
            "2. COMPLETENESS: Implementation of all specified elements and annotations\n"
            "3. CLARITY: Appropriate emphasis of key points through visual hierarchy\n"
            "4. AESTHETICS: Proper spacing, alignment, and visual balance\n"
            "5. TECHNICAL CORRECTNESS: Code structure and execution reliability\n\n"
            "If the implementation perfectly meets specifications and educational goals, return the original code unchanged. "
            "If modifications are needed, return a COMPLETE corrected version with no explanations or comments about your changes."
        )
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
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
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
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
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def visual_refinement_prompt(self, goal, general_description, code):
        system_prompt = (
            "You are a visualization artist specializing in educational graphics and data presentation. "
            "Your expertise lies in transforming technically correct visualizations into visually compelling, "
            "aesthetically pleasing educational tools through expert application of color theory, typography, "
            "spacing, and visual hierarchy principles."
            f"Your goal is to make sure that the {code} aligns with the {goal} and meets the requirements of {general_description}"
        )
        
        user_prompt = (
            f"Enhance the visual appeal and instructional effectiveness of this code:\n\n{code}\n\n"
            "Apply these specific refinements while preserving all functionality:\n"
            "1. Improve COLOR SCHEME using a harmonious palette that enhances conceptual understanding\n"
            "2. Refine TYPOGRAPHY with appropriate font sizes, weights, and styles for hierarchical clarity\n"
            "3. Optimize LAYOUT with proper spacing, alignment, and proportions\n"
            "4. Enhance ANNOTATIONS with strategic positioning and visual connections to relevant elements\n"
            "5. Add POLISH with appropriate figure size, aspect ratio, and resolution\n\n"
            "Return ONLY the refined code without explanation. Your enhancements should be subtle yet impactful, "
            "focusing on making the visualization more engaging and easier to comprehend."
        )
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]


    def visualization_judge_prompt(self, goal, general_description, code):
        system_prompt = (
            f"You are an expert in visualizations with deep knowledge in {self.topic}, data visualization best practices, "
            "and educational design principles. Your task is to objectively score the quality of a visualization based on how well it achieves "
            "its educational purpose and follows visualization best practices. Your evaluation must be rigorous, consistent, and fair from a score 0(awful) to 20(excellent)"
        )
        
        user_prompt = (
            f"YOU MUST EVALUATE THE VISUALIZATION FOLLOWING THIS RUBRIC:\n\n"
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
            
<<<<<<< Updated upstream
            "4. PEDAGOGICAL EFFECTIVENESS (0-5 points)\n"
            "   - Does the visualization facilitate understanding of the concept?\n"
            "   - Are complexity and detail appropriate for the stated student background?\n"
            "   - Does it provide insight beyond what text alone could convey?\n\n"
            "   - Does everything in'the' visualization'conveys some' information'to'the'viewer.'\n\n"
            "   - Is the visualization to the intended audience. Is the audience properly considered in terms of visual design, conveyed?"
            "   - Do the Legends should describe and explain every graphic variable type employed."
            
            "IMPORTANT: Return ONLY a single numerical score between 0-100. Do not include any explanation, "
            "comments, or other text. Just the final score as a single number."
=======
            "5. TECHNICAL IMPLEMENTATION (0-10 points)\n"
            "   - Appropriate use of visualization libraries\n"
            "   - Code is executable\n"

            "RETURN ONLY A SINGLE INTEGER SCORE BETWEEN 0-100."
            "Do not return any extreneous text, only return a number. There shouls be no text in your response."
>>>>>>> Stashed changes
        )
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    # def visualization_judge_prompt(self, goal, general_description, code):
    #     system_prompt = (
    #         f"You are a senior evaluator of educational visualizations specializing in {self.topic}. "
    #         "Your task is to objectively assess visualization quality using established principles of "
    #         "instructional design, visual perception, and domain accuracy. Your ratings must be consistent, "
    #         "calibrated against professional standards, and focused on educational effectiveness."
    #     )
        
    #     user_prompt = (
    #         f"Evaluate this visualization against its intended purpose:\n\n"
    #         f"LEARNING GOAL: {goal}\n\n"
    #         f"DESIGN SPECIFICATION: {general_description}\n\n"
    #         f"IMPLEMENTATION: {code}\n\n"
    #         "Score this visualization on a scale of 0-100 based on these weighted criteria:\n\n"
    #         "1. CONCEPTUAL PRECISION (0-20 points)\n"
    #         "   - Perfect alignment with the stated learning goal\n"
    #         "   - Accurate representation of domain knowledge\n"
    #         "   - Appropriate emphasis on key concepts\n\n"
            
    #         "2. TECHNICAL ACCURACY (0-20 points)\n"
    #         "   - Mathematical/scientific correctness\n"
    #         "   - Appropriate scales, units, and relationships\n"
    #         "   - Absence of misleading visual elements\n\n"
            
    #         "3. COGNITIVE EFFECTIVENESS (0-25 points)\n"
    #         "   - Reduction of cognitive load through clear visual organization\n"
    #         "   - Intuitive representation of complex relationships\n"
    #         "   - Strategic use of visual elements to guide attention\n\n"
            
    #         "4. INSTRUCTIONAL DESIGN (0-25 points)\n"
    #         "   - Appropriate complexity for stated student background\n"
    #         "   - Effective highlighting of crucial insights\n"
    #         "   - Support for knowledge construction and transfer\n\n"
            
    #         "5. TECHNICAL IMPLEMENTATION (0-10 points)\n"
    #         "   - Code quality, efficiency, and maintainability\n"
    #         "   - Appropriate use of visualization libraries\n"
    #         "   - Robustness and error handling\n\n"
            
    #         "RETURN ONLY A SINGLE INTEGER SCORE BETWEEN 0-100."
    #     )
        
    #     return [
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": user_prompt}
    #     ]
        
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