from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

class PromptGenerator:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained("llama3_3b_instruct")
        self.tokenizer = AutoTokenizer.from_pretrained("llama3_3b_instruct")
        self.pipeline = transformers.pipeline(
                            "text-generation",
                            model="meta-llama/Llama-3.2-3B-Instruct",
                            model_kwargs={"torch_dtype": torch.bfloat16},
                        )
        # self.run()
        topic = 'statistics'
        student_level = 'high school'
        created_questions = self.create_questions_by_maturity_level(topic, student_level)
        self.create_structure_from_questions(topic, created_questions)


    def create_structure_from_questions(self, topic, category_to_question):
        message_list = []
        for category, questions in category_to_question.items():
            for question in questions:
                messages = [
                    {
                        "role": "system",
                        "content": (
                            f"You are an expert professor in the topic of {topic}, specializing in {category}. "
                            "A student is seeking guidance on a specific question related to this topic and aims to create a visualization. "
                            "Your task is to assist them by structuring the necessary information in a JSON format."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"A student presents the following question: '{question}'. "
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
                            "Ensure that the response is well-structured, concise, and clear to assist the student effectively."
                        ),
                    }
                ]
                outputs = self.pipeline(messages, max_new_tokens=512)
                response  = outputs[0]["generated_text"][-1]['content']
                print(response)
                message_list.append(messages)
                break;
        # print(message_list)
        return message_list


    def create_questions_by_maturity_level(self, topic, maturity_level):
        sub_categories = self.create_subcategories(topic)
        # sub_categories = self.create_subcategories('Statistics')
        topic_to_prompts = {}
        for category in sub_categories: 
            messages = [
                {"role": "system", "content": f"You are a expert professor in the topic of {topic} with a specialty in {category}. You are teaching {maturity_level} level student, therefore cater your questions towards thier general level of understanding"},
                {"role": "user", "content": f"Given the specific topic of {category}, list 10 prompts, students would input in order to create visualizations regarding this topic. You should only output the prompts seperated by '*' so i can parse the data easily. Do not give me anything else in the output just the prompts a student would input into create visuals surrounding this topic."}
            ]
            outputs = self.pipeline(messages, max_new_tokens=512)
            response  = outputs[0]["generated_text"][-1]['content']
            prompts = [p.strip() for p in response.split("\n") if p.strip()]
            prompts = [p.lstrip("*").strip() for p in prompts]
            # print(prompts[0])
            topic_to_prompts[category] = prompts

        print(topic_to_prompts)

        return topic_to_prompts

    def create_subcategories(self, topic):
        messages = [
            {"role": "system", "content": f"You are a expert professor in topic of {topic}!"},
            {"role": "user", "content": "Give me the most important subcategories within this topic that students must know. The output should be in a list seperated by commas. For example the output should look like this topic1, topic2, ..., topicN. Do not give me anything else in the output just the list of topics in my desired format"},
        ]
        outputs = self.pipeline(
                        messages,
                        max_new_tokens=256)
        response  = outputs[0]["generated_text"][-1]['content']
        # print(response)
        topics = set(response.split(','))
        return topics
    
PromptGenerator()
