from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import os
import json

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
        topic = 'Gradient Descent in regards to Machine Learning'
        student_level = 'Univertity Undergraduate'
        subcategories, subcategory_to_questions = self.create_questions_by_maturity_level(topic, student_level)
        self.save_data(subcategories, subcategory_to_questions)

    def save_data(self, subcategories, subcategory_to_questions): 
        # os.makedirs("data", exist_ok=True)
        with open("data/subcategory_to_questions.json", "w") as f:
            json.dump(subcategory_to_questions, f, indent=4)
        pass

    def create_questions_by_maturity_level(self, topic, maturity_level):
        sub_categories = self.create_subcategories(topic)
        # sub_categories = self.create_subcategories('Statistics')
        topic_to_prompts = {}
        for category in sub_categories: 
            messages = [
                {"role": "system", "content": f"You are a expert professor in the topic of {topic} with a specialty in {category}. You are teaching {maturity_level} level student, therefore cater your questions towards thier general level of understanding"},
                {"role": "user", "content": f"Given the specific topic of {category}, list many prompts (you have a max of 2048 tokens using lama3_3b_instruct auto tokenizer), students would input in order to create visualizations regarding this topic. You should only output the prompts seperated by '*' so i can parse the data easily. Do not give me anything else in the output just the prompts a student would input into create visuals surrounding this topic."}
            ]
            outputs = self.pipeline(messages, max_new_tokens=2048)
            response  = outputs[0]["generated_text"][-1]['content']
            prompts = [p.strip() for p in response.split("\n") if p.strip()]
            prompts = [p.lstrip("*").strip() for p in prompts]
            # print(prompts[0])
            topic_to_prompts[category] = prompts

        # print(topic_to_prompts)

        return sub_categories, topic_to_prompts

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
