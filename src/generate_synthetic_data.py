import os
import json
import traceback
from datetime import datetime
from openai import OpenAI

OPENAI = True

class SyntheticDataGenerator:
    def __init__(self, num_generations, verbose=False):
        # self.original_data_point = original_data_point
        self.num_generations = num_generations
        self.verbose = verbose
        self.output_file = self._create_output_file()

        self._load_model()
        # self._generate_data()

    def _create_output_file(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/synthetic_data_{timestamp}.jsonl"
        return filename

    def _load_model(self):
        if OPENAI:
            self.openai_client = OpenAI()
            self.openai_client.api_key = os.environ.get("OPENAI_API_KEY")

    # def _generate_data(self):
    #     model = "gpt-4o-mini"
    #     system_prompt = "You are an expert in creating synthetic data."
    #     user_prompt = (
    #         f"Given this data point: {self.original_data_point}\n"
    #         "Create a different version of this data point so that I can expand my dataset with varied examples. Your answer should be only the paraphrased text without any additional explanation or headers. The paraphrased version must retain all the key information from the paragraph."
    #     )

    #     if self.verbose:
    #         print(f"Starting data generation for {self.num_generations} iterations...")

    #     with open(self.output_file, 'w') as f:
    #         # Write original data point at the top
    #         header = {
    #             "original_data_point": self.original_data_point,
    #             "generation_time": datetime.now().isoformat()
    #         }
    #         f.write(json.dumps(header) + "\n")

    #         # Generate synthetic examples
    #         for i in range(self.num_generations):
    #             try:
    #                 response = self.openai_client.chat.completions.create(
    #                     model=model,
    #                     messages=[
    #                         {"role": "system", "content": system_prompt},
    #                         {"role": "user", "content": user_prompt}
    #                     ],
    #                     temperature=1.5,
    #                 )
    #                 output_text = response.choices[0].message.content.strip()
    #                 json.dump({"index": i, "synthetic_data": output_text}, f)
    #                 f.write('\n')

    #                 if self.verbose:
    #                     print(f"[✓] Generation {i + 1} completed.")
    #             except Exception as e:
    #                 if self.verbose:
    #                     print(f"[!] Generation {i + 1} failed: {str(e)}")
    #                 json.dump({"index": i, "synthetic_data": "COULD NOT GENERATE", "error": str(e)}, f)
    #                 f.write('\n')

    #     if self.verbose:
    #         print(f"\nAll data saved to: {self.output_file}")
            
    def _generate_data2(self, original_data_point) -> list[str]:
        model = "gpt-4o-mini"
        system_prompt = "You are an expert in creating synthetic data."
        user_prompt = (
            f"Given this data point: {original_data_point}\n"
            "Create a different version of this data point so that I can expand my dataset with varied examples."
            "Your task is to generate a paraphrased version the data point (paragraph) in a way that must preserve the core educational content and key visualizable concepts"
            "The goal is to create diverse yet faithful variations that can support synthetic data generation for educational visualizations."
            "Your answer should be only the paraphrased text without any additional explanation or headers. "
        )

        if self.verbose:
            print(f"Starting data generation for {self.num_generations} iterations...")

        results = []
        for i in range(self.num_generations):
            try:
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt}
                    ],
                    temperature=1.5,
                )
                text = response.choices[0].message.content.strip()
                results.append(text)

                if self.verbose:
                    print(f"[✓] Generation {i + 1} completed.")
            except Exception as e:
                if self.verbose:
                    print(f"[!] Generation {i + 1} failed: {e!r}")
                # append a placeholder or re‑raise, as you prefer
                results.append(None)

        if self.verbose:
            print("All generations done.")

        return results

