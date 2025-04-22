import pandas as pd
import numpy as np
from openai import OpenAI, APIError, AuthenticationError 
import os
import json 
import logger
from logger import Logger

OPENAI_ENABLED = True
DEFAULT_OPENAI_MODEL = "gpt-4o-mini" #
DEFAULT_TEMPERATURE = 0.2 

class RecommendationClient():
    """
    Client to get recommendations on whether a text chunk
    would benefit from a visualization, using an LLM agent.
    """

    def __init__(self, model_name=DEFAULT_OPENAI_MODEL, temperature=DEFAULT_TEMPERATURE):
        """
        Initializes the RecommendationClient.
        """
        self.logger = Logger()
        self.openai_enabled = OPENAI_ENABLED
        self.openai_client = None
        self.model_name = model_name
        self.temperature = temperature

        # Attempt to load models immediately upon initialization
        self.load_models()

    def load_models(self):
        """
        Loads and initializes the required models (currently OpenAI client).
        """
        self.logger.info("Attempting to load models...")
        if self.openai_enabled:
            try:
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    self.logger.error("OPENAI_API_KEY environment variable not set.")
                    self.openai_enabled = False
                    self.logger.warning("OpenAI functionality disabled due to missing API key.")
                    return

                self.openai_client = OpenAI(api_key=api_key)
                self.logger.info(f"OpenAI client loaded successfully for model: {self.model_name}")
            except Exception as e:
                self.logger.error(f"An unexpected error occurred loading OpenAI client: {e}")
                self.openai_enabled = False

    def _build_prompt(self, text_chunk):
        """
        Constructs the prompt for the OpenAI API call.
        """
        system_prompt = """
            You are an expert analyst evaluating text to determine if adding a visualization would significantly improve its clarity, impact, or reader understanding. Your goal is to assess the 'visualizability' of the text. My goal is to build mathematical visualizations. 

            Provide a score from 0.0 to 1.0 indicating how much the text would benefit from a visualization (0.0 = no benefit, 1.0 = high benefit). Also provide a brief justification for your score based on the criteria.

            Respond ONLY with a JSON object containing two keys:
            1.  "score": A float between 0.0 and 1.0 (e.g., 0.75).
            2.  "reasoning": A brief string (1-2 sentences) explaining your score based on the criteria mentioned above.

            Example response format:
            {
            "score": 0.8,
            "reasoning": "The text contains multiple numerical comparisons over time and lists several key statistics, making it highly suitable for a line chart or summary table."
            }
        """

        user_prompt = f"""
            Please analyze the following text chunk:

            --- TEXT START ---
            {text_chunk}
            --- TEXT END ---

            Provide your analysis in the specified JSON format.
            """
        return system_prompt, user_prompt

    def get_recommendation(self, text_chunk):
        system_prompt, user_prompt = self._build_prompt(text_chunk)

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=200,
                response_format={"type": "json_object"} 
            )

            raw_response_content = response.choices[0].message.content
            self.logger.info(f"Raw OpenAI response: {raw_response_content}")

            # Parse the JSON response
            try:
                result = json.loads(raw_response_content)
                # Basic validation of the parsed result
                if isinstance(result, dict) and 'score' in result and 'reasoning' in result and isinstance(result['score'], (float, int)):
                    result['score'] = max(0.0, min(1.0, float(result['score'])))
                    self.logger.info(f"Recommendation received: Score={result['score']:.2f}, Reasoning='{result['reasoning']}'")
                    return result
                else:
                    self.logger.error(f"Parsed JSON does not match expected format: {result}")
                    return None
            except json.JSONDecodeError:
                self.logger.error(f"Failed to parse JSON response from OpenAI: {raw_response_content}")
                return None
            except Exception as parse_err:
                self.logger.error(f"Error processing LLM response content: {parse_err}")
                return None

        except Exception as e:
            self.logger.error(f"An unexpected error occurred during recommendation: {e}")
            return None

if __name__ == "__main__":

    INPUT_JSON_PATH = "/Users/andrewchoy/Desktop/CS Projects/Visard/data/data100/chunked_feature_engineering.json"
    OUTPUT_JSON_PATH = "recommendations_output_poc.json"
    print("Initializing Recommendation Client (Simplified PoC)...")
    client = RecommendationClient()

    results = {}

    print(f"\nLoading input data from: {INPUT_JSON_PATH}")
    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        content = json.load(f)
    print(f"Successfully loaded {len(content)} text chunks.")
    print("\nStarting recommendation process...")
    for i, (key, value) in enumerate(content.items()):
        print(f"--- Processing item {i+1}/{len(content)}: Key='{key}' ---")
        if isinstance(value, str) and value.strip():
            recommendation = client.get_recommendation(value)
            score = recommendation['score']
            justification = recommendation['reasoning']
            
            results[key] = {"original_text": value, "score": score, "reasoning":justification}
        else:
            print(f"Skipping Key='{key}': Value is not valid text.")
            results[key] = {"score": 0.0, "reasoning": "Input was not valid text."} 

    print("\nRecommendation process finished.")
    print(f"Writing results to: {OUTPUT_JSON_PATH}")
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as outfile:
        json.dump(results, outfile, indent=4, ensure_ascii=False)
    print(f"\nSuccessfully wrote {len(results)} recommendations to {OUTPUT_JSON_PATH}")




