import os
import json
import matplotlib.pyplot as plt
import traceback
import re
import numpy as np
from datetime import datetime
from prompts_v3 import Prompts
from logger import Logger
from openai import OpenAI
import time 
from collections import defaultdict


# --- Configuration ---
OPENAI = True 
ANTHROPIC = False 
OPEN_SOURCE = False 

# --- Constants ---
MAX_DEBUG_ATTEMPTS = 8 

class Pipeline():
    def __init__(self, topic):
        self.output_dir = "research_results"
        self.logger = Logger()
        self.logger.info("Pipeline initialized.")
        self.load_models()
        self.prompts = Prompts(topic)
        self.metrics = {}
        self._reset_metrics() # Initialize metrics structure

    def _reset_metrics(self):
        """Resets metrics for a new run."""
        self.metrics = {
            'run_start_time_iso': datetime.now().isoformat(),
            'run_end_time_iso': None,
            'topic': None,
            'pipeline_success': False,
            'end_to_end_latency_seconds': None,
            'total_api_calls': 0,
            'api_calls_per_agent': defaultdict(int),
            'initial_code_generation_success': None, # True/False after first attempt
            'code_execution_attempts': 0,
            'debugging_failed': False, # True if MAX_ATTEMPTS reached in run_code
            'judge_feedback_loops': 0,
            'initial_goal_alignment_score': None,
            'initial_visual_clarity_score': None,
            'goal_alignment_scores': [], # Store all scores received
            'visual_clarity_scores': [], # Store all scores received
            'final_code_generated': False,
            'error_message': None # Store final error if pipeline fails
        }

    def run(self, data, topic, img_filename=None):
        """
        Executes the pipeline and returns results along with collected metrics.
        """
        self._reset_metrics() 
        start_time = time.time()
        self.metrics['topic'] = topic
        self.metrics['input_data_snippet'] = data

        simple_goal = None
        final_code = None

        try:
            self.logger.info(f"Starting pipeline run for topic: {topic}")

            # 1. Simple Query Agent
            simple_goal = self.simple_query_agent(data)
            self.logger.info(f"Simple goal generated: {simple_goal}")
            if not simple_goal: raise ValueError("Failed to generate simple goal")

            # 2. Visualization Code Generator Agent
            initial_code = self.visualization_code_generator_agent(simple_goal, self.output_dir)
            self.logger.info(f"Initial visualization code generated: {initial_code}")
            if not initial_code: raise ValueError("Failed to generate initial code")

            # 3. Run Code (includes debugging loop)
            # run_code now returns (corrected_code, attempts, success)
            executed_code, attempts, initial_success, debug_fail = self.run_code(initial_code)
            self.metrics['code_execution_attempts'] = attempts
            self.metrics['initial_code_generation_success'] = initial_success
            self.metrics['debugging_failed'] = debug_fail

            self.logger.info(f"Code after execution attempts: {executed_code}")
            if debug_fail:
                raise RuntimeError("Code execution failed after maximum debug attempts.")
            if executed_code == "NO CODE GENERATED": 
                 raise RuntimeError("run_code returned NO CODE GENERATED")


            # 4. Run Sequence of Judges (includes feedback loop)
            # run_sequence_of_judges returns (final_code, judge_metrics)
            final_code, judge_metrics = self.run_sequence_of_judges(simple_goal, executed_code) # Start judging with runnable code
            self.metrics.update(judge_metrics) # Merge judge metrics

            self.logger.info(f"Final code after all judges: {final_code}")
            if not final_code: raise ValueError("Failed to get final code from judge sequence")

            # 5. Run Final Code & Save
            self.run_final_code(final_code, img_filename)
            self.metrics['final_code_generated'] = True
            self.logger.info("Completed Pipeline ✅")
            self.metrics['pipeline_success'] = True # Mark as successful

        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            self.metrics['error_message'] = traceback.format_exc() # Store stack trace
            traceback.print_exc()
            self.metrics['pipeline_success'] = False # Mark as failed

        finally:
            end_time = time.time()
            self.metrics['run_end_time_iso'] = datetime.now().isoformat()
            self.metrics['end_to_end_latency_seconds'] = round(end_time - start_time, 2)
            # Return results along with the collected metrics
            return simple_goal, final_code, self.metrics

    def load_models(self):
        self.logger.info("Loading models...")
        if OPENAI:
            try:
                self.openai_client = OpenAI()
                if not os.environ.get("OPENAI_API_KEY"):
                    self.logger.warning("OPENAI_API_KEY environment variable not set.")
                self.logger.info("OpenAI client potentially loaded (API key check at runtime).")
            except Exception as e:
                 self.logger.error(f"Failed to initialize OpenAI client: {e}")
        else:
            self.logger.warning("OPENAI is not enabled in configuration.")

    def run_inference(self, agent_name, prompt, model, max_tokens):
        self.logger.info(f"Running inference via agent '{agent_name}' with model: {model}")

        # --- Metric Tracking ---
        self.metrics['total_api_calls'] += 1
        self.metrics['api_calls_per_agent'][agent_name] += 1
        # --- End Metric Tracking ---

        if OPENAI:
            try:
                system_prompt, user_prompt = prompt
                model_instructions, model_input = prompt
                response = self.openai_client.responses.create(
                    model=model,
                    instructions=model_instructions,
                    input=model_input,
                    # max_tokens=max_tokens
                    )
                result_text = response.output[0].content[0].text
                self.logger.info(f"Inference successful for agent {agent_name}.")
                return result_text
            except Exception as e:
                self.logger.error(f"Error during OpenAI API call for agent {agent_name}: {e}")
                self.metrics['error_message'] = f"API call failed for {agent_name}: {str(e)}" 
                return None 
        else:
            self.logger.warning(f"OPENAI is not enabled or client not loaded, cannot run inference for {agent_name}.")
            return None

    def run_code(self, initial_code):
        """
        Tries to execute code, applying debugging agents if errors occur.
        Returns: (executed_code, attempts, initial_success, debug_failed)
        """
        self.logger.info("Executing Code with Debug Loop")
        code_to_run = initial_code
        attempts = 0
        initial_success = None
        debug_failed = False

        while attempts < MAX_DEBUG_ATTEMPTS:
            attempts += 1
            self.logger.info(f"Execution Attempt: {attempts}")
            try:
                cleaned_code = code_to_run.strip().replace('```python', '').replace('```', '').strip()
                if not cleaned_code: # Handle empty code generation
                    raise ValueError("Generated code is empty after cleaning.")

                self.logger.info(f"Attempting to execute cleaned code (attempt {attempts}):\n{cleaned_code}")
                local_vars = {}
                exec(cleaned_code, globals(), local_vars)

                self.logger.info(f"Code executed successfully on attempt {attempts}")
                if attempts == 1:
                    initial_success = True
                else:
                     if initial_success is None: initial_success = False # Mark initial as failed if success is on later attempt
                return cleaned_code, attempts, initial_success, debug_failed 

            except Exception as e:
                error_str = str(e)
                error_trace = traceback.format_exc()
                self.logger.warning(f"Error on attempt {attempts}: {error_str}\nTrace:\n{error_trace}")

                if initial_success is None: initial_success = False # Mark initial as failed if error on first attempt

                if attempts >= MAX_DEBUG_ATTEMPTS:
                    self.logger.error("Maximum debugging attempts reached. Failed to execute code.")
                    debug_failed = True
                    return code_to_run, attempts, initial_success, debug_failed

                # --- Debugging Agent Sequence ---
                try:
                    self.logger.info("Attempting automated error correction...")
                    error_explanation = self.code_error_identifier_agent(code_to_run, error_trace)
                    if not error_explanation:
                        self.logger.warning("Code Error Identifier Agent failed to provide explanation.")
                        continue

                    self.logger.info(f"Error Explanation: {error_explanation}")
                    corrected_code_suggestion = self.code_error_correction_agent(code_to_run, error_trace, error_explanation)

                    if not corrected_code_suggestion:
                        self.logger.warning("Code Error Correction Agent failed to provide suggestion.")
                        continue

                    self.logger.info("Received corrected code suggestion. Will use for next attempt.")
                    code_to_run = corrected_code_suggestion 

                except Exception as correction_error:
                    self.logger.error(f"Error during code correction agent sequence: {correction_error}")
                    continue

        self.logger.error("Exited run_code loop unexpectedly.")
        debug_failed = True
        return "NO CODE GENERATED", attempts, initial_success, debug_failed


    def run_final_code(self, code, img_filename):
        self.logger.info("Running final code and saving visualization.")
        if not code or code == "NO CODE GENERATED":
            self.logger.error("Cannot run final code: Code is empty or marked as failed.")
            raise ValueError("Invalid code provided to run_final_code")

        cleaned_code = code.strip().replace('```python', '').replace('```', '').strip()

        if img_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_filename = os.path.join(self.output_dir, f"visualization_{timestamp}.png")
        else:
            img_dir = os.path.dirname(img_filename)
            if img_dir: 
                os.makedirs(img_dir, exist_ok=True)

        os.makedirs(self.output_dir, exist_ok=True)
        absolute_img_filename = os.path.abspath(img_filename)
        save_command = f"\nimport matplotlib.pyplot as plt\nplt.savefig(r\"{absolute_img_filename}\")\nplt.close() # Close plot to free memory"
        final_script = cleaned_code + save_command

        self.logger.info(f"Executing final script to save to: {absolute_img_filename}")
        try:
            local_vars = {}
            exec(final_script, globals(), local_vars)
            self.logger.info(f"Final visualization saved successfully to {absolute_img_filename}")
        except Exception as e:
            self.logger.error(f"Failed to execute final code or save visualization: {e}")
            self.logger.error(f"Problematic final script snippet:\n{final_script[:500]}...") 
            raise RuntimeError(f"Final code execution failed: {e}") from e


    def parse_judge_response(self, response):
        """Parses judge response expected as '[1-5]\nFeedback...'"""
        if not response:
            self.logger.warning("Received empty response from judge.")
            return 0, "No response received from judge." # Default to lowest score

        lines = response.strip().split('\n', 1)
        score_str = lines[0].strip()
        feedback = lines[1].strip() if len(lines) > 1 else "No feedback provided."

        try:
            score = int(re.match(r"\[?(\d)\]?", score_str).group(1))
            if not 1 <= score <= 5:
                raise ValueError("Score out of range")
        except (AttributeError, ValueError, IndexError):
            self.logger.warning(f"Could not parse score from judge response line: '{score_str}'. Defaulting to 1.")
            score = 1 

        self.logger.info(f"Parsed Judge Score: {score}, Feedback: {feedback[:100]}...")
        return score, feedback

    def run_sequence_of_judges(self, goal, initial_runnable_code):
        """
        Runs code through judges, triggers regeneration if needed.
        Returns: (final_code, judge_metrics)
        """
        self.logger.info("Executing Sequence of Judges")
        current_code = initial_runnable_code
        max_judge_loops = 5 
        judge_loop_count = 0

        # --- Metrics for this sequence ---
        judge_metrics = {
            'judge_feedback_loops': 0,
            'initial_goal_alignment_score': None,
            'initial_visual_clarity_score': None,
            'goal_alignment_scores': [],
            'visual_clarity_scores': []
        }

        while judge_loop_count < max_judge_loops:
            judge_loop_count += 1
            self.logger.info(f"Judge Sequence Loop: {judge_loop_count}")

            # --- Goal Alignment Judge ---
            self.logger.info("Running Goal Alignment Judge...")
            goal_response = self.goal_alignment_judge_agent(goal, current_code)

            goal_score, goal_feedback = self.parse_judge_response(goal_response)
            judge_metrics['goal_alignment_scores'].append(goal_score)
            if judge_metrics['initial_goal_alignment_score'] is None:
                 judge_metrics['initial_goal_alignment_score'] = goal_score

            if goal_score < 3:
                self.logger.info(f"Goal Alignment Judge failed (Score: {goal_score}). Regenerating code from feedback...")
                judge_metrics['judge_feedback_loops'] += 1
                regenerated_code = self.code_generator_from_judge_feedback_agent(current_code, goal_feedback)
                if not regenerated_code:
                    self.logger.warning("Failed to regenerate code from goal feedback. Re-using previous code.")
                    raise RuntimeError("Failed to regenerate code from Goal Alignment feedback.")
                
                self.logger.info("Checking executability of regenerated code (from goal feedback)...")
                new_code, attempts, _, debug_fail = self.run_code(regenerated_code)
                self.metrics['code_execution_attempts'] += attempts 

                if debug_fail or new_code == "NO CODE GENERATED":
                     self.logger.error("Regenerated code (from goal feedback) failed execution checks.")
                     raise RuntimeError("Regenerated code (from goal feedback) failed execution.")
                current_code = new_code 
                continue

            self.logger.info("Passed Goal Alignment Judge ✅")

            # --- Visual Clarity Judge ---
            self.logger.info("Running Visual Clarity Judge...")
            clarity_response = self.visual_clarity_judge_agent(current_code)
            clarity_score, clarity_feedback = self.parse_judge_response(clarity_response)
            judge_metrics['visual_clarity_scores'].append(clarity_score)
            if judge_metrics['initial_visual_clarity_score'] is None:
                judge_metrics['initial_visual_clarity_score'] = clarity_score

            if clarity_score < 3:
                self.logger.info(f"Visual Clarity Judge failed (Score: {clarity_score}). Regenerating code from feedback...")
                judge_metrics['judge_feedback_loops'] += 1
                regenerated_code = self.execute_agent(
                    agent_name='code_generator_from_judge_feedback_agent',
                    pipeline='code_generation_model',
                    max_new_tokens=1024,
                    prompt_method=self.prompts.code_generator_from_judge_feedback_prompt,
                    args=[current_code, f"Visual Clarity Feedback (Score {clarity_score}): {clarity_feedback}"]
                )
                if not regenerated_code:
                    self.logger.warning("Failed to regenerate code from clarity feedback. Re-using previous code.")
                    raise RuntimeError("Failed to regenerate code from Visual Clarity feedback.")

                self.logger.info("Checking executability of regenerated code (from clarity feedback)...")
                new_code, attempts, _, debug_fail = self.run_code(regenerated_code)
                self.metrics['code_execution_attempts'] += attempts # Add attempts to overall count

                if debug_fail or new_code == "NO CODE GENERATED":
                     self.logger.error("Regenerated code (from clarity feedback) failed execution checks.")
                     raise RuntimeError("Regenerated code (from clarity feedback) failed execution.")

                current_code = new_code
                continue 

            self.logger.info("Passed Visual Clarity Judge ✅")

            # --- Both judges passed ---
            self.logger.info("All judges passed. Finalizing code.")
            return current_code, judge_metrics 

        # --- Max loops reached ---
        self.logger.error(f"Judge sequence failed to converge after {max_judge_loops} loops.")
        return current_code, judge_metrics 


    def execute_agent(self, agent_name, pipeline, max_new_tokens, prompt_method, args):
        """Executes a specific agent, handling inference and logging."""
        self.logger.info(f"Executing Agent: {agent_name}")
        messages = prompt_method(*args)
        model = 'gpt-4o-mini' if OPENAI else 'dummy-model'
        response = self.run_inference(agent_name, messages, model, max_tokens=max_new_tokens)
        if response is None:
            self.logger.warning(f"Agent {agent_name} failed to get response from inference.")
            return None
        
        if not response.strip():
             self.logger.warning(f"Agent {agent_name} returned an empty response.")
             return None 
        
        self.logger.info(f"Agent {agent_name} response received.")
        return response

    def simple_query_agent(self, data):
        return self.execute_agent('simple_query_agent', 'base_model', 512, self.prompts.simple_query_prompt, [data])

    def visualization_code_generator_agent(self, simple_goal, output_dir):
        return self.execute_agent('visualization_code_generator_agent', 'code_generation_model', 1024, self.prompts.visualization_code_generator_prompt, [simple_goal, output_dir])

    def code_error_identifier_agent(self, code, error_message):
         # Increase token limit for potentially long tracebacks
        return self.execute_agent('code_error_identifier_agent', 'base_model', 768, self.prompts.code_error_identifier_prompt, [code, error_message])

    def code_error_correction_agent(self, original_code, error_message, explanation):
         # Increase token limit to allow generating full corrected code
        return self.execute_agent('code_error_correction_agent', 'code_generation_model', 1536, self.prompts.code_error_correction_prompt, [original_code, error_message, explanation])

    def goal_alignment_judge_agent(self, goal, code):
        # Judges likely need fewer tokens
        return self.execute_agent('goal_alignment_judge_agent', 'base_model', 128, self.prompts.goal_alignment_judge_prompt, [goal, code])

    def visual_clarity_judge_agent(self, code):
        return self.execute_agent('visual_clarity_judge_agent', 'base_model', 128, self.prompts.visual_clarity_judge_prompt, [code])

    def code_generator_from_judge_feedback_agent(self, code, feedback):
        return self.execute_agent('code_generator_from_judge_feedback_agent', 'code_generation_model', 1536, self.prompts.code_generator_from_judge_feedback_prompt, [code, feedback])


# --- Example Usage ---
if __name__ == "__main__":
    # Example input data and topic
    math_topic = "Gradient Descent Optimization"
    input_text = "Gradient Descent Algorithm Gradient Descent Algorithm iteratively calculates the next point using gradient at the current position, scales it (by a learning rate) and subtracts obtained value from the current position (makes a step). It subtracts the value because we want to minimise the function (to maximise it would be adding). This process can be written as:p_{n+1} = p_n - η * ∇f(p_n) There’s an important parameter η which scales the gradient and thus controls the step size. In machine learning, it is called learning rate and have a strong influence on performance. The smaller learning rate the longer GD converges, or may reach maximum iteration before reaching the optimum point If learning rate is too big the algorithm may not converge to the optimal point (jump around) or even to diverge completely. In summary, Gradient Descent method’s steps are: 1-choose a starting point (initialisation), 2-calculate gradient at this point, 3-make a scaled step in the opposite direction to the gradient (objective: minimise), 4-repeat points 2 and 3 until one of the criteria is met: maximum number of iterations reached step size is smaller than the tolerance (due to scaling or a small gradient)."

    # --- Create Pipeline Instance ---
    pipeline_instance = Pipeline(topic=math_topic)

    # --- Run the Pipeline ---
    # Pass data to the run method
    simple_goal, final_code, run_metrics = pipeline_instance.run(data=input_text, topic=math_topic)

    # --- Process Results ---
    print("\n" + "="*30 + " PIPELINE EXECUTION FINISHED " + "="*30)

    if run_metrics['pipeline_success']:
        print("Pipeline Run: SUCCESS")
        print(f"Simple Goal Generated:\n{simple_goal}")
        print(f"\nFinal Code Generated:\n{final_code}")
        # Image saved to self.output_dir
        print(f"\nVisualization potentially saved in '{pipeline_instance.output_dir}' directory.")
    else:
        print("Pipeline Run: FAILED")
        print(f"Error Message:\n{run_metrics.get('error_message', 'No specific error message captured.')}")

    print("\n" + "="*30 + " COLLECTED METRICS " + "="*30)
    # Pretty print the metrics dictionary
    for key, value in run_metrics.items():
        if key == 'api_calls_per_agent':
            print(f"  {key}:")
            for agent, count in value.items():
                print(f"    - {agent}: {count}")
        else:
            print(f"  {key}: {value}")

    # Example of how to analyze metrics across multiple runs (pseudo-code)
    # all_run_metrics = []
    # for input_data_item in my_dataset:
    #     _, _, metrics = pipeline_instance.run(data=input_data_item['text'], topic=input_data_item['topic'])
    #     all_run_metrics.append(metrics)
    #
    # # Now analyze all_run_metrics list:
    # total_runs = len(all_run_metrics)
    # success_rate = sum(m['pipeline_success'] for m in all_run_metrics) / total_runs * 100
    # avg_latency = np.mean([m['end_to_end_latency_seconds'] for m in all_run_metrics if m['pipeline_success']])
    # avg_debug_attempts = np.mean([m['code_execution_attempts'] for m in all_run_metrics if m['pipeline_success']])
    # print(f"\nOverall Success Rate: {success_rate:.2f}%")
    # print(f"Average Latency (successful runs): {avg_latency:.2f} seconds")
    # print(f"Average Code Execution Attempts (successful runs): {avg_debug_attempts:.2f}")
    # # ... calculate other aggregate stats ...