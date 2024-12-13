import os
import threading
from transformers import pipeline
from typing import List
from math import ceil
# Import vllm if using vLLM backend

# constants
BRANCH_DELIMITER = "###"
BRANCHING_PROMPT_TEMPLATE = (
    "Given the following question and incomplete answer, "
    "provide {num_candidates} different options for the immediate next step following the given prompt and "
    f'separate them with delimiter "{BRANCH_DELIMITER}".\n'
    f"{BRANCH_DELIMITER}\n{{prompt}}\n{BRANCH_DELIMITER}\n"
)

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("vLLM not installed. Install it if you wish to use it as a model backend.")

# Set the environment variable for the endpoint
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class LLMService:
    """
    A class to manage a large language model (LLM) service using Hugging Face's transformers library.
    """

    def __init__(self, model_name: str = "/root/.cache/modelscope/hub/Qwen/Qwen2___5-Math-7B-Instruct",
                 device: str = "cuda", max_new_tokens: int = 2048,
                 temperature: float = 0.7, top_k: int = 30, top_p: float = 0.9, model_type: str="hf"):
        """
        Initialize the LLMService with model parameters and sampling settings.

        Parameters:
        - model_name (str): Path or Hugging Face hub model name.
        - device (str): Device for computation, e.g., 'cuda' or 'cpu'.
        - max_new_tokens (int): Maximum number of new tokens to generate.
        - temperature (float): Sampling temperature for response generation.
        - top_k (int): Top-K sampling parameter for response diversity.
        - top_p (float): Top-P sampling parameter for response diversity.
        """
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.model_type = model_type.lower()
        self.pipe = None
        self.llm = None
        self.load_lock = threading.Lock()

    def start_service(self):
        """
        Start the LLM service by loading the model into the chosen pipeline if it's not already loaded.
        Ensures thread-safe loading using a lock.
        """
        with self.load_lock:
            if self.model_type == "hf":
                if self.pipe is None:
                    print(f"Loading Hugging Face model '{self.model_name}' on device '{self.device}'...")
                    self.pipe = pipeline(
                        "text-generation",
                        model=self.model_name,
                        torch_dtype="auto",
                        device_map=self.device
                    )
                    print("Hugging Face model loaded successfully.")
            elif self.model_type == "vllm":
                if self.llm is None:
                    print(f"Loading vLLM model '{self.model_name}' on device '{self.device}'...")
                    self.llm = LLM(self.model_name, tensor_parallel_size=1)
                    print("vLLM model loaded successfully.")
            else:
                raise ValueError("Unsupported model_type. Choose 'hf' for Hugging Face or 'vllm' for vLLM.")

    def generate_response(self, prompt: str, num_copies: int = 2) -> List[str]:
        """
        Generate responses from the model based on the provided prompt, duplicated to form a batch.

        Parameters:
        - prompt (str): The input prompt to generate responses for.
        - num_copies (int): The number of copies of the prompt to create for batch processing (default is 16).

        Returns:
        - List[str]: A list of generated responses, each corresponding to a duplicate of the input prompt.
        """
        if self.model_type == "hf":
            return self._generate_response_hf(prompt, num_copies)
        elif self.model_type == "vllm":
            return self._generate_response_vllm(prompt, num_copies)
        else:
            raise ValueError("Unsupported model_type. Choose 'hf' for Hugging Face or 'vllm'.")

    def _generate_response_hf(self, prompt: str, num_copies: int = 2, single_string_branches: int = 1, outer_batch_size: int = 1) -> List[str]:
        """
        Generate responses from the model based on the provided prompt, duplicated to form a batch.

        Parameters:
        - prompt (str): The input prompt to generate responses for.
        - num_copies (int): The number of copies of the prompt to create for batch processing (default is 16).

        Returns:
        - List[str]: A list of generated responses, each corresponding to a duplicate of the input prompt.
        """
        if self.pipe is None:
            raise ValueError("LLM service not started. Please call start_service() first.")
        
        collated_batch = []

        # batching setup
        # - we need a total of `n_copies` candidates
        # - we generate `single_string_branches` candidates in a single generated string
        # - hf generation pipeline allows for [n strings in/n strings out], let batch_size = n
        # ceil(n_copies / single_string_branches) = number of individual strings needed 
        # ceil(strings needed / batch_size) = number of LLM calls

        # total = n
        # ssb = k
        # max parallel = m
        # total parallel = ceil(n / k)
        # total calls = ceil(total parallel / max parallel)


        if single_string_branches == 1:
            # Create a batch of the same prompt
            prompts = [prompt] * num_copies

            # Generate responses from the model
            responses = self.pipe(
                prompts,
                max_new_tokens=self.max_new_tokens,
                batch_size=num_copies,
                do_sample=True,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                return_full_text=False
            )
            response_message_batch = [result[0]["generated_text"] for result in responses]
            
            # Extract and return the generated text for each response
            return response_message_batch
        
        response_messages = []
        num_llm_output_strings = ceil(num_copies / single_string_branches)
        for outer_batch_idx in range(0, num_llm_output_strings, outer_batch_size):
            curr_outer_batch_size = min(outer_batch_size, num_llm_output_strings - outer_batch_idx)
            branching_prompts = [BRANCHING_PROMPT_TEMPLATE.format(
                num_candidates=single_string_branches,
                prompt=prompt
            )] * (curr_outer_batch_size
            if (curr_outer_batch_size * single_string_branches) + len(response_messages) > num_copies:

        

    def _generate_response_vllm(self, prompt: str, num_copies: int) -> List[str]:
        """
        Generate responses using vLLM.
        """
        if self.llm is None:
            raise ValueError("LLM service not started for vLLM model. Please call start_service() first.")

        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens
        )

        prompts = [prompt] * num_copies
        responses = self.llm.generate(prompts, sampling_params=sampling_params)

        return [response.outputs[0].text for response in responses]


if __name__ == "__main__":
    # Initialize the service for vLLM
    llm_service = LLMService(model_type="vllm")
    llm_service.start_service()

    prompt = "What is game theory?"
    responses = llm_service.generate_response(prompt, num_copies=3)

    print(responses)

