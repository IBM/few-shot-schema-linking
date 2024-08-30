from typing import List, Tuple
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


class VLLM:
    def __init__(self, model_name_or_path):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        # drop device_map if running on CPU
        if "deepseek" in model_name_or_path.lower():
            self.llm = LLM(
                model=model_name_or_path,
                # tensor_parallel_size=2,
                # distributed_executor_backend="mp",
                # disable_custom_all_reduce=True,
                # load_format="safetensors",
                max_model_len=16384,
            )
        else:
            self.llm = LLM(
                model=model_name_or_path,
                # tensor_parallel_size=2,
                # distributed_executor_backend="mp",
                # disable_custom_all_reduce=True,
                # load_format="safetensors",
                # max_model_len=16384,
            )
        self.model_name_or_path = model_name_or_path.lower()

    def tokenize(self, instructions, question):
        # tokenize the text
        input_tokens = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": instructions},
                {"role": "user", "content": question},
            ],
            add_generation_prompt=True,
            tokenize=False,
        )

        return input_tokens

    def few_shot_tokenize(
        self, instructions, question, demonstration_prompts: List[Tuple[str, str]]
    ):
        chat = []

        if "codestral" in self.model_name_or_path or "gemma" in self.model_name_or_path:
            # NOTE: Some models do not accept the 'system' role
            # Add demonstrations
            for i, (demo_question_prompt, demo_expected_response) in enumerate(
                demonstration_prompts
            ):
                if i == 0:
                    # The chat starts with the task instructions
                    chat.append(
                        {
                            "role": "user",
                            "content": instructions + "\n####\n" + demo_question_prompt,
                        }
                    )
                else:
                    chat.append({"role": "user", "content": demo_question_prompt})

                chat.append({"role": "assistant", "content": demo_expected_response})
        else:
            # The chat starts with the task instructions
            chat.append({"role": "system", "content": instructions})

            # Add demonstrations
            for demo_question_prompt, demo_expected_response in demonstration_prompts:
                chat.append({"role": "user", "content": demo_question_prompt})
                chat.append({"role": "assistant", "content": demo_expected_response})

        # Add the final question
        chat.append({"role": "user", "content": question})

        # tokenize the text
        input_prompt = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
        )

        return input_prompt

    def generate(self, input_prompt, num_return_sequences=1) -> str | List[str]:
        if "finetuned" in self.model_name_or_path:
            sampling_parameters = SamplingParams(
                n=num_return_sequences,
                max_tokens=1024,
                temperature=0.2,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1,
                stop="<|EOT|>",
            )
        else:
            sampling_parameters = SamplingParams(
                n=num_return_sequences,
                max_tokens=1024,
                temperature=0.2,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1,
            )

        # generate output tokens
        outputs = self.llm.generate(
            input_prompt, sampling_params=sampling_parameters, use_tqdm=False
        )
        outputs = outputs[0].outputs

        generated_texts = [output.text for output in outputs]

        if num_return_sequences == 1:
            return generated_texts[0]
        else:
            return generated_texts

    def get_tokenized_length(self, text: str) -> int:
        return len(self.tokenizer(text, return_tensors="np"))

    def get_max_length(self) -> int:
        return self.llm.llm_engine.get_model_config().max_model_len
