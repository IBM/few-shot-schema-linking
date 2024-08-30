from typing import List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from dec_sl.llm.llm_base import LLM


class Mixtral(LLM):
    def __init__(self, model_name_or_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # drop device_map if running on CPU
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            torch_dtype=torch.float16,
            use_flash_attention_2=True,
        )

    def tokenize(self, instructions, question):
        # tokenize the text
        # NOTE: For the Mixtral models only user and assistant roles are supported!
        # NOTE: Also, conversation roles must alternate user/assistant/user/assistant/...
        input_tokens = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": instructions + "\n####\n" + question},
            ],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        return input_tokens

    def few_shot_tokenize(
        self, instructions, question, demonstration_prompts: List[Tuple[str, str]]
    ):
        # The chat starts with the task instructions
        chat = []

        # Add demonstrations
        for i, (demo_question_prompt, demo_expected_response) in enumerate(
            demonstration_prompts
        ):
            if i == 0:
                chat.append(
                    {
                        "role": "user",
                        "content": instructions + "\n####\n" + demo_question_prompt,
                    }
                )
            else:
                chat.append({"role": "user", "content": demo_question_prompt})

            chat.append({"role": "assistant", "content": demo_expected_response})

        # Add the final question
        chat.append({"role": "user", "content": question})

        # tokenize the text
        input_tokens = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        return input_tokens

    def generate(self, input_tokens):
        # generate output tokens
        output = self.model.generate(
            input_tokens, max_new_tokens=1024, pad_token_id=self.tokenizer.eos_token_id
        )
        # Calculate prompt length
        prompt_length = input_tokens.shape[-1]

        # Remove the prompt from the output
        output = output[0][prompt_length:]

        # decode output tokens into text
        output = self.tokenizer.batch_decode([output], skip_special_tokens=True)
        prediction = output[0]

        return prediction
