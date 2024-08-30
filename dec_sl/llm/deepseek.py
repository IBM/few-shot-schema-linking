from typing import List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from dec_sl.llm.llm_base import LLM


class DeepseekCoder(LLM):
    def __init__(self, model_name_or_path):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        # drop device_map if running on CPU
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

    def tokenize(self, instructions, question):
        # tokenize the text
        input_tokens = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": instructions},
                {"role": "user", "content": question},
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
        chat.append({"role": "system", "content": instructions})

        # Add demonstrations
        for demo_question_prompt, demo_expected_response in demonstration_prompts:
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

    def generate(self, input_tokens, num_return_sequences=1) -> str | List[str]:
        # generate output tokens
        output = self.model.generate(
            input_tokens,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.2,
            repetition_penalty=1,
            top_k=50,
            top_p=0.95,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.convert_tokens_to_ids("<pad>"),
            num_return_sequences=num_return_sequences,
        )

        # Calculate prompt length
        prompt_length = input_tokens.shape[-1]
        # len(input_tokens[0])

        # Remove the prompt from the output
        output = output[:, prompt_length:]

        # decode output tokens into text
        output = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        if num_return_sequences == 1:
            return output[0]
        else:
            return output
