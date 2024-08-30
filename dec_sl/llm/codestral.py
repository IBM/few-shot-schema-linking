from typing import List, Tuple

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage, AssistantMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
import torch
from transformers import AutoModelForCausalLM

from dec_sl.llm.llm_base import LLM


class Codestral(LLM):
    def __init__(self, model_name_or_path):
        self.tokenizer = MistralTokenizer.v3()
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

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
        completion_request = ChatCompletionRequest(
            messages=[UserMessage(content=instructions + "\n####\n" + question)]
        )

        input_tokens = self.tokenizer.encode_chat_completion(completion_request).tokens
        input_tokens = torch.as_tensor([input_tokens], device=self.model.device)

        # input_tokens = self.tokenizer.apply_chat_template(
        #     [
        #         {"role": "user", "content": instructions + "\n\n" + question},
        #     ],
        #     add_generation_prompt=True,
        #     return_tensors="pt",
        # ).to(self.model.device)

        return input_tokens

    def few_shot_tokenize(
        self, instructions, question, demonstration_prompts: List[Tuple[str, str]]
    ):
        # Add demonstrations
        messages = []
        for i, (demo_question_prompt, demo_expected_response) in enumerate(
            demonstration_prompts
        ):
            if i == 0:
                # The chat starts with the task instructions
                messages.append(
                    UserMessage(
                        content=instructions + "\n####\\n" + demo_question_prompt
                    )
                )
            else:
                messages.append(UserMessage(content=demo_question_prompt))

            messages.append(AssistantMessage(content=demo_expected_response))

        # Add the final question
        messages.append(UserMessage(content=question))

        # Create the request
        completion_request = ChatCompletionRequest(messages=messages)

        # tokenize the text
        input_tokens = self.tokenizer.encode_chat_completion(completion_request).tokens
        input_tokens = torch.as_tensor([input_tokens], device=self.model.device)

        return input_tokens

    def generate(self, input_tokens, num_return_sequences=1):

        # generate output tokens
        output = self.model.generate(
            input_tokens,
            max_new_tokens=1024,
            do_sample=True,
            # temperature=0.0,
            num_return_sequences=num_return_sequences,
            eos_token_id=self.tokenizer.instruct_tokenizer.tokenizer.eos_id,
            pad_token_id=self.tokenizer.instruct_tokenizer.tokenizer.eos_id,
        )
        # Calculate prompt length
        prompt_length = input_tokens.shape[-1]

        # Remove the prompt from the output
        output = output[:, prompt_length:]

        # decode output tokens into text
        output = self.tokenizer.decode(output.tolist())

        if num_return_sequences == 1:
            return output[0]
        else:
            return output

    def get_tokenized_length(self, text: str) -> int:
        completion_request = ChatCompletionRequest(messages=[UserMessage(content=text)])
        input_tokens = self.tokenizer.encode_chat_completion(completion_request).tokens
        return len(input_tokens)
