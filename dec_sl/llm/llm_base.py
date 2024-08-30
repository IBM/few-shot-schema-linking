from transformers import AutoModelForCausalLM, AutoTokenizer


class LLM:
    def __init__(self, model_name_or_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # drop device_map if running on CPU
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, device_map="auto", torch_dtype="auto"
        )

    def tokenize(self, instructions, question):
        prompt = f"{instructions}\n{question}"
        # tokenize the text
        input_tokens = self.tokenizer(prompt, return_tensors="pt")
        # transfer tokenized inputs to the device
        for i in input_tokens:
            input_tokens[i] = input_tokens[i].to(self.model.device)

        return input_tokens

    def generate(self, input_tokens):
        # generate output tokens
        output = self.model.generate(**input_tokens, max_new_tokens=1024)
        # Calculate prompt length
        prompt_length = input_tokens["input_ids"].shape[1]

        # Remove the prompt from the output
        output = output[0][prompt_length:]

        # decode output tokens into text
        output = self.tokenizer.batch_decode([output], skip_special_tokens=True)
        prediction = output[0]

        return prediction

    def get_tokenized_length(self, text: str) -> int:
        return len(self.tokenizer(text, return_tensors="np"))
    
    def get_max_length(self) -> int:
        return self.model.config.max_position_embeddings
