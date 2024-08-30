import json
from dec_sl.task.schema_linking import prepare_prompt

input_paths = [
    "data/processed/bird_train_text2sql.json",
    "data/processed/bird_dev_text2sql.json",
    "data/processed/spider_train_text2sql.json",
    "data/processed/spider_dev_text2sql.json",
]
output_paths = [
    "data/finetuning/bird/schema_linking_train_set.jsonl",
    "data/finetuning/bird/schema_linking_dev_set.jsonl",
    "data/finetuning/spider/schema_linking_train_set.jsonl",
    "data/finetuning/spider/schema_linking_dev_set.jsonl",
]


for input_path, output_path in zip(input_paths, output_paths):
    # Load dataset
    with open(input_path, "r") as fp:
        train_data = json.load(fp)

    # Create fine-tuning set
    with open(output_path, "w") as fp:
        for example in train_data:
            instruction, question, answer = prepare_prompt(example)
            json_obj = json.dumps({"instruction": question, "output": answer})
            fp.write(json_obj + "\n")
