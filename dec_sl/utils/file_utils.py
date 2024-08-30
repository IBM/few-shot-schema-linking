from typing import Dict, List
import json


def load_json(file_path: str) -> List[Dict]:
    with open(file_path, "r") as fp:
        examples = json.load(fp)
    return examples


def write_json(examples, file_path: str):
    with open(file_path, "w") as fp:
        json.dump(examples, fp, indent=4)
