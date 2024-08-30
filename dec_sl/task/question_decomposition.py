import argparse
from typing import List
from tqdm.auto import tqdm

from dec_sl.llm.llama import Llama
from dec_sl.llm.mixtral import Mixtral
from dec_sl.llm.llm_base import LLM
from dec_sl.processing.postprocess import extract_list
from dec_sl.processing.prompting import create_decomposition_prompt
from dec_sl.utils.file_utils import load_json, write_json


DEMONSTRATION_PROMPTS = [
    # demonstration = (question_prompt, expected_response)
    (
        "Among the countries whose GDP is over 1000000, how many of them have a population groth rate of over 3%?",
        (
            "Decomposed questions:\n\n"
            "- What is the GDP of each country?\n"
            "- What is the population growth rate of each country?\n"
            "- Which countries have a GDP over 1000000?\n"
            "- How many of those countries have a population growth rate of over 3%?\n"
        ),
    ),
    (
        "Provide the country with its full name which has the most ethnic group? List them all ethnic group together with its percentage.",
        (
            "Decomposed questions:\n\n"
            "- What are the ethnic groups in each country?\n"
            "- What is the percentage of each ethnic group in each country?\n"
            "- Which country has the most ethnic groups?\n"
            "- What is the full name of each country?\n"
            "- What are the names of all ethnic groups?\n"
        ),
    ),
    (
        "Among the employees, give me the full names of those who have less than 4 territories.",
        (
            "Decomposed questions:\n\n"
            "- What are the territories of each employee?\n"
            "- How many territories does each employee have?\n"
            "- Which employees have less than 4 territories?\n"
            "- What are the full names of the employees?\n"
        ),
    ),
    (
        "Among all the current legislators whose religion is Roman Catholic, what is the percentage of the ones without an instagram account?",
        (
            "Decomposed questions:\n\n"
            "- What is the religion of each legislator?\n"
            "- Which legislators have Roman Catholic as their religion?\n"
            "- Which legislators have an Instagram account?\n"
            "- What is the percentage of Roman Catholic legislators without an Instagram account?\n"
        ),
    ),
    (
        "What are the language and title of the ordered books with price less than 20%% of the average price of all ordered books?",
        (
            "Decomposed questions:\n\n"
            "- What is the price of each ordered book?\n"
            "- What is the average price of all ordered books?\n"
            "- Which ordered books have a price less than 20%% of the average price?\n"
            "- What is the language of each book?\n"
            "- What is the title of each book?\n"
        ),
    ),
    (
        "In 2014, what is the shortest duration of trips by subscribers which started at 2nd at Folsom and ended in the 5th at Howard stations, and by how much shorter than the average? Give me the minimum temperature, maximum gust speed and weather event on that trip.",
        (
            "Decomposed questions:\n\n"
            "- What is the duration of each trip?\n"
            "- Show the trips for 2014.\n"
            "- Which trips started at 2nd at Folsom station?\n"
            "- Which trips ended in the 5th at Howard station?\n"
            "- What is the average trip duration?\n"
            "- How much shorter was each trip than the average trip duration?\n"
            "- What is the minimum temperature of each trip?\n"
            "- What is the maximum gust speed of each trip?\n"
            "- What is the weather event of each trip?\n"
        ),
    ),
]


def predict(model_name_or_path: str, prompts: List[str]):
    if "llama" in model_name_or_path.lower():
        llm = Llama(model_name_or_path)
    elif "mixtral" in model_name_or_path.lower():
        llm = Mixtral(model_name_or_path)
    else:
        llm = LLM(model_name_or_path)

    llm.model.eval()

    predictions = []
    for instructions, question in tqdm(prompts):

        input_tokens = llm.few_shot_tokenize(
            instructions, question, DEMONSTRATION_PROMPTS
        )
        prediction = llm.generate(input_tokens)

        list_prediction = extract_list(prediction, "Decomposed questions:")

        predictions.append(list_prediction)

    return predictions


def parse_args():
    parser = argparse.ArgumentParser(
        description="Decompose Natural Language Questions for Text-to-SQL"
    )
    parser.add_argument(
        "-i", "--input_file", help="The input .json file,", required=True
    )
    parser.add_argument(
        "-o",
        "--output_file",
        help="The output file to write the generated prediction.",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--model",
        help="The LLM model to use for predicting.",
        required=True,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # Get Text-to-SQL examples
    examples = load_json(args.input_file)

    # Create prompts
    prompts = []
    for example in examples:
        db_id = example["db_id"]
        evidence = example["evidence"]
        question = example["question"]
        sql = example["sql"]

        prompts.append(create_decomposition_prompt(question, simple=True))

    # Get predictions
    predictions = predict(args.model, prompts)
    # Write predictions
    write_json(predictions, args.output_file)
