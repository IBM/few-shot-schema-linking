"""Usefull text utilities.

Adapted from the CodeS github repositiry. 
https://github.com/RUCKBReasoning/codes/blob/main/prepare_sft_datasets.py
"""

import re
import nltk
from simcse import SimCSE
import numpy as np


def str_replace_ignore_case(evidence, schema_item_name):
    evidence = re.sub(
        re.escape(schema_item_name), schema_item_name, evidence, 0, re.IGNORECASE
    )

    return evidence


def preprocess_evidence(
    evidence,
    # schema_items,
):
    if evidence.strip() == "":
        return ""

    evidence = evidence.strip()
    # if evidence does not end with ";", add a ";" char
    if not evidence.endswith(";"):
        evidence += ";"

    # lowercase schema items appeared in the evidence
    # for table in schema_items:
    #     if table["table_name"] in evidence.lower():
    #         evidence = str_replace_ignore_case(evidence, table["table_name"])

    #     for column_name in table["column_names"]:
    #         if column_name in evidence.lower():
    #             evidence = str_replace_ignore_case(evidence, column_name)

    evidence = evidence.replace("< =", "<=").replace("> =", ">=")

    evidence = evidence.replace("\n", " ")

    return evidence


def extract_skeleton(text: str) -> str:
    """Extract the context-independant skeleton of a NLQ."""
    tokens_and_tags = nltk.pos_tag(nltk.word_tokenize(text))

    output_tokens = []
    for token, tag in tokens_and_tags:
        if tag in ["NN", "NNP", "NNS", "NNPS", "CD", "SYM", "FW", "IN"]:
            output_tokens.append("_")
        elif token in ["$", "''", "(", ")", ",", "--", ".", ":"]:
            pass
        else:
            output_tokens.append(token)

    text_skeleton = " ".join(output_tokens)
    text_skeleton = text_skeleton.replace("_ 's", "_")
    text_skeleton = text_skeleton.replace(" 's", "'s")

    while "_ _" in text_skeleton:
        text_skeleton = text_skeleton.replace("_ _", "_")
    while "_ , _" in text_skeleton:
        text_skeleton = text_skeleton.replace("_ , _", "_")

    if text_skeleton.startswith("_ "):
        text_skeleton = text_skeleton[2:]

    return text_skeleton

def get_demonstration_silimarities(examples, demonstrations, sim_model_path):
    # Get question and skeletons from examples
    example_questions = [data["question"] for data in examples]
    example_question_skeletons = [
        extract_skeleton(question) for question in example_questions
    ]

    # Get question and skeletons from demonstrations
    demonstration_questions = [data["question"] for data in demonstrations]
    demonstration_question_skeletons = [
        extract_skeleton(question) for question in demonstration_questions
    ]

    # compute similarities between questions in the evaluation set and the demonstration pool
    simsce_model = SimCSE(sim_model_path)
    question_similarities = simsce_model.similarity(
        example_questions, demonstration_questions
    )
    question_skeleton_similarities = simsce_model.similarity(
        example_question_skeletons, demonstration_question_skeletons
    )
    similarities = np.maximum(question_similarities, question_skeleton_similarities)

    del simsce_model

    return similarities