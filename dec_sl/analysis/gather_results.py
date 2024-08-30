import pandas as pd
import json
import os
import itertools
from tqdm.auto import tqdm

from dec_sl.evaluation.schema_linking_eval import schema_linking_eval

DATASET = "bird"
PREDICTIONS_DIR = "data/predictions/schema-linking"

with open(f"data/processed/{DATASET}_dev_text2sql.json", "r") as fp:
    dev_set = json.load(fp)

schema_linking_models = [
    "deepseek-coder-6.7b-instruct",
    "finetuned-deepseek-6.7b",
    "Meta-Llama-3-8B-Instruct",
    "finetuned-llama-3-8b",
    "granite-8b-code-instruct",
    "finetuned-granite-8b",
    "Codestral-22B-v0.1",
    "finetuned-codestral-22b",
    "deepseek-coder-33b-instruct",
    "finetuned-deepseek-33b",
    "granite-20b-code-instruct",
    "finetuned-granite-20b",
    "granite-34b-code-instruct",
    "finetuned-granite-34b",
]

decomposition_models = [
    "No Decomposition",
    "Meta-Llama-3-70B-Instruct",
    # "Mixtral-8x22B-Instruct-v0.1",
]

demonstrations_count = [0, 1, 3, 5, 7]
samples_count = [1, 2, 4, 8]

df = pd.DataFrame(
    columns=[
        "Model",
        "Question Decomposition",
        "Num. Return Seq.",
        "Example Shots",
        "Refinement",
        "Schema Linking Recall",
        "Schema Linking Precision",
        "Table Precision",
        "Table Recall",
        "Column Precision",
        "Column Recall",
        "Failed Extraction",
        "Unmatched Tables",
        "Unmatched Column",
    ]
)

model_combinations = list(
    itertools.product(
        schema_linking_models, decomposition_models, demonstrations_count, samples_count
    )
)

for sl_model, dec_model, shots, samples in tqdm(model_combinations):
    if dec_model == "No Decomposition":
        predictions_file = f"{PREDICTIONS_DIR}/{sl_model}/{DATASET}_dev_{shots}_shot_{samples}_return.json"
    else:
        predictions_file = f"{PREDICTIONS_DIR}/{sl_model}/{DATASET}_dev_decomposition_by_{dec_model}_{shots}_shot_{samples}_return.json"

    # Check if we have results for given experiment
    if not os.path.isfile(predictions_file):
        continue

    # Load predictions
    with open(predictions_file, "r") as fp:
        predictions_json = json.load(fp)

    for refinement in [True, False]:
        # Evaluate predictions
        (
            table_precision,
            table_recall,
            column_precision,
            column_recall,
            schema_linking_precision,
            schema_linking_recall,
            failed_extraction,
            total_unmatched_tables,
            total_unmatched_columns,
        ) = schema_linking_eval(predictions_json, dev_set, use_refinement=refinement)

        # Add results to dataframe
        entry = pd.DataFrame.from_dict(
            {
                "Model": [sl_model],
                "Question Decomposition": [dec_model],
                "Example Shots": [shots],
                "Num. Return Seq.": [samples],
                "Refinement": [refinement],
                "Schema Linking Precision": [schema_linking_precision],
                "Schema Linking Recall": [schema_linking_recall],
                "Table Precision": [table_precision],
                "Table Recall": [table_recall],
                "Column Precision": [column_precision],
                "Column Recall": [column_recall],
                "Failed Extraction": [failed_extraction],
                "Unmatched Tables": [total_unmatched_tables],
                "Unmatched Column": [total_unmatched_columns],
            }
        )
        df = pd.concat([df, entry], ignore_index=True)

# Format precision and recall
df["Table Precision"] = df.apply(lambda x: x["Table Precision"] * 100, axis=1)
df["Table Recall"] = df.apply(lambda x: x["Table Recall"] * 100, axis=1)
df["Column Precision"] = df.apply(lambda x: x["Column Precision"] * 100, axis=1)
df["Column Recall"] = df.apply(lambda x: x["Column Recall"] * 100, axis=1)
df["Schema Linking Precision"] = df.apply(
    lambda x: x["Schema Linking Precision"] * 100, axis=1
)
df["Schema Linking Recall"] = df.apply(
    lambda x: x["Schema Linking Recall"] * 100, axis=1
)

# Calculate F1 Scores
df["Table F1"] = df.apply(
    lambda x: 2
    * (x["Table Precision"] * x["Table Recall"])
    / (x["Table Precision"] + x["Table Recall"]),
    axis=1,
)
df["Column F1"] = df.apply(
    lambda x: 2
    * (x["Column Precision"] * x["Column Recall"])
    / (x["Column Precision"] + x["Column Recall"]),
    axis=1,
)

# Save experiment results
df.to_csv(f"dec_sl/analysis/{DATASET}_experiments.csv")
