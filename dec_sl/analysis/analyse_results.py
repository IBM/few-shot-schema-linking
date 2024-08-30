# %%
import pandas as pd

DATASET = "spider"
df = pd.read_csv(f"{DATASET}_experiments.csv", index_col=0)
# %%
# Calculate the average difference between using refinement and not
diff = pd.DataFrame(
    columns=[
        "Schema Linking Precision",
        "Schema Linking Recall",
        "Table Precision",
        "Table Recall",
        "Table F1",
        "Column Precision",
        "Column Recall",
        "Column F1",
    ]
)

for name, group in df.groupby(
    ["Model", "Question Decomposition", "Example Shots", "Num. Return Seq."],
):
    with_refinement = group[group["Refinement"] == True]
    without_refinement = group[group["Refinement"] == False]
    assert len(with_refinement) == len(without_refinement) == 1

    entry = pd.DataFrame.from_dict(
        {
            key: with_refinement[key].values - without_refinement[key].values
            for key in diff.columns
        }
    )
    diff = pd.concat([diff, entry], ignore_index=True)

# %%
df.describe()

# %%
diff.describe()

# %%
# Find best configuration per model
best_per_model = pd.DataFrame(columns=df.columns)

for name, group in df[df["Refinement"] == True].groupby(
    ["Model", "Question Decomposition"]
):
    cur_best = group.sort_values(
        "Schema Linking Recall", axis=0, ascending=False, ignore_index=True
    ).iloc[[0]]

    best_per_model = pd.concat([best_per_model, cur_best], ignore_index=True)


# %%
# Print best models
best_per_model[
    [
        "Model",
        "Question Decomposition",
        "Num. Return Seq.",
        "Example Shots",
        # "Refinement",
        "Schema Linking Precision",
        "Schema Linking Recall",
        # "Table Precision",
        # "Table Recall",
        # "Table F1",
        # "Column Precision",
        # "Column Recall",
        # "Column F1",
    ]
]
