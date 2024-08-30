# %%
import pandas as pd
import matplotlib.pyplot as plt

dataset = "bird"

df = pd.read_csv(f"{dataset}_experiments.csv")

# Keep only results with refinement
df = df[df["Refinement"] == True]

# Merge QD in model's name
df["Model"] = df.apply(
    lambda x: (
        x["Model"]
        if x["Question Decomposition"] == "No Decomposition"
        else f"{x['Model']} (QD)"
    ),
    axis=1,
)

# Keep only certain models
keep_models = [
    "Codestral-22B-v0.1",
    "deepseek-coder-6.7b-instruct",
    "deepseek-coder-33b-instruct",
    "granite-8b-code-instruct",
    "granite-34b-code-instruct",
]
df = df[df["Model"].isin(keep_models)]


df = df[
    [
        "Model",
        # "Question Decomposition",
        "Num. Return Seq.",
        "Example Shots",
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

df = df.rename(
    columns={"Num. Return Seq.": "samples", "Example Shots": "demonstrations"}
)

df = df.dropna()

# %%
# Plot w.r.t. demonstration number - keep samples constant at 1
fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(10, 4))
fig.tight_layout()

for name, group in df[(df["samples"] == 1)].groupby("Model"):
    group.plot(
        x="demonstrations",
        y="Schema Linking Precision",
        ax=axes[0],
        label=name,
        title="Schema Linking Precision",
        legend=False,
    )
    group.plot(
        x="demonstrations",
        y="Schema Linking Recall",
        ax=axes[1],
        label=name,
        title="Schema Linking Recall",
        legend=False,
    )

axes[-1].legend(
    loc="lower center",
    bbox_to_anchor=(-0.10, -0.35),
    ncol=3,
    fancybox=True,
    shadow=False,
)

fig.savefig(f"{dataset}_demonstrations.pdf", bbox_inches="tight")

# %%
# Plot w.r.t. samples number - keep demonstrations constant at 1
fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(10, 4))
fig.tight_layout()

for name, group in df[df["demonstrations"] == 1].groupby("Model"):
    group.plot(
        x="samples",
        y="Schema Linking Precision",
        ax=axes[0],
        label=name,
        title="Schema Linking Precision",
        legend=False,
    )
    group.plot(
        x="samples",
        y="Schema Linking Recall",
        ax=axes[1],
        label=name,
        title="Schema Linking Recall",
        legend=False,
    )

axes[-1].legend(
    loc="lower center",
    bbox_to_anchor=(-0.10, -0.35),
    ncol=3,
    fancybox=True,
    shadow=False,
)


fig.savefig(f"{dataset}_samples.pdf", bbox_inches="tight")