# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("bird_text2sql.csv")
# %%
df = df.dropna()
df = df[(df["Model"] != "No Schema Linking") & (df["Model"] != "Oracle Schema Linking")]
# %%
# df.columns
data = {
    "Model": [],
    "Schema Linking Precision": [],
    "Schema Linking Recall": [],
    "F1 Score": [],
    "Execution Accuracy": [],
    "Text-to-SQL Model": [],
}

for _, row in df.iterrows():
    for model in ["Codestral", "Deepseek", "Granite"]:
        data["Model"].append(row["Model"])
        data["Schema Linking Precision"].append(row["Schema Linking Precision"])
        data["Schema Linking Recall"].append(row["Schema Linking Recall"])
        data["F1 Score"].append(
            2
            * (row["Schema Linking Recall"] * row["Schema Linking Precision"])
            / (row["Schema Linking Recall"] + row["Schema Linking Precision"])
        )
        data["Execution Accuracy"].append(row[model])
        data["Text-to-SQL Model"].append(model)

data = pd.DataFrame(data=data)
# %%

# f, axes = plt.subplots(1, 2)

# df.plot(x=" Column Recall", y=[" Codestral"], kind="scatter")
sns.lmplot(
    x="Schema Linking Recall",
    # x="Schema Linking Precision",
    y="Execution Accuracy",
    hue="Text-to-SQL Model",
    markers=["x", "o", "+"],
    # data=data,
    data=data[data["Schema Linking Precision"] > 50],
    # data=data[data["Schema Linking Recall"] > 80],
    fit_reg=True,
    legend_out=False,
    order=2,
)

# %%
df