# Few-shot Schema Linking with Question Decomposition

The goal of this project is to enable schema linking without the need for fine-tuning
a model, by taking advantage of the question decomposition capabilities of LLMs.
This allows us to break down the complex task of schema linking into easier and
more manageable sub-tasks.

## Data Preparation

We must first download the datasets and pre-process them to the appropriate formats.

### Download datasets

This project uses the BIRD and Spider datasets that can be found in the following links:

- BIRD: https://bird-bench.github.io/
- Spider: https://yale-lily.github.io/spider

Extract the datasets and add them under the data directory.
Your directory structure should look like this:

```
├── project-root
│   ├── data
│   │   ├── bird
│   │   │   ├── dev
│   │   │   │   ├── dev.json
│   │   │   │   ├── ...
│   │   │   ├── train
│   │   │   │   ├── train.json
│   │   │   │   ├── ...
│   │   ├── spider
│   │   │   ├── train_spider.json
│   │   │   ├── dev.json
│   ├── dec_sl
│   │   ├── llm
│   │   ├── task
│   │   ├── ...
│   ├── ...
│   ├── README.md
│   └── .gitignore
```

### Build databases indexes and prepare datasets

Now that the datasets are in place, we need to build the content indexes that are
used to find relevant values for each question and then pre-process the files in
a format that can be used more easily.

To do this you can run the following commands:

```bash
python -m dec_sl.processing.build_contents_index
python -m dec_sl.processing.prepare_datasets
```

The processed datasets will be saved under `data/processed/`.

### Download Similarity Model

In order to perform few-shot prompting for schema linking or Text-to-SQL you will
also need to download a similarity model that helps retrieve demonstration
examples based on their similarity to the test example.

You will need to download the `princeton-nlp/sup-simcse-roberta-base` model from
[huggingface](https://huggingface.co/princeton-nlp/sup-simcse-roberta-base) and 
save it in a directory named `models` on the same level as the project-root.

```
├── project-root
│   ├── data
│   ├── dec_sl
│   ├── ...
│   ├── README.md
│   └── .gitignore
├── models
│   └── princeton-nlp
│   │   └── sup-simcse-roberta-base
```

## Run Question Decomposition

To generate the decomposed questions you can use the following command.
This is an optional step that can improve schema linking recall, but you can also
perform schema linking without it.

```bash
python -m dec_sl.task.question_decomposition \
--model path/to/huggingface/model/ \
--input_file data/processed/bird_dev_text2sql.json \
--output_file path/to/output/dev.json
```

## Run Schema Linking

To generate schema linking predictions you can use the following command.

If you want to use question decomposition during schema linking you can pass the
output file from the previous script with the `--decomposed_file` arguments.

If you want to perform few-shot linking you can pass the train set with the
`--demonstrations_file` argument, and specify the number of demonstrations in
each input with `--num_of_demonstrations`.

If you want to sample more than 1 outputs per prediction you can specify the
number of samples with the `--num_return_sequence` argument.

```bash
python3 -m dec_sl.task.schema_linking \
--model path/to/huggingface/model/ \
--input_file data/processed/bird_dev_text2sql.json \
--decomposed_file  path/to/decomposition/predictions/dev.json \
--demonstrations_file data/processed/bird_train_text2sql.json \
--num_of_demonstrations 3 \
--num_return_sequence 8 \
--output_file path/to/output/dev.json
```

### Evaluate Schema Linking Predictions

To evaluate the predictions of schema linking you can use the following command.

If you also want to use the refinement techniques that improve prediction quality
you can pass the flag `--use_refinement`.

```bash
python3 -m dec_sl.evaluation.schema_linking_eval \
--input_file path/to/predictions/dev.json \
--ground_truth_file data/processed/bird_dev_text2sql.json \
--use_refinement
```

## Run Text-to-SQL

To generate Text-to-SQL predictions you can used the following command.

If you want to use the schema linking predictions from the previous script you
can pass them with the `--schema_links_file` argument.

If you want to perform few-shot Text-to-SQL you can pass the train set with the
`--demonstrations_file` argument, and specify the number of demonstrations in
each input with `--num_of_demonstrations`.

```bash
python3 -m dec_sl.task.text_to_sql \
--model path/to/huggingface/model/ \
--input_file data/processed/bird_dev_text2sql.json \
--schema_links_file path/to/schema/linking/predictions/dev.json  \
--demonstrations_file data/processed/bird_train_text2sql.json \
--num_of_demonstrations 3 \
--output_file path/to/output/dev.sql
```