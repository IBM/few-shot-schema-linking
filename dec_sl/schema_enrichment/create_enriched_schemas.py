import json
from dec_sl.schema_enrichment.schema_enrichment import SchemaEnrichment
from dec_sl.processing.postprocess import flatten_list

bird_dev_path = "data/processed/bird_with_evidence_dev_text2sql.json"
REPRESENTATION_PERCENT = 0.0
IS_DB_SCAN = True
PLUS_OUTLIERS = False

# Initialize enricher
enricher = SchemaEnrichment(bird_dev_path)

# For each question load ground truth schema links and enrich them
with open(bird_dev_path, "r") as fp:
    bird_dev = json.load(fp)

enriched_schemas = []

for example in bird_dev:
    # Load schema and schema linking labels
    table_labels = example["table_labels"]
    column_labels = flatten_list(example["column_labels"])
    schema = example["schema"]
    flat_table_names = [
        schema_item["table_name"] for schema_item in schema["schema_items"]
    ]
    flat_column_names = [
        f"{schema_item['table_name']}.{column_name}"
        for schema_item in schema["schema_items"]
        for column_name in schema_item["column_names"]
    ]

    # Create ground truth table and column
    ground_truth = {}
    ground_truth["tables"] = [
        table_name
        for table_name, label in zip(flat_table_names, table_labels)
        if label == 1
    ]
    ground_truth["columns"] = [
        column_name
        for column_name, label in zip(flat_column_names, column_labels)
        if label == 1
    ]

    # Get enriched columns
    enriched_columns = enricher.enrich_schema(
        example["db_id"],
        ground_truth["columns"],
        representation_percent=REPRESENTATION_PERCENT,
        is_dbscan=IS_DB_SCAN,
        plus_outliers=PLUS_OUTLIERS,
    )

    # Create enriched schema for given example
    enriched_schema = {
        "tables": ground_truth["tables"],
        "columns": list(set(enriched_columns).union(ground_truth["columns"])),
    }

    enriched_schemas.append(enriched_schema)

# Write enriched schemas to output
output_path = "data/predictions/schema-linking/enriched/enriched_schemas"
if REPRESENTATION_PERCENT != 0.0:
    output_path += f"_{REPRESENTATION_PERCENT}"
if IS_DB_SCAN:
    output_path += f"_dbscan"
if PLUS_OUTLIERS:
    output_path += f"_outliers"
output_path += ".json"

with open(output_path, "w") as fp:
    json.dump(enriched_schemas, fp, indent=4)
