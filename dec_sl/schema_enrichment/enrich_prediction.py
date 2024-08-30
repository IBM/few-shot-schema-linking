import json
from dec_sl.schema_enrichment.schema_enrichment import SchemaEnrichment
from dec_sl.processing.postprocess import (
    extract_labels_from_json,
    flatten_list,
    refine_schema_link_predictions,
)

bird_dev_path = "data/processed/bird_with_evidence_dev_text2sql.json"
predictions_path = "/dccstor/flowpilot-vol/gka/decomposed-linking/data/predictions/schema-linking/Codestral-22B-v0.1/bird_dev_decomposition_by_Meta-Llama-3-70B-Instruct_3_shot_8_return.json"
output_path = "data/predictions/schema-linking/enriched/codestral-22b-qd.enriched_0.4.json"
REPRESENTATION_PERCENT = 0.4
IS_DB_SCAN = False
PLUS_OUTLIERS = False

# Initialize enricher
enricher = SchemaEnrichment(bird_dev_path)

# Load the dataset
with open(bird_dev_path, "r") as fp:
    bird_dev = json.load(fp)

# Load the predictions
with open(predictions_path, "r") as fp:
    predictions = json.load(fp)

enriched_schemas = []

for example, prediction in zip(bird_dev, predictions):
    schema = example["schema"]
    flat_table_names = [
        schema_item["table_name"] for schema_item in schema["schema_items"]
    ]
    flat_column_names = [
        f"{schema_item['table_name']}.{column_name}"
        for schema_item in schema["schema_items"]
        for column_name in schema_item["column_names"]
    ]

    # Perform refinement on the prediction
    table_labels, column_labels, _, _ = extract_labels_from_json(
        prediction, schema["schema_items"], fuzzy_matching=True
    )
    table_labels, column_labels = refine_schema_link_predictions(
        table_labels, column_labels, schema
    )
    
    refined_prediction = {}
    refined_prediction["tables"] = [
        table_name
        for table_name, label in zip(flat_table_names, table_labels)
        if label == 1
    ]
    refined_prediction["columns"] = [
        column_name
        for column_name, label in zip(flat_column_names, column_labels)
        if label == 1
    ]

    prediction = refined_prediction

    # Get enriched columns
    enriched_columns = enricher.enrich_schema(
        example["db_id"],
        prediction["columns"],
        representation_percent=REPRESENTATION_PERCENT,
        is_dbscan=IS_DB_SCAN,
        plus_outliers=PLUS_OUTLIERS,
    )

    # Create enriched schema for given example
    enriched_schema = {
        "tables": prediction["tables"],
        "columns": list(set(enriched_columns).union(prediction["columns"])),
    }

    enriched_schemas.append(enriched_schema)

# Write enriched schemas to output
with open(output_path, "w") as fp:
    json.dump(enriched_schemas, fp, indent=4)
