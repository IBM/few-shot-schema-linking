from dec_sl.utils.file_utils import load_json
from dec_sl.processing.postprocess import (
    extract_json,
    extract_labels_from_json,
    flatten_list,
    refine_schema_link_predictions,
)
import argparse
from sklearn.metrics import precision_score, recall_score


def schema_linking_eval(predictions_json, bird_dev, use_refinement: bool = True):
    schemas = [example["schema"] for example in bird_dev]
    ground_truth_labels = [
        (example["table_labels"], example["column_labels"]) for example in bird_dev
    ]

    failed_extraction = sum([1 if pred is None else 0 for pred in predictions_json])

    prediction_labels = []
    total_unmatched_tables = 0
    total_unmatched_columns = 0
    for prediction_json, schema in zip(predictions_json, schemas):
        table_labels, column_labels, unmatched_tables, unmatched_columns = (
            extract_labels_from_json(
                prediction_json, schema["schema_items"], fuzzy_matching=use_refinement
            )
        )

        if use_refinement:
            # Refinements for better predictions
            table_labels, column_labels = refine_schema_link_predictions(
                table_labels, column_labels, schema
            )

        prediction_labels.append((table_labels, column_labels))
        total_unmatched_tables += unmatched_tables
        total_unmatched_columns += unmatched_columns

    table_ground_truth_labels = [labels[0] for labels in ground_truth_labels]
    flat_table_ground_truth_labels = flatten_list(table_ground_truth_labels)

    table_prediction_labels = [labels[0] for labels in prediction_labels]
    flat_table_prediction_labels = flatten_list(table_prediction_labels)

    column_ground_truth_labels = [labels[1] for labels in ground_truth_labels]
    flat_column_ground_truth_labels = flatten_list(
        flatten_list(column_ground_truth_labels)
    )

    column_prediction_labels = [labels[1] for labels in prediction_labels]
    flat_column_prediction_labels = flatten_list(column_prediction_labels)

    per_example_precision = []
    per_example_recall = []
    for col_gt_labels, col_pred_labels in zip(
        column_ground_truth_labels, column_prediction_labels
    ):
        # Calculate schema linking precision
        if sum(flatten_list(col_gt_labels)) == sum(col_pred_labels) == 0:
            # Edge case where there are no columns in the GT and no columns are predicted
            # This should be considered correct and not set to 0.0 by zero_division
            per_example_precision.append(1.0)
        else:
            per_example_precision.append(
                precision_score(
                    flatten_list(col_gt_labels), col_pred_labels, zero_division=0.0
                )
            )
        # Calculate schema linking recall
        per_example_recall.append(
            recall_score(
                flatten_list(col_gt_labels), col_pred_labels, zero_division=1.0
            )
        )

    schema_linking_precision = sum(per_example_precision) / len(per_example_precision)
    schema_linking_recall = per_example_recall.count(1.0) / len(per_example_recall)

    table_precision = precision_score(
        flat_table_ground_truth_labels,
        flat_table_prediction_labels,
    )

    table_recall = recall_score(
        flat_table_ground_truth_labels,
        flat_table_prediction_labels,
    )

    column_precision = precision_score(
        flat_column_ground_truth_labels,
        flat_column_prediction_labels,
    )

    column_recall = recall_score(
        flat_column_ground_truth_labels,
        flat_column_prediction_labels,
    )

    return (
        table_precision,
        table_recall,
        column_precision,
        column_recall,
        schema_linking_precision,
        schema_linking_recall,
        failed_extraction,
        total_unmatched_tables,
        total_unmatched_columns,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare examples for Text-to-SQL")
    parser.add_argument(
        "-i", "--input_file", help="The input .json file,", required=True
    )
    parser.add_argument(
        "-g", "--ground_truth_file", help="The ground truth .json file,", required=True
    )
    parser.add_argument("--use_refinement", action="store_true", default=False)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    predictions_json = load_json(args.input_file)
    bird_dev = load_json(args.ground_truth_file)

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
    ) = schema_linking_eval(
        predictions_json, bird_dev, use_refinement=args.use_refinement
    )

    print(
        f"Number of predictions that failed extraction: {failed_extraction}/{len(predictions_json)}"
    )
    print(
        f"Number of predicted table names that were not matched to schema tables: {total_unmatched_tables}"
    )
    print(
        f"Number of predicted colum names that were not matched to schema columns: {total_unmatched_columns}"
    )

    print(f"Table Prec.:  {table_precision * 100}")
    print(f"Table Rec.:   {table_recall * 100}")
    print(f"Column Prec.: {column_precision * 100}")
    print(f"Column Rec.:  {column_recall * 100}")
    print(f"SL Prec.:     {schema_linking_precision * 100}")
    print(f"SL Rec.:      {schema_linking_recall * 100}")
