import json
import re
import itertools
from typing import Dict, List, Tuple, Union

from rapidfuzz.utils import default_process
from rapidfuzz import fuzz


def flatten_list(list2d):
    """Flatten a list of lists into a simple list."""
    return list(itertools.chain(*list2d))


def remove_table_name(column_name: str) -> str:
    """Remove the table specification from a column name."""
    return default_process(column_name.split(".", maxsplit=1)[-1])


def extract_fenced_code_block(generation: str, language: str = "") -> str | None:
    """Extract code from a markdown fenced code blocks."""
    # NOTE: The *? symbol matches as little text as possible
    pattern_recipe = f"```{language}([\S\s]*?)```"
    match = re.search(pattern_recipe, generation, flags=re.IGNORECASE)
    if match is None:
        return None
    match_text = match.group(1)

    return match_text


def extract_sql(generation: str) -> str | None:
    """Extract a SQL query from an LLM generation."""
    pattern_recipe = "```sql([\S\s]*)```"
    match = re.search(pattern_recipe, generation)

    if match is None:
        return None
    else:
        match = match.group(1)
        return match


def extract_json(generation: str) -> Dict | None:
    """Extract a json object from an LLM generation."""
    # Ideally the model will have generated the json in a markdown fenced code block
    possible_json = extract_fenced_code_block(generation, "json")

    if possible_json is None:
        # If the model did not generate a fenced code block try to extract from the original generation
        possible_json = generation

    # Try to extract json object from text
    try:
        match_json = json.loads(possible_json)
    except:
        match_json = None

    if not isinstance(match_json, dict):
        match_json = None

    return match_json


def extract_list(generation: str, expected_title: str) -> Dict | None:
    """Extract a list from an LLM generation."""
    # Chek that the title of the list is generated correctly and get text after
    pattern_recipe = f"{expected_title}\n[\s]*([\S\s]*)"
    match = re.search(pattern_recipe, generation)

    if match is None:
        return None
    else:
        match_text = match.group(1)

    # Split the text after into items
    items = match_text.split("\n")
    # Remove list dashes and numbering at the start of each item
    items = [item.lstrip(" -1234567890.") for item in items]

    return items


def remove_special_chars(text: str) -> str:
    return text.replace("`", "").replace("_", "").replace("-", "")


def group_column_labels_by_table(
    column_labels: List[int], schema_items: List[Dict]
) -> List[List[int]]:
    grouped_column_labels = []
    column_index = 0

    for table in schema_items:
        column_count = len(table["column_names"])
        grouped_column_labels.append(
            column_labels[column_index : column_index + column_count]
        )
        column_index += column_count

    return grouped_column_labels


def find_column_prediction_index(
    prediction: str, column_names: List[str], fuzzy_matching: bool = True
) -> int:
    """Find the index of str column prediction in a list of column names."""
    # Check if there is an exact match
    if prediction in column_names:
        return column_names.index(prediction)

    if fuzzy_matching:
        # Calculate fuzzy ratios
        fuzzy_ratios = [
            0.5
            * fuzz.ratio(
                prediction, column_name, processor=default_process, score_cutoff=50.0
            )
            + 0.5
            * fuzz.ratio(
                prediction, column_name, processor=remove_table_name, score_cutoff=50.0
            )
            for column_name in column_names
        ]
        # If any column name had a score above the cutoff
        max_ratio = max(fuzzy_ratios)
        if max_ratio > 0:
            return fuzzy_ratios.index(max_ratio)
    else:
        # Remove special characters from all names and check again
        prediction = remove_special_chars(prediction)
        column_names = [
            remove_special_chars(column_name) for column_name in column_names
        ]
        if prediction in column_names:
            return column_names.index(prediction)

    return -1


def find_table_prediction_index(
    prediction: str, table_names: List[str], fuzzy_matching: bool = True
):
    """Find the index of str table prediction in a list of table names."""
    # Check if there is an exact match
    if prediction in table_names:
        return table_names.index(prediction)

    if fuzzy_matching:
        # Calculate fuzzy ratios
        fuzzy_ratios = [
            fuzz.ratio(
                prediction, table_name, processor=default_process, score_cutoff=50.0
            )
            for table_name in table_names
        ]
        # If any table name had a score above the cutoff
        max_ratio = max(fuzzy_ratios)
        if max_ratio > 0:
            return fuzzy_ratios.index(max_ratio)
    else:
        # Remove special characters from all names and check again
        prediction = remove_special_chars(prediction)
        table_names = [remove_special_chars(table_name) for table_name in table_names]
        if prediction in table_names:
            return table_names.index(prediction)

    return -1


def extract_labels_from_json(
    predicted_json: Union[Dict, None],
    schema_items: List[Dict],
    fuzzy_matching: bool = True,
) -> Tuple[List, List, int, int]:
    """Extract column and talbe labels for schema linking from a predicted json."""
    # Get all table and column names in flat lists
    flat_table_names: List[str] = [
        schema_item["table_name"] for schema_item in schema_items
    ]
    flat_column_names: List[str] = [
        f"{schema_item['table_name']}.{column_name}"
        for schema_item in schema_items
        for column_name in schema_item["column_names"]
    ]

    # Create label lists of same length an initialize with zeros
    pred_table_labels = [0 for _ in range(len(flat_table_names))]
    pred_column_labels = [0 for _ in range(len(flat_column_names))]

    if predicted_json is None:
        # If the prediction has failed, return predicted labels with only zeros
        return pred_table_labels, pred_column_labels, 0, 0

    pred_tables = predicted_json.get("tables", [])
    pred_columns = predicted_json.get("columns", [])

    # NOTE: Debugging
    unmatched_tables = []
    unmatched_columns = []

    for pred_table in pred_tables:
        table_index = find_table_prediction_index(
            pred_table, flat_table_names, fuzzy_matching=fuzzy_matching
        )
        if table_index >= 0:
            pred_table_labels[table_index] = 1
        else:
            # NOTE: Debugging
            unmatched_tables.append(pred_table)

    for pred_column in pred_columns:
        column_index = find_column_prediction_index(
            pred_column, flat_column_names, fuzzy_matching=fuzzy_matching
        )
        if column_index >= 0:
            pred_column_labels[column_index] = 1
        else:
            # NOTE: Debugging
            unmatched_columns.append(pred_column)

    # NOTE: Debugging
    if False:
        if len(unmatched_tables) > 0 or len(unmatched_columns) > 0:
            print(f"flat_table_names: {flat_table_names}")
            print(f"flat_column_names: {flat_column_names}")
            print("\n")
            print(f"unmatched_tables: {unmatched_tables}")
            print(f"unmatched_columns: {unmatched_columns}")
            print("\n")
            print(f"pred_tables: {pred_tables}")
            print(f"pred_columns: {pred_columns}")
            print(f"{15*'='}")
            print("\n")

    # Calculate unmatched tables and columns
    unmatched_tables = len(pred_tables) - sum(pred_table_labels)
    unmatched_columns = len(pred_columns) - sum(pred_column_labels)

    return pred_table_labels, pred_column_labels, unmatched_tables, unmatched_columns


def get_all_possbile_fks(schema):
    """Identify additional FK that may not be declared explicitely in the database."""
    foreign_keys = schema["foreign_keys"]
    # Obtain additional foreign keys that are not explicit
    # If a PK column appears in another table with the same name and type add a FK connection
    for cur_schema_item in schema["schema_items"]:
        cur_table_name = cur_schema_item["table_name"]
        # Get PKs of current table
        primary_keys = [
            (col_name, col_type)
            for col_name, col_type, is_pk in zip(
                cur_schema_item["column_names"],
                cur_schema_item["column_types"],
                cur_schema_item["pk_indicators"],
            )
            if is_pk
        ]

        for schema_item in schema["schema_items"]:
            # Search all other tables
            other_table_name = schema_item["table_name"]
            if other_table_name == cur_table_name:
                continue

            # Get column names and type of other table
            columns = [
                (column_name, column_type)
                for column_name, column_type in zip(
                    schema_item["column_names"], schema_item["column_types"]
                )
            ]

            # Look for columns with the same name and type as the PKs
            for pk_name, pk_type in primary_keys:
                for col_name, col_type in columns:
                    if pk_name == col_name and pk_type == col_type:
                        foreign_keys.append(
                            [cur_table_name, pk_name, other_table_name, col_name]
                        )

    return foreign_keys


def create_db_graph(foreign_keys: List[List[str]]) -> Dict[str, List[str]]:
    """Create a graph of DB tables based on their FK connections."""
    db_graph = {}

    for t1_name, _, t2_name, _ in foreign_keys:
        if t1_name in db_graph:
            db_graph[t1_name].append(t2_name)
        else:
            db_graph[t1_name] = [t2_name]

        if t2_name in db_graph:
            db_graph[t2_name].append(t1_name)
        else:
            db_graph[t2_name] = [t1_name]

    # Remove duplicates
    db_graph = {
        table_name: list(set(neighbors)) for table_name, neighbors in db_graph.items()
    }

    return db_graph


def sub_graph_is_connected(sub_graph: Dict) -> bool:
    """Check if all the nodes of a subgraph are connected."""
    for node, neighbours in sub_graph.items():
        # Perform BFS and check if we can visit all the nodes of the sub-graph
        visited = [node]
        queue = [n for n in neighbours if n in sub_graph and not n in visited]
        while len(queue) > 0:
            cur_node = queue.pop(0)
            visited.append(cur_node)
            queue.extend(
                [
                    n
                    for n in sub_graph[cur_node]
                    if n in sub_graph and not n in visited and not n in queue
                ]
            )

        if len(visited) != len(sub_graph):
            return False

    return True


def get_sub_graph_neighborhoods(sub_graph: Dict) -> List:
    """Get the neighborhoods of a subgraph with no connections between them."""
    # At first every node is assigned to its own neighborhood
    inv_neighborhoods = {node: i for i, node in enumerate(sub_graph.keys())}

    for node, index in inv_neighborhoods.items():
        # Add all the node's neighbors to its neighborhood
        for neighbor in sub_graph[node]:
            if neighbor in sub_graph:
                inv_neighborhoods[neighbor] = index

    # Create neighborhoods from inverted index
    neighborhoods = [([], []) for _ in range(len(inv_neighborhoods))]
    for node, index in inv_neighborhoods.items():
        neighborhoods[index][0].append(node)
        neighborhoods[index][1].extend(sub_graph[node])

    # Remove empty neighborhoods
    neighborhoods = [x for x in neighborhoods if x != ([], [])]

    # Clean up duplicate nodes in neighborhoods
    for i in range(len(neighborhoods)):

        nodes, neighbors = neighborhoods[i]
        neighbors = [n for n in neighbors if not n in nodes]
        neighbors = list(set(neighbors))
        neighborhoods[i] = (nodes, neighbors)

    return neighborhoods


def refine_schema_link_predictions(
    table_labels: List[int], column_labels: List[int], schema: Dict
) -> Tuple[List[int], List[int]]:
    """Automatic refinements to improve the schema linking prediction."""

    flat_table_names: List[str] = [
        schema_item["table_name"] for schema_item in schema["schema_items"]
    ]
    flat_column_names: List[str] = [
        f"{schema_item['table_name']}.{column_name}"
        for schema_item in schema["schema_items"]
        for column_name in schema_item["column_names"]
    ]

    foreign_keys = get_all_possbile_fks(schema)

    # REFINEMENT 1
    # Make sure that if a column is predicted its table is also predicted
    table_idx, column_idx = 0, 0
    for table in schema["schema_items"]:
        if table_labels[table_idx] == 1:
            # If the table is already predicted there is no need to check, only move the index
            column_idx += len(table["column_names"])
        else:
            for _ in table["column_names"]:
                if column_labels[column_idx] == 1:
                    table_labels[table_idx] = 1
                column_idx += 1

        table_idx += 1

    # REFINEMENT 2
    # Make sure that all predicted tables have a connecting FK path.
    # Create a DB graph based on FK connections and add any missing tables to
    # connect the predicted tables.
    tables_to_add = []

    # Create DB graph
    db_graph = create_db_graph(foreign_keys)

    # NOTE: This refinement can only work if all tables are connected in some way
    if all([table_name in db_graph for table_name in flat_table_names]):
        # Create sub-graph with only predicted tables
        sub_graph = {
            table_name: db_graph[table_name]
            for table_name, label in zip(flat_table_names, table_labels)
            if label == 1
        }

        # If any predicted table has only one neighbor that does not appear in
        # the predicted tables then it must be added to connect the sub-graph
        for table, neighbors in sub_graph.items():
            if len(neighbors) == 1 and not neighbors[0] in sub_graph:
                tables_to_add.append(neighbors[0])
        for table in tables_to_add:
            sub_graph[table] = db_graph[table]

        # Check if predicted sub-graph is still not connected
        if not sub_graph_is_connected(sub_graph):
            # Find the neighborhoods in the sub-graph
            neighborhoods = get_sub_graph_neighborhoods(sub_graph)

            # Find the common neighbours between all neighborhoods
            neighbours_sets = [
                set([n for n in neighbors if n not in sub_graph])
                for nodes, neighbors in neighborhoods
            ]

            common_neighbors = neighbours_sets[0]
            for i in range(1, len(neighbours_sets)):
                common_neighbors = common_neighbors.intersection(neighbours_sets[i])

            # Add all common neighbors
            tables_to_add.extend(list(common_neighbors))

        for table_name in tables_to_add:
            table_labels[flat_table_names.index(table_name)] = 1

    # REFINEMENT 3
    # Make sure that if a table is predicted then its PKs are also predicted
    # NOTE: Increases col. recall slightly but decreases col. precision
    # table_idx, column_idx = 0, 0
    # for table in schema["schema_items"]:
    #     if table_labels[table_idx] == 0:
    #         # If the table is not predicted, move to the next one
    #         column_idx += len(table["column_names"])
    #     else:
    #         for pk_indicator in table["pk_indicators"]:
    #             # pk_indicator: int = 1 if column is pk else 0
    #             if pk_indicator == 1:
    #                 column_labels[column_idx] = 1
    #             column_idx += 1

    #     table_idx += 1

    # REFINEMENT 4
    # Make sure that if two predicted tables can be joined by FKs, they are in the column predictions
    foreign_keys = [
        (
            (
                flat_table_names.index(fk[0]),
                flat_column_names.index(f"{fk[0]}.{fk[1]}"),
            ),
            (
                flat_table_names.index(fk[2]),
                flat_column_names.index(f"{fk[2]}.{fk[3]}"),
            ),
        )
        for fk in schema["foreign_keys"]
    ]

    for fk1, fk2 in foreign_keys:
        if table_labels[fk1[0]] == 1 and table_labels[fk2[0]] == 1:
            # If both tables are in the prediction, add the FK columns
            column_labels[fk1[1]] = 1
            column_labels[fk2[1]] = 1

    return table_labels, column_labels


def remove_extra_whitespace(sql_query: str) -> str:
    """Replace all whitespace in the string with simple spaces."""
    sql_query = " ".join(sql_query.split())
    return sql_query
