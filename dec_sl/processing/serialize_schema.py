from typing import Dict, List, Tuple, Literal
from os import path
import argparse
import json

from dec_sl.utils.db_utils import get_db_schema, get_db_schema_sequence, get_db_comments


def simple_serialization(table_to_columns: Dict[str, List]) -> str:
    serialized_tables = [
        f"{table_name}: {', '.join(table_columns)}"
        for table_name, table_columns in table_to_columns.items()
    ]
    serialized_schema = " | ".join(serialized_tables)
    return serialized_schema


def verbose_serialization(table_to_columns: Dict[str, List]) -> str:
    serialized_tables = []
    for table_name, table_columns in table_to_columns.items():
        # Prepare the serialization of the columns
        serialized_columns = [
            f"\t- `{column_name}` ({column_type})"
            for column_name, column_type in table_columns
        ]
        joined_serialized_columns = "\n".join(serialized_columns)
        # Serialize the table with its columns
        serialized_tables.append(
            f"Table `{table_name}` contains the following columns:"
            "\n"
            f"{joined_serialized_columns}"
        )

    serialized_schema = "\n".join(serialized_tables)
    return serialized_schema


def ddl_serialization(table_to_columns: Dict[str, List]) -> str:

    serialized_tables = []
    for table_name, table_columns in table_to_columns.items():
        serialized_table = f"CREATE TABLE {table_name} (\n"
        for column_name, column_type in table_columns:
            serialized_table += f"\t{column_name} {column_type.upper()},\n"
        serialized_table += ");"

        serialized_tables.append(serialized_table)

    serialized_schema = "\n\n".join(serialized_tables)

    return serialized_schema


def codes_serialization(
    db_id: str, db_sqlite_path: str, db_comments: Dict, include_fk: bool
):
    db_schema = get_db_schema(db_sqlite_path, db_comments, db_id)
    serialized_schema = get_db_schema_sequence(db_schema, include_fk=include_fk)
    return serialized_schema


def serialize_schema(
    schema_dict: Dict,
    db_sqlite_path: str,
    mode: Literal["simple", "verbose", "ddl", "codeS"] = "codeS",
    include_fk: bool = False,
    db_comments: Dict = {},
) -> str:
    """
    Generates a serialization of DB schema from a schema dict like the ones in
    a tables.json format of Spider/BIRD

    Args:
        schema_dict (Dict): The dictionary representation of a DB schema.
        db_sqlite_path (str): The path to the .sqlite file of the DB.
        mode (Literal["simple", "verbose", "ddl", "codeS"], optional): The mode
            to use when serializing the DB schema. Defaults to "codeS".
        include_fk (bool): Wheter or not to include foreign key information.
            Defaults to False.
        db_comments(Dict): Comment about column meanings (CodeS)

    Returns:
        str: The serialized representation of the DB schema.
    """

    # The database id
    db_id = schema_dict["db_id"]
    # A list of table names strings
    table_names = schema_dict["table_names_original"]
    # A lits of tuples (table_num, column name)
    column_names = schema_dict["column_names_original"]
    # A list of column types
    column_types = schema_dict["column_types"]

    # Get the columns that belong to each table
    table_to_columns: Dict[str, List[Tuple[str, str]]] = {}
    # table_to_colums = {table_name : [(column_name, column_type), ...]}
    for table_num, table_name in enumerate(table_names):
        table_to_columns[table_name] = [
            (column_name, column_type)
            for (cur_table_num, column_name), column_type in zip(
                column_names, column_types
            )
            if table_num == cur_table_num
        ]

    # Create the serialized schema string
    if mode == "simple":
        serialized_schema = simple_serialization(table_to_columns)
    elif mode == "verbose":
        serialized_schema = verbose_serialization(table_to_columns)
    elif mode == "ddl":
        serialized_schema = ddl_serialization(table_to_columns)
    elif mode == "codeS":
        serialized_schema = codes_serialization(
            db_id, db_sqlite_path, db_comments, include_fk
        )

    return serialized_schema


def serialize_db_schemas(
    tables_file_path: str, databases_dir_path: str
) -> Dict[str, str]:
    """
    Generates a serialized version of DB schemas from a path to a
    Spider/BIRD-like tables.json file.

    Args:
        tables_file_path (str): Path to a tables.json file containing DB schemas.
        databases_dir_path (str): Path to the databases directory associated with
            the given tables.json file. It should contain a directory for each
            database that contains a .sqlite file and a database_description
            directory with a .csv file for each table.

    Returns:
        Dict[str, str]: A dictionary where the keys are the db_ids and the values
            are the serialized DB schemas.
    """

    # Load schemas
    with open(tables_file_path, "r") as fp:
        db_schemas = json.load(fp)

    # Load DB comments
    db_comments = get_db_comments(tables_file_path)

    serialized_schemas = {}
    for schema in db_schemas:
        db_id = schema["db_id"]
        db_sqlite_path = path.join(databases_dir_path, f"{db_id}/{db_id}.sqlite")
        db_description_dir = path.join(
            databases_dir_path, f"{db_id}/database_description/"
        )
        serialized_schemas[db_id] = serialize_schema(
            schema, db_sqlite_path, db_comments=db_comments
        )

    return serialized_schemas


def parse_args():
    parser = argparse.ArgumentParser(
        description="Serialize DB schemas from Spider/BIRD tables.json files."
    )
    parser.add_argument(
        "-i", "--input_file", help="The input tables.json file,", required=True
    )
    parser.add_argument(
        "-d",
        "--databases_dir",
        help="The path to the databases directory associated with the tables.json file.",
        required=True,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    serialized_schemas = serialize_db_schemas(args.input_file, args.databases_dir)
    print(serialized_schemas["card_games"])
