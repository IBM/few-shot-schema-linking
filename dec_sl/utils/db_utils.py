"""Usefull DB utilities.

Adapted from the CodeS github repositiry. 
https://github.com/RUCKBReasoning/codes/blob/main/utils/db_utils.py
"""

import os
import sqlite3
import json
from typing import List, Optional

from func_timeout import func_set_timeout, FunctionTimedOut


# get the database cursor for a sqlite database path
def get_cursor_from_path(sqlite_path):
    try:
        if not os.path.exists(sqlite_path):
            print("Openning a new connection %s" % sqlite_path)
        connection = sqlite3.connect(sqlite_path, check_same_thread=False)
    except Exception as e:
        print(sqlite_path)
        raise e
    connection.text_factory = lambda b: b.decode(errors="ignore")
    cursor = connection.cursor()
    return cursor


# execute predicted sql with a time limitation
@func_set_timeout(200)
def execute_sql(cursor, sql):
    cursor.execute(sql)

    return cursor.fetchall()


# execute predicted sql with a long time limitation (for buiding content index)
@func_set_timeout(2000)
def execute_sql_long_time_limitation(cursor, sql):
    cursor.execute(sql)

    return cursor.fetchall()


def check_sql_executability(generated_sql, db):
    if generated_sql.strip() == "":
        return "Error: empty string"
    try:
        cursor = get_cursor_from_path(db)
        # use `EXPLAIN QUERY PLAN` to avoid actually executing
        execute_sql(cursor, "EXPLAIN QUERY PLAN " + generated_sql)
        execution_error = None
    except FunctionTimedOut as fto:
        print("SQL execution time out error: {}.".format(fto))
        execution_error = "SQL execution times out."
    except Exception as e:
        print("SQL execution runtime error: {}.".format(e))
        execution_error = str(e)

    return execution_error


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def detect_special_char(name):
    for special_char in ["(", "-", ")", " ", "/"]:
        if special_char in name:
            return True

    return False


def add_quotation_mark(s):
    return "`" + s + "`"


def get_column_contents(column_name, table_name, cursor):
    select_column_sql = (
        "SELECT DISTINCT `{}` FROM `{}` WHERE `{}` IS NOT NULL LIMIT 2;".format(
            column_name, table_name, column_name
        )
    )
    results = execute_sql_long_time_limitation(cursor, select_column_sql)
    column_contents = [str(result[0]).strip() for result in results]
    # remove empty and extremely-long contents
    column_contents = [
        content
        for content in column_contents
        if len(content) != 0 and len(content) <= 25
    ]

    return column_contents


def remove_similar_comments(names, comments):
    """
    Remove table (or column) comments that have a high degree of similarity with their names

    Arguments:
        names: a list of table (or column) names
        comments: a list of table (or column) comments

    Returns:
        new_comments: a list of new table (or column) comments
    """
    new_comments = []
    for name, comment in zip(names, comments):
        if name.replace("_", "").replace(" ", "") == comment.replace("_", "").replace(
            " ", ""
        ):
            new_comments.append("")
        else:
            new_comments.append(comment)

    return new_comments


def get_db_comments(table_json_path):
    additional_db_info = json.load(open(table_json_path))

    db_comments = dict()
    # record comments for tables and columns
    for db_info in additional_db_info:
        comment_dict = dict()

        column_names = [
            column_name.lower() for _, column_name in db_info["column_names_original"]
        ]
        table_idx_of_each_column = [
            t_idx for t_idx, _ in db_info["column_names_original"]
        ]
        column_comments = [
            column_comment.lower() for _, column_comment in db_info["column_names"]
        ]

        assert len(column_names) == len(column_comments)
        column_comments = remove_similar_comments(column_names, column_comments)

        table_names = [
            table_name.lower() for table_name in db_info["table_names_original"]
        ]
        table_comments = [
            table_comment.lower() for table_comment in db_info["table_names"]
        ]

        assert len(table_names) == len(table_comments)
        table_comments = remove_similar_comments(table_names, table_comments)

        # enumerate each table and its columns
        for table_idx, (table_name, table_comment) in enumerate(
            zip(table_names, table_comments)
        ):
            comment_dict[table_name] = {
                "table_comment": table_comment,
                "column_comments": dict(),
            }
            for t_idx, column_name, column_comment in zip(
                table_idx_of_each_column, column_names, column_comments
            ):
                # record columns in current table
                if t_idx == table_idx:
                    comment_dict[table_name]["column_comments"][
                        column_name
                    ] = column_comment

        db_comments[db_info["db_id"]] = comment_dict

    return db_comments


def filter_schema(schema, table_labels, column_labels):
    # Filter out tables and columns based on schema linking labels

    filtered_schema_items = []

    for table, table_label, table_column_labels in zip(
        schema["schema_items"], table_labels, column_labels
    ):
        if table_label == 0:
            # Skip the table if it is not in the schema linking predictions
            continue
        else:
            filtered_table = table.copy()
            # Make a list of indexes of columns we want to keep
            keep_column_indexes = [
                i for i, label in enumerate(table_column_labels) if label == 1
            ]

            # Keep only information about columns we want to keep
            filtered_table["column_names"] = []
            filtered_table["column_types"] = []
            filtered_table["column_comments"] = []
            filtered_table["column_contents"] = []
            filtered_table["pk_indicators"] = []

            for i in keep_column_indexes:
                filtered_table["column_names"].append(table["column_names"][i])
                filtered_table["column_types"].append(table["column_types"][i])
                filtered_table["column_comments"].append(table["column_comments"][i])
                filtered_table["column_contents"].append(table["column_contents"][i])
                filtered_table["pk_indicators"].append(table["pk_indicators"][i])

            filtered_schema_items.append(filtered_table)

    # Keep foreign keys only if both columns are kept in the filtered schema
    flat_column_names = [
        f"{schema_item['table_name']}.{column_name}"
        for schema_item in filtered_schema_items
        for column_name in schema_item["column_names"]
    ]

    filtered_foreign_keys = [
        [t1, c1, t2, c2]
        for t1, c1, t2, c2 in schema["foreign_keys"]
        if f"{t1}.{c1}" in flat_column_names and f"{t2}.{c2}" in flat_column_names
    ]

    # Update schema with filtered information
    filtered_schema = schema.copy()
    filtered_schema["schema_items"] = filtered_schema_items
    filtered_schema["foreign_keys"] = filtered_foreign_keys

    return filtered_schema


def get_db_schema_sequence(
    schema,
    include_fk: bool,
    include_content: bool = True,
    table_labels: Optional[List[int]] = None,
    column_labels: Optional[List[List[int]]] = None,
):
    """Create the serialized string sequence of the DB schema.

    Args:
        include_fk (bool): Wheter or not to include foreign key information.
    """
    if table_labels is not None and column_labels is not None:
        schema = filter_schema(schema, table_labels, column_labels)

    schema_sequence = ""

    for table in schema["schema_items"]:
        table_name, table_comment = table["table_name"], table["table_comment"]
        if detect_special_char(table_name):
            table_name = add_quotation_mark(table_name)

        if table_comment != "":
            table_name += " ( comment : " + table_comment + " )"

        column_info_list = []
        for (
            column_name,
            column_type,
            column_comment,
            column_content,
            pk_indicator,
        ) in zip(
            table["column_names"],
            table["column_types"],
            table["column_comments"],
            table["column_contents"],
            table["pk_indicators"],
        ):
            if detect_special_char(column_name):
                column_name = add_quotation_mark(column_name)
            additional_column_info = []
            # column type
            additional_column_info.append(column_type)
            # pk indicator
            if pk_indicator != 0:
                additional_column_info.append("primary key")
            # column comment
            if column_comment != "":
                additional_column_info.append("comment : " + column_comment)
            # representive column values
            if len(column_content) != 0 and include_content:
                additional_column_info.append("values : " + " , ".join(column_content))

            column_info_list.append(
                table_name
                + "."
                + column_name
                + " ( "
                + " | ".join(additional_column_info)
                + " )"
            )

        schema_sequence += (
            "table "
            + table_name
            + " , columns = [ "
            + " , ".join(column_info_list)
            + " ]\n"
        )

    if include_fk:
        if len(schema["foreign_keys"]) != 0:
            schema_sequence += "foreign keys :\n"
            for foreign_key in schema["foreign_keys"]:
                for i in range(len(foreign_key)):
                    if detect_special_char(foreign_key[i]):
                        foreign_key[i] = add_quotation_mark(foreign_key[i])
                schema_sequence += "{}.{} = {}.{}\n".format(
                    foreign_key[0], foreign_key[1], foreign_key[2], foreign_key[3]
                )
        else:
            schema_sequence += "foreign keys : None\n"

    return schema_sequence.strip()


def get_matched_content_sequence(matched_contents):
    content_sequence = ""
    if len(matched_contents) != 0:
        content_sequence += "matched contents :\n"
        for tc_name, contents in matched_contents.items():
            table_name = tc_name.split(".")[0]
            column_name = tc_name.split(".")[1]
            if detect_special_char(table_name):
                table_name = add_quotation_mark(table_name)
            if detect_special_char(column_name):
                column_name = add_quotation_mark(column_name)

            content_sequence += (
                table_name + "." + column_name + " ( " + " , ".join(contents) + " )\n"
            )
    else:
        content_sequence = "matched contents : None"

    return content_sequence.strip()


def get_db_schema(db_path, db_comments, db_id):
    if db_id in db_comments:
        db_comment = db_comments[db_id]
    else:
        db_comment = None

    cursor = get_cursor_from_path(db_path)

    # obtain table names
    results = execute_sql(cursor, "SELECT name FROM sqlite_master WHERE type='table';")
    table_names = [result[0].lower() for result in results]

    schema = dict()
    schema["schema_items"] = []
    foreign_keys = []
    # for each table
    for table_name in table_names:
        # skip SQLite system table: sqlite_sequence
        if table_name == "sqlite_sequence":
            continue
        # obtain column names in the current table
        results = execute_sql(
            cursor,
            "SELECT name, type, pk FROM PRAGMA_TABLE_INFO('{}')".format(table_name),
        )
        column_names_in_one_table = [result[0].lower() for result in results]
        column_types_in_one_table = [result[1].lower() for result in results]
        pk_indicators_in_one_table = [result[2] for result in results]

        column_contents = []
        for column_name in column_names_in_one_table:
            column_contents.append(get_column_contents(column_name, table_name, cursor))

        # obtain foreign keys in the current table
        results = execute_sql(
            cursor, "SELECT * FROM pragma_foreign_key_list('{}');".format(table_name)
        )
        for result in results:
            if None not in [result[3], result[2], result[4]]:
                foreign_keys.append(
                    [
                        table_name.lower(),
                        result[3].lower(),
                        result[2].lower(),
                        result[4].lower(),
                    ]
                )

        # obtain comments for each schema item
        if db_comment is not None:
            if table_name in db_comment:  # record comments for tables and columns
                table_comment = db_comment[table_name]["table_comment"]
                column_comments = [
                    (
                        db_comment[table_name]["column_comments"][column_name]
                        if column_name in db_comment[table_name]["column_comments"]
                        else ""
                    )
                    for column_name in column_names_in_one_table
                ]
            else:  # current database has comment information, but the current table does not
                table_comment = ""
                column_comments = ["" for _ in column_names_in_one_table]
        else:  # current database has no comment information
            table_comment = ""
            column_comments = ["" for _ in column_names_in_one_table]

        schema["schema_items"].append(
            {
                "table_name": table_name,
                "table_comment": table_comment,
                "column_names": column_names_in_one_table,
                "column_types": column_types_in_one_table,
                "column_comments": column_comments,
                "column_contents": column_contents,
                "pk_indicators": pk_indicators_in_one_table,
            }
        )

    schema["foreign_keys"] = foreign_keys

    return schema
