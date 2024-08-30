from typing import Tuple, List, Dict
import sqlglot
import sqlglot.expressions as exp

# NOTE: Maybe using qualify could also help
# import sqlglot.optimizer.qualify
# sqlglot.optimizer.qualify.qualify()


def extract_tables_and_columns(
    sql_query: str, schema: Dict = None, dialect: str = "sqlite"
) -> Tuple[List[str], List[str]]:

    def get_subquery_tables_and_columns(expression, cte_aliases):
        # Get tables and check that they are not CTE aliases
        tables = [
            t.name.lower()
            for t in expression.find_all(exp.Table)
            if not t.name.lower() in cte_aliases
        ]

        # Get table aliases for later disambiguation
        table_aliases = {
            t.alias.lower(): t.name.lower()
            for t in expression.find_all(exp.Table)
            if t.alias != ""
        }

        # Get columns
        columns = []
        for c in expression.find_all(exp.Column):
            column_name = c.name.lower()
            table_name_or_alias = c.table.lower()

            if table_name_or_alias == "":
                # If table name is an empty string, there should only be one table
                if len(tables) == 1:
                    table_name = tables[0]
                else:
                    # If there are more tables, we try to disambiguate the column
                    table_name = ""
                    if schema:
                        for table in schema["schema_items"]:
                            if (
                                column_name in table["column_names"]
                                and table["table_name"] in tables
                            ):
                                table_name = table["table_name"]
                                break
                    # If we can not find the column, then it might be from an intermediate result and we can drop it
                    if table_name == "":
                        continue
            elif table_name_or_alias in table_aliases:
                # If an alias is found then use it
                table_name = table_aliases[table_name_or_alias]
            elif table_name_or_alias in tables:
                # If an alias is not found then we should have the original talbe name
                table_name = table_name_or_alias
            else:
                # If the table name is still not found, this is probably an alias of an intermediate result and we can drop it
                continue

            columns.append(f"{table_name}.{column_name}")

        return tables, columns

    # Parse the query
    expression = sqlglot.parse_one(sql_query, read=dialect)

    # Get CTE aliases to avoid potentially mistaking them for schema tables
    cte_aliases = [cte.alias for cte in expression.find_all(exp.CTE)]

    # Work on each CTE and sub-query separately to avoid confusing names between scopes
    # Traverse using DFS and reverse order to start from most nested sub-query
    sub_queries = list(expression.find_all((exp.Subquery, exp.CTE), bfs=False))
    sub_queries.reverse()
    # Also add the root query to the list
    sub_queries.append(expression)

    tables = []
    columns = []
    for sub_query in sub_queries:
        sub_tables, sub_columns = get_subquery_tables_and_columns(
            sub_query, cte_aliases
        )
        # Remove subquery from its AST to avoid duplicate processing
        sub_query.pop()
        # Add tables and column from sub-query to global lists
        tables.extend(sub_tables)
        columns.extend(sub_columns)

    # Avoid duplicate elements
    tables = list(set(tables))
    columns = list(set(columns))

    return tables, columns
