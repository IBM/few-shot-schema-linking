from typing import List, Optional, Tuple

simple_decomposition_prompt_template = (
    # Instructions
    """\
You goal is to decompose the following question into simpler questions that when joined will match the original.

Carefully consider the following guidelines:
- Make sure that no question can be further decomposed.
- Try to avoid any overlap between the decomposed questions, if necessary it is preferable to refer to a previous question than to repeat it.
- Do not add any explanations or comments. Your answer should only be a list of decomposed questions.

Your answer should follow this template:

Decomposed questions:

- Question 1
- Question 2
- ...
""",
    # Question
    """\
Decompose the following question:

{nl_question}
""",
)

decomposition_prompt_template = (
    # Instructions
    """\
You goal is to decompose the a question into simpler questions that when joined will still form the original.

Carefully consider the following guidelines:
- Start by rephrasing the question in a simpler way that will help you achieve better decomposition.
- Decompose the question into simpler questions.
- Make sure that no question can be further decomposed.
- Try to avoid any overlap between the decomposed questions, if necessary you must to refer to a previous question instead of repeating it.
- Reformulate the decomposed questions, re-ordering where necessary, so that simpler questions are first and more complex question come later and use the previous questions when necessary.
- Do not add any explanations or comments. Only follow the template bellow.

Your answer should follow this template:

Rephrased question: ...

Preliminary decomposition:

- Question 1
- Question 2
- ...

Decomposed questions:

- Question 1
- Question 2
- ...
""",
    # Question
    """\
Decompose the following question:

{nl_question}
""",
)

simple_decomposition_prompt_template = (
    # Instructions
    """\
Your goal is to decompose a question into simpler sub-questions.
""",
    # Question
    """\
{nl_question}
""",
)

schema_linking_prompt_template = (
    # Instructions
    """\
You are an expert database assistant.
You will be given a Natural Language Question and a relational database schema.
Your goal is to identify the schema elements (tables and columns) of the schema that are necessary to translate the given Natural Language Question into a SQL query.
Identify the tables and columns of the schema that are needed to build the equivalent SQL query. 
Do not create the SQL query, only identify the necessary tables and columns.
Create a json object in the following format:

```json
{"tables": ["table_name", "table_name", ...], "columns": ["column_name", "column_name", ...]}
```
""",
    # Question
    """\
For the following database schema:

{serialized_db_schema}

Find the the tables and columns needed to answer the following natural language question:

{nl_question}

Which are the needed tables and columns?
""",
)

text_to_sql_prompt_template = (
    # Instuctions
    """\
You are a database expert that always generates correct and executable SQLite queries. 
You have been asked to write a SQL query based on a natural language question and the database schema.
""",
    # Question
    """\
For the following database schema:

{serialized_db_schema}

Translate this natural language question into a SQL query:

{nl_question}
""",
)


def create_decomposition_prompt(
    nl_question: str, simple: bool = False
) -> Tuple[str, str]:
    """Creates a LLM prompt for decomposing a NL Question."""
    # Get task templates
    if simple:
        instructions, question = simple_decomposition_prompt_template
    else:
        instructions, question = decomposition_prompt_template
    # Add instance parameters to create the prompt
    question = question.format(nl_question=nl_question)

    return instructions, question


def create_schema_linking_prompt(
    serialized_db_schema: str,
    nl_question: str,
    decomposed_questions: Optional[List[str]] = None,
) -> str:
    """Creates a LLM prompt for decomposing a NL Question."""
    # Get task templates
    instructions, question = schema_linking_prompt_template
    # Add instance parameters to create the prompt
    question = question.format(
        serialized_db_schema=serialized_db_schema, nl_question=nl_question
    )

    return instructions, question


def create_text_to_sql_prompt(
    serialized_db_schema: str, nl_question: str
) -> Tuple[str, str]:
    """Creates a LLM prompt for Text-to-SQL."""
    instructions, question = text_to_sql_prompt_template

    question = question.format(
        serialized_db_schema=serialized_db_schema,
        nl_question=nl_question,
    )

    return instructions, question


def render_list(item_list: List[str]) -> str:
    """Render a list of items into a readable format."""
    rendered_list = "\n".join([f"- {item}" for item in item_list])
    return rendered_list
