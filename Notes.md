## SQL Queries with Unecessary Tables

In some examples the SQL query includes more tables than what is necessary, 
leading to decreased recall scores when the schema linker does not identify them.

Examples in the BIRD dev set:

- Example #87: The `frpm` table is JOINed but never used and removing it does not change the result.
- Example #99: The `disp` table is JOINed but never used and removing it does not change the result.
- Example #158: The `district` table is JOINed to get the district_id of accounts, 
but this id also exists in the `account` table as a FK.
- Example #365: The `foreign_data` table is JOINed but neber used and removing it does not change the result.
- Example #441: It is not really clear why the `set_translations` table is used.
- Example #442: It is not really clear why the `set_translations` table is used.
- Example #443: It is not really clear why the `set_translations` table is used.
- Example #447: It is not really clear why the `set_translations` table is used.
- Example #594: The SQL seems to not match the NL.
- Example #970: NL refers to lap times but SQL uses pit stop time
- Example #973: NL refers to lap times but SQL uses pit stop time
- Example #1240: The `patient` table is JOINed but neber used and removing it does not change the result.
- Example #1245: The `patient` table is JOINed but neber used and removing it does not change the result.


## Unclear Connection Between Tables

- In the `financial` dev database there is no clearly diffened connection path
(i.e., FK relations) between the tables `client` and `loan`.
This can be seen in example #113.