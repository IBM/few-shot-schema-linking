import unittest
import json

from dec_sl.utils.sql import extract_tables_and_columns

# Load needed schemas for certain test cases
with open("dec_sl/tests/assets/schemas.json", "r") as fp:
    schemas = json.load(fp)


class TestExtractTablesAndColumns(unittest.TestCase):

    def test_extract_tables_and_columns(self):

        test_cases = [
            (
                0,
                (
                    "SELECT DATETIME() - T2.birthday age"
                    " FROM Player_Attributes AS t1 INNER JOIN Player AS t2 ON t1.player_api_id = t2.player_api_id"
                    " WHERE SUBSTR(t1.`date`, 1, 10) BETWEEN '2013-01-01' AND '2015-12-31' AND t1.sprint_speed >= 97"
                ),
                None,
                {"Player_Attributes", "Player"},
                {
                    "Player_Attributes.player_api_id",
                    "Player_Attributes.date",
                    "Player_Attributes.sprint_speed",
                    "Player.birthday",
                    "Player.player_api_id",
                },
            ),
            (
                1,
                (
                    "SELECT DISTINCT t4.team_long_name"
                    " FROM Team_Attributes AS t3 INNER JOIN Team AS t4 ON t3.team_api_id = t4.team_api_id"
                    " WHERE SUBSTR(t3.`date`, 1, 4) = '2012' AND t3.buildUpPlayPassing >"
                    " ( SELECT CAST(SUM(t2.buildUpPlayPassing) AS REAL) / COUNT(t1.id)"
                    " FROM Team AS t1 INNER JOIN Team_Attributes AS t2 ON t1.team_api_id = t2.team_api_id"
                    " WHERE SUBSTR(t2.`date`, 1, 4) = '2012' )"
                ),
                None,
                {"Team", "Team_Attributes"},
                {
                    "Team.team_long_name",
                    "Team.team_api_id",
                    "Team.id",
                    "Team_Attributes.team_api_id",
                    "Team_Attributes.date",
                    "Team_Attributes.buildUpPlayPassing",
                },
            ),
            (
                2,
                (
                    "SELECT COUNT(CDSCode) FROM schools WHERE City = 'San Joaquin'"
                    " AND MailState = 'CA' AND StatusType = 'Active'"
                ),
                None,
                {"schools"},
                {
                    "schools.CDSCode",
                    "schools.City",
                    "schools.MailState",
                    "schools.StatusType",
                },
            ),
            (
                3,
                (
                    "SELECT disp.account_id FROM card INNER JOIN disp ON"
                    " card.disp_id = disp.disp_id WHERE card.type IN ('gold', 'junior')"
                ),
                None,
                {"card", "disp"},
                {
                    "disp.account_id",
                    "disp.disp_id",
                    "card.disp_id",
                    "card.type",
                },
            ),
            # NOTE: in the following case `account_id` must be disabmiguated
            (
                4,
                (
                    "SELECT cast(sum(district.a2 = 'Decin') AS REAL) * 100 / count(account_id) FROM district INNER JOIN"
                    " ACCOUNT ON district.district_id = ACCOUNT.district_id WHERE strftime('%Y', ACCOUNT.date) = '1993'"
                ),
                schemas["financial"],
                {"district", "ACCOUNT"},
                {
                    "district.a2",
                    "district.district_id",
                    "ACCOUNT.account_id",
                    "ACCOUNT.district_id",
                    "ACCOUNT.date",
                },
            ),
            # NOTE: in the following case `a2` must be disabmiguated
            (
                5,
                (
                    "SELECT district.district_id FROM district INNER JOIN ACCOUNT ON district.district_id = ACCOUNT.district_id"
                    " INNER JOIN trans ON ACCOUNT.account_id = trans.account_id WHERE trans.type = 'VYDAJ' AND"
                    " ACCOUNT.date LIKE '1996-01%' ORDER BY a2 ASC LIMIT 10"
                ),
                schemas["financial"],
                {"district", "ACCOUNT", "trans"},
                {
                    "district.district_id",
                    "district.a2",
                    "ACCOUNT.account_id",
                    "ACCOUNT.district_id",
                    "ACCOUNT.date",
                    "trans.account_id",
                    "trans.type",
                },
            ),
            # NOTE: in the following case `oxygen_count` is not an actual column
            (
                6,
                (
                    "SELECT avg(oxygen_count) FROM (SELECT atom.molecule_id, count(atom.element) AS oxygen_count FROM"
                    " atom INNER JOIN bond ON atom.molecule_id = bond.molecule_id  WHERE bond.bond_type = '-' AND"
                    " atom.element = 'o'  GROUP BY atom.molecule_id) AS oxygen_counts"
                ),
                schemas["toxicology"],
                {"atom", "bond"},
                {
                    "atom.molecule_id",
                    "atom.element",
                    "bond.molecule_id",
                    "bond.bond_type",
                },
            ),
            # NOTE: in the following case `single_bond_count` is not an actual column
            (
                7,
                (
                    "SELECT avg(single_bond_count) FROM (SELECT molecule.molecule_id, count(bond.bond_type) AS single_bond_count"
                    " FROM bond  INNER JOIN atom ON bond.molecule_id = atom.molecule_id INNER JOIN molecule"
                    " ON molecule.molecule_id = atom.molecule_id WHERE bond.bond_type = '-' AND molecule.label = '+'"
                    " GROUP BY molecule.molecule_id) AS subquery"
                ),
                schemas["toxicology"],
                {"molecule", "atom", "bond"},
                {
                    "molecule.molecule_id",
                    "molecule.label",
                    "atom.molecule_id",
                    "bond.molecule_id",
                    "bond.bond_type",
                },
            ),
            # NOTE: In the following case `t` is not an actual table
            (
                8,
                (
                    "SELECT userid FROM ( SELECT userid, count(name) AS num FROM badges GROUP BY userid ) t WHERE t.num > 5"
                ),
                schemas["codebase_community"],
                {"badges"},
                {
                    "badges.userid",
                    "badges.name",
                },
            ),
            # NOTE: In the following case `name` must be disabmiguated
            (
                9,
                (
                    "SELECT name FROM badges INNER JOIN comments ON badges.userid = comments.userid"
                    " GROUP BY comments.userid ORDER BY count(comments.userid) DESC LIMIT 1"
                ),
                schemas["codebase_community"],
                {"badges", "comments"},
                {
                    "badges.userid",
                    "badges.name",
                    "comments.userid",
                },
            ),
            # NOTE: Example with CTEs that should not be confused for actual tables
            (
                10,
                (
                    "WITH time_in_seconds AS ( SELECT results.positionorder, CASE WHEN results.positionorder = 1 THEN"
                    " (cast(substr(results.time, 1, 1) AS REAL) * 3600) + (cast(substr(results.time, 3, 2) AS REAL) * 60)"
                    " + cast(substr(results.time, 6) AS REAL) ELSE cast(substr(results.time, 2) AS REAL) END AS time_seconds"
                    " FROM results INNER JOIN races ON results.raceid = races.raceid WHERE races.name = 'Australian Grand Prix'"
                    " AND results.time IS NOT NULL AND races.year = 2008 ), champion_time AS ( SELECT time_seconds"
                    " FROM time_in_seconds WHERE positionorder = 1), last_driver_incremental AS ( SELECT time_seconds"
                    " FROM time_in_seconds WHERE positionorder = (SELECT max(positionorder) FROM time_in_seconds) )"
                    " SELECT (cast((SELECT time_seconds FROM last_driver_incremental) AS REAL) * 100) / (SELECT time_seconds + (SELECT time_seconds FROM last_driver_incremental) FROM champion_time)"
                ),
                schemas["formula_1"],
                {"results", "races"},
                {
                    "results.positionorder",
                    "results.time",
                    "results.raceid",
                    "races.raceid",
                    "races.name",
                    "races.year",
                },
            ),
            # NOTE: Example with CTEs that should not be confused for actual tables
            (
                11,
                (
                    "WITH time_in_seconds AS ( SELECT races.year, races.raceid, results.positionorder, CASE WHEN"
                    " results.positionorder = 1 THEN (cast(substr(results.time, 1, 1) AS REAL) * 3600) +"
                    " (cast(substr(results.time, 3, 2) AS REAL) * 60) + cast(substr(results.time, 6) AS REAL) ELSE"
                    " cast(substr(results.time, 2) AS REAL) END AS time_seconds FROM results INNER JOIN races"
                    " ON results.raceid = races.raceid WHERE results.time IS NOT NULL ), champion_time AS"
                    " ( SELECT YEAR, raceid, time_seconds FROM time_in_seconds WHERE positionorder = 1 )"
                    " SELECT YEAR, avg(time_seconds) FROM champion_time GROUP BY YEAR HAVING avg(time_seconds) IS NOT NULL"
                ),
                schemas["formula_1"],
                {"results", "races"},
                {
                    "results.positionorder",
                    "results.time",
                    "results.raceid",
                    "races.raceid",
                    "races.year",
                },
            ),
            # NOTE: Another example with a CTE
            (
                12,
                (
                    "WITH lap_times_in_seconds AS ( SELECT driverid, (CASE WHEN instr(TIME, ':') <>"
                    " instr(substr(TIME, instr(TIME, ':') + 1), ':') + instr(TIME, ':') THEN"
                    " cast(substr(TIME, 1, instr(TIME, ':') - 1) AS REAL) * 3600 ELSE 0 END) +"
                    " (cast(substr(TIME, instr(TIME, ':') - 2 * (instr(TIME, ':') = instr(substr(TIME, instr(TIME, ':') + 1), ':')"
                    " + instr(TIME, ':')), instr(TIME, ':') - 1) AS REAL) * 60) + (cast(substr(TIME, instr(TIME, ':') + 1, instr(TIME, '.')"
                    " - instr(TIME, ':') - 1) AS REAL)) + (cast(substr(TIME, instr(TIME, '.') + 1) AS REAL) / 1000) AS time_in_seconds"
                    " FROM laptimes) SELECT drivers.forename, drivers.surname FROM ( SELECT driverid, min(time_in_seconds)"
                    " AS min_time_in_seconds FROM lap_times_in_seconds GROUP BY driverid) AS t1 INNER JOIN drivers ON"
                    " t1.driverid = drivers.driverid ORDER BY t1.min_time_in_seconds ASC LIMIT 1"
                ),
                schemas["formula_1"],
                {"laptimes", "drivers"},
                {
                    "laptimes.driverid",
                    "laptimes.time",
                    "drivers.forename",
                    "drivers.surname",
                    "drivers.driverid",
                },
            ),
            # NOTE: Another example with a CTE
            (
                13,
                (
                    "SELECT `date` FROM ( SELECT player_attributes.crossing, player_attributes.`date` FROM player"
                    " INNER JOIN player_attributes ON player.player_fifa_api_id = player_attributes.player_fifa_api_id"
                    " WHERE player.player_name = 'Kevin Constant' ORDER BY player_attributes.crossing DESC) ORDER BY date DESC LIMIT 1"
                ),
                None,
                {"player", "player_attributes"},
                {
                    "player.player_fifa_api_id",
                    "player.player_name",
                    "player_attributes.player_fifa_api_id",
                    "player_attributes.date",
                    "player_attributes.crossing",
                },
            ),
            (
                14,
                (
                    "SELECT a FROM ( SELECT avg(finishing) RESULT, 'Max' a FROM player INNER JOIN player_attributes ON"
                    " player.player_api_id = player_attributes.player_api_id WHERE player.height = ( SELECT max(height)"
                    " FROM player ) UNION SELECT avg(finishing) RESULT, 'Min' a FROM player INNER JOIN player_attributes"
                    " ON player.player_api_id = player_attributes.player_api_id WHERE player.height = ( SELECT min(height)"
                    " FROM player ) ) ORDER BY RESULT DESC LIMIT 1"
                ),
                schemas["european_football_2"],
                {"player", "player_attributes"},
                {
                    "player.height",
                    "player.player_api_id",
                    "player_attributes.player_api_id",
                    "player_attributes.finishing",
                },
            ),
        ]

        for (
            i,
            sql_query,
            schema,
            ground_truth_tables,
            ground_truth_columns,
        ) in test_cases:
            with self.subTest(i=i):
                tables, columns = extract_tables_and_columns(sql_query, schema=schema)
                # Lower-case everything
                tables = set([x.lower() for x in tables])
                columns = set([x.lower() for x in columns])
                ground_truth_tables = set([x.lower() for x in ground_truth_tables])
                ground_truth_columns = set([x.lower() for x in ground_truth_columns])
                # Check assertions
                self.assertSetEqual(set(tables), ground_truth_tables)
                self.assertSetEqual(set(columns), ground_truth_columns)


if __name__ == "__main__":
    unittest.main()
