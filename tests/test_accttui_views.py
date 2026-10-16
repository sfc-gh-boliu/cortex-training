# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A job's location (rows from the account-wide list): its label, search and preview."""

from __future__ import annotations

from datetime import datetime
from datetime import timezone

import pytest

from cortex_training.tui import views as v

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


def _job(db=None, schema=None, **extra) -> dict:
    job = {"job_id": "11111111-2222", "status": "RUNNING", "created_at": "2026-10-01T10:00:00Z", **extra}
    if db is not None:
        job["database_name"] = db
    if schema is not None:
        job["schema_name"] = schema
    return job


@pytest.mark.parametrize("db, schema, label", [
    ("CORTEX_TRAINING_DB", "PUBLIC", "CORTEX_TRAINING_DB.PUBLIC"),
    ("ml", "dev", '"ml"."dev"'),  # lowercase: quoted, as SQL resolves it
    ("Team DB", 'we"ird', '"Team DB"."we""ird"'),  # a space, an embedded quote doubled
    ("TEAM_DB", "EXP$1", "TEAM_DB.EXP$1"),
])
def test_the_label_quotes_each_name_as_sql_needs(db, schema, label):
    job = _job(db, schema)
    assert v.location_names(job) == (db, schema)
    assert v.location_label(job) == label


@pytest.mark.parametrize("job", [_job(), _job("DB"), _job(None, "S"), _job("", "S"), _job("DB", 7), "row", None])
def test_a_row_without_both_names_has_no_location(job):
    assert v.location_names(job) is None and v.location_label(job) == ""


def test_the_label_never_carries_control_characters():
    label = v.location_label(_job("db\x1b[31m", "s‮"))
    assert "\x1b" not in label and "‮" not in label


def test_plain_words_match_the_location():
    team, lower, plain = _job("TEAM_DB", "EXPERIMENTS"), _job("ml", "dev"), _job()
    for words, matches in (("team_db", [team]), ("experiments", [team]), ("team_db.exp", [team]),
                           ("team", [team]), ('"ml"', [lower]), ("dev", [lower]),
                           # As the Location column shows it, and bare: the search box's shell-style
                           # quoting turns a typed "ml"."dev" into ml.dev.
                           ('"ml"."dev"', [lower]), ("ml.dev", [lower]), ("""'"ml"."dev"'""", [lower])):
        query = v.parse_query(words)
        assert [j for j in (team, lower, plain) if v.job_matches_query(j, query, "")] == matches, words


def test_a_word_matches_the_start_of_a_location_name_not_its_middle():
    # The documented database CORTEX_TRAINING_DB contains "training": a word in the
    # middle of a name must not match every job in it.
    sampling = _job("CORTEX_TRAINING_DB", "PUBLIC", sub_jobs=[{"sub_job_id": "x:sampling:0", "job_type": "SAMPLING"}])
    training = _job("CORTEX_TRAINING_DB", "PUBLIC", sub_jobs=[{"sub_job_id": "y:training:0", "job_type": "TRAINING"}])
    for words, matches in (("training", [training]), ("cortex", [sampling, training]),
                           ("public", [sampling, training]), ("cortex_training_db.pub", [sampling, training]),
                           ("ublic", [])):
        query = v.parse_query(words)
        assert [j for j in (sampling, training) if v.job_matches_query(j, query, "")] == matches, words


@pytest.mark.parametrize("text, expected", [
    ("db:team", ["team"]),
    ("db=TEAM_DB", ["team"]),
    ("database:team_db", ["team"]),
    ("database_name=team_db", ["team"]),
    ("schema:exp", ["team"]),
    ("schema_name=EXPERIMENTS", ["team"]),
    ("db=ml schema=dev", ["lower"]),
    ("""db='"ml"'""", ["lower"]),  # the quoted spelling matches too
    ("db!=TEAM_DB", ["lower", "plain"]),  # a row without a location isn't in TEAM_DB either
    ("db:nowhere", []),
    # = and != follow SQL identifier rules: a bare name is its uppercase form, so
    # ML is not "ml"; the name written exactly as stored matches too.
    ("db=ML", []),
    ("schema=DEV", []),
    ("db=team_db", ["team"]),
    ("db!=ML", ["team", "lower", "plain"]),
    ("""db!='"ml"'""", ["team", "plain"]),
    ("db:ML", ["lower"]),  # ':' stays a case-insensitive part of the name
])
def test_db_and_schema_fields(text, expected):
    jobs = {"team": _job("TEAM_DB", "EXPERIMENTS"), "lower": _job("ml", "dev"), "plain": _job()}
    query = v.parse_query(text)
    assert not query.errors
    assert [name for name, job in jobs.items() if v.job_matches_query(job, query, "")] == expected


@pytest.mark.parametrize("text, expected", [
    ("db=CORTEX_TRAINING_DB", ["home", "public"]),
    ("db:cortex", ["home", "public"]),
    ("schema=public", ["home", "public"]),
    ("db!=CORTEX_TRAINING_DB", ["team"]),
    ("db=TEAM_DB", ["team"]),
])
def test_rows_without_a_location_are_in_the_configured_one_for_db_and_schema(text, expected):
    # The configured schema's own list (the fallback, --schema-only) names no location.
    jobs = {"home": _job(), "public": _job("CORTEX_TRAINING_DB", "PUBLIC"), "team": _job("TEAM_DB", "EXPERIMENTS")}
    query = v.parse_query(text)
    home = ("CORTEX_TRAINING_DB", "PUBLIC")
    assert [name for name, job in jobs.items() if v.job_matches_query(job, query, "", home=home)] == expected
    # Plain words don't: the list shows no location for those rows.
    assert not v.job_matches_query(jobs["home"], v.parse_query("public"), "", home=home)


def test_the_new_fields_are_listed_for_the_search_help():
    assert {"db", "schema"} <= set(v.SEARCH_FIELDS)


def test_the_no_match_message_says_when_the_list_holds_only_the_newest_jobs():
    query = v.parse_query("sweep")
    whole = v.no_match_message("sweep", query, 120)
    assert whole.startswith("No jobs match “sweep” among 120 job(s) in the last 6 months.")
    assert "more exist" not in whole and "location's database or schema name" in whole
    cut = v.no_match_message("sweep", query, 2003, cut="the newest 2,000 of the last 6 months and every active "
                             "job; more exist")
    assert cut.startswith("No jobs match “sweep” among the 2,003 job(s) loaded: the newest 2,000 of the last 6 "
                          "months and every active job; more exist.")
    assert "Search covers only the loaded jobs" in cut and "--schema-only" in cut


@pytest.mark.parametrize("identifier, name", [
    ("TEAM_DB", "TEAM_DB"), ("team_db", "TEAM_DB"), ('"ml"', "ml"), (' "ml" ', "ml"), ('"we""ird"', 'we"ird'),
    ('"Team DB"', "Team DB"),
])
def test_an_identifier_resolves_by_sql_rules_and_back(identifier, name):
    assert v.identifier_name(identifier) == name
    assert v.identifier_name(v.sql_identifier(name)) == name


def test_a_clients_location_is_its_resolved_names():
    class Client:
        database, schema = "cortex_training_db", '"dev"'

    assert v.client_location(Client()) == ("CORTEX_TRAINING_DB", "dev")
    assert v.client_location(object()) is None


@pytest.mark.parametrize("location, command", [
    ({}, "cortex-training download-log JOB"),
    ({"database": "TEAM_DB", "schema": "EXPERIMENTS"},
     "cortex-training --database TEAM_DB --schema EXPERIMENTS download-log JOB"),
    ({"database": '"ml"', "schema": '"dev"', "connection": "training"},
     """cortex-training --connection training --database '"ml"' --schema '"dev"' download-log JOB"""),
    ({"database": "DB; rm -rf ~", "schema": None, "connection": ""},
     "cortex-training --database 'DB; rm -rf ~' download-log JOB"),
])
def test_cli_commands_name_the_jobs_location_quoted_for_the_shell(location, command):
    assert v.cli_command("download-log JOB", **location) == command


def test_the_preview_shows_the_location_after_the_job_id():
    labels = [label for label, _ in v.summary_fields(_job("ml", "dev"), "a job", NOW)]
    assert labels[:3] == ["Name", "Job ID", "Location"]
    assert dict(v.summary_fields(_job("ml", "dev"), "a job", NOW))["Location"] == '"ml"."dev"'
    assert "Location" not in dict(v.summary_fields(_job(), "a job", NOW))  # the configured schema's own rows


def test_a_fresh_get_keeps_the_rows_location():
    row = _job("TEAM_DB", "EXPERIMENTS", reason="quota_filled")
    fresh = {"job_id": row["job_id"], "status": "RUNNING"}  # a per-schema get names no location
    assert v.location_label(v.refresh_job(row, fresh)) == "TEAM_DB.EXPERIMENTS"
