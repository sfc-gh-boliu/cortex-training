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

"""Pure view rules for job names, UNKNOWN rows, requested GPUs and server reasons (no textual)."""

from datetime import datetime
from datetime import timedelta
from datetime import timezone

from cortex_training.tui import views as v

NOW = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)
JID = "fake-job-0001"


def _iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _job(**extra):
    job = {"job_id": JID, "status": "RUNNING", "created_at": _iso(NOW - timedelta(hours=2)),
           "updated_at": _iso(NOW - timedelta(minutes=1)), "hardware": "H200", "comment": None,
           "sub_jobs": [{"sub_job_id": f"{JID}:training:0", "job_type": "TRAINING", "model_name": "Qwen/Qwen3-8B",
                         "training_config": {"n_gpus": 8.0}}]}
    job.update(extra)
    return job


def _unknown(sub_jobs=None, reason="Job status unavailable: Control Plane is unavailable"):
    """The service's UNKNOWN fallback row: configs only, no sub-job ids, no times."""
    if sub_jobs is None:
        sub_jobs = [
            {"status": "UNKNOWN", "job_type": "TRAINING", "model_name": "Qwen/Qwen3-8B", "training_config": {"n_gpus": 8}},
            {"status": "UNKNOWN", "job_type": "SAMPLING", "model_name": "Qwen/Qwen3-8B", "inference_config": {"n_gpus": 2}},
            {"status": "UNKNOWN", "job_type": "SAMPLING", "model_name": "Qwen/Qwen3-8B", "inference_config": {"n_gpus": 2}},
        ]
    return {"job_id": JID, "status": "UNKNOWN", "reason": reason, "comment": "ablation [/b] no warmup",
            "owner_role": "ML", "sub_jobs": sub_jobs}


# ─── names ───────────────────────────────────────────────────────────────────


def test_display_name_is_the_alias_then_the_comments_first_line_then_the_id():
    job = _job(comment="  qwen3-8b sweep #3  \n lr 1e-4, warmup 100 ")
    assert v.display_name(job) == "qwen3-8b sweep #3 …"  # more lines follow
    assert v.display_name(job, {JID: "my alias"}) == "my alias"
    assert v.display_name(_job()) == "" and v.shown_name(_job()) == JID
    assert v.shown_name(job, {JID: "my alias"}) == "my alias"
    assert v.display_name(_job(comment="\n\n  \n")) == ""


def test_one_line_trims_collapses_and_ellipsizes():
    assert v.one_line("a   b\tc") == "a b c"
    long = "x" * 200
    assert v.one_line(long) == "x" * 79 + "…" and len(v.one_line(long)) == v.NAME_WIDTH
    assert v.one_line("short\nmore") == "short …"
    assert len(v.one_line("y" * 79 + "\nmore")) <= v.NAME_WIDTH


def test_comment_is_printable_and_whole():
    job = _job(comment="line one\r\nline\ttwo \x1b[31mred\x07 \x9b2J")
    text = v.comment_of(job)
    assert text.split("\n") == ["line one", "line    two �[31mred� �2J"]
    assert "\x1b" not in v.display_name(job) and "\x9b" not in text
    assert v.comment_of({"comment": None}) == "" and v.comment_of({}) == ""


def test_search_matches_alias_comment_and_id_case_insensitively():
    job = _job(comment="Contract-Extraction SFT\nfrozen March export")
    aliases = {JID: "my run"}
    name = v.display_name(job, aliases)
    assert name == "my run"
    for words in ("MY RUN", "contract-extraction", "MARCH EXPORT", JID[:8].upper(), "name:extraction", "name:my"):
        assert v.job_matches_query(job, v.parse_query(words), name), words
    assert not v.job_matches_query(job, v.parse_query("april"), name)
    assert v.SEARCH_FIELDS["name"] == "display name or comment"
    assert "comment" in v.no_match_message("april", v.parse_query("april"), 3)


# ─── UNKNOWN rows ────────────────────────────────────────────────────────────


def test_unknown_rows_get_ids_derived_by_the_control_planes_rule():
    subs = v.sub_jobs(_unknown())
    assert [s["sub_job_id"] for s in subs] == [f"{JID}:training:0", f"{JID}:sampling:0", f"{JID}:sampling:1"]
    assert [v.sub_job_label(s) for s in subs] == ["training #0", "sampling #0", "sampling #1"]
    log_prob = _unknown([{"job_type": "LOG_PROBABILITY", "inference_config": {"n_gpus": 1}}])
    assert v.sub_jobs(log_prob)[0]["sub_job_id"] == f"{JID}:log_prob:0"


def test_ids_are_not_derived_without_enough_data():
    untyped = _unknown([{"status": "UNKNOWN", "model_name": "m"}, {"status": "UNKNOWN", "job_type": "SAMPLING"}])
    assert all("sub_job_id" not in s for s in v.sub_jobs(untyped))
    partial = _job(sub_jobs=[{"sub_job_id": "x:training:0", "job_type": "TRAINING"}, {"job_type": "SAMPLING"}])
    assert [s.get("sub_job_id") for s in v.sub_jobs(partial)] == ["x:training:0", None]
    no_id = {**_unknown(), "job_id": ""}
    assert all("sub_job_id" not in s for s in v.sub_jobs(no_id))
    assert v.sub_jobs(_unknown([])) == []


def test_unknown_rows_render_without_times_and_say_status_unavailable():
    job = _unknown()
    assert v.is_unknown(job) and v.is_live(job) and v.within_retention(job, NOW)
    fields = dict(v.summary_fields(job, v.display_name(job), NOW))
    assert fields["Created"] == "—" and fields["Elapsed"] == "—" and fields["Last update"].startswith("—")
    assert fields["GPUs"] == "12 requested  (8 training #0 + 2 sampling #0 + 2 sampling #1)"
    assert v.status_unavailable_text(job) == "Job status unavailable: Control Plane is unavailable"
    assert v.status_unavailable_text(_unknown(reason="")) == (
        "status unavailable: the service could not read the live state")
    assert v.status_unavailable_text(_unknown(reason="timed out")) == "status unavailable: timed out"
    ordered = v.sort_job_list([job, _job(job_id="b")], "created")
    assert [j["job_id"] for j in ordered] == ["b", JID]  # no created_at sorts last, without an error


# ─── requested GPUs ──────────────────────────────────────────────────────────


def _subs(*entries):
    return _job(sub_jobs=list(entries))


def test_requested_gpus_follow_the_control_planes_rule():
    train = {"sub_job_id": "j:training:0", "job_type": "training", "training_config": {"n_gpus": 8}}
    sample = {"sub_job_id": "j:sampling:0", "job_type": "sampling", "inference_config": {"n_gpus": 4.0}}
    mux = {"sub_job_id": "j:sampling:1", "job_type": "sampling",
           "inference_config": {"n_gpus": 4, "multiplex_job_id": "other-job:training:0"}}
    log_prob = {"sub_job_id": "j:log_prob:0", "job_type": "log_probability", "inference_config": {"n_gpus": 2}}
    job = _subs(train, sample, mux, log_prob)
    assert [v.requested_gpus(s) for s in v.sub_jobs(job)] == [8, 4, 0, 0]
    assert v.total_gpus(job) == 12
    assert v.multiplex_target(mux) == "other-job:training:0" and v.multiplex_target(train) == ""
    assert [v.sub_job_gpu_text(s) for s in (train, sample, mux, log_prob)] == ["8", "4", "shared", "0"]
    text = v.gpu_request_text(job)
    assert text.startswith("12 × H200 requested  (8 training #0 + 4 sampling #0; ")
    assert "sampling #1 runs on the GPUs of other-job:training:0" in text and "uses none" in text


def test_a_job_on_another_jobs_gpus_requests_none():
    mux = {"sub_job_id": "j:training:0", "job_type": "TRAINING",
           "training_config": {"n_gpus": 8, "multiplex_job_id": "host:training:0"}}
    job = _subs(mux)
    assert v.total_gpus(job) == 0
    assert v.gpu_request_text(job) == "no GPUs of its own requested  (training #0 runs on the GPUs of host:training:0)"
    # The search field counts requested GPUs too.
    assert v.job_matches_query(job, v.parse_query("gpus=0"), "")


def test_summary_labels_gpus_as_requested_and_elapsed_as_including_queue_time():
    job = _job(comment="first line\nsecond line")
    fields = v.summary_fields(job, v.display_name(job), NOW)
    labels = [label for label, _ in fields]
    assert labels[:3] == ["Name", "Comment", "Job ID"]
    values = dict(fields)
    assert values["Comment"] == "first line\nsecond line"
    assert values["GPUs"] == "8 × H200 requested"
    assert values["Elapsed"] == "2h 00m  (includes queue time; running, so far)"
    done = _job(status="FAILED", state_changed_at=_iso(NOW - timedelta(hours=1)))
    assert dict(v.summary_fields(done, "", NOW))["Elapsed"] == "1h 00m  (includes queue time)"
    assert "Comment" not in dict(v.summary_fields(_job(), "", NOW))
    assert dict(v.summary_fields(_job(), "", NOW))["Name"] == JID


# ─── reasons and fresh answers ───────────────────────────────────────────────


def test_reason_text_joins_reason_and_details_on_one_line():
    job = _job(status="PENDING", reason="quota_filled", reason_details="needs 8 GPUs;\n64 currently active")
    assert v.reason_text(job) == "quota_filled — needs 8 GPUs; 64 currently active"
    assert v.reason_text(_job()) == ""
    assert v.reason_text(_job(reason_details="only details")) == "only details"


def test_a_fresh_get_job_replaces_the_services_fields():
    shown = _job(status="PENDING", reason="quota_filled", reason_details="needs 8 GPUs",
                 database_name="CORTEX_TRAINING_DB", schema_name="PUBLIC")
    fresh = {"job_id": JID, "status": "RUNNING", "created_at": shown["created_at"], "sub_jobs": shown["sub_jobs"]}
    merged = v.refresh_job(shown, fresh)
    assert merged["status"] == "RUNNING"
    assert "reason" not in merged and "reason_details" not in merged and "updated_at" not in merged
    assert merged["database_name"] == "CORTEX_TRAINING_DB" and merged["hardware"] == "H200"  # not service-owned
    assert v.refresh_job(None, fresh) == fresh


# ─── terminal cells, invisible characters, renaming ──────────────────────────

WIDE = "日本語の名前テスト・"  # ten characters, twenty cells


def test_names_are_cut_by_terminal_cells_not_characters():
    name = v.one_line(WIDE * 10)
    assert v.cell_width(name) <= v.NAME_WIDTH and name.endswith("…")
    assert v.cell_width(v.one_line(WIDE * 10, 30)) <= 30
    more = v.one_line(WIDE + "\nsecond line")
    assert more == WIDE + " …"
    assert v.fit_cells(WIDE, 9) == "日本語の…" and v.fit_cells("abc", 3) == "abc"
    assert v.first_line("  \n  first   line \n x") == "first line"  # whole, never cut
    assert v.first_line("y" * 300) == "y" * 300


def test_wrapping_counts_cells_breaks_long_words_and_keeps_indentation():
    rows = v.wrap_cells(WIDE * 5, 25)
    assert all(v.cell_width(row) <= 25 for row in rows) and "".join(rows) == WIDE * 5
    rows = v.wrap_cells("サポート チャット 評価 " * 6, 21)
    assert all(v.cell_width(row) <= 21 for row in rows)
    assert " ".join(rows).split() == ("サポート チャット 評価 " * 6).split()
    assert v.wrap_cells("    indented words and more words", 16) == ["    indented", "words and more", "words"]
    assert v.wrap_cells("x" * 45, 20) == ["x" * 20, "x" * 20, "x" * 5]
    assert v.wrap_cells("", 10) == [""] and v.wrap_cells("a  b", 10) == ["a  b"]
    family = chr(0x1F468) + chr(0x200D) + chr(0x1F469)  # one emoji: never split
    assert family in v.wrap_cells("ab " + family * 3, 4)[1:]


def test_printable_drops_bidi_overrides_and_invisible_characters():
    rlo, pdf, lri, pdi, zwsp, bom, zwj = (chr(c) for c in (0x202E, 0x202C, 0x2066, 0x2069, 0x200B, 0xFEFF, 0x200D))
    text = v.printable(f"a{rlo}evil{pdf} {lri}x{pdi} ad{zwsp}min{bom}")
    assert all(ch not in text for ch in (rlo, pdf, lri, pdi, zwsp, bom))
    assert text.count(chr(0xFFFD)) == 6
    assert zwj in v.printable(f"a{zwj}b"), "zero-width joiners stay (emoji, scripts)"
    assert v.printable("one" + chr(0x2028) + "two") == "one\ntwo"
    assert v.one_line(f"{rlo}name") == chr(0xFFFD) + "name"
    assert v.collapse("a\n b\t c") == "a b c" and v.collapse(None) == ""


def test_rename_starts_from_the_alias_or_shows_the_comment_as_a_hint():
    job = _job(comment="qwen3-8b sweep #3 · a first line well over eighty cells long, " + "x" * 40 + "\nmore")
    value, hint = v.rename_prompt(job, {})
    assert value == "" and hint.startswith("qwen3-8b sweep #3") and v.cell_width(hint) <= 70
    assert v.rename_prompt(job, {JID: "  my   alias "}) == ("my alias", hint)
    assert v.rename_prompt(_job(), {}) == ("", "e.g. qwen3-8b lora sweep #3")
    # Saving the comment's own first line (or nothing) keeps no alias.
    assert v.alias_to_store(job, v.first_line(job["comment"])) == ""
    assert v.alias_to_store(job, "  ") == "" and v.alias_to_store(job, " my  run ") == "my run"


# ─── sub-jobs: derived ids for UNKNOWN rows only, labels never the raw config ─


def test_only_unknown_rows_get_derived_ids():
    configs = [{"job_type": "TRAINING", "training_config": {"n_gpus": 8}}, {"job_type": "SAMPLING"}]
    running = _job(sub_jobs=configs)
    assert [s.get("sub_job_id") for s in v.sub_jobs(running)] == [None, None]
    assert [v.sub_job_label(s) for s in v.sub_jobs(running)] == ["training", "sampling"]
    assert [s["sub_job_id"] for s in v.sub_jobs(_unknown(configs))] == [f"{JID}:training:0", f"{JID}:sampling:0"]


def test_a_config_without_an_id_or_a_type_is_labelled_by_its_place():
    untyped = _unknown([{"status": "UNKNOWN", "model_name": "openai/gpt-oss-20b", "inference_config": {"n_gpus": 4}},
                        {"status": "UNKNOWN", "job_type": "SAMPLING"}])
    assert [v.sub_job_label(s) for s in v.sub_jobs(untyped)] == ["config #0", "sampling"]
    assert "model_name" not in v.gpu_request_text(untyped)


# ─── fresh answers, GPU cells ────────────────────────────────────────────────


def test_a_fresh_unknown_answer_keeps_the_creation_time():
    shown = _job(status="RUNNING", reason="r")
    merged = v.refresh_job(shown, _unknown())
    assert merged["status"] == "UNKNOWN" and merged["created_at"] == shown["created_at"]  # it never changes
    assert "updated_at" not in merged and merged["reason"].startswith("Job status unavailable")
    elapsed = dict(v.summary_fields(merged, "", NOW))["Elapsed"]
    assert elapsed == "2h 00m  (includes queue time; status unavailable)"


def test_job_gpu_cell_is_the_request_or_shared():
    assert v.job_gpu_cell(_job(), "H200") == "8 H200" and v.job_gpu_cell(_job(), "requested") == "8 requested"
    mux = _job(sub_jobs=[{"sub_job_id": "j:training:0", "job_type": "TRAINING",
                          "training_config": {"n_gpus": 8, "multiplex_job_id": "host:training:0"}}])
    assert v.job_gpu_cell(mux, "H200") == "shared"
    assert v.job_gpu_cell(_job(sub_jobs=[]), "H200") == "—"


# ─── review follow-ups: cutting, whole-name search, cheap rows, printable ids ─


def _chop_cells_reversed(text, max_size, position=0):
    """``rich.cells.chop_cells`` as rich 13.3-13.6 shipped it: pieces in reverse order."""
    from rich.cells import get_character_cell_size

    lines, total = [[]], position
    for character in reversed(text):
        size = get_character_cell_size(character)
        if total + size > max_size:
            lines.append([character])
            total = size
        else:
            total += size
            lines[-1].append(character)
    return ["".join(line) for line in lines]


def test_names_are_cut_from_their_start_whatever_rich_chop_cells_does(monkeypatch):
    import rich.cells

    monkeypatch.setattr(rich.cells, "chop_cells", _chop_cells_reversed)
    monkeypatch.setattr(v, "chop_cells", _chop_cells_reversed, raising=False)
    assert v.one_line("qwen3-8b lora sweep number three", 12) == "qwen3-8b lo…"
    assert v.wrap_cells("abcdefghij klm", 6) == ["abcdef", "ghij", "klm"]
    assert v.fit_cells("日本語テキスト", 7) == "日本語…"  # never half a wide character
    assert v._cut_cells("日本語", 1) == ""


def test_search_matches_the_whole_alias_not_the_cut_display_name():
    alias = "あ" * 60 + "最後の部分は検索できない"  # 72 characters, 144 cells: the display name is cut
    job, aliases = _job(comment="the comment"), {JID: alias}
    assert v.display_name(job, aliases).endswith("…") and "できない" not in v.display_name(job, aliases)
    name = v.search_name(job, aliases)
    assert name == alias
    assert v.job_matches_query(job, v.parse_query("できない"), name)
    assert v.job_matches_query(job, v.parse_query("name:できない"), name)
    assert v.job_matches_query(job, v.parse_query(f"name={alias}"), name)
    assert v.job_matches_query(job, v.parse_query("name:comment"), name)  # the comment still counts
    assert v.search_name(_job(comment="first line\nsecond")) == "first line"


def test_the_usual_row_is_listed_without_deriving_or_relabelling(monkeypatch):
    job = _job()

    def no_kind(sj):  # a row whose entries all have ids needs no type
        raise AssertionError("sub_job_kind called for a row with sub-job ids")

    monkeypatch.setattr(v, "sub_job_kind", no_kind)
    subs = v.sub_jobs(job)
    assert subs == job["sub_jobs"] and subs[0] is job["sub_jobs"][0]


def test_multiplex_ids_are_printable_on_one_line():
    sj = {"sub_job_id": f"{JID}:training:0", "job_type": "TRAINING",
          "training_config": {"n_gpus": 8, "multiplex_job_id": "host:training:0\x1b[31m\n[/b]"}}
    job = _job(sub_jobs=[sj])
    assert v.multiplex_target(sj) == "host:training:0�[31m [/b]"
    assert "\x1b" not in v.gpu_request_text(job) and "\n" not in v.gpu_request_text(job)
    assert v.requested_gpus(sj) == 0 and v.sub_job_gpu_text(sj) == "shared"
    blank = {**sj, "training_config": {"n_gpus": 8, "multiplex_job_id": "  "}}
    assert v.multiplex_target(blank) == "" and v.requested_gpus(blank) == 8  # no id: GPUs of its own


def test_a_log_prob_only_job_requests_zero_and_says_why():
    lp = _job(sub_jobs=[{"sub_job_id": f"{JID}:log_prob:0", "job_type": "LOG_PROBABILITY",
                         "inference_config": {"n_gpus": 2}}])
    # The sub-job id's spelling (log_prob), as the list, the preview and Logs name it.
    assert v.gpu_request_text(lp) == "0 requested  (log_prob #0 uses none)"
    assert v.job_gpu_cell(lp, "H200") == "0" and v.sub_job_gpu_text(lp["sub_jobs"][0]) == "0"
    assert v.SEARCH_FIELDS["gpus"] == "requested GPU count"
    assert v.job_matches_query(lp, v.parse_query("gpus=0"))
