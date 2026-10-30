# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""pytester suite of the conformance plugin: outcome table, mode record, CSV, --shard, --fake."""

import os
import textwrap

import pytest

import aotriton_layout
import hipdnn_aotriton_plugin as plugin
import reconcile

PLUGIN = ["-p", "hipdnn_aotriton_plugin"]


@pytest.fixture
def run(pytester, fake_sdpa, monkeypatch, tmp_path):
    """Run the plugin in-process over synthetic test files; returns (result, rows by test name)."""
    monkeypatch.setattr(plugin, "VENDOR_DIR", pytester.path / "vendor")
    out = tmp_path / "results"

    def go(source, *args, extra_files=None):
        pytester.makepyfile(**{"test_synthetic": textwrap.dedent(source)})
        for name, body in (extra_files or {}).items():
            pytester.makepyfile(**{name: textwrap.dedent(body)})
        result = pytester.runpytest(*PLUGIN, f"--results-dir={out}", *args)
        rows = {
            r["nodeid"].split("::")[-1]: r
            for r in reconcile.read_rows(out / "results.csv")
        }
        return result, rows

    return go


PRELUDE = """
import threading
import pytest
import hipdnn_sdpa
import hipdnn_aotriton_plugin as plug
"""


def test_hipdnn_forward_and_backward_pass(run):
    result, rows = run(
        PRELUDE
        + """
def test_both():
    plug.record('fwd', 'hipdnn', 'eng_f')
    plug.record('bwd', 'hipdnn', 'eng_b')

def test_forward_only():
    plug.record('fwd', 'hipdnn', 'eng_f')

def test_forward_only_failing():
    plug.record('fwd', 'hipdnn', 'eng_f')
    assert False, 'mismatch 3.5x'
"""
    )
    result.assert_outcomes(passed=2, failed=1)
    assert (
        rows["test_both"]["outcome"] == "pass" and rows["test_both"]["mode"] == "hipdnn"
    )
    assert (rows["test_both"]["fwd"], rows["test_both"]["bwd"]) == (
        "hipdnn:eng_f",
        "hipdnn:eng_b",
    )
    assert (
        rows["test_forward_only"]["mode"] == "hipdnn"
        and rows["test_forward_only"]["bwd"] == ""
    )
    failed = rows["test_forward_only_failing"]
    assert (
        failed["outcome"] == "fail"
        and failed["mode"] == "hipdnn"
        and "mismatch 3.5x" in failed["reason"]
    )


def test_backward_decline_skips_and_keeps_the_forward_engine(run):
    result, rows = run(
        PRELUDE
        + """
def test_bwd_declined():
    plug.record('fwd', 'hipdnn', 'eng_f')
    plug.record('bwd', 'declined', 'no bwd engine')
    raise hipdnn_sdpa.EngineDeclined('bwd', 'no bwd engine')
"""
    )
    result.assert_outcomes(skipped=1)
    row = rows["test_bwd_declined"]
    assert (row["outcome"], row["reason"], row["fwd"]) == (
        "declined",
        "bwd no bwd engine",
        "hipdnn:eng_f",
    )
    assert row["bwd"] == "declined:no bwd engine" and row["mode"] == ""


def test_reference_forward_with_hipdnn_backward_is_recorded_as_ref_fwd(run):
    result, rows = run(
        PRELUDE
        + """
def test_ref_pass():
    plug.record('fwd', 'ref', 'no dropout engine')
    plug.record('bwd', 'hipdnn', 'eng_b')

def test_ref_fail():
    plug.record('fwd', 'ref', 'no dropout engine')
    plug.record('bwd', 'hipdnn', 'eng_b')
    assert False
"""
    )
    result.assert_outcomes(passed=1, failed=1)
    assert (rows["test_ref_pass"]["outcome"], rows["test_ref_pass"]["mode"]) == (
        "pass",
        "ref-fwd",
    )
    assert (rows["test_ref_fail"]["outcome"], rows["test_ref_fail"]["mode"]) == (
        "fail",
        "ref-fwd",
    )


def test_reference_forward_without_a_hipdnn_backward_is_declined_whatever_the_result(
    run,
):
    result, rows = run(
        PRELUDE
        + """
def test_ref_bwd_not_run_passes():
    plug.record('fwd', 'ref', 'r')

def test_ref_bwd_not_run_fails():
    plug.record('fwd', 'ref', 'r')
    assert False

def test_ref_bwd_declined():
    plug.record('fwd', 'ref', 'r')
    plug.record('bwd', 'declined', 'b')
    raise hipdnn_sdpa.EngineDeclined('bwd', 'b')
"""
    )
    result.assert_outcomes(skipped=3)
    assert rows["test_ref_bwd_not_run_passes"]["reason"] == "fwd r"
    assert rows["test_ref_bwd_not_run_fails"]["reason"] == "fwd r"
    assert rows["test_ref_bwd_declined"]["reason"] == "fwd r; bwd b"
    assert all(r["outcome"] == "declined" for r in rows.values())


def test_unrecorded_engine_decline_exception_becomes_declined(run):
    """A case that calls hipdnn_sdpa directly (no shim entries) still gets a declined row."""
    result, rows = run(
        PRELUDE
        + """
def test_direct():
    raise hipdnn_sdpa.EngineDeclined('fwd', 'span rule')
"""
    )
    result.assert_outcomes(skipped=1)
    assert (rows["test_direct"]["outcome"], rows["test_direct"]["reason"]) == (
        "declined",
        "fwd span rule",
    )


def test_native_skip_needs_vendor_location_unmarked_skip_fails(run):
    result, rows = run(
        PRELUDE
        + """
def test_outside_vendor():
    pytest.skip('because')

@pytest.mark.skip(reason='marked outside')
def test_marked_outside_vendor():
    pass

def test_declined_prefix():
    pytest.skip('declined: packed layout')
""",
        extra_files={
            "vendor/test_native": """
import pytest

def test_raised():
    pytest.skip('hdim > 256 on gfx11')

@pytest.mark.skip(reason='marked native')
def test_marked():
    pass
"""
        },
    )
    result.assert_outcomes(
        passed=0, failed=1, errors=1, skipped=3
    )  # the skip mark reports at setup
    assert (
        rows["test_outside_vendor"]["outcome"] == "fail"
        and "unmarked skip" in rows["test_outside_vendor"]["reason"]
    )
    assert rows["test_marked_outside_vendor"]["outcome"] == "fail"
    assert (
        rows["test_declined_prefix"]["outcome"],
        rows["test_declined_prefix"]["reason"],
    ) == ("declined", "packed layout")
    assert (rows["test_raised"]["outcome"], rows["test_raised"]["reason"]) == (
        "native-skip",
        "hdim > 256 on gfx11",
    )
    assert (rows["test_marked"]["outcome"], rows["test_marked"]["reason"]) == (
        "native-skip",
        "marked native",
    )


def test_skipped_setup_follows_the_same_rule(run):
    result, rows = run(
        PRELUDE
        + """
@pytest.fixture
def needs_something():
    pytest.skip('fixture says no')

def test_setup_skip(needs_something):
    pass

@pytest.fixture
def broken():
    raise RuntimeError('fixture exploded')

def test_setup_error(broken):
    pass
"""
    )
    assert rows["test_setup_skip"]["outcome"] == "fail"
    assert (
        rows["test_setup_error"]["outcome"] == "fail"
        and "fixture exploded" in rows["test_setup_error"]["reason"]
    )


def test_xfail_is_not_a_recordable_outcome(run):
    result, rows = run(
        PRELUDE
        + """
def test_imperative():
    pytest.xfail('calibration')

@pytest.mark.xfail(reason='x')
def test_xpass():
    pass
"""
    )
    assert (
        rows["test_imperative"]["outcome"] == "fail"
        and rows["test_xpass"]["outcome"] == "fail"
    )
    assert result.ret != 0


def test_entries_from_other_threads_and_repeated_calls_fold(run):
    result, rows = run(
        PRELUDE
        + """
def worker(direction, kind, detail):
    t = threading.Thread(target=plug.record, args=(direction, kind, detail))
    t.start()
    t.join()

def test_threads():
    worker('fwd', 'hipdnn', 'eng_f')
    worker('bwd', 'hipdnn', 'eng_b')

def test_all_calls_hipdnn_joins_engines():
    for engine in ('e1', 'e1', 'e2'):
        plug.record('fwd', 'hipdnn', engine)

def test_any_reference_call_makes_the_direction_ref():
    plug.record('fwd', 'hipdnn', 'e1')
    plug.record('fwd', 'ref', 'why')
    plug.record('bwd', 'hipdnn', 'eb')

def test_any_decline_wins_over_ref():
    plug.record('fwd', 'ref', 'why')
    plug.record('fwd', 'declined', 'worse')
    plug.record('bwd', 'hipdnn', 'eb')

def test_entries_do_not_leak_between_cases():
    pass
"""
    )
    assert (rows["test_threads"]["fwd"], rows["test_threads"]["bwd"]) == (
        "hipdnn:eng_f",
        "hipdnn:eng_b",
    )
    assert rows["test_all_calls_hipdnn_joins_engines"]["fwd"] == "hipdnn:e1+e2"
    ref = rows["test_any_reference_call_makes_the_direction_ref"]
    assert (ref["fwd"], ref["mode"]) == ("ref:why", "ref-fwd")
    declined = rows["test_any_decline_wins_over_ref"]
    assert (declined["outcome"], declined["reason"]) == ("declined", "fwd worse")
    leak = rows["test_entries_do_not_leak_between_cases"]
    assert (leak["fwd"], leak["bwd"], leak["mode"], leak["outcome"]) == (
        "",
        "",
        "",
        "no-hipdnn",
    )


def test_fake_always_decline_replaces_both_directions_and_nothing_else(run, fake_sdpa):
    result, rows = run(
        PRELUDE
        + """
def test_fwd():
    hipdnn_sdpa.sdpa_fwd(None, None, None, o=None, stats=None, scale=1.0)

def test_bwd():
    hipdnn_sdpa.sdpa_bwd(None, None, None, None, None, None, dq=None, dk=None, dv=None, scale=1.0)

def test_helpers_stay_real():
    assert hipdnn_sdpa.philox_mask.__name__ == 'keep_pattern'
    assert hipdnn_sdpa.Mask('tl', None, 0).alignment == 'tl'
""",
        "--fake=always-decline",
    )
    result.assert_outcomes(skipped=2, passed=1)
    assert rows["test_fwd"]["reason"] == "fwd always-decline fake"
    assert rows["test_bwd"]["reason"] == "bwd always-decline fake"


def test_engine_lists_are_recorded_only_for_a_direction_hipdnn_served(run, fake_sdpa):
    fake_sdpa.engine_lists = {"fwd": ("ENG_A", "ENG_B"), "bwd": ("ENG_C",)}
    result, rows = run(
        PRELUDE
        + """
import torch
def test_both():
    hipdnn_sdpa.sdpa_fwd(None, None, None, o=torch.zeros(1), stats=None, scale=1.0)
    hipdnn_sdpa.sdpa_bwd(None, None, None, None, None, None, dq=None, dk=None, dv=None, scale=1.0)

def test_bwd_declined():
    hipdnn_sdpa.sdpa_fwd(None, None, None, o=torch.zeros(1), stats=None, scale=1.0)
    plug.record('bwd', 'declined', 'no engine')

def test_nothing_planned():
    pass
"""
    )
    both, half, none = (
        rows[n] for n in ("test_both", "test_bwd_declined", "test_nothing_planned")
    )
    assert (both["fwd_engines"], both["bwd_engines"]) == ("ENG_A;ENG_B", "ENG_C")
    assert (half["outcome"], half["fwd_engines"], half["bwd_engines"]) == (
        "declined",
        "ENG_A;ENG_B",
        "",
    )
    assert (none["fwd_engines"], none["bwd_engines"]) == ("", "")


def test_autouse_device_fixture_is_a_noop_without_cuda(run, monkeypatch):
    """Cases without a `gpu_id` fixture still run: the plugin only asks for it when CUDA exists."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    result, rows = run(PRELUDE + "def test_x():\n    pass\n")
    result.assert_outcomes(passed=1)


@pytest.mark.parametrize(
    "knob, value",
    [
        ("USE_ADIFFS_TXT", "/x"),
        ("RECORD_ADIFFS_TO", "/x"),
        ("SKIP_BWD", "1"),
        ("FWD_IMPL", "triton"),
        ("BWD_IMPL", "flyc"),
        ("SMALL_HEADDIM_ONLY", "1"),
        ("LARGE_HEADDIM_ONLY", "1"),
        ("AOTRITON_TORCH_ONLY_USE_CPU", "1"),
        ("PROBE_UNSUPPORTED", "1"),
        ("IGNORE_BACKWARD_IMPORT", "1"),
    ],
)
def test_refused_knobs_abort_at_configure(pytester, monkeypatch, knob, value):
    monkeypatch.setenv(knob, value)
    pytester.makepyfile("def test_x():\n    pass\n")
    result = pytester.runpytest(*PLUGIN)
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines([f"*{knob}*"])


def test_forwarded_and_neutral_knobs_are_accepted(pytester, monkeypatch):
    for knob, value in (
        ("FOR_RELEASE", "2"),
        ("SMALL_VRAM", "1"),
        ("SKIP_BWD", "0"),
        ("HIPDNN_FWD_ENGINE", "e"),
        ("AOTRITON_REF_DEVICE_OPTION", "cpu"),
    ):
        monkeypatch.setenv(knob, value)
    pytester.makepyfile("def test_x():\n    pass\n")
    result = pytester.runpytest(*PLUGIN)
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(
        ["sdpa-conformance: *FOR_RELEASE=2*HIPDNN_FWD_ENGINE=e*"]
    )


# ------------------------------------------------------------------ sharding

SHARD_SOURCE = "\n".join(f"def test_{c}():\n    pass\n" for c in "abcdefg")


def selected(pytester, shard):
    result = pytester.runpytest(*PLUGIN, f"--shard={shard}", "--collect-only", "-q")
    return [line.split("::")[-1] for line in result.stdout.lines if "::" in line]


def test_shard_keeps_every_nth_sorted_node_id_and_shards_partition_the_suite(pytester):
    pytester.makepyfile(SHARD_SOURCE)
    shards = [selected(pytester, f"{i}/3") for i in (1, 2, 3)]
    assert shards[0] == ["test_a", "test_d", "test_g"]
    assert shards[1] == ["test_b", "test_e"] and shards[2] == ["test_c", "test_f"]
    assert sorted(sum(shards, [])) == [f"test_{c}" for c in "abcdefg"]


def test_nodeids_file_keeps_exactly_the_listed_cases_before_sharding(
    pytester, tmp_path
):
    pytester.makepyfile(SHARD_SOURCE)
    listing = tmp_path / "ids.txt"
    listing.write_text(
        "test_nodeids_file_keeps_exactly_the_listed_cases_before_sharding.py::test_c\n"
        "test_nodeids_file_keeps_exactly_the_listed_cases_before_sharding.py::test_e\n"
        "gone.py::test_zz\n\n"
    )
    base = [*PLUGIN, f"--hipdnn-nodeids-file={listing}", "--collect-only", "-q"]

    def names(*extra):
        lines = pytester.runpytest(*base, *extra).stdout.lines
        return [line.split("::")[-1] for line in lines if "::" in line]

    assert names() == ["test_c", "test_e"]
    assert names("--shard=2/2") == ["test_e"]  # shards the listed cases, not the suite


@pytest.mark.parametrize("bad", ["0/3", "4/3", "x", "1/0", "1"])
def test_shard_rejects_malformed_specs(pytester, bad):
    pytester.makepyfile(SHARD_SOURCE)
    assert (
        pytester.runpytest(*PLUGIN, f"--shard={bad}").ret == pytest.ExitCode.USAGE_ERROR
    )


def test_shard_selection_is_identical_in_every_xdist_process(
    pytester, tmp_path, monkeypatch
):
    monkeypatch.setenv("PYTHONPATH", aotriton_layout.pythonpath())
    pytester.makepyfile(SHARD_SOURCE)
    out = tmp_path / "out"
    result = pytester.runpytest_subprocess(
        *PLUGIN, "--shard=2/3", "-n", "2", f"--results-dir={out}"
    )
    result.assert_outcomes(passed=2)
    assert sorted(
        r["nodeid"].split("::")[-1] for r in reconcile.read_rows(out / "results.csv")
    ) == ["test_b", "test_e"]


# ------------------------------------------------------------------ xdist, crashes, kills


def test_xdist_controller_writes_each_row_once_and_a_dead_worker_is_a_crash_row(
    pytester, tmp_path, monkeypatch
):
    monkeypatch.setenv("PYTHONPATH", aotriton_layout.pythonpath())
    pytester.makepyfile(
        """
import os
import hipdnn_aotriton_plugin as plug

def test_ok_1():
    plug.record('fwd', 'hipdnn', 'e')

def test_ok_2():
    plug.record('fwd', 'hipdnn', 'e')

def test_dies():
    os._exit(139)

def test_ok_3():
    pass

def test_bad():
    assert False
"""
    )
    out = tmp_path / "out"
    result = pytester.runpytest_subprocess(
        *PLUGIN, "-n", "2", "--max-worker-restart=5", f"--results-dir={out}"
    )
    assert result.ret != 0
    lines = (out / "results.csv").read_text().splitlines()
    assert lines[0] == ",".join(reconcile.FIELDS)
    rows = {
        r["nodeid"].split("::")[-1]: r for r in reconcile.read_rows(out / "results.csv")
    }
    assert len(lines) - 1 == len(rows) == 5
    assert rows["test_dies"]["outcome"] == "crash"
    assert {rows[n]["outcome"] for n in ("test_ok_1", "test_ok_2")} == {"pass"}
    assert rows["test_ok_3"]["outcome"] == "no-hipdnn"
    assert (
        rows["test_ok_1"]["fwd"] == "hipdnn:e" and rows["test_bad"]["outcome"] == "fail"
    )


def test_killed_run_leaves_flushed_rows_and_reconcile_reports_the_rest_not_run(
    pytester, tmp_path, monkeypatch
):
    """SIGKILL of the whole run: rows written so far survive; the rest become not-run."""
    monkeypatch.setenv("PYTHONPATH", aotriton_layout.pythonpath())
    pytester.makepyfile(
        """
import os, signal

def test_1():
    pass

def test_2():
    pass

def test_3_killed():
    os.kill(os.getpid(), signal.SIGKILL)

def test_4():
    pass
"""
    )
    shard = tmp_path / "run" / "shard-1of1"
    collect = pytester.runpytest_subprocess(*PLUGIN, "--collect-only", "-q")
    shard.mkdir(parents=True)
    (shard / "collected.txt").write_text(
        "".join(line + "\n" for line in collect.outlines if "::" in line)
    )
    pytester.runpytest_subprocess(*PLUGIN, f"--results-dir={shard}")
    rows = {
        r["nodeid"].split("::")[-1]: r for r in reconcile.reconcile(tmp_path / "run")
    }
    assert [
        rows[n]["outcome"] for n in ("test_1", "test_2", "test_3_killed", "test_4")
    ] == ["no-hipdnn", "no-hipdnn", "not-run", "not-run"]


# ------------------------------------------------------------------ hipdnn_sdpa wrappers

CALLS = (
    PRELUDE
    + """
import torch

def call_fwd():
    o, stats = torch.zeros(1), torch.zeros(1)
    return hipdnn_sdpa.sdpa_fwd(o, o, o, o=o, stats=stats, scale=1.0)

def call_bwd():
    t = torch.zeros(1)
    return hipdnn_sdpa.sdpa_bwd(t, t, t, t, t, t, dq=t, dk=t, dv=t, scale=1.0)
"""
)


def test_pass_needs_a_hipdnn_call_otherwise_no_hipdnn(run):
    result, rows = run(
        CALLS
        + """
def test_never_calls_hipdnn():
    pass

def test_calls_forward():
    assert call_fwd() == 'fake_fwd'

def test_calls_both_from_threads():
    for fn in (call_fwd, call_bwd):
        t = threading.Thread(target=fn)
        t.start()
        t.join()

def test_failing_without_a_call_stays_fail():
    assert False
"""
    )
    result.assert_outcomes(passed=3, failed=1)
    assert rows["test_never_calls_hipdnn"]["outcome"] == "no-hipdnn"
    called = rows["test_calls_forward"]
    assert (called["outcome"], called["mode"], called["fwd"]) == (
        "pass",
        "hipdnn",
        "hipdnn:fake_fwd",
    )
    both = rows["test_calls_both_from_threads"]
    assert (both["outcome"], both["fwd"], both["bwd"]) == (
        "pass",
        "hipdnn:fake_fwd",
        "hipdnn:fake_bwd",
    )
    assert rows["test_failing_without_a_call_stays_fail"]["outcome"] == "fail"


def test_wrapped_decline_is_recorded_and_a_reference_fallback_replaces_it(
    run, fake_sdpa
):
    fake_sdpa.decline.update(fwd="no fwd", bwd="no bwd")
    result, rows = run(
        CALLS
        + """
def test_bwd_decline_propagates():
    with pytest.raises(hipdnn_sdpa.EngineDeclined):
        call_bwd()
    raise hipdnn_sdpa.EngineDeclined('bwd', 'no bwd')

def test_forward_fallback():
    try:
        call_fwd()
    except hipdnn_sdpa.EngineDeclined as e:
        plug.record('fwd', 'ref', e.reason)   # what the shim does after a forward decline
    plug.record('bwd', 'hipdnn', 'fake_bwd')
"""
    )
    assert (
        rows["test_bwd_decline_propagates"]["outcome"],
        rows["test_bwd_decline_propagates"]["reason"],
    ) == ("declined", "bwd no bwd")
    fallback = rows["test_forward_fallback"]
    assert (fallback["outcome"], fallback["mode"], fallback["fwd"]) == (
        "pass",
        "ref-fwd",
        "ref:no fwd",
    )
