# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""tools/limbo_audit.py finds each test no CI lane ran that the registry does
not account for, from the run records of real pytest runs.

Each case writes test files at the paths of real registry entries in a
scratch tree, runs them in pytest processes that stand for CI lanes (each
lane's environment picks the skip reason a test gives), and audits the
records against ``rocke.tests.pending_lanes``. A test declares the lane it
waits for with ``rocke.tests.lanes.skip_until``. The registry entries it relies
on: a library test file waits for the build variant alone,
``platform/tests/instances/test_gfx950_mxfp4_numeric.py`` for the runner
hardware, and ``platform/tests/portable_ir/test_portable_ir.py::test_record_coverage``
is quarantined.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from rocke.assets import platform_root
from rocke.tests.lanes import (
    BUILD_VARIANT,
    RUNNER_HARDWARE,
    SOURCE_TREE,
    skip_until_reason,
)
from rocke.tests.pending_lanes import PENDING
from rocke.tests.run_records import RECORDS_DIR_ENV, UNFINISHED, read_run_records
from rocke.tests.subprocess_launcher import rocke_subprocess_env

ROCKE = platform_root().parent
_CHILD_TIMEOUT_SECONDS = 120
# A library test file the registry lists under the build variant alone, derived
# so the example follows the registry: the audit tests below need one whose
# only recorded reason is that lane.
BUILD_VARIANT_FILE = min(
    path
    for path in {entry.path for entry in PENDING}
    if path.startswith("library/tests/")
    and {entry.lane.name for entry in PENDING if entry.path == path}
    == {BUILD_VARIANT.name}
)
# A test file whose cases the registry lists under the source tree alone, as
# cases that declare it rather than as the whole file.
SOURCE_TREE_FILE = min(
    path
    for path in {entry.path for entry in PENDING}
    if {
        (entry.lane.name, entry.whole_file, entry.node)
        for entry in PENDING
        if entry.path == path
    }
    == {(SOURCE_TREE.name, False, "")}
)
GFX950_FILE = "platform/tests/instances/test_gfx950_mxfp4_numeric.py"
QUARANTINED_TEST = "platform/tests/portable_ir/test_portable_ir.py"
# A source test file that no registry entry names.
UNREGISTERED_FILE = "platform/tests/analysis/test_kernel_arguments.py"


def _load_audit_tool():
    spec = importlib.util.spec_from_file_location(
        "limbo_audit", ROCKE / "tools" / "limbo_audit.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module
    spec.loader.exec_module(module)
    return module


limbo_audit = _load_audit_tool()


def test_the_registry_entries_this_audit_relies_on_hold():
    lanes = {(entry.path, entry.lane.name) for entry in PENDING if not entry.node}
    assert (BUILD_VARIANT_FILE, BUILD_VARIANT.name) in lanes
    assert (GFX950_FILE, RUNNER_HARDWARE.name) in lanes
    assert (SOURCE_TREE_FILE, SOURCE_TREE.name) in lanes
    assert not any(entry.path == UNREGISTERED_FILE for entry in PENDING)


def _tree_with(tmp_path: Path, files: dict[str, str]) -> Path:
    """A scratch rocke directory holding ``files`` (path -> source)."""
    tree = tmp_path / "rocke"
    for path, source in files.items():
        (tree / path).parent.mkdir(parents=True, exist_ok=True)
        (tree / path).write_text(textwrap.dedent(source), encoding="utf-8")
    return tree


def _run_lane(
    tree: Path,
    records: Path,
    lane: str,
    *arguments: str,
    environment: dict[str, str] | None = None,
    statuses: tuple[int, ...] = (0, 5),
) -> Path:
    """Run pytest over ``tree`` as the CI lane ``lane`` (the tests read it
    from ``FAKE_LANE``), writing run records into ``records``, with
    ``environment`` added to the child's; the run exits with one of
    ``statuses``."""
    child_environment = rocke_subprocess_env()
    child_environment.update(
        PYTEST_PLUGINS="rocke.tests.run_records",
        FAKE_LANE=lane,
        **{RECORDS_DIR_ENV: str(records)},
        **(environment or {}),
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", *arguments],
        cwd=tree,
        env=child_environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=_CHILD_TIMEOUT_SECONDS,
    )
    assert result.returncode in statuses, result.stdout[-3000:] + result.stderr[-3000:]
    return records


def _audit(tree: Path, lanes: dict[str, Path], universe: Path | None = None):
    source_paths = limbo_audit.SourcePaths(limbo_audit.python_files(tree))
    runs = [
        limbo_audit.lane_run(name, read_run_records(records), source_paths)
        for name, records in lanes.items()
    ]
    tests = (
        limbo_audit.universe_of(read_run_records(universe), source_paths)
        if universe
        else ()
    )
    return {
        (test.path, test.node): dict(test.reasons)
        for test in limbo_audit.audit(runs, tests)
    }


# Skips by lane: the test reads FAKE_LANE and skips with that lane's reason.
_SKIPS_BY_LANE = """
import os
import pytest
from rocke.tests.lanes import (
    BUILD_VARIANT,
    RUNNER_HARDWARE,
    SOURCE_TREE,
    skip_until_reason,
)

def skip_for(reasons):
    reason = reasons.get(os.environ["FAKE_LANE"])
    if reason:
        pytest.skip(reason)
"""
_DECLARED_BUILD_VARIANT = skip_until_reason(BUILD_VARIANT, reason="no rocke_engine")


def test_a_skip_that_declares_the_files_lane_is_accounted_for_and_no_other(tmp_path):
    tree = _tree_with(
        tmp_path,
        {
            GFX950_FILE: _SKIPS_BY_LANE
            + """
def test_runs():
    pass

def test_waits_for_gfx950():
    skip_for({"gfx942": skip_until_reason(RUNNER_HARDWARE, reason="needs a gfx950 GPU")})

def test_only_mentions_gfx950():
    skip_for({"gfx942": "the gfx950 kernel is refused here"})

def test_waits_for_the_build_variant():
    skip_for({"gfx942": skip_until_reason(BUILD_VARIANT, reason="no rocke_engine")})
""",
        },
    )
    records = _run_lane(tree, tmp_path / "gfx942", "gfx942")
    # The file waits for the runner hardware, not the build variant; a skip that names
    # gfx950 without declaring the lane is not booked as waiting for it.
    assert _audit(tree, {"gfx942": records}) == {
        (GFX950_FILE, "test_only_mentions_gfx950"): {
            "gfx942": "the gfx950 kernel is refused here"
        },
        (GFX950_FILE, "test_waits_for_the_build_variant"): {
            "gfx942": _DECLARED_BUILD_VARIANT
        },
    }


def test_a_skip_that_names_another_lanes_gpu_leaves_the_test_to_that_lane(tmp_path):
    tree = _tree_with(
        tmp_path,
        {
            BUILD_VARIANT_FILE: _SKIPS_BY_LANE
            + """
def test_explained_where_its_gpu_is():
    skip_for({
        "gfx942": skip_until_reason(BUILD_VARIANT, reason="no rocke_engine"),
        "gfx1100": "needs a gfx942 GPU, found gfx1100",
    })

def test_unexplained_where_its_gpu_is():
    skip_for({
        "gfx942": "the harness is not installed",
        "gfx1100": "needs a gfx942 GPU, found gfx1100",
    })

def test_each_lane_leaves_it_to_the_other():
    skip_for({
        "gfx942": "needs a gfx1100 GPU, found gfx942",
        "gfx1100": "needs a gfx942 GPU, found gfx1100",
    })
""",
        },
    )
    lanes = {
        lane: _run_lane(tree, tmp_path / lane, lane) for lane in ("gfx942", "gfx1100")
    }
    assert set(_audit(tree, lanes)) == {
        (BUILD_VARIANT_FILE, "test_unexplained_where_its_gpu_is"),
        (BUILD_VARIANT_FILE, "test_each_lane_leaves_it_to_the_other"),
    }


def test_a_lane_run_in_part_leaves_the_other_shards_tests_to_the_other_lanes(
    tmp_path,
):
    """A lane whose records hold one shard of an entry has no verdict on the
    tests the other shards run: the other lanes decide those."""
    tree = _tree_with(
        tmp_path,
        {
            GFX950_FILE: _SKIPS_BY_LANE
            + "".join(
                f"""
def test_waits_for_gfx950_{index}():
    skip_for({{
        "gfx942": skip_until_reason(RUNNER_HARDWARE, reason="needs a gfx950 GPU"),
        "gfx1100": skip_until_reason(RUNNER_HARDWARE, reason="needs a gfx950 GPU"),
    }})
"""
                for index in range(6)
            ),
        },
    )
    shard_lane = _run_lane(
        tree,
        tmp_path / "gfx942",
        "gfx942",
        "-p",
        "rocke.tests.ctest_shard",
        environment={"ROCKE_TEST_SHARD": "0/3"},
    )
    full_lane = _run_lane(tree, tmp_path / "gfx1100", "gfx1100")
    assert _audit(tree, {"gfx942": shard_lane, "gfx1100": full_lane}) == {}


def test_a_test_no_lane_collected_is_seen_through_the_universe(tmp_path):
    tree = _tree_with(
        tmp_path,
        {
            UNREGISTERED_FILE: """
def test_selected():
    pass

def test_left_out():
    pass
""",
        },
    )
    records = _run_lane(tree, tmp_path / "gfx942", "gfx942", "-k", "not left_out")
    universe = _run_lane(tree, tmp_path / "universe", "none", "--collect-only")
    assert _audit(tree, {"gfx942": records}) == {
        (UNREGISTERED_FILE, "test_left_out"): {
            "gfx942": "deselected by -k 'not left_out'"
        }
    }
    selected_only = _run_lane(
        tree, tmp_path / "selected", "gfx942", f"{UNREGISTERED_FILE}::test_selected"
    )
    assert _audit(tree, {"gfx942": selected_only}, universe) == {
        (UNREGISTERED_FILE, "test_left_out"): {}
    }


def test_a_test_whose_process_ended_first_is_in_limbo_as_not_run(tmp_path):
    """A test the lane's process was to run but ended before (a per-test time
    limit's os._exit) is not run, with the reason that says so."""
    tree = _tree_with(
        tmp_path,
        {
            UNREGISTERED_FILE: """
import os

def test_ends_the_process():
    os._exit(1)
""",
        },
    )
    records = _run_lane(tree, tmp_path / "gfx942", "gfx942", statuses=(1,))
    assert _audit(tree, {"gfx942": records}) == {
        (UNREGISTERED_FILE, "test_ends_the_process"): {"gfx942": UNFINISHED}
    }


def test_a_quarantined_test_is_accounted_for(tmp_path):
    tree = _tree_with(
        tmp_path,
        {
            QUARANTINED_TEST: """
import pytest

@pytest.mark.skip(reason="quarantined: AICK-2863")
def test_record_coverage():
    pass
""",
        },
    )
    assert (
        _audit(tree, {"gfx942": _run_lane(tree, tmp_path / "gfx942", "gfx942")}) == {}
    )


def test_an_empty_parameter_set_where_other_cases_ran_is_no_missing_test(tmp_path):
    tree = _tree_with(
        tmp_path,
        {
            UNREGISTERED_FILE: """
import os
import pytest

CASES = {"gfx942": [1, 2], "gfx1100": []}[os.environ["FAKE_LANE"]]

@pytest.mark.parametrize("case", CASES)
def test_per_gpu(case):
    pass

@pytest.mark.parametrize("case", [])
def test_with_no_case_anywhere(case):
    pass
""",
        },
    )
    lanes = {
        lane: _run_lane(tree, tmp_path / lane, lane) for lane in ("gfx942", "gfx1100")
    }
    assert list(_audit(tree, lanes)) == [
        (UNREGISTERED_FILE, "test_with_no_case_anywhere[NOTSET]")
    ]


def test_an_empty_parameter_set_skipped_by_a_skip_mark_is_no_missing_test(tmp_path):
    """A lane whose parameter set is empty skips the placeholder case with the
    reason of the test's own skip mark, not pytest's empty-set reason: the
    placeholder's id (``[NOTSET]``, or a ``NOTSET`` part of a stacked
    parametrize's id) still says no case is missing where other cases ran."""
    tree = _tree_with(
        tmp_path,
        {
            UNREGISTERED_FILE: """
import os
import pytest

CASES = {"gfx942": [1, 2], "gfx1100": []}[os.environ["FAKE_LANE"]]

NOT_HERE = pytest.mark.skipif(
    os.environ["FAKE_LANE"] != "gfx942", reason="needs a gfx942 GPU, found gfx1100"
)

@NOT_HERE
@pytest.mark.parametrize("case", CASES)
def test_per_gpu(case):
    pass

@NOT_HERE
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("case", CASES)
def test_per_gpu_and_dtype(case, dtype):
    pass
""",
        },
    )
    lanes = {
        lane: _run_lane(tree, tmp_path / lane, lane) for lane in ("gfx942", "gfx1100")
    }
    assert _audit(tree, lanes) == {}


@pytest.mark.parametrize("last", ["undeclared", "declared"])
def test_a_lane_keeps_the_recorded_reason_of_one_entry_that_left_a_test_out(
    tmp_path, last
):
    """A lane's host entry leaves an integration test out with a reason that
    declares the source tree, and its GPU entry leaves it out as no GPU test:
    whichever entry ran last, the lane's reason is the recorded one."""
    tree = _tree_with(
        tmp_path,
        {
            SOURCE_TREE_FILE: """
import pytest

@pytest.mark.integration
def test_reads_the_source_tree():
    pass
""",
        },
    )
    records = tmp_path / "gfx942"
    selections = {
        "declared": "not integration and not gpu",
        "undeclared": "gpu and not integration",
    }
    first = "declared" if last == "undeclared" else "undeclared"
    for entry in (first, last):
        _run_lane(
            tree,
            records,
            "gfx942",
            "-m",
            selections[entry],
            statuses=(pytest.ExitCode.NO_TESTS_COLLECTED,),
        )
    assert _audit(tree, {"gfx942": records}) == {}


def test_an_installed_path_maps_to_its_source_file():
    source_paths = limbo_audit.SourcePaths(
        [
            "platform/tests/core/test_dtypes.py",
            "library/tests/test_dtypes.py",
            "platform/python/rocke/benchmark/gemm/tests/test_fp16_rcr_multigpu.py",
        ]
    )
    assert (
        source_paths.source_of("/x/bin/hip_kernel_provider/tests/core/test_dtypes.py")
        == "platform/tests/core/test_dtypes.py"
    )
    assert (
        source_paths.source_of(
            "/x/hip_kernel_provider/tests/library/tests/test_dtypes.py"
        )
        == "library/tests/test_dtypes.py"
    )
    # Installed under another directory: its name is unique.
    assert (
        source_paths.source_of("/x/tests/benchmark/test_fp16_rcr_multigpu.py")
        == "platform/python/rocke/benchmark/gemm/tests/test_fp16_rcr_multigpu.py"
    )
    # Two source files of that name and no longer suffix in common: none.
    assert source_paths.source_of("/elsewhere/test_dtypes.py") is None


def test_the_command_fails_on_a_test_in_limbo_and_writes_its_report(tmp_path):
    tree = _tree_with(
        tmp_path,
        {
            UNREGISTERED_FILE: "def test_skips():\n    import pytest; pytest.skip('why')\n"
        },
    )
    records = _run_lane(tree, tmp_path / "gfx942", "gfx942")
    report = tmp_path / "limbo.md"
    arguments = [
        "--lane",
        f"gfx942={records}",
        "--tree",
        str(tree),
        "--report",
        str(report),
    ]
    assert limbo_audit.main(arguments) == 1
    assert (
        f"| `{UNREGISTERED_FILE}` | `test_skips` | gfx942: why |" in report.read_text()
    )
    with pytest.raises(SystemExit):
        limbo_audit.main(["--lane", "gfx942"])
