# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rocke.tests.lanes: a skipped case names the lane it waits for, and only the
helpers write that declaration."""

from __future__ import annotations

import pytest
from rocke.tests.lanes import (
    BUILD_VARIANT,
    LANES,
    RUNNER_HARDWARE,
    SOURCE_TREE,
    TICKET,
    Lane,
    arch_skip_reason,
    import_or_skip_until,
    lanes_waited_for,
    runner_lanes_for,
    skip_needing_arch,
    skip_until,
    skip_until_installed,
    skip_until_reason,
)


def test_there_are_four_lanes_each_a_different_kind_of_wait():
    """What CI lacks comes in four kinds: hardware no runner has, a build or
    test environment CI does not provide, work a ticket tracks, and a source
    checkout. A finer distinction (which arch, which tool) is in the reason."""
    assert LANES == (RUNNER_HARDWARE, BUILD_VARIANT, TICKET, SOURCE_TREE)
    assert [lane.name for lane in LANES] == [
        "runner-hardware",
        "build-variant",
        "ticket",
        "source-tree",
    ]
    assert {lane.marker for lane in LANES if lane.marker} == {
        "exhaustive",
        "integration",
    }


def test_a_declared_reason_reads_back_as_its_lanes():
    reason = skip_until_reason(RUNNER_HARDWARE, reason="needs a gfx950 GPU")
    assert reason == "waits for runner-hardware: needs a gfx950 GPU"
    assert lanes_waited_for(reason) == (RUNNER_HARDWARE,)
    both = skip_until_reason(RUNNER_HARDWARE, BUILD_VARIANT, reason="needs both")
    assert lanes_waited_for(both) == (RUNNER_HARDWARE, BUILD_VARIANT)


def test_a_reason_that_only_mentions_a_lane_declares_none():
    for reason in (
        "needs a gfx950 GPU, found gfx942",
        "could not import 'rocke_engine'",
        "the runner-hardware waits for: nothing",
        "",
    ):
        assert lanes_waited_for(reason) == (), reason


def test_a_declaration_of_a_lane_that_does_not_exist_fails_loudly():
    with pytest.raises(ValueError, match=r"lanes \['gfx950-runner'\] are not in"):
        lanes_waited_for("waits for gfx950-runner: renamed lane")


def test_only_a_listed_lane_can_be_declared_once():
    with pytest.raises(ValueError, match="expected at least one lane"):
        skip_until_reason(reason="why")
    with pytest.raises(ValueError, match="expected one of rocke.tests.lanes.LANES"):
        skip_until_reason(Lane("unlisted", "a lane"), reason="why")
    with pytest.raises(ValueError, match="expected each lane once"):
        skip_until_reason(TICKET, TICKET, reason="why")
    with pytest.raises(ValueError, match="reason is empty"):
        skip_until_reason(RUNNER_HARDWARE, reason=" ")


def test_a_lane_is_validated_at_construction():
    with pytest.raises(ValueError, match="expected lowercase words"):
        Lane("GFX950 runner", "a lane")
    with pytest.raises(ValueError, match="expected a tuple of known gfx arches"):
        Lane("x", "a lane", arches=("gfx9999",))
    with pytest.raises(ValueError, match="expected a tuple of known gfx arches"):
        Lane("x", "a lane", arches="gfx950")
    assert len({lane.name for lane in LANES}) == len(LANES)
    runner_arches = [arch for lane in LANES for arch in lane.arches]
    assert len(set(runner_arches)) == len(runner_arches), "an arch has two lanes"


def test_skip_until_skips_with_the_declared_reason():
    with pytest.raises(pytest.skip.Exception) as skipped:
        skip_until(BUILD_VARIANT, reason="no rocke_engine")
    assert skipped.value.msg == "waits for build-variant: no rocke_engine"
    assert skipped.value.allow_module_level


def test_import_or_skip_until_returns_the_module_or_declares_its_lane():
    assert import_or_skip_until("json", BUILD_VARIANT).__name__ == "json"
    with pytest.raises(pytest.skip.Exception) as missing:
        import_or_skip_until("a_module_that_is_not_installed", BUILD_VARIANT)
    assert missing.value.msg.startswith(
        "waits for build-variant: could not import 'a_module_that_is_not_installed'"
    )


def test_skip_until_installed_reads_the_requirements_file(tmp_path):
    """A requirements file every line of which is met runs on; one with a
    package missing, or installed at a version its specifier refuses, skips
    the case as waiting for the lane and names each unmet line."""
    met = tmp_path / "met.txt"
    met.write_text("# comment line\npytest>=1.0  # trailing comment\n\nnumpy\n")
    skip_until_installed(met, BUILD_VARIANT)
    unmet = tmp_path / "unmet.txt"
    unmet.write_text("pytest<1.0\na-distribution-that-is-not-installed\n")
    with pytest.raises(pytest.skip.Exception) as skipped:
        skip_until_installed(unmet, BUILD_VARIANT)
    assert skipped.value.msg.startswith("waits for build-variant: unmet.txt: pytest ")
    assert skipped.value.msg.endswith(
        " does not satisfy pytest<1.0; a-distribution-that-is-not-installed is not "
        "installed"
    )
    assert skipped.value.allow_module_level


def test_a_module_that_is_installed_but_fails_to_import_is_not_a_lane_wait(
    tmp_path, monkeypatch
):
    """A broken build (an installed module whose import fails, or that imports
    a module that is missing) raises: it is a defect, not a wait for the lane
    that installs the module."""
    (tmp_path / "broken_engine.py").write_text(
        "raise ImportError('undefined symbol')\n"
    )
    (tmp_path / "engine_missing_a_dependency.py").write_text(
        "import a_dependency_that_is_not_installed\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ImportError, match="undefined symbol"):
        import_or_skip_until("broken_engine", BUILD_VARIANT)
    with pytest.raises(ModuleNotFoundError, match="a_dependency_that_is_not"):
        import_or_skip_until("engine_missing_a_dependency", BUILD_VARIANT)


def test_a_case_waits_for_runner_hardware_only_when_no_ci_runner_has_its_arch():
    assert runner_lanes_for(("gfx950",)) == (RUNNER_HARDWARE,)
    assert runner_lanes_for(("gfx950", "gfx1250")) == (RUNNER_HARDWARE,)
    assert runner_lanes_for(("gfx1151", "gfx1201")) == (RUNNER_HARDWARE,)
    assert runner_lanes_for(("gfx90a",)) == (RUNNER_HARDWARE,)
    # gfx942 has a CI runner: the case runs there, so it waits for no lane.
    assert runner_lanes_for(("gfx942", "gfx950")) == ()
    assert runner_lanes_for(()) == ()
    assert arch_skip_reason(("gfx950",), "needs gfx950") == (
        "waits for runner-hardware: needs gfx950"
    )
    assert arch_skip_reason(("gfx942", "gfx950"), "needs gfx942/gfx950") == (
        "needs gfx942/gfx950"
    )
    with pytest.raises(pytest.skip.Exception) as gfx950_only:
        skip_needing_arch(("gfx950",), reason="needs gfx950, found gfx942")
    assert gfx950_only.value.msg == (
        "waits for runner-hardware: needs gfx950, found gfx942"
    )
    with pytest.raises(pytest.skip.Exception) as on_a_ci_runner:
        skip_needing_arch(("gfx942",), reason="needs gfx942, found gfx1100")
    assert on_a_ci_runner.value.msg == "needs gfx942, found gfx1100"
