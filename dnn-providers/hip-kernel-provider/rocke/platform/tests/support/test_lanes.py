# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rocke.tests.lanes: a skipped case names the lane it waits for, and only the
helpers write that declaration."""

from __future__ import annotations

import sys
import types

import pytest
from rocke.tests.lanes import (
    BINDING_LANE,
    CI_TEST_TOOLS,
    GFX950_RUNNER,
    GFX1151_GFX12_RUNNER,
    GFX1250_RUNNER,
    LANES,
    NIGHTLY_SOURCE,
    Lane,
    arch_skip_reason,
    import_or_skip_until,
    lanes_waited_for,
    runner_lanes_for,
    skip_needing_arch,
    skip_until,
    skip_until_reason,
)


def test_a_declared_reason_reads_back_as_its_lanes():
    reason = skip_until_reason(GFX950_RUNNER, reason="needs a gfx950 GPU")
    assert reason == "waits for gfx950-runner: needs a gfx950 GPU"
    assert lanes_waited_for(reason) == (GFX950_RUNNER,)
    both = skip_until_reason(GFX950_RUNNER, GFX1250_RUNNER, reason="needs either")
    assert lanes_waited_for(both) == (GFX950_RUNNER, GFX1250_RUNNER)


def test_a_reason_that_only_mentions_a_lane_declares_none():
    for reason in (
        "needs a gfx950 GPU, found gfx942",
        "could not import 'rocke_engine'",
        "the gfx950-runner waits for: nothing",
        "",
    ):
        assert lanes_waited_for(reason) == (), reason


def test_a_declaration_of_a_lane_that_does_not_exist_fails_loudly():
    with pytest.raises(ValueError, match=r"lanes \['gfx9999-runner'\] are not in"):
        lanes_waited_for("waits for gfx9999-runner: renamed lane")


def test_only_a_listed_lane_ci_lacks_can_be_declared():
    with pytest.raises(ValueError, match="expected at least one lane"):
        skip_until_reason(reason="why")
    with pytest.raises(ValueError, match="expected one of rocke.tests.lanes.LANES"):
        skip_until_reason(Lane("unlisted", "a lane", enrolled=True), reason="why")
    with pytest.raises(ValueError, match="expected a lane CI lacks"):
        skip_until_reason(NIGHTLY_SOURCE, reason="why")
    with pytest.raises(ValueError, match="reason is empty"):
        skip_until_reason(GFX950_RUNNER, reason=" ")


def test_a_lane_is_validated_at_construction():
    with pytest.raises(ValueError, match="expected lowercase words"):
        Lane("GFX950 runner", "a lane", enrolled=True)
    with pytest.raises(ValueError, match="expected neither for a lane whose files"):
        Lane("x", "a lane", enrolled=False, arches=("gfx950",))
    with pytest.raises(ValueError, match="expected neither for a lane whose files"):
        Lane("x", "a lane", enrolled=False, marker="exhaustive")
    with pytest.raises(ValueError, match="expected a tuple of known gfx arches"):
        Lane("x", "a lane", enrolled=True, arches=("gfx9999",))
    assert len({lane.name for lane in LANES}) == len(LANES)
    runner_arches = [arch for lane in LANES for arch in lane.arches]
    assert len(set(runner_arches)) == len(runner_arches), "an arch has two lanes"


def test_skip_until_skips_with_the_declared_reason():
    with pytest.raises(pytest.skip.Exception) as skipped:
        skip_until(BINDING_LANE, reason="no rocke_engine")
    assert skipped.value.msg == "waits for binding-lane: no rocke_engine"
    assert skipped.value.allow_module_level


def test_import_or_skip_until_returns_the_module_or_declares_its_lane(monkeypatch):
    assert import_or_skip_until("json", CI_TEST_TOOLS).__name__ == "json"
    with pytest.raises(pytest.skip.Exception) as missing:
        import_or_skip_until("a_module_that_is_not_installed", CI_TEST_TOOLS)
    assert missing.value.msg.startswith(
        "waits for ci-test-tools: could not import 'a_module_that_is_not_installed'"
    )
    old = types.ModuleType("old_tool")
    old.__version__ = "0.5.4"
    monkeypatch.setitem(sys.modules, "old_tool", old)
    assert import_or_skip_until("old_tool", CI_TEST_TOOLS, minversion="0.5")
    with pytest.raises(pytest.skip.Exception) as too_old:
        import_or_skip_until("old_tool", CI_TEST_TOOLS, minversion="0.6.0")
    assert too_old.value.msg == (
        "waits for ci-test-tools: old_tool 0.5.4 is older than the 0.6.0 this "
        "case needs"
    )
    unversioned = types.ModuleType("unversioned_tool")
    monkeypatch.setitem(sys.modules, "unversioned_tool", unversioned)
    with pytest.raises(pytest.skip.Exception) as no_version:
        import_or_skip_until("unversioned_tool", CI_TEST_TOOLS, minversion="1.0")
    assert no_version.value.msg == (
        "waits for ci-test-tools: unversioned_tool has no __version__ to compare "
        "with the 1.0 this case needs"
    )


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
        import_or_skip_until("broken_engine", BINDING_LANE)
    with pytest.raises(ModuleNotFoundError, match="a_dependency_that_is_not"):
        import_or_skip_until("engine_missing_a_dependency", BINDING_LANE)


def test_a_case_waits_for_runner_lanes_only_when_no_ci_runner_has_its_arch():
    assert runner_lanes_for(("gfx950",)) == (GFX950_RUNNER,)
    assert runner_lanes_for(("gfx950", "gfx1250")) == (GFX950_RUNNER, GFX1250_RUNNER)
    assert runner_lanes_for(("gfx1151", "gfx1201")) == (GFX1151_GFX12_RUNNER,)
    # gfx942 has a CI runner: the case runs there, so it waits for no lane.
    assert runner_lanes_for(("gfx942", "gfx950")) == ()
    assert runner_lanes_for(()) == ()
    assert arch_skip_reason(("gfx950",), "needs gfx950") == (
        "waits for gfx950-runner: needs gfx950"
    )
    assert arch_skip_reason(("gfx942", "gfx950"), "needs gfx942/gfx950") == (
        "needs gfx942/gfx950"
    )
    with pytest.raises(pytest.skip.Exception) as gfx950_only:
        skip_needing_arch(("gfx950",), reason="needs gfx950, found gfx942")
    assert (
        gfx950_only.value.msg == "waits for gfx950-runner: needs gfx950, found gfx942"
    )
    with pytest.raises(pytest.skip.Exception) as on_a_ci_runner:
        skip_needing_arch(("gfx942",), reason="needs gfx942, found gfx1100")
    assert on_a_ci_runner.value.msg == "needs gfx942, found gfx1100"
