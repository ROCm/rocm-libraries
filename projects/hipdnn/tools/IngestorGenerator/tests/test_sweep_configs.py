# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""A committed sweep config must be data, and its exclusion set must be the miner's.

Two independent things are checked here, both of which have already gone wrong.

`exclude_tensors` must name the exact set `mine_shapes.py` filters backward graphs
on. The two lived as independent literals once, and drifted: the shipped example
declared `d_query,d_key,d_value,d_output` while `mine_shapes.py`'s own filter had
moved on to also catch `dq`/`dk`/`dv`/`do`. The sweep's exclusion gate is a
SEPARATE, cruder filter -- a flat tensor-name check with no node-type fallback and
no dtype backstop -- guarding a corpus staged straight from disk, nowhere near
`mine_shapes.py`'s dispatcher resolution path. A run with the stale value logged
`0 graphs carrying [...]` against a real backward graph: protection it was not
providing, on a class that has already faulted a device mid-sweep.

The key surface must match what `sweep.py` actually parses. A config is read with a
safe loader that rejects unknown and missing keys outright, so an example carrying a
key the parser does not know -- or missing one it requires -- cannot be copied and
run at all. Checking it against `sweep.REQUIRED_KEYS`/`OPTIONAL_KEYS` rather than
against a second list here keeps one source of truth.

Every `configs/*.sweep.yaml` and `configs/*.sweep.yaml.example` is covered, so a
future arch's sweep config is checked automatically -- no new test needed, just a
file matching the existing naming convention.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

_CONFIGS = Path(__file__).resolve().parents[1] / "configs"
_TOOLS = Path(__file__).resolve().parents[1] / "tools"

sys.path.insert(0, str(_TOOLS))

import sweep  # noqa: E402
from mine_shapes import BACKWARD_GRADIENT_TENSOR_NAMES  # noqa: E402


def _sweep_config_files() -> list[Path]:
    """Every committed sweep config, by the two names the harness uses.

    `sweep.py` takes `--config <file>`; the shipped worked example is
    `sweep-isolation.sweep.yaml.example` and a real per-arch config is
    `<slug>.sweep.yaml`. Globbing rather than naming each file is the point: a third
    arch's config lands under one of these patterns without anyone having to
    remember to extend a list.
    """
    return sorted(_CONFIGS.glob("*.sweep.yaml")) + sorted(
        _CONFIGS.glob("*.sweep.yaml.example")
    )


def _load(path: Path) -> dict:
    """Read the config as DATA, with the driver's own loader.

    `sweep.UniqueSafeLoader` refuses duplicate and non-string keys, which plain
    `safe_load` silently accepts by keeping the last one -- a config declaring
    `min_served` twice would then be gated on whichever copy came last.
    """
    return yaml.load(path.read_text(), Loader=sweep.UniqueSafeLoader)  # nosec B506


def test_at_least_one_sweep_config_is_committed():
    """A test iterating an empty file list passes trivially and proves nothing. If
    this fires, the glob patterns above no longer match anything committed -- fix
    the glob, not this assertion."""
    assert _sweep_config_files(), (
        "no configs/*.sweep.yaml or configs/*.sweep.yaml.example found -- "
        "the checks below have nothing to check"
    )


@pytest.mark.parametrize("path", _sweep_config_files(), ids=lambda p: p.name)
class TestACommittedSweepConfigIsData:
    def test_it_declares_exactly_the_surface_the_parser_accepts(self, path):
        """`sweep.load_config` rejects both unknown and missing top-level keys, so a
        config that does not match this set cannot be copied and run."""
        declared = set(_load(path))
        required, optional = set(sweep.REQUIRED_KEYS), set(sweep.OPTIONAL_KEYS)
        assert not required - declared, (
            f"{path.name} is missing required key(s) {sorted(required - declared)}; "
            f"sweep.load_config refuses a config that omits any of them"
        )
        assert not declared - required - optional, (
            f"{path.name} declares unknown key(s) "
            f"{sorted(declared - required - optional)}; sweep.load_config refuses "
            f"them rather than ignoring them"
        )

    def test_the_benchmark_command_is_an_argument_array(self, path):
        """Not a shell string. The whole point of replacing the sourced `.env` is
        that a config can no longer smuggle in a command line to be word-split."""
        argv = (_load(path).get("benchmark") or {}).get("argv")
        assert (
            isinstance(argv, list) and argv
        ), f"{path.name}: benchmark.argv must be a nonempty list"
        assert all(
            isinstance(a, str) for a in argv
        ), f"{path.name}: benchmark.argv must be strings"

    def test_the_exclusion_set_matches_the_miner_exactly(self, path):
        declared = _load(path)["exclude_tensors"]
        if declared == "none":
            return
        assert isinstance(declared, list), (
            f"{path.name}: exclude_tensors must be the string 'none' or a list of "
            f"tensor names, got {type(declared).__name__}"
        )
        names = {str(t).strip().lower() for t in declared}
        assert names == BACKWARD_GRADIENT_TENSOR_NAMES, (
            f"{path.name}'s exclude_tensors does not match "
            f"mine_shapes.BACKWARD_GRADIENT_TENSOR_NAMES "
            f"({sorted(BACKWARD_GRADIENT_TENSOR_NAMES)}) -- "
            f"missing: {sorted(BACKWARD_GRADIENT_TENSOR_NAMES - names)}, "
            f"extra: {sorted(names - BACKWARD_GRADIENT_TENSOR_NAMES)}. "
            f"A sweep gate checking the wrong set reports protection it is not "
            f"providing (see this module's docstring)."
        )
