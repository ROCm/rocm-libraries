# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""EXCLUDE_TENSORS must name the exact set `mine_shapes.py` filters backward graphs
on -- one source of truth, checked here rather than trusted.

The two lived as independent literals once, and drifted: the shipped
`sweep-isolation.env.example` declared `d_query,d_key,d_value,d_output` while
`mine_shapes.py`'s own filter (by then fixed to also catch `dq`/`dk`/`dv`/`do` --
see `TestGradientMarkersCatchBothSpellings` in test_mine_shapes.py) had moved on
without it. `tools/sweep.sh`'s exclusion gate is a SEPARATE, cruder filter -- a flat
tensor-name check with no node-type fallback and no dtype backstop -- guarding a
sweep corpus staged straight from disk, nowhere near `mine_shapes.py`'s dispatcher
resolution path. A sweep run with the stale value logged `0 graphs carrying [...]`
against a real backward graph: protection it was not providing, on a class that has
already faulted a device mid-sweep.

Every `configs/*.env` and `configs/*.env.example` file that declares
EXCLUDE_TENSORS (rather than the literal `none`) is checked here against
`mine_shapes.BACKWARD_GRADIENT_TENSOR_NAMES`, so a future third arch's sweep config
is covered automatically -- no new test needed, just a file matching the existing
naming convention.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_CONFIGS = Path(__file__).resolve().parents[1] / "configs"
_TOOLS = Path(__file__).resolve().parents[1] / "tools"

sys.path.insert(0, str(_TOOLS))

from mine_shapes import BACKWARD_GRADIENT_TENSOR_NAMES  # noqa: E402

_EXCLUDE_TENSORS_RE = re.compile(r"^EXCLUDE_TENSORS=(.*)$", re.MULTILINE)


def _sweep_env_files() -> list[Path]:
    """Every committed sweep config, by the two extensions the harness uses.

    `sweep.sh` takes `SWEEP_CONFIG=<file>`; the shipped worked example is
    `sweep-isolation.env.example` and a real per-arch config is `<slug>.sweep.env`
    (see `configs/gfx950_attention_dense.sweep.env`). Globbing rather than naming
    each file is the point: a third arch's config lands under one of these two
    patterns without anyone having to remember to extend this list.
    """
    return sorted(_CONFIGS.glob("*.sweep.env")) + sorted(_CONFIGS.glob("*.env.example"))


def _declared_exclude_tensors(path: Path) -> str | None:
    """The RHS of the first uncommented `EXCLUDE_TENSORS=` assignment, or None.

    A regex over the text, not a bash source: these files are read as data, and
    `sweep.sh` itself is the thing that gives the shell assignment meaning. Lines
    commented out with a leading `#` are not a declaration.
    """
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        match = _EXCLUDE_TENSORS_RE.match(stripped)
        if match:
            return match.group(1)
    return None


class TestExcludeTensorsMatchesTheMinerFilter:
    def test_at_least_one_sweep_config_is_committed(self):
        """A test iterating an empty file list passes trivially and proves
        nothing. If this ever fires, the glob patterns above no longer match
        anything committed -- fix the glob, not this assertion."""
        assert _sweep_env_files(), (
            "no configs/*.sweep.env or configs/*.env.example found -- "
            "the drift check below has nothing to check"
        )

    @pytest.mark.parametrize("path", _sweep_env_files(), ids=lambda p: p.name)
    def test_declared_set_matches_mine_shapes_exactly(self, path):
        raw = _declared_exclude_tensors(path)
        assert raw is not None, (
            f"{path.name} declares no EXCLUDE_TENSORS= line -- sweep.sh's "
            f': "${{EXCLUDE_TENSORS:?...}}" guard means an unset value refuses '
            f"to run, but a committed config should say so explicitly, even if "
            f"only as EXCLUDE_TENSORS=none for an op with no dangerous class"
        )
        if raw == "none":
            return
        declared = {t.strip().lower() for t in raw.split(",") if t.strip()}
        assert declared == BACKWARD_GRADIENT_TENSOR_NAMES, (
            f"{path.name}'s EXCLUDE_TENSORS={raw!r} does not match "
            f"mine_shapes.BACKWARD_GRADIENT_TENSOR_NAMES "
            f"({sorted(BACKWARD_GRADIENT_TENSOR_NAMES)}) -- "
            f"missing: {sorted(BACKWARD_GRADIENT_TENSOR_NAMES - declared)}, "
            f"extra: {sorted(declared - BACKWARD_GRADIENT_TENSOR_NAMES)}. "
            f"A sweep gate checking the wrong set reports protection it is not "
            f"providing (see this module's docstring)."
        )
