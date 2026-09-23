# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""A committed sweep config must be data, and its exclusion set must be the miner's.

The sweep's exclusion gate is a flat tensor-name check with no node-type fallback
and no dtype backstop, cruder than `mine_shapes.py`'s. A set that drifts from
`BACKWARD_GRADIENT_TENSOR_NAMES` reports protection it is not providing.
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
    """Every committed sweep config, by the two names the harness uses. Globbing covers
    a future arch's config without extending a list."""
    return sorted(_CONFIGS.glob("*.sweep.yaml")) + sorted(
        _CONFIGS.glob("*.sweep.yaml.example")
    )


def _load(path: Path) -> dict:
    """Read the config as DATA, with the driver's own loader: `sweep.UniqueSafeLoader`
    refuses the duplicate and non-string keys plain `safe_load` accepts."""
    return yaml.load(path.read_text(), Loader=sweep.UniqueSafeLoader)  # nosec B506


def test_at_least_one_sweep_config_is_committed():
    """If this fires, the glob patterns above no longer match anything committed."""
    assert _sweep_config_files(), (
        "no configs/*.sweep.yaml or configs/*.sweep.yaml.example found -- "
        "the checks below have nothing to check"
    )


@pytest.mark.parametrize("path", _sweep_config_files(), ids=lambda p: p.name)
class TestACommittedSweepConfigIsData:
    def test_it_declares_exactly_the_surface_the_parser_accepts(self, path):
        """`sweep.load_config` rejects unknown and missing top-level keys, so a config
        outside this set cannot be copied and run."""
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
        """Not a shell string: a config must not smuggle in a word-split command
        line."""
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
