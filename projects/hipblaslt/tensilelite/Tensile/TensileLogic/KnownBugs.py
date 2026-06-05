################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
################################################################################

"""
Load optional known-bugs YAML for TensileLogic --check-all.

Paths in the file are relative to the library logic root (the LogicPath argument),
using forward slashes — the same form as validation error messages.
"""

from __future__ import annotations

from pathlib import Path
from typing import FrozenSet, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

KnownBugKey = Tuple[str, int]


def normalize_logic_relative_path(path: Path) -> str:
    """Normalize to POSIX-style relative path for lookup keys."""
    return "/".join(Path(path).parts)


def load_known_bugs(config_path: Optional[Path]) -> FrozenSet[KnownBugKey]:
    """
    Parse known-bugs YAML into a set of (relative_path, solution_index).

    If config_path is None or the file is missing, returns an empty frozenset.
    If a file path is given but PyYAML is not installed, raises RuntimeError.
    """
    if config_path is None:
        return frozenset()
    if not config_path.is_file():
        return frozenset()
    if yaml is None:
        raise RuntimeError(
            "Known-bugs YAML requires PyYAML. Install with: pip install PyYAML\n"
            f"  (file was: {config_path})"
        )

    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        return frozenset()

    if not isinstance(raw, dict):
        raise ValueError(
            f"Known-bugs file must be a mapping at the top level: {config_path}"
        )

    skips = raw.get("skips")
    if skips is None:
        return frozenset()

    if not isinstance(skips, list):
        raise ValueError(f"Known-bugs 'skips' must be a list: {config_path}")

    out: set[KnownBugKey] = set()
    for i, entry in enumerate(skips):
        if not isinstance(entry, dict):
            raise ValueError(f"Known-bugs skips[{i}] must be a mapping: {config_path}")
        path_str = entry.get("path")
        if not path_str or not isinstance(path_str, str):
            raise ValueError(
                f"Known-bugs skips[{i}] requires string 'path': {config_path}"
            )
        sol_idx = entry.get("solution_index")
        if sol_idx is None:
            raise ValueError(
                f"Known-bugs skips[{i}] requires integer 'solution_index': {config_path}"
            )
        if not isinstance(sol_idx, int):
            raise ValueError(
                f"Known-bugs skips[{i}].solution_index must be int: {config_path}"
            )
        key = (normalize_logic_relative_path(Path(path_str)), sol_idx)
        out.add(key)

    return frozenset(out)


def is_known_bug(
    known: FrozenSet[KnownBugKey], rel_file: Path, solution_index: int
) -> bool:
    return (normalize_logic_relative_path(rel_file), int(solution_index)) in known
