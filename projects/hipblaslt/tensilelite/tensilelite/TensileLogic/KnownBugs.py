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

"""Load the known-bugs YAML for TensileLogic --check-all.

Each skip is keyed on (relative_path, solution_name), where solution_name is the
solution's ``SolutionNameMin`` — a canonical, content-derived name (macro tile,
MatrixInstruction, all kernel params). Unlike a positional ``SolutionIndex``,
the name is stable across library re-tuning/regeneration, so a documented skip
keeps matching the same buggy kernel without hand-editing the file when indices
shift. If the kernel is genuinely changed or removed, the name no longer
matches, which is the correct signal to prune the entry.

Paths in the file are relative to the library logic root (the LogicPath
argument), using forward slashes — the same form as validation error messages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, FrozenSet, Optional, Tuple

from Tensile.resources import known_bugs_text

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

# A known-bug lookup key: (normalized relative path, solution_name).
KnownBugKey = Tuple[str, str]
_BUNDLED_SOURCE = "bundled TensileLogic/known_bugs.yaml"


def normalize_logic_relative_path(path: Path) -> str:
    """Normalize to POSIX-style relative path for lookup keys."""
    return "/".join(Path(path).parts)


def _load_known_bugs(
    read_text: Callable[[], str], source: str
) -> FrozenSet[KnownBugKey]:
    """Read and parse known-bugs YAML from a named source."""
    if yaml is None:
        raise RuntimeError(
            "Known-bugs YAML requires PyYAML. Install with: pip install PyYAML\n"
            f"  (source was: {source})"
        )

    raw = yaml.safe_load(read_text())

    if raw is None:
        return frozenset()

    if not isinstance(raw, dict):
        raise ValueError(
            f"Known-bugs file must be a mapping at the top level: {source}"
        )

    skips = raw.get("skips")
    if skips is None:
        return frozenset()

    if not isinstance(skips, list):
        raise ValueError(f"Known-bugs 'skips' must be a list: {source}")

    out: set[KnownBugKey] = set()
    for i, entry in enumerate(skips):
        if not isinstance(entry, dict):
            raise ValueError(f"Known-bugs skips[{i}] must be a mapping: {source}")
        path_str = entry.get("path")
        if not path_str or not isinstance(path_str, str):
            raise ValueError(
                f"Known-bugs skips[{i}] requires string 'path': {source}"
            )
        sol_name = entry.get("solution_name")
        if not sol_name or not isinstance(sol_name, str):
            raise ValueError(
                f"Known-bugs skips[{i}] requires string 'solution_name': {source}"
            )
        key = (normalize_logic_relative_path(Path(path_str)), sol_name)
        out.add(key)

    return frozenset(out)


def load_known_bugs(config_path: Optional[Path]) -> FrozenSet[KnownBugKey]:
    """Read known-bug pairs from an explicit YAML file, if it exists.

    If config_path is None or the file is missing, returns an empty frozenset.
    If a file path is given but PyYAML is not installed, raises RuntimeError.
    """
    if config_path is None:
        return frozenset()

    config_path = Path(config_path)
    if not config_path.is_file():
        return frozenset()

    return _load_known_bugs(
        lambda: config_path.read_text(encoding="utf-8"), str(config_path)
    )


def load_bundled_known_bugs() -> FrozenSet[KnownBugKey]:
    """Load known-bug pairs from the bundled package resource."""
    return _load_known_bugs(known_bugs_text, _BUNDLED_SOURCE)


def is_known_bug(
    known: FrozenSet[KnownBugKey], rel_file: Path, solution_name: Optional[str]
) -> bool:
    if solution_name is None:
        return False
    return (normalize_logic_relative_path(rel_file), solution_name) in known
