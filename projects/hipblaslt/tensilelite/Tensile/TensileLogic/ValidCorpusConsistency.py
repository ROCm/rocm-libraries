# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
ValidCorpusConsistency
---
Architecture-independent, whole-corpus consistency checks for the library
logic tree, run unconditionally by ``TensileLogic --check-all`` regardless of
which ``--architecture`` values a given build targets.

These checks are different in kind from the other ``Valid*.py`` validators:
they are cheap (header-only YAML reads, or a bare predicate call -- no
per-solution parsing) and inherently need whole-corpus visibility. A sibling
mismatch or an overlay-shape violation in one architecture's tree should be
caught by *any* build that has the corpus checked out, not just one that
happens to target that architecture. So every function here walks the full
``logic_root`` and ignores any ``--architecture`` filtering the caller may be
applying to its own per-solution validation.

No known-bugs / quarantine escape hatch exists for these checks (unlike the
per-solution validators, which can accept a documented ``known_bugs.yaml``
entry). A violation here is always a hard failure. If a future violation
needs a documented, temporary exception, extend the known-bugs schema (see
``KnownBugs.py``) deliberately -- do not assume one already covers these
checks.
"""

import re
import sys

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from Tensile.Common.Architectures import supportsChipIdPredicate
from Tensile.CustomYamlLoader import load_logic_gfx_arch, load_logic_schedule_name

GFX1250 = "gfx1250"
GFX1250V0 = "gfx1250v0"

# Logic YAMLs come in two header dialects: the positional list form, where
# DeviceNames is the 4th sequence entry (``- [Device ...]``), and the mapping
# form used by e.g. Origami (``DeviceNames: [Device ...]``). Match both, or
# divergence in the mapping-form files is silently skipped (see #11442).
_DEVICE_NAMES_RE = re.compile(r"^\s*(?:-|DeviceNames:)\s*\[\s*Device\s+([^\]]+)\]\s*$")


def iter_arch_dirs(logic_root: Path) -> Iterator[Tuple[str, Path]]:
    """Yield (codename, arch_dir) for every ``gfx*`` directory one level below
    a codename directory directly under ``logic_root`` (the corpus's
    ``<codename>/<gfx_arch>/...`` layout)."""
    for codename_dir in logic_root.iterdir():
        if not codename_dir.is_dir():
            continue
        for arch_dir in codename_dir.iterdir():
            if arch_dir.is_dir() and arch_dir.name.startswith("gfx"):
                yield codename_dir.name, arch_dir


def all_arch_names(logic_root: Path) -> List[str]:
    """Sorted, de-duplicated set of arch directory names present anywhere in
    the corpus (e.g. ``gfx950``, ``gfx950_id75a3``)."""
    return sorted({arch_dir.name for _, arch_dir in iter_arch_dirs(logic_root)})


def read_device_names(yaml_path: Path) -> Optional[Tuple[str, ...]]:
    """Return the sorted ``DeviceNames`` tuple from a logic-YAML header (the
    ``- [Device ...]`` or ``DeviceNames: [Device ...]`` line within the first
    few lines), or ``None`` if it can't be found/read. Intentionally a cheap
    line scan, not a full parse: this runs over the whole corpus on every
    check-all invocation."""
    try:
        with yaml_path.open("r") as f:
            for _ in range(8):
                line = f.readline()
                if not line:
                    return None
                m = _DEVICE_NAMES_RE.match(line)
                if m:
                    parts = [p.strip() for p in m.group(1).split(",")]
                    parts = [
                        p[len("Device "):].strip() if p.startswith("Device ") else p
                        for p in parts
                    ]
                    return tuple(sorted(parts))
    except OSError:
        return None
    return None


def find_sibling_device_names_violations(logic_root: Path) -> List[str]:
    """Same-basename logic YAMLs within one arch directory must declare
    identical ``DeviceNames``; a divergence (e.g. one sibling missing a chip
    ID the other declares) shipped invisibly before this check existed --
    see https://github.com/ROCm/rocm-libraries/issues/11397."""
    violations: List[str] = []
    for codename, arch_dir in iter_arch_dirs(logic_root):
        by_basename: Dict[str, Dict[Tuple[str, ...], List[Path]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for yaml_path in arch_dir.rglob("*.yaml"):
            names = read_device_names(yaml_path)
            if names is None:
                continue
            by_basename[yaml_path.name][names].append(yaml_path)
        for basename, dn_map in by_basename.items():
            if len(dn_map) > 1:
                detail = {
                    str(names): [str(p.relative_to(logic_root)) for p in paths]
                    for names, paths in dn_map.items()
                }
                violations.append(
                    f"Divergent sibling DeviceNames: {codename}/{arch_dir.name}/"
                    f"{basename}: {detail}"
                )
    return violations


def find_chip_id_arch_lock_violations(logic_root: Path) -> List[str]:
    """Lock chip-ID-aware architectures to the current, audited set
    (``gfx950`` only). ``supportsChipIdPredicate`` gates both logic-file
    placement rules (``ValidChipId.py``) and the ``SolutionLibrary`` placeholder
    suffix; a new architecture silently becoming chip-ID-aware (or ``gfx950``
    silently stopping being one) needs a deliberate re-audit of both, not a
    registry edit that just happens to flip this predicate."""
    violations: List[str] = []
    for arch in all_arch_names(logic_root):
        base_arch = arch.split("_", 1)[0]
        expected = base_arch == "gfx950"
        actual = supportsChipIdPredicate(base_arch)
        if actual is not expected:
            violations.append(
                f"Chip-ID-arch-lock violation: {arch} (base {base_arch}): "
                f"supportsChipIdPredicate={actual}, expected={expected} -- new "
                "chip-ID-aware architectures require a re-audit of logic YAML "
                "placement rules and the SolutionLibrary suffix gate"
            )
    return violations


def find_gfx1250v0_overlay_violations(logic_root: Path) -> List[str]:
    """gfx1250 ships as two silicon revisions (v0, v1) sharing one ISA, arch
    name, and compiler target; the runtime tells them apart only via
    ``hipDeviceProp_t::asicRevision``, and ``TensileCreateLibrary`` globs one
    tree, separating the revisions by ``ScheduleName`` alone. A mis-tagged
    file fails silently -- dropped from v0, or leaked into every v1 build --
    so this checks the invariant against the tree that actually ships:

    1. the ``gfx1250v0`` overlay directory exists and ships at least one
       logic file (an empty overlay means a v0 build reports success having
       written a library with no solutions in it);
    2. every file inside the overlay declares ``ScheduleName: gfx1250v0``;
    3. every file inside the overlay keeps ``ArchitectureName: gfx1250``
       (a stepping there is rejected by ``TensileCreateLibrary``, and
       ``library/gfx1250v0/`` is a directory the runtime never reads); and
    4. no file *outside* the overlay claims ``ScheduleName: gfx1250v0``.
    """
    violations: List[str] = []
    overlay_root = logic_root / GFX1250V0
    overlay_files = sorted(overlay_root.rglob("*.yaml")) if overlay_root.is_dir() else []

    if not overlay_files:
        violations.append(
            f"{GFX1250V0} overlay ships no logic under {overlay_root} -- an "
            "empty overlay means a v0 build reports success having written a "
            "library with no solutions in it"
        )

    for p in overlay_files:
        schedule_name = load_logic_schedule_name(p)
        if schedule_name != GFX1250V0:
            violations.append(
                f"{p.relative_to(logic_root)}: overlay file declares "
                f"ScheduleName {schedule_name!r}, expected {GFX1250V0!r}"
            )
        gfx_arch = load_logic_gfx_arch(p)
        if gfx_arch != GFX1250:
            violations.append(
                f"{p.relative_to(logic_root)}: overlay file declares "
                f"ArchitectureName {gfx_arch!r}, expected {GFX1250!r}"
            )

    for p in sorted(logic_root.rglob("*.yaml")):
        if overlay_root.is_dir() and p.is_relative_to(overlay_root):
            continue
        if load_logic_schedule_name(p) == GFX1250V0:
            violations.append(
                f"{p.relative_to(logic_root)}: declares ScheduleName "
                f"{GFX1250V0!r} outside the {GFX1250V0} overlay directory"
            )

    return violations


def check_corpus_invariants(logic_root: Path) -> List[str]:
    """Aggregate every corpus-wide invariant check into one flat violation
    list. Returns an empty list (rather than raising) when ``logic_root``
    isn't a directory -- a single-file ``LogicPath`` invocation has no corpus
    to walk, so these checks are inapplicable rather than failing."""
    logic_root = Path(logic_root)
    if not logic_root.is_dir():
        return []
    return [
        *find_sibling_device_names_violations(logic_root),
        *find_chip_id_arch_lock_violations(logic_root),
        *find_gfx1250v0_overlay_violations(logic_root),
    ]


def report_corpus_invariant_violations(violations: List[str]) -> None:
    for violation in violations:
        print(f"Error: {violation}", file=sys.stderr)
