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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################
"""
Regression tests for the placeholder/predicate merge collision that produced
``matmul_test.matmul/pre_checkin_matmul_f8_bf8_fnuz_dst_fp16_*`` "NO solution
found" failures on gfx942.

Two independent defects combined to silently orphan a leaf library:

1. *Data*: a sibling logic YAML for the same gfx942 problem-type declared a
   ``DeviceNames`` list that disagreed with every other sibling. The two
   YAMLs nevertheless produced ``__eq__`` ``HardwarePredicate``s on gfx942
   (chip-id is not part of the predicate on gfx942), so they collided in
   ``PredicateLibrary.merge``.

2. *Code*: ``SolutionLibrary.MasterSolutionLibrary.FromOriginalState``'s
   ``hardware()`` closure appended an ``_ID<chipid>`` suffix to the
   placeholder filename whenever the YAML carried ``DeviceNames``, *without*
   gating on ``supportsChipIdPredicate``. The predicate gate
   (``Hardware.HardwarePredicate.FromHardware``) and the filename gate must
   agree; otherwise non-gfx950 archs produce distinct on-disk filenames for
   colliding predicates, and ``PlaceholderLibrary.merge`` (a no-op) silently
   drops one leaf, leaving its ``.dat`` file unreferenced and its kernels
   unreachable at runtime.

These tests guard both invariants so the failure mode cannot recur silently.
"""
import inspect
import re
from collections import defaultdict
from pathlib import Path

import pytest

from Tensile import SolutionLibrary
from Tensile.Common.Architectures import supportsChipIdPredicate


# Path from this test file to the hipBLASLt logic YAML root.
_LOGIC_ROOT = (
    Path(__file__).resolve().parents[4]
    / "library"
    / "src"
    / "amd_detail"
    / "rocblaslt"
    / "src"
    / "Tensile"
    / "Logic"
    / "asm_full"
)


def _iter_arch_dirs():
    """Yield every (codename, arch_dir) under asm_full/<codename>/<gfxN>."""
    if not _LOGIC_ROOT.is_dir():
        return
    for codename_dir in _LOGIC_ROOT.iterdir():
        if not codename_dir.is_dir():
            continue
        for arch_dir in codename_dir.iterdir():
            if arch_dir.is_dir() and arch_dir.name.startswith("gfx"):
                yield codename_dir.name, arch_dir


# Matches a logic-YAML header DeviceNames line, e.g. "- [Device 0049, Device 0050]".
_DEVICE_NAMES_RE = re.compile(r"^\s*-\s*\[\s*Device\s+([^\]]+)\]\s*$")


def _read_device_names(yaml_path: Path):
    """Extract the DeviceNames list from a logic YAML header without parsing
    the whole (large) document. Returns a tuple of normalized device strings,
    or None if the header has no DeviceNames entry."""
    try:
        with yaml_path.open("r") as f:
            # DeviceNames is line 4 of the header in every current logic YAML;
            # scan a few extra lines defensively in case the header layout
            # ever shifts.
            for _ in range(8):
                line = f.readline()
                if not line:
                    return None
                m = _DEVICE_NAMES_RE.match(line)
                if m:
                    raw = m.group(1)
                    parts = [p.strip() for p in raw.split(",")]
                    # Strip any leading "Device " that survived the split.
                    parts = [p[len("Device "):].strip() if p.startswith("Device ") else p
                             for p in parts]
                    return tuple(sorted(parts))
    except OSError:
        return None
    return None


# ---------------------------------------------------------------------------
# Test 1: data-side invariant. Within a single arch directory, any YAMLs that
# share a basename across library subfolders (FreeSize/GridBased/Equality) are
# describing the same logical kernel-set and MUST declare identical DeviceNames.
#
# This is the precise collision hazard: same arch + same problem-type +
# divergent DeviceNames -> equal HardwarePredicates on archs without
# supportsChipIdPredicate -> PredicateLibrary.merge collides the rows ->
# PlaceholderLibrary.merge (no-op) drops one leaf.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _LOGIC_ROOT.is_dir(),
    reason=f"Logic YAML root not found at {_LOGIC_ROOT}",
)
def test_logic_yaml_sibling_device_names_consistent():
    """Sibling logic YAMLs (same arch dir, same basename) must agree on DeviceNames.

    A basename collision across library subfolders means the YAMLs target the
    same logical problem-type. If their DeviceNames disagree on an arch where
    chip-id is not part of the HardwarePredicate, PredicateLibrary.merge will
    collapse the rows and PlaceholderLibrary.merge (no-op) will silently drop
    one leaf, leaving its .dat file unreachable at runtime."""
    violations = []
    for codename, arch_dir in _iter_arch_dirs():
        # by_basename: basename -> {device-names-tuple: [paths]}
        by_basename = defaultdict(lambda: defaultdict(list))
        for yaml_path in arch_dir.rglob("*.yaml"):
            names = _read_device_names(yaml_path)
            if names is None:
                continue
            by_basename[yaml_path.name][names].append(yaml_path)
        for basename, dn_map in by_basename.items():
            if len(dn_map) > 1:
                detail = {
                    str(names): [str(p.relative_to(_LOGIC_ROOT)) for p in paths]
                    for names, paths in dn_map.items()
                }
                violations.append(
                    f"  {codename}/{arch_dir.name}/{basename}: {detail}"
                )

    assert not violations, (
        "Sibling logic YAMLs share a basename within the same arch directory "
        "but declare divergent DeviceNames. On any arch without "
        "supportsChipIdPredicate (e.g. gfx942), these will collide in "
        "PredicateLibrary.merge and one PlaceholderLibrary leaf will be "
        "silently dropped, producing 'NO solution found' at runtime for "
        "problem dimensions only the dropped leaf could serve.\n"
        + "\n".join(violations)
    )


# ---------------------------------------------------------------------------
# Test 2: code-side invariant. The placeholder _ID<chipid> suffix in
# SolutionLibrary.hardware() must be gated on supportsChipIdPredicate so the
# filename gate matches the predicate gate in Hardware.HardwarePredicate.
# ---------------------------------------------------------------------------

def test_solution_library_gates_placeholder_chip_id_suffix():
    """``hardware()`` must only append ``_ID<chipid>`` when the runtime
    HardwarePredicate also discriminates on chip-id (gfx950 today).

    Otherwise two YAMLs that differ only in DeviceNames produce identical
    predicates but distinct placeholder filenames; the predicate collision
    drops one leaf via PlaceholderLibrary.merge while the orphan placeholder
    keeps its uniquely-named .dat on disk, unreferenced.
    """
    src = inspect.getsource(SolutionLibrary.MasterSolutionLibrary.FromOriginalState)
    assert "supportsChipIdPredicate" in src, (
        "SolutionLibrary.MasterSolutionLibrary.FromOriginalState no longer "
        "references supportsChipIdPredicate. The placeholder _ID<chipid> "
        "suffix MUST be gated on supportsChipIdPredicate(devicePart) to "
        "mirror Hardware.HardwarePredicate.FromHardware; otherwise non-gfx950 "
        "archs regress to the F8NHS TN orphan-leaf failure pattern."
    )
    # Tight pattern: the gate must be conjunctive with the pciChipId truthiness
    # check, on the same conditional branch that emits the suffix.
    assert re.search(r"pciChipId\s+and\s+supportsChipIdPredicate", src), (
        "The placeholder _ID<chipid> suffix branch in "
        "SolutionLibrary.MasterSolutionLibrary.FromOriginalState.hardware() "
        "must be guarded by `if pciChipId and supportsChipIdPredicate(...)`. "
        "A bare `if pciChipId:` allows the filename to diverge while the "
        "predicate stays equal, causing PredicateLibrary.merge to collide "
        "the rows and PlaceholderLibrary.merge (no-op) to drop one leaf."
    )


# ---------------------------------------------------------------------------
# Test 3: invariant on the gating function itself. Any future widening of
# supportsChipIdPredicate requires re-evaluating both the predicate side
# (Hardware.HardwarePredicate.FromHardware) and the filename side
# (SolutionLibrary.hardware()). This test ensures the two stay in lockstep
# by failing loudly if a non-gfx950 arch is added to the gate without an
# accompanying audit.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "arch",
    [
        "gfx900",
        "gfx906",
        "gfx908",
        "gfx90a",
        "gfx940",
        "gfx941",
        "gfx942",
        "gfx1030",
        "gfx1100",
        "gfx1101",
        "gfx1102",
        "gfx1200",
        "gfx1201",
        "gfx1250",
    ],
)
def test_supports_chip_id_predicate_only_gfx950(arch):
    """Lock the current scope of chip-id-aware archs.

    If this test starts failing because ``supportsChipIdPredicate`` was
    widened, that is fine -- but the change author must also confirm:

    * ``SolutionLibrary.MasterSolutionLibrary.FromOriginalState.hardware()``
      still gates the ``_ID<chipid>`` suffix on the same predicate, and
    * existing logic YAMLs for the newly-included arch declare consistent
      ``DeviceNames`` (see ``test_logic_yaml_device_names_uniform_per_arch``).
    """
    assert supportsChipIdPredicate(arch) is False, (
        f"{arch} now claims chip-id support; verify SolutionLibrary.hardware() "
        "still gates the placeholder _ID<chipid> suffix on this predicate and "
        "that all logic YAMLs for this arch declare uniform DeviceNames."
    )


def test_supports_chip_id_predicate_includes_gfx950():
    assert supportsChipIdPredicate("gfx950") is True
