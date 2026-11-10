# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
valid_corpus_consistency
---
Cross-file consistency checks for the library logic tree, run by
``TensileLogic --check-all`` in addition to the existing per-file/per-solution
validators.

These checks are different in kind from the other ``Valid*.py`` validators:
they are cheap (header-only YAML reads -- no per-solution parsing) but need
visibility across more than one file to do their job. They are also where a
cross-file fact about the shipped corpus belongs, rather than a unit test that
reads the corpus: a unit test only sees the corpus in lanes that happen to have
it, and only runs when tensilelite changes, not when the logic does. The checks
below are scoped the same way the caller's own per-solution validation already is: to
the files ``TensileLogic`` selected for the requested ``--architecture``
(the full corpus when ``--architecture all`` is given). There's no separate
"whole corpus" mode here -- if a gfx942-only build doesn't touch gfx1250
data, a gfx1250-only violation should not fail it.

No known-bugs / quarantine escape hatch exists for these checks (unlike the
per-solution validators, which can accept a documented ``known_bugs.yaml``
entry). A violation here is always a hard failure. If a future violation
needs a documented, temporary exception, extend the known-bugs schema (see
``known_bugs.py``) deliberately -- do not assume one already covers these
checks.
"""

import copy
import sys

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..CustomYamlLoader import (
    load_logic_cu_count,
    load_logic_device_names,
    load_logic_gfx_arch,
    load_logic_perf_metric,
    load_logic_problem_type,
    load_logic_schedule_name,
)

GFX1250 = "gfx1250"
GFX1250_STRICT = "gfx1250-strict"

# The corpus's ``<codename>/<gfx_arch>/...`` layout always lives under a
# directory named ``asm_full``. Callers rarely pass that exact directory,
# though: ``TensileLogic``'s own ``LogicPath`` CLI argument defaults (via
# ``HIPBLASLT_LIBLOGIC_PATH``) to the whole ``library/`` tree, several
# directories above the real corpus. Resolve down to it explicitly instead of
# assuming a fixed depth below whatever ``logic_root`` the caller passed in.
_ASM_FULL_DIRNAME = "asm_full"


def _resolve_corpus_root(logic_root: Path) -> Path:
    """Return the actual ``asm_full`` corpus directory, whether ``logic_root``
    already is it (as in these checks' own unit tests, and any direct-path
    CLI invocation) or is a higher ancestor of it (as in the default,
    CMake-driven build). Falls back to ``logic_root`` unchanged if no
    ``asm_full`` directory can be found anywhere below it, so hermetic test
    fixtures that build a synthetic corpus under an arbitrarily-named
    ``tmp_path`` keep working."""
    if logic_root.name == _ASM_FULL_DIRNAME:
        return logic_root
    # Deliberately unsorted: rglob() is a lazy walk, and library/ can be big
    # under --check-all, so take the first match and stop walking rather than
    # forcing sorted() to materialize (and order) every match in the tree.
    # There is exactly one asm_full corpus below a project's library/, so
    # first-found is unambiguous in practice.
    for candidate in logic_root.rglob(_ASM_FULL_DIRNAME):
        if candidate.is_dir():
            return candidate
    return logic_root


def read_device_names(yaml_path: Path) -> Optional[Tuple[str, ...]]:
    """Return the sorted ``DeviceNames`` tuple parsed from a logic YAML's
    header (list index 3 in the positional dialect, or the ``DeviceNames``
    mapping key), or ``None`` if the file itself can't be opened or parsed at
    all. A header that parses fine but has no ``DeviceNames`` field returns
    an *empty* tuple rather than ``None`` -- a distinct, comparable value, so
    it still participates in sibling comparison (and is flagged as a
    divergence from any sibling that does declare device names) instead of
    being silently dropped, the same fail-open gap that let a divergence go
    unnoticed before this module existed."""
    try:
        names = load_logic_device_names(yaml_path)
    except (OSError, RuntimeError):
        return None
    if not names:
        return ()
    parts = []
    for name in names:
        name = str(name)
        parts.append(name[len("Device "):].strip() if name.startswith("Device ") else name.strip())
    return tuple(sorted(parts))


def _hashable(value):
    """Coerce a parsed header value into something hashable. Header fields
    are normally scalars (str/int/None), but a malformed file can put a
    mapping where a scalar is expected (e.g. a stray ``{Architecture: ...,
    CUCount: ...}`` in the ScheduleName slot) -- stringify it rather than
    letting the whole check crash on an unhashable dict; the file's own
    logic tree becomes its own singleton group and any real sibling still
    compares against it (correctly reporting a divergence, since a
    malformed header can never equal a well-formed sibling's)."""
    try:
        hash(value)
        return value
    except TypeError:
        return repr(value)


def _chip_id_dir_suffix(yaml_path: Path, base_arch: str) -> Optional[str]:
    """Return the chip-ID-variant directory component nearest the file (e.g.
    ``"gfx950_id75a3"``), or ``None`` if the file lives under the bare
    ``base_arch`` directory instead (the default/fallback tree). A chip-ID
    variant's header declares the *same* ScheduleName/ArchitectureName as the
    default tree's -- ``valid_chip_id.py``'s placement rules, not header
    content, are what distinguish per-chip-ID logic files, so the
    sibling-DeviceNames comparison must not merge them just because their
    headers otherwise match. Mirrors ``valid_chip_id.py``'s own
    ``_chipIdDirFromPath``: walk ancestors nearest-first so an outer
    ``base_arch``-named segment (e.g. an enclosing checkout path) can't
    shadow a real variant directory closer to the file."""
    if not base_arch:
        return None
    for part in reversed(yaml_path.parts[:-1]):
        if part == base_arch:
            return None
        if part.startswith(base_arch + "_"):
            return part
    return None


def _arch_variant_key(yaml_path: Path) -> Tuple[object, object, object, object]:
    """(ScheduleName, base gfx arch, CUCount, chip-ID directory variant)
    identifies one logic *tree* -- e.g. a CU-limited SKU (``gfx942`` at 20
    CUs) legitimately supports a different device subset than the full chip,
    and a chip-ID-specific directory (``gfx950_id75a3``) legitimately covers
    a different device subset than the default ``gfx950`` tree, so neither
    must be merged into a comparison with its unrelated sibling just because
    they share a bare gfx-arch string. ScheduleName/arch/CUCount are read
    from each file's own header rather than inferred from directory depth:
    the corpus does not use one fixed directory shape (some architectures
    nest under a codename directory, e.g. ``aldebaran/gfx90a/...``; others
    use the gfx name as both the top-level directory and the arch, e.g.
    ``gfx1201/...``), and grouping by header content instead of a
    directory-depth assumption is what actually identifies "the same tree"
    in both shapes. The chip-ID variant is the one part of this key that is
    *not* in the header (see ``_chip_id_dir_suffix``), so it's still read
    from the path."""
    try:
        schedule = load_logic_schedule_name(yaml_path)
    except (OSError, RuntimeError):
        schedule = None
    try:
        arch = load_logic_gfx_arch(yaml_path)
    except (OSError, RuntimeError):
        arch = None
    try:
        cu_count = load_logic_cu_count(yaml_path)
    except (OSError, RuntimeError):
        cu_count = None
    chip_id_suffix = _chip_id_dir_suffix(yaml_path, arch) if isinstance(arch, str) else None
    return (_hashable(schedule), _hashable(arch), _hashable(cu_count), chip_id_suffix)


def find_sibling_device_names_violations(
    files: Sequence[Path], logic_root: Path
) -> List[str]:
    """Same-basename logic YAMLs within one logic tree (same ``ScheduleName``,
    gfx arch, and CU count -- see ``_arch_variant_key``) must declare
    identical ``DeviceNames``; a divergence (e.g. one sibling missing a chip
    ID the other declares) shipped invisibly before this check existed --
    see https://github.com/ROCm/rocm-libraries/issues/11397.

    ``files`` is whatever the caller already selected (the full corpus for
    ``--architecture all``, or just the files matching a requested
    architecture otherwise) -- this only compares within that set, it does
    not independently re-walk ``logic_root`` for more files."""
    logic_root = _resolve_corpus_root(logic_root)
    by_variant_and_basename: Dict[
        Tuple[Tuple, str], Dict[Tuple[str, ...], List[Path]]
    ] = defaultdict(lambda: defaultdict(list))
    for yaml_path in files:
        names = read_device_names(yaml_path)
        if names is None:
            continue
        key = (_arch_variant_key(yaml_path), yaml_path.name)
        by_variant_and_basename[key][names].append(yaml_path)

    violations: List[str] = []
    for (variant, basename), dn_map in by_variant_and_basename.items():
        if len(dn_map) > 1:
            schedule, arch, cu_count, chip_id_suffix = variant
            variant_label = (
                f"{schedule}/{arch}"
                + (f"@{cu_count}CU" if cu_count else "")
                + (f" [{chip_id_suffix}]" if chip_id_suffix else "")
            )
            detail = {
                str(names): [str(p.relative_to(logic_root)) for p in paths]
                for names, paths in dn_map.items()
            }
            violations.append(
                f"Divergent sibling DeviceNames: {variant_label}/{basename}: {detail}"
            )
    return violations


_RANGE_DIRNAME = "Range"
_EQUALITY_DIRNAME = "Equality"


def _placeholder_identity(yaml_path: Path) -> Dict[str, object]:
    """The header fields, effective ``PerfMetric`` and effective
    ``ProblemType`` that a Range file and its Equality sibling must share.
    ``ProblemType`` is compared after TensileLite fills in its defaults, so a
    file that writes a default out explicitly and one that omits it compare
    equal; ``PerfMetric`` likewise defaults to ``DeviceEfficiency``, the one
    value that adds nothing to the name. ``DeviceNames`` keeps the file's
    order, unlike ``read_device_names``: ``MasterSolutionLibrary.hardware``
    joins them in that order into the ``_ID`` placeholder suffix."""
    from ..SolutionStructs.Problem import ProblemType

    identity: Dict[str, object] = {
        "ScheduleName": load_logic_schedule_name(yaml_path),
        "ArchitectureName": load_logic_gfx_arch(yaml_path),
        "CUCount": load_logic_cu_count(yaml_path),
        "DeviceNames": [str(name) for name in load_logic_device_names(yaml_path) or []],
        "PerfMetric": load_logic_perf_metric(yaml_path) or "DeviceEfficiency",
    }
    raw = load_logic_problem_type(yaml_path) or {}
    for key, value in ProblemType(copy.deepcopy(raw), False).state.items():
        identity[f"ProblemType.{key}"] = value
    return identity


def find_range_sibling_violations(
    files: Sequence[Path], logic_root: Path
) -> List[str]:
    """A ``Range`` logic file with a same-basename ``Equality`` sibling in its
    logic tree must declare the same ``ScheduleName``, ``ArchitectureName``,
    ``CUCount``, ``DeviceNames``, effective ``PerfMetric`` and effective
    ``ProblemType``. The Range rows are meant to sit in the Equality file's
    placeholder library beside its rows (the gfx950 skinny TN custom kernels
    rely on this), and a Range file whose placeholder fields drift from its
    sibling's gets a placeholder of its own instead. The check is stricter
    than the placeholder name: it also requires fields the name leaves out,
    such as ``ScheduleName`` and every ``ProblemType`` field, to match.

    Only Range files are held to this. Same-basename Equality, GridBased and
    Origami siblings do diverge in the shipped corpus today, so a general
    sibling rule would fail builds over files nothing here is about.

    As with ``find_sibling_device_names_violations``, only pairs where both
    files are in ``files`` are compared."""
    logic_root = _resolve_corpus_root(logic_root)
    selected = set(files)
    violations: List[str] = []
    for range_path in files:
        if range_path.parent.name != _RANGE_DIRNAME:
            continue
        sibling = range_path.parent.parent / _EQUALITY_DIRNAME / range_path.name
        if sibling not in selected:
            continue
        ours = _placeholder_identity(range_path)
        theirs = _placeholder_identity(sibling)
        differing = sorted(
            key for key in ours.keys() | theirs.keys() if ours.get(key) != theirs.get(key)
        )
        if differing:
            detail = ", ".join(
                f"{key}: {ours.get(key)!r} != {theirs.get(key)!r}" for key in differing
            )
            violations.append(
                f"Range logic diverges from its Equality sibling: "
                f"{range_path.relative_to(logic_root)} vs "
                f"{sibling.relative_to(logic_root)}: {detail}"
            )
    return violations


def check_corpus_invariants(
    logic_root: Path,
    files: Optional[Sequence[Path]] = None,
) -> List[str]:
    """Aggregate the corpus-wide invariant checks into one flat violation
    list. Returns an empty list (rather than raising) when ``logic_root``
    isn't a directory -- a single-file ``LogicPath`` invocation has no
    cross-file comparison to make, so these checks are inapplicable rather
    than failing.

    ``files`` should be the caller's own already-selected file list, and
    defaults to "the whole resolved corpus" for hermetic tests and other
    callers that just want to check everything.

    The chip-ID-architecture-lock check (locking chip-ID-aware dispatch to
    ``gfx950``) is intentionally *not* included here: it guards a future
    source-policy change, not the artifact a given build selects, so it's a
    hermetic pytest assertion over the architecture registry (see
    ``Tests/unit/test_PlaceholderMerge.py`` and
    ``find_chip_id_arch_lock_violations``), not a per-build gate."""
    logic_root = Path(logic_root)
    if not logic_root.is_dir():
        return []
    resolved_root = _resolve_corpus_root(logic_root)
    if files is None:
        files = sorted(resolved_root.rglob("*.yaml"))

    return find_sibling_device_names_violations(
        files, resolved_root
    ) + find_range_sibling_violations(files, resolved_root)


def find_chip_id_arch_lock_violations(files: Sequence[Path]) -> List[str]:
    """Lock chip-ID-aware architectures to the current, audited set
    (``gfx950`` only). ``supportsChipIdPredicate`` gates both logic-file
    placement rules (``valid_chip_id.py``) and the ``SolutionLibrary`` placeholder
    suffix; a new architecture silently becoming chip-ID-aware (or ``gfx950``
    silently stopping being one) needs a deliberate re-audit of both, not a
    registry edit that just happens to flip this predicate.

    This is a source-policy check over the architectures the given files
    declare, not a property of any one build's artifact, so it is exercised only
    as a pytest assertion rather than wired into ``check_corpus_invariants()`` /
    ``TensileLogic --check-all`` -- every from-source build re-checking the
    whole architecture registry on every invocation would add no additional
    safety over running it once in CI's unit-test job. The registry-wide form
    of the lock, which needs no corpus, is
    ``test_PlaceholderMerge.py::test_supports_chip_id_predicate_only_gfx950``."""
    from ..Common.Architectures import supportsChipIdPredicate

    violations: List[str] = []
    seen_archs = set()
    for yaml_path in files:
        arch = load_logic_gfx_arch(yaml_path)
        if not isinstance(arch, str) or arch in seen_archs:
            continue
        seen_archs.add(arch)
        expected = arch == "gfx950"
        actual = supportsChipIdPredicate(arch)
        if actual is not expected:
            violations.append(
                f"Chip-ID-arch-lock violation: {arch}: "
                f"supportsChipIdPredicate={actual}, expected={expected} -- new "
                "chip-ID-aware architectures require a re-audit of logic YAML "
                "placement rules and the SolutionLibrary suffix gate"
            )
    return violations


def report_corpus_invariant_violations(violations: List[str]) -> None:
    for violation in violations:
        print(f"Error: {violation}", file=sys.stderr)
