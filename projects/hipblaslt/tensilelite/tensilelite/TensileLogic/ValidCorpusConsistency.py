# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
ValidCorpusConsistency
---
Cross-file consistency checks for the library logic tree, run by
``TensileLogic --check-all`` in addition to the existing per-file/per-solution
validators.

These checks are different in kind from the other ``Valid*.py`` validators:
they are cheap (header-only YAML reads -- no per-solution parsing) but need
visibility across more than one file to do their job. Both checks below are
scoped the same way the caller's own per-solution validation already is: to
the files ``TensileLogic`` selected for the requested ``--architecture``
(the full corpus when ``--architecture all`` is given). There's no separate
"whole corpus" mode here -- if a gfx942-only build doesn't touch gfx1250
data, a gfx1250-only violation should not fail it.

No known-bugs / quarantine escape hatch exists for these checks (unlike the
per-solution validators, which can accept a documented ``known_bugs.yaml``
entry). A violation here is always a hard failure. If a future violation
needs a documented, temporary exception, extend the known-bugs schema (see
``KnownBugs.py``) deliberately -- do not assume one already covers these
checks.
"""

import sys

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from Tensile.CustomYamlLoader import (
    load_logic_cu_count,
    load_logic_device_names,
    load_logic_gfx_arch,
    load_logic_schedule_name,
)

GFX1250 = "gfx1250"
GFX1250V0 = "gfx1250v0"

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
    default tree's -- ``ValidChipId.py``'s placement rules, not header
    content, are what distinguish per-chip-ID logic files, so the
    sibling-DeviceNames comparison must not merge them just because their
    headers otherwise match. Mirrors ``ValidChipId.py``'s own
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


def find_gfx1250v0_overlay_violations(
    logic_root: Path,
    files: Optional[Sequence[Path]] = None,
    overlay_required: bool = False,
) -> List[str]:
    """gfx1250 ships as two silicon revisions (v0, v1) sharing one ISA, arch
    name, and compiler target; the runtime tells them apart only via
    ``hipDeviceProp_t::asicRevision``, and ``TensileCreateLibrary`` globs one
    tree, separating the revisions by ``ScheduleName`` alone. A mis-tagged
    file fails silently -- dropped from v0, or leaked into every v1 build --
    so this checks the invariant against the tree that actually ships.

    ``files`` is the same caller-selected list ``find_sibling_device_names_
    violations`` takes (the full corpus by default, or the already
    ``--architecture``/``Experimental``-filtered set ``Run.py`` passes) --
    both the overlay-contents and outside-overlay scans below are scoped to
    it, not a fresh unconditional walk of ``logic_root``, so an excluded
    file can't trip a violation the caller never selected. The
    directory-*exists* requirement is independent of file selection, since
    it is checking that the tree the runtime will actually read exists at
    all, not something about the selected files:

    1. if ``overlay_required`` -- an explicit caller policy, not implied by
       ``--architecture`` (see below) -- the ``gfx1250v0`` overlay directory
       must exist on disk. Not every ``TensileLogic``-checked corpus does a
       v0/v1 split for gfx1250 in the first place -- e.g. hipSPARSELt's
       corpus has no ``gfx1250v0`` directory at all -- so a missing overlay
       is only a violation when the caller actually owns the split and says
       so;
    2. an *existing but empty* overlay (empty within the selected ``files``)
       is always a violation, regardless of ``overlay_required``: a v0 build
       reports success having written a library with no solutions in it,
       and any tree deliberately doing the split should never leave the
       overlay directory present-but-empty;
    3. every selected file inside the overlay declares
       ``ScheduleName: gfx1250v0``;
    4. every selected file inside the overlay keeps
       ``ArchitectureName: gfx1250`` (a stepping there is rejected by
       ``TensileCreateLibrary``, and ``library/gfx1250v0/`` is a directory
       the runtime never reads); and
    5. no selected file *outside* the overlay claims
       ``ScheduleName: gfx1250v0`` (checked regardless of whether the
       overlay directory exists).

    ``overlay_required`` must be an explicit, caller-owned policy rather
    than inferred from whether ``gfx1250v0`` was requested: a shared
    invocation (e.g. hipSPARSELt's, which never did a gfx1250 v0/v1 split)
    can still be asked to build for architecture ``gfx1250v0`` without that
    request implying its corpus owns the overlay. Only hipBLASLt's dedicated
    gfx1250v0 device-library build (see ``device-library/CMakeLists.txt``,
    and ``--require-gfx1250v0-overlay`` in ``ParseArguments.py``) sets this
    ``True``.
    """
    logic_root = _resolve_corpus_root(logic_root)
    if files is None:
        files = sorted(logic_root.rglob("*.yaml"))
    violations: List[str] = []
    overlay_root = logic_root / GFX1250V0
    overlay_exists = overlay_root.is_dir()
    overlay_files = (
        sorted(p for p in files if p.is_relative_to(overlay_root)) if overlay_exists else []
    )

    if overlay_required and not overlay_exists:
        violations.append(
            f"{GFX1250V0} overlay required but missing under {overlay_root}"
        )
    elif overlay_exists and not overlay_files:
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

    for p in sorted(files):
        if overlay_exists and p.is_relative_to(overlay_root):
            continue
        if load_logic_schedule_name(p) == GFX1250V0:
            violations.append(
                f"{p.relative_to(logic_root)}: declares ScheduleName "
                f"{GFX1250V0!r} outside the {GFX1250V0} overlay directory"
            )

    return violations


def check_corpus_invariants(
    logic_root: Path,
    files: Optional[Sequence[Path]] = None,
    archs: Optional[Sequence[str]] = None,
    overlay_required: bool = False,
) -> List[str]:
    """Aggregate the corpus-wide invariant checks into one flat violation
    list. Returns an empty list (rather than raising) when ``logic_root``
    isn't a directory -- a single-file ``LogicPath`` invocation has no
    cross-file comparison to make, so these checks are inapplicable rather
    than failing.

    ``files`` and ``archs`` should be the caller's own already-selected file
    list and requested ``--architecture`` values (``["all"]`` when no
    filtering applies); both default to "the whole resolved corpus" for
    hermetic tests and other callers that just want to check everything.

    The gfx1250v0-overlay check runs whenever ``gfx1250v0`` or ``all`` is in
    ``archs``, or ``overlay_required`` is set. ``overlay_required`` is a
    separate, explicit caller policy, deliberately *not* inferred from
    ``archs`` -- requesting architecture ``gfx1250v0`` does not by itself
    mean the corpus being validated owns a gfx1250/gfx1250v0 split (e.g. a
    shared invocation can ask for that architecture against hipSPARSELt's
    corpus, which never did the split and has no overlay directory at all).
    Only pass ``True`` when the caller actually owns that split (hipBLASLt's
    dedicated gfx1250v0 device-library build; see
    ``find_gfx1250v0_overlay_violations``).

    The chip-ID-architecture-lock check (locking chip-ID-aware dispatch to
    ``gfx950``) is intentionally *not* included here: it guards a future
    source-policy change, not the artifact a given build selects, so it's a
    hermetic pytest assertion over the architecture registry (see
    ``Tests/unit/test_valid_corpus_consistency.py`` and
    ``find_chip_id_arch_lock_violations``), not a per-build gate."""
    logic_root = Path(logic_root)
    if not logic_root.is_dir():
        return []
    resolved_root = _resolve_corpus_root(logic_root)
    if files is None:
        files = sorted(resolved_root.rglob("*.yaml"))
    if archs is None:
        archs = ["all"]

    violations = list(find_sibling_device_names_violations(files, resolved_root))
    if overlay_required or GFX1250V0 in archs or "all" in archs:
        violations += find_gfx1250v0_overlay_violations(
            resolved_root, files, overlay_required=overlay_required
        )
    return violations


def find_chip_id_arch_lock_violations(files: Sequence[Path]) -> List[str]:
    """Lock chip-ID-aware architectures to the current, audited set
    (``gfx950`` only). ``supportsChipIdPredicate`` gates both logic-file
    placement rules (``ValidChipId.py``) and the ``SolutionLibrary`` placeholder
    suffix; a new architecture silently becoming chip-ID-aware (or ``gfx950``
    silently stopping being one) needs a deliberate re-audit of both, not a
    registry edit that just happens to flip this predicate.

    This is a source-policy check over the set of architectures the corpus
    currently contains, not a property of any one build's artifact, so it is
    exercised only as a pytest assertion (hermetically, and against the real
    corpus in ``test_PlaceholderMerge.py``) rather than wired into
    ``check_corpus_invariants()`` / ``TensileLogic --check-all`` -- every
    from-source build re-checking the whole architecture registry on every
    invocation would add no additional safety over running it once in CI's
    unit-test job."""
    from Tensile.Common.Architectures import supportsChipIdPredicate

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
