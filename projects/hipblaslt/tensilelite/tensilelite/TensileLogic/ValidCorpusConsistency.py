# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
ValidCorpusConsistency
---
Cross-file consistency checks for the library logic tree, run by
``TensileLogic --check-all`` in addition to the existing per-file/per-solution
validators.

These checks are different in kind from the other ``Valid*.py`` validators:
they are cheap (one streaming YAML read per file -- no solution construction)
and mostly need visibility across more than one file to do their job. The
rocRoller macrotile check is the exception on both counts: it is per-file, and
it reads the ``Solutions`` list rather than only the header. It lives here
because it shares the same "cheap pre-build gate, no known-bugs escape hatch"
contract as the rest. The checks below are scoped the same way the caller's
own per-solution validation already is: to
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

import re
import sys

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..CustomYamlLoader import (
    DEFAULT_YAML_LOADER,
    load_logic_cu_count,
    load_logic_device_names,
    load_logic_gfx_arch,
    load_logic_schedule_name,
    load_yaml_dict_item,
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


_RR_KERNEL_PREFIX = "RR_GEMM_"
_WGT_TOKEN = re.compile(r"_WGT_(\d+)x(\d+)x(\d+)_")


def find_rocroller_macrotile_violations(
    files: Sequence[Path],
    logic_root: Path,
) -> List[str]:
    """Check that each rocRoller custom kernel declares a usable ``macrotile``
    that agrees with its own name.

    These kernels are pre-assembled objects captured at one tile. The size
    predicates that stop a kernel being handed a shape it was never captured
    for are derived at library-build time from this ``macrotile`` value, so a
    missing or malformed one does not fail the build; it silently yields a
    kernel with nothing bounding its M and K. The value is written into the
    logic file at tuning time and is not reconciled against the assembly
    afterwards, which is what makes an independent check worth having.

    The kernel's own name carries the same tile as a ``_WGT_<m>x<n>x<k>_``
    token, so the two can be checked against each other without reading any
    assembly. That catches a disagreement between them; it cannot catch both
    copies being wrong in the same way, which is what the corpus predicate test
    in ``Tests/unit/test_rocroller_predicate_corpus.py`` covers from the other
    end.

    Unlike the other checks here this one reads the ``Solutions`` list rather
    than the file header, so it is not header-only. It is still one streaming
    read per file with no solution construction.
    """
    violations: List[str] = []
    for yaml_path in files:
        # ``Solutions`` is the last top-level key, so asking the streaming
        # loader for it parses the whole file -- about 50 ms each, and the
        # corpus has thousands of files, most of them holding no rocRoller
        # kernel at all. Scanning the raw bytes for the prefix first skips the
        # parse for those. It keys on the same prefix the parse does rather
        # than on a directory layout, so a kernel registered somewhere new is
        # still checked.
        try:
            if _RR_KERNEL_PREFIX.encode() not in yaml_path.read_bytes():
                continue
        except OSError:
            continue
        try:
            solutions = load_yaml_dict_item(yaml_path, DEFAULT_YAML_LOADER, "Solutions")
        except Exception:  # noqa: BLE001 - malformed files are another check's job
            continue
        if not isinstance(solutions, list):
            continue
        for solution in solutions:
            if not isinstance(solution, dict):
                continue
            custom_kernel = solution.get("CustomKernel")
            if not isinstance(custom_kernel, dict):
                continue
            name = custom_kernel.get("name")
            if not isinstance(name, str) or not name.startswith(_RR_KERNEL_PREFIX):
                continue

            try:
                where = f"{yaml_path.relative_to(logic_root)}: {name}"
            except ValueError:
                where = f"{yaml_path}: {name}"

            macrotile = custom_kernel.get("macrotile")
            if (
                not isinstance(macrotile, list)
                or len(macrotile) != 3
                or not all(isinstance(v, int) and v > 0 for v in macrotile)
            ):
                violations.append(
                    f"rocRoller macrotile violation: {where}: macrotile is "
                    f"{macrotile!r}, expected three positive integers -- the M "
                    "and K size predicates are derived from it"
                )
                continue

            token = _WGT_TOKEN.search(name)
            if token is None:
                violations.append(
                    f"rocRoller macrotile violation: {where}: name carries no "
                    "_WGT_<m>x<n>x<k>_ token to check the macrotile against"
                )
                continue

            from_name = [int(group) for group in token.groups()]
            if from_name != list(macrotile):
                violations.append(
                    f"rocRoller macrotile violation: {where}: macrotile "
                    f"{list(macrotile)} disagrees with the {from_name} in its "
                    "own name"
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
    ``Tests/unit/test_valid_corpus_consistency.py`` and
    ``find_chip_id_arch_lock_violations``), not a per-build gate."""
    logic_root = Path(logic_root)
    if not logic_root.is_dir():
        return []
    resolved_root = _resolve_corpus_root(logic_root)
    if files is None:
        files = sorted(resolved_root.rglob("*.yaml"))

    return list(find_sibling_device_names_violations(files, resolved_root)) + list(
        find_rocroller_macrotile_violations(files, resolved_root)
    )


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
