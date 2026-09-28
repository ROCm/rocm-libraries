# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Hermetic unit tests for ``Tensile.TensileLogic.ValidCorpusConsistency``.

Everything here builds its own tiny corpus under ``tmp_path`` -- no dependency
on the real ``Logic/asm_full`` checkout, unlike ``test_PlaceholderMerge.py`` /
``test_GpuRevisionTarget.py``, whose corpus-backed copies of these same checks
are skipped when that directory is absent. This is new logic being pinned,
not existing behavior being characterized, so plain asserts are used rather
than snapshots.
"""

import importlib.util
import sys
import types

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


# Load ValidCorpusConsistency.py via importlib to bypass
# Tensile/TensileLogic/__init__.py, which transitively imports joblib / heavy
# build deps via Run.py (see test_ValidChipId.py for the same pattern).
def _load_vcc_mod():
    p = Path(__file__).resolve().parents[2] / "TensileLogic" / "ValidCorpusConsistency.py"
    spec = importlib.util.spec_from_file_location("ValidCorpusConsistency_under_test", p)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _install_rocisa_stub(monkeypatch):
    # When the rocisa C-extension is not importable (e.g. CI lint job), install
    # a minimal fixture-scoped stub so this module does not pollute sys.modules
    # for the rest of the pytest session.
    try:  # pragma: no cover - environment-dependent
        from rocisa import rocIsa  # noqa: F401
        return
    except ImportError:  # pragma: no cover
        _rocisa_stub = types.ModuleType("rocisa")

        class _RocIsaInstanceStub:  # noqa: D401 - test helper
            @staticmethod
            def getData():
                return {}

        class _RocIsaStub:  # noqa: D401 - test helper
            @staticmethod
            def getInstance():
                return _RocIsaInstanceStub()

        _rocisa_stub.rocIsa = _RocIsaStub
        monkeypatch.setitem(sys.modules, "rocisa", _rocisa_stub)


@pytest.fixture
def vcc(monkeypatch):
    _install_rocisa_stub(monkeypatch)
    return _load_vcc_mod()


def _all_yaml(root: Path):
    return sorted(root.rglob("*.yaml"))


# ===========================================================================
# Shared helpers
# ===========================================================================

def _write_header_yaml(path, *, schedule="schedule", gfx="gfx942", devices="Device 74a0"):
    """A minimal logic YAML: just enough header lines for read_device_names /
    load_logic_schedule_name / load_logic_gfx_arch to find what they need."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "- MinimumRequiredVersion: 4.33.0",
                f"- {schedule}",
                f"- {gfx}",
                f"- [{devices}]",
                "",
            ]
        )
    )
    return path


def _write_header_yaml_no_device_names(path, *, schedule="schedule", gfx="gfx942"):
    """A structurally valid header that simply omits the DeviceNames item --
    an unmet precondition, not a parse failure; distinct from a genuinely
    unparseable file (see test_read_device_names_returns_none_for_unparseable_content)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "- MinimumRequiredVersion: 4.33.0",
                f"- {schedule}",
                f"- {gfx}",
                "",
            ]
        )
    )
    return path


def _write_multiline_header_yaml(path, *, schedule="schedule", gfx="gfx942", device_lines=("Device 0050,", "    Device 740c")):
    """A DeviceNames header that wraps onto multiple physical lines, the way
    the real aldebaran corpus files do. A regex line-scan limited to a single
    line silently drops these; the event-based YAML parser must not."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "- MinimumRequiredVersion: 4.33.0",
        f"- {schedule}",
        f"- {gfx}",
        f"- [{device_lines[0]}",
    ]
    lines.extend(f"    {line}" for line in device_lines[1:-1])
    lines.append(f"    {device_lines[-1]}]")
    path.write_text("\n".join(lines) + "\n")
    return path


def _write_mapping_form_header_yaml(path, *, schedule="schedule", gfx="gfx942", devices="Device 74a0"):
    """A logic YAML in the whole-file mapping dialect used by e.g. Origami
    (every header field, including ``DeviceNames``, is a top-level mapping
    key), as opposed to the positional-list-of-sequence-items form
    ``_write_header_yaml`` produces. Mirrors a real Origami file's header
    (``MinimumRequiredVersion``/``ScheduleName``/``ArchitectureName``/
    ``DeviceNames`` all as mapping keys) rather than mixing a top-level
    sequence with a top-level mapping key, which is not valid YAML. See
    #11442."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "MinimumRequiredVersion: 4.33.0",
                f"ScheduleName: {schedule}",
                f"ArchitectureName: {gfx}",
                f"DeviceNames: [{devices}]",
                "",
            ]
        )
    )
    return path


def _write_cu_variant_header_yaml(path, *, schedule="schedule", gfx="gfx942", cu_count=20, devices="Device 74a0"):
    """A CU-limited SKU's header: the positional dialect nests CUCount inside
    the same sequence item as the gfx arch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "- MinimumRequiredVersion: 4.33.0",
                f"- {schedule}",
                f"- {{Architecture: {gfx}, CUCount: {cu_count}}}",
                f"- [{devices}]",
                "",
            ]
        )
    )
    return path


def _write_overlay_yaml(path, *, schedule, gfx):
    """gfx1250v0-overlay tests only care about ScheduleName / gfx arch, but
    write the same full header shape ``_write_header_yaml`` does (rather than
    a bare positional list of scalars) so these stay representative of the
    real logic-file dialect that ``load_logic_schedule_name()`` /
    ``load_logic_gfx_arch()`` parse."""
    return _write_header_yaml(path, schedule=schedule, gfx=gfx)


# ===========================================================================
# read_device_names
# ===========================================================================

def test_read_device_names_parses_header_line(tmp_path, vcc):
    f = _write_header_yaml(tmp_path / "a.yaml", devices="Device 74a0, Device 74a1")
    assert vcc.read_device_names(f) == ("74a0", "74a1")


def test_read_device_names_parses_mapping_form_header_line(tmp_path, vcc):
    # Origami and similar logic files write DeviceNames as a mapping
    # (``DeviceNames: [Device ...]``) rather than the positional list form;
    # both dialects must parse to the same result. See #11442.
    f = _write_mapping_form_header_yaml(tmp_path / "a.yaml", devices="Device 74a0, Device 74a1")
    assert vcc.read_device_names(f) == ("74a0", "74a1")


def test_read_device_names_parses_a_multiline_device_list(tmp_path, vcc):
    # The real, checked-in aldebaran corpus wraps DeviceNames onto a second
    # physical line; a single-line regex scan would silently miss these.
    f = _write_multiline_header_yaml(
        tmp_path / "a.yaml",
        device_lines=("Device 0050,", "Device 0051,", "Device 740c"),
    )
    assert vcc.read_device_names(f) == ("0050", "0051", "740c")


def test_read_device_names_returns_empty_tuple_when_field_absent(tmp_path, vcc):
    # A structurally valid header missing the DeviceNames field is a distinct,
    # comparable value (empty tuple) -- not dropped from sibling comparison
    # the way an unreadable file is.
    f = _write_header_yaml_no_device_names(tmp_path / "a.yaml")
    assert vcc.read_device_names(f) == ()


def test_read_device_names_returns_none_for_unparseable_content(tmp_path, vcc):
    f = tmp_path / "a.yaml"
    f.write_text("not a logic header at all\n")
    assert vcc.read_device_names(f) is None


def test_read_device_names_returns_none_for_missing_file(tmp_path, vcc):
    assert vcc.read_device_names(tmp_path / "does_not_exist.yaml") is None


# ===========================================================================
# _resolve_corpus_root / ancestor-root invocation
#
# ``TensileLogic``'s own ``LogicPath`` CLI argument does not, in the default
# CMake-driven build, point at the ``asm_full`` corpus root directly -- it
# defaults (via ``HIPBLASLT_LIBLOGIC_PATH``) to the whole ``library/`` tree,
# several directories above the real corpus. Every finder must still work
# when called with that higher ancestor, not just with the corpus root
# itself (which is all the other tests in this file exercise).
# ===========================================================================

def test_resolve_corpus_root_returns_input_unchanged_when_already_asm_full(tmp_path, vcc):
    asm_full = tmp_path / "asm_full"
    asm_full.mkdir()
    assert vcc._resolve_corpus_root(asm_full) == asm_full


def test_resolve_corpus_root_finds_a_nested_asm_full_directory(tmp_path, vcc):
    # Mirrors HIPBLASLT_LIBLOGIC_PATH's default: LogicPath is `library`, and
    # the real corpus is nested several directories below it.
    library = tmp_path / "library"
    asm_full = library / "src" / "amd_detail" / "rocblaslt" / "src" / "Tensile" / "Logic" / "asm_full"
    asm_full.mkdir(parents=True)
    assert vcc._resolve_corpus_root(library) == asm_full


def test_resolve_corpus_root_falls_back_to_input_when_no_asm_full_exists(tmp_path, vcc):
    # Hermetic tmp_path fixtures (as used throughout this file) build a
    # synthetic corpus directly under tmp_path, with no `asm_full` directory
    # anywhere -- resolution must be a no-op rather than raising or finding a
    # false match.
    (tmp_path / "aldebaran" / "gfx950").mkdir(parents=True)
    assert vcc._resolve_corpus_root(tmp_path) == tmp_path


def test_resolve_corpus_root_skips_a_non_directory_asm_full_match(tmp_path, vcc):
    # A stray *file* named "asm_full" (e.g. a leftover build artifact) must
    # not be mistaken for the real corpus directory -- rglob() matches it by
    # name, but the is_dir() check must reject it and keep looking, falling
    # back to the input if nothing else matches.
    (tmp_path / "aldebaran" / "asm_full").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "aldebaran" / "asm_full").write_text("n/a")
    assert vcc._resolve_corpus_root(tmp_path) == tmp_path


def test_gfx1250v0_overlay_violations_identical_via_ancestor_or_direct_corpus_root(tmp_path, vcc):
    # The exact shape that broke PR #11447's own CI: TensileLogic invoked
    # with `library` (an ancestor of asm_full), not asm_full directly. An
    # unresolved ancestor previously reported both a false "ships no logic"
    # violation (overlay_root computed at the wrong, nonexistent path) and a
    # false "outside the overlay" violation for every real overlay file.
    asm_full = tmp_path / "library" / "src" / "amd_detail" / "rocblaslt" / "src" / "Tensile" / "Logic" / "asm_full"
    _write_overlay_yaml(
        asm_full / vcc.GFX1250V0 / "Equality" / "logic.yaml",
        schedule=vcc.GFX1250V0, gfx=vcc.GFX1250,
    )
    _write_overlay_yaml(
        asm_full / "gfx1250" / "Equality" / "logic.yaml",
        schedule="gfx1250", gfx=vcc.GFX1250,
    )
    assert vcc.find_gfx1250v0_overlay_violations(asm_full) == []
    assert vcc.find_gfx1250v0_overlay_violations(tmp_path / "library") == []


def test_sibling_device_names_violations_identical_via_ancestor_or_direct_corpus_root(tmp_path, vcc):
    asm_full = tmp_path / "library" / "src" / "amd_detail" / "rocblaslt" / "src" / "Tensile" / "Logic" / "asm_full"
    _write_header_yaml(
        asm_full / "aldebaran" / "gfx950" / "Equality" / "logic.yaml",
        devices="Device 75a0",
    )
    _write_header_yaml(
        asm_full / "aldebaran" / "gfx950" / "GridBased" / "logic.yaml",
        devices="Device 75a3",
    )
    direct = vcc.find_sibling_device_names_violations(_all_yaml(asm_full), asm_full)
    via_ancestor = vcc.find_sibling_device_names_violations(
        _all_yaml(asm_full), tmp_path / "library"
    )
    assert len(direct) == 1
    assert direct == via_ancestor


# ===========================================================================
# find_sibling_device_names_violations
# ===========================================================================

def test_sibling_device_names_clean_when_consistent(tmp_path, vcc):
    _write_header_yaml(
        tmp_path / "aldebaran" / "gfx950" / "Equality" / "logic.yaml",
        devices="Device 75a0",
    )
    _write_header_yaml(
        tmp_path / "aldebaran" / "gfx950" / "GridBased" / "logic.yaml",
        devices="Device 75a0",
    )
    assert vcc.find_sibling_device_names_violations(_all_yaml(tmp_path), tmp_path) == []


def test_sibling_device_names_flags_mismatched_siblings(tmp_path, vcc):
    # Same basename ("logic.yaml"), same arch tree, divergent DeviceNames --
    # exactly the shape of https://github.com/ROCm/rocm-libraries/issues/11397.
    _write_header_yaml(
        tmp_path / "aldebaran" / "gfx950" / "Equality" / "logic.yaml",
        devices="Device 75a0",
    )
    _write_header_yaml(
        tmp_path / "aldebaran" / "gfx950" / "GridBased" / "logic.yaml",
        devices="Device 75a0, Device 75a3",
    )
    violations = vcc.find_sibling_device_names_violations(_all_yaml(tmp_path), tmp_path)
    assert len(violations) == 1
    assert "logic.yaml" in violations[0]
    assert "aldebaran/gfx950" in violations[0]


def test_sibling_device_names_flags_mismatch_against_a_mapping_form_sibling(tmp_path, vcc):
    # The exact shape #11442 found and fixed: one sibling in the positional
    # list dialect, the other (e.g. an Origami file) in the mapping dialect.
    # Before the regex matched both, the mapping-form file's DeviceNames read
    # as None, so this divergence was silently skipped rather than flagged.
    _write_header_yaml(
        tmp_path / "gfx1250" / "gfx1250" / "Equality" / "logic.yaml",
        devices="Device 73f0, Device 0073, Device 75c1",
    )
    _write_mapping_form_header_yaml(
        tmp_path / "gfx1250" / "gfx1250" / "Origami" / "logic.yaml",
        devices="Device 73f0",
    )
    violations = vcc.find_sibling_device_names_violations(_all_yaml(tmp_path), tmp_path)
    assert len(violations) == 1
    assert "logic.yaml" in violations[0]


def test_sibling_device_names_flags_mismatch_against_a_multiline_sibling(tmp_path, vcc):
    # A real, present-day corpus shape: one sibling's DeviceNames wraps onto
    # a second physical line. Before load_logic_device_names() replaced the
    # 8-line single-line regex scan, this file's DeviceNames read as None and
    # the divergence went unnoticed.
    _write_header_yaml(
        tmp_path / "aldebaran" / "gfx90a" / "Equality" / "logic.yaml",
        gfx="gfx90a",
        devices="Device 0050",
    )
    _write_multiline_header_yaml(
        tmp_path / "aldebaran" / "gfx90a" / "GridBased" / "logic.yaml",
        gfx="gfx90a",
        device_lines=("Device 0050,", "Device 0051"),
    )
    violations = vcc.find_sibling_device_names_violations(_all_yaml(tmp_path), tmp_path)
    assert len(violations) == 1


def test_sibling_device_names_flags_a_missing_device_names_field_against_a_declared_sibling(tmp_path, vcc):
    # A sibling with a structurally valid header but no DeviceNames field at
    # all must not be silently dropped -- it's a distinct, comparable ()
    # value, so it diverges from a sibling that does declare device names.
    _write_header_yaml(
        tmp_path / "aldebaran" / "gfx950" / "Equality" / "logic.yaml",
        devices="Device 75a0",
    )
    _write_header_yaml_no_device_names(
        tmp_path / "aldebaran" / "gfx950" / "GridBased" / "logic.yaml",
    )
    violations = vcc.find_sibling_device_names_violations(_all_yaml(tmp_path), tmp_path)
    assert len(violations) == 1


def test_sibling_device_names_checks_a_direct_gfx_named_layout(tmp_path, vcc):
    # The real, checked-in corpus shape for e.g. gfx1201/navi31/gfx1200: the
    # top-level directory *is* the gfx arch itself (no separate codename
    # directory above it), with category directories directly beneath it.
    # A directory-depth-based walker that assumes a codename directory
    # always sits above the arch directory misses this shape entirely.
    _write_header_yaml(
        tmp_path / "gfx1201" / "Equality" / "logic.yaml",
        schedule="gfx1201", gfx="gfx1201", devices="Device 1111",
    )
    _write_header_yaml(
        tmp_path / "gfx1201" / "GridBased" / "logic.yaml",
        schedule="gfx1201", gfx="gfx1201", devices="Device 2222",
    )
    violations = vcc.find_sibling_device_names_violations(_all_yaml(tmp_path), tmp_path)
    assert len(violations) == 1
    assert "logic.yaml" in violations[0]


def test_sibling_device_names_ignores_different_basenames(tmp_path, vcc):
    # Different basenames in the same arch dir may legitimately declare
    # different DeviceNames; only same-basename siblings are compared.
    _write_header_yaml(
        tmp_path / "aldebaran" / "gfx950" / "Equality" / "a.yaml",
        devices="Device 75a0",
    )
    _write_header_yaml(
        tmp_path / "aldebaran" / "gfx950" / "Equality" / "b.yaml",
        devices="Device 75a3",
    )
    assert vcc.find_sibling_device_names_violations(_all_yaml(tmp_path), tmp_path) == []


def test_sibling_device_names_does_not_merge_different_cu_variants(tmp_path, vcc):
    # A CU-limited SKU (e.g. gfx942 @ 20 CUs) legitimately supports a
    # different device subset than the full chip -- these must not be
    # compared against each other just because they share a bare gfx arch.
    _write_header_yaml(
        tmp_path / "aquavanjaram" / "gfx942" / "Equality" / "logic.yaml",
        schedule="aquavanjaram", gfx="gfx942", devices="Device 74a0, Device 74a1",
    )
    _write_cu_variant_header_yaml(
        tmp_path / "aquavanjaram" / "gfx942_20cu" / "Equality" / "logic.yaml",
        schedule="aquavanjaram", gfx="gfx942", cu_count=20, devices="Device 74a2",
    )
    assert vcc.find_sibling_device_names_violations(_all_yaml(tmp_path), tmp_path) == []


def test_sibling_device_names_does_not_merge_chip_id_directory_variants(tmp_path, vcc):
    # The real, checked-in gfx950 corpus: a chip-ID-specific directory
    # (gfx950_id75a3) sits alongside the default gfx950 tree, and both
    # declare *identical* ScheduleName/ArchitectureName headers (the header
    # carries no chip-ID information at all -- ValidChipId.py's placement
    # rules are what distinguish them). These must not be compared against
    # each other even though (schedule, arch, CUCount) alone would collide.
    _write_header_yaml(
        tmp_path / "gfx950" / "gfx950" / "Equality" / "logic.yaml",
        schedule="gfx950", gfx="gfx950", devices="Device 75a0",
    )
    _write_header_yaml(
        tmp_path / "gfx950" / "gfx950_id75a3" / "Equality" / "logic.yaml",
        schedule="gfx950", gfx="gfx950", devices="Device 75a2, Device 75a3",
    )
    assert vcc.find_sibling_device_names_violations(_all_yaml(tmp_path), tmp_path) == []


def test_sibling_device_names_still_flags_true_divergence_within_a_chip_id_variant(tmp_path, vcc):
    _write_header_yaml(
        tmp_path / "gfx950" / "gfx950_id75a3" / "Equality" / "logic.yaml",
        schedule="gfx950", gfx="gfx950", devices="Device 75a3",
    )
    _write_header_yaml(
        tmp_path / "gfx950" / "gfx950_id75a3" / "GridBased" / "logic.yaml",
        schedule="gfx950", gfx="gfx950", devices="Device 75a3, Device 75a2",
    )
    violations = vcc.find_sibling_device_names_violations(_all_yaml(tmp_path), tmp_path)
    assert len(violations) == 1


def test_sibling_device_names_tolerates_a_malformed_unhashable_header(tmp_path, vcc):
    # Two real, checked-in hipBLASLt files put a {Architecture, CUCount}
    # mapping in the ScheduleName slot instead of a plain codename string --
    # a pre-existing corpus data quirk unrelated to this check. It must not
    # crash the whole comparison; the malformed file becomes its own
    # singleton group (no sibling has the same malformed key) rather than
    # raising on an unhashable dict.
    malformed = tmp_path / "aquavanjaram" / "gfx942_152cu" / "GridBased" / "logic.yaml"
    malformed.parent.mkdir(parents=True, exist_ok=True)
    malformed.write_text(
        "\n".join(
            [
                "- {MinimumRequiredVersion: 5.0.0}",
                "- {Architecture: gfx942, CUCount: 152}",
                "- gfx942",
                "- [Device 0049]",
                "",
            ]
        )
    )
    _write_header_yaml(
        tmp_path / "aquavanjaram" / "gfx942_152cu" / "Equality" / "logic.yaml",
        schedule="aquavanjaram", gfx="gfx942", devices="Device 0049",
    )
    violations = vcc.find_sibling_device_names_violations(_all_yaml(tmp_path), tmp_path)
    assert violations == []


def test_sibling_device_names_flags_mismatched_cu_variant_siblings(tmp_path, vcc):
    # Two files that do share the same (schedule, arch, CUCount) tree must
    # still be compared against each other.
    _write_cu_variant_header_yaml(
        tmp_path / "aquavanjaram" / "gfx942_20cu" / "Equality" / "logic.yaml",
        schedule="aquavanjaram", gfx="gfx942", cu_count=20, devices="Device 74a2",
    )
    _write_cu_variant_header_yaml(
        tmp_path / "aquavanjaram" / "gfx942_20cu" / "GridBased" / "logic.yaml",
        schedule="aquavanjaram", gfx="gfx942", cu_count=20, devices="Device 74a3",
    )
    violations = vcc.find_sibling_device_names_violations(_all_yaml(tmp_path), tmp_path)
    assert len(violations) == 1


def test_sibling_device_names_skips_a_file_that_is_entirely_unparseable(tmp_path, vcc):
    # A same-basename sibling that read_device_names() can't parse at all
    # (e.g. a malformed or unusually-shaped header) must be skipped rather
    # than treated as an empty-tuple DeviceNames that would falsely diverge
    # from every real sibling. Distinct from
    # test_sibling_device_names_flags_a_missing_device_names_field_against_a_declared_sibling,
    # where the header itself is valid but simply omits DeviceNames.
    _write_header_yaml(
        tmp_path / "aldebaran" / "gfx950" / "Equality" / "logic.yaml",
        devices="Device 75a0",
    )
    unparseable = tmp_path / "aldebaran" / "gfx950" / "GridBased" / "logic.yaml"
    unparseable.parent.mkdir(parents=True, exist_ok=True)
    unparseable.write_text("not a logic header at all\n")
    assert vcc.find_sibling_device_names_violations(_all_yaml(tmp_path), tmp_path) == []


def test_sibling_device_names_respects_a_caller_supplied_file_subset(tmp_path, vcc):
    # This is how Run.py scopes the check to a requested --architecture: the
    # caller passes only the files it already selected, not a fresh
    # unfiltered walk of logic_root. A divergent pair that both live outside
    # that subset must not be reported.
    _write_header_yaml(
        tmp_path / "aldebaran" / "gfx950" / "Equality" / "logic.yaml",
        devices="Device 75a0",
    )
    divergent_sibling = _write_header_yaml(
        tmp_path / "aldebaran" / "gfx950" / "GridBased" / "logic.yaml",
        devices="Device 75a3",
    )
    only_one_side = [
        f for f in _all_yaml(tmp_path) if f != divergent_sibling
    ]
    assert vcc.find_sibling_device_names_violations(only_one_side, tmp_path) == []


# ===========================================================================
# find_chip_id_arch_lock_violations
#
# Not wired into check_corpus_invariants() / TensileLogic --check-all -- see
# that function's docstring. Exercised directly here (and against the real
# corpus in test_PlaceholderMerge.py) as a source-policy assertion.
# ===========================================================================

def test_chip_id_arch_lock_clean_for_real_predicate(tmp_path, vcc):
    # gfx950 (chip-ID-aware) and gfx942 (not) both match the real,
    # unpatched supportsChipIdPredicate -- no violations.
    f1 = _write_header_yaml(tmp_path / "aldebaran" / "gfx950" / "Equality" / "a.yaml", gfx="gfx950")
    f2 = _write_header_yaml(tmp_path / "aquavanjaram" / "gfx942" / "Equality" / "a.yaml", gfx="gfx942")
    assert vcc.find_chip_id_arch_lock_violations([f1, f2]) == []


def test_chip_id_arch_lock_flags_a_newly_gated_architecture(tmp_path, vcc, monkeypatch):
    # Simulate a registry edit that makes a non-gfx950 architecture report
    # chip-ID awareness without the corresponding re-audit -- the lock must
    # catch it even though no real logic file changed.
    f = _write_header_yaml(tmp_path / "codename" / "gfx1200" / "Equality" / "a.yaml", gfx="gfx1200")
    import Tensile.Common.Architectures as arch_mod

    monkeypatch.setattr(arch_mod, "supportsChipIdPredicate", lambda gfx: gfx == "gfx1200")
    violations = vcc.find_chip_id_arch_lock_violations([f])
    assert len(violations) == 1
    assert "gfx1200" in violations[0]
    assert "expected=False" in violations[0]


def test_chip_id_arch_lock_flags_gfx950_losing_its_gate(tmp_path, vcc, monkeypatch):
    # The lock is symmetric: gfx950 silently losing chip-ID awareness is
    # just as much a violation as another arch silently gaining it.
    f = _write_header_yaml(tmp_path / "aldebaran" / "gfx950" / "Equality" / "a.yaml", gfx="gfx950")
    import Tensile.Common.Architectures as arch_mod

    monkeypatch.setattr(arch_mod, "supportsChipIdPredicate", lambda gfx: False)
    violations = vcc.find_chip_id_arch_lock_violations([f])
    assert len(violations) == 1
    assert "gfx950" in violations[0]
    assert "expected=True" in violations[0]


def test_chip_id_arch_lock_dedupes_repeated_archs(tmp_path, vcc):
    # Many files declare the same arch (e.g. every gfx950 logic file); the
    # lock is a property of the arch, so it must only be reported once even
    # though a real corpus has far more than one file per arch.
    f1 = _write_header_yaml(tmp_path / "aldebaran" / "gfx950" / "Equality" / "a.yaml", gfx="gfx950")
    f2 = _write_header_yaml(tmp_path / "aldebaran" / "gfx950" / "GridBased" / "a.yaml", gfx="gfx950")
    assert vcc.find_chip_id_arch_lock_violations([f1, f2]) == []


# ===========================================================================
# find_gfx1250v0_overlay_violations
# ===========================================================================

def test_gfx1250v0_overlay_clean(tmp_path, vcc):
    _write_overlay_yaml(
        tmp_path / vcc.GFX1250V0 / "Equality" / "logic.yaml",
        schedule=vcc.GFX1250V0, gfx=vcc.GFX1250,
    )
    # A sibling v1 file elsewhere in the corpus, correctly *not* claiming the
    # v0 schedule name, must not trip the "leaked outside" check.
    _write_overlay_yaml(
        tmp_path / "gfx1250" / "Equality" / "logic.yaml",
        schedule="gfx1250", gfx=vcc.GFX1250,
    )
    assert vcc.find_gfx1250v0_overlay_violations(tmp_path) == []


def test_gfx1250v0_overlay_no_split_at_all_is_not_a_violation_when_not_required(tmp_path, vcc):
    # No gfx1250v0 directory anywhere, and the caller isn't specifically
    # requesting the gfx1250v0 architecture -- this corpus simply hasn't done
    # a v0/v1 split for gfx1250 (e.g. hipSPARSELt's corpus, which ships only a
    # unified gfx1250 tree with no per-revision overlay at all, and never
    # requests architecture gfx1250v0). Not every TensileLogic-checked corpus
    # is hipBLASLt's, so this is inapplicable, not a violation.
    (tmp_path / "gfx1250" / "Equality").mkdir(parents=True)
    assert vcc.find_gfx1250v0_overlay_violations(tmp_path, overlay_required=False) == []


def test_gfx1250v0_overlay_missing_is_a_violation_when_required(tmp_path, vcc):
    # hipBLASLt's dedicated gfx1250v0 build (device-library/CMakeLists.txt
    # invokes TensileLogic with --architecture gfx1250v0 specifically for
    # this) must find the overlay -- a corpus that does the v0/v1 split for
    # gfx1250 elsewhere cannot silently lose the overlay directory itself.
    (tmp_path / "gfx1250" / "Equality").mkdir(parents=True)
    violations = vcc.find_gfx1250v0_overlay_violations(tmp_path, overlay_required=True)
    assert any("required" in v for v in violations)


def test_gfx1250v0_overlay_existing_but_empty_is_a_violation(tmp_path, vcc):
    # The overlay directory exists on disk but ships no logic files -- this
    # is the actually-broken case: something started the v0/v1 split for
    # this corpus but the overlay ended up empty. A violation regardless of
    # overlay_required.
    (tmp_path / vcc.GFX1250V0).mkdir(parents=True)
    (tmp_path / "gfx1250" / "Equality").mkdir(parents=True)
    violations = vcc.find_gfx1250v0_overlay_violations(tmp_path)
    assert len(violations) == 1
    assert "ships no logic" in violations[0]


def test_gfx1250v0_overlay_wrong_schedule_name_is_a_violation(tmp_path, vcc):
    _write_overlay_yaml(
        tmp_path / vcc.GFX1250V0 / "Equality" / "logic.yaml",
        schedule="gfx1250",  # should be "gfx1250v0"
        gfx=vcc.GFX1250,
    )
    violations = vcc.find_gfx1250v0_overlay_violations(tmp_path)
    assert any("ScheduleName" in v and "expected 'gfx1250v0'" in v for v in violations)


def test_gfx1250v0_overlay_wrong_architecture_name_is_a_violation(tmp_path, vcc):
    _write_overlay_yaml(
        tmp_path / vcc.GFX1250V0 / "Equality" / "logic.yaml",
        schedule=vcc.GFX1250V0,
        gfx="gfx1250v0",  # must stay the base arch, "gfx1250"
    )
    violations = vcc.find_gfx1250v0_overlay_violations(tmp_path)
    assert any("ArchitectureName" in v and "expected 'gfx1250'" in v for v in violations)


def test_gfx1250v0_overlay_leaking_outside_is_a_violation(tmp_path, vcc):
    _write_overlay_yaml(
        tmp_path / vcc.GFX1250V0 / "Equality" / "logic.yaml",
        schedule=vcc.GFX1250V0, gfx=vcc.GFX1250,
    )
    # A file outside the overlay wrongly claims the v0 schedule name.
    _write_overlay_yaml(
        tmp_path / "gfx1250" / "Equality" / "logic.yaml",
        schedule=vcc.GFX1250V0, gfx=vcc.GFX1250,
    )
    violations = vcc.find_gfx1250v0_overlay_violations(tmp_path)
    assert any("outside the gfx1250v0 overlay" in v for v in violations)


def test_gfx1250v0_overlay_respects_a_caller_supplied_file_subset_outside_scan(tmp_path, vcc):
    # Regression: Run.py excludes "Experimental" logic from the file list it
    # passes to check_corpus_invariants(), mirroring _runChecks()'s own
    # per-file loop. An excluded Experimental file that wrongly claims the
    # v0 schedule name outside the overlay must not trip this check just
    # because find_gfx1250v0_overlay_violations() re-walked logic_root on
    # its own instead of honoring the caller's selection.
    _write_overlay_yaml(
        tmp_path / vcc.GFX1250V0 / "Equality" / "logic.yaml",
        schedule=vcc.GFX1250V0, gfx=vcc.GFX1250,
    )
    _write_overlay_yaml(
        tmp_path / "gfx1250" / "Equality" / "logic.yaml",
        schedule="gfx1250", gfx=vcc.GFX1250,
    )
    _write_overlay_yaml(
        tmp_path / "gfx1250" / "Experimental" / "probe.yaml",
        schedule=vcc.GFX1250V0, gfx=vcc.GFX1250,
    )
    selected = [p for p in _all_yaml(tmp_path) if "Experimental" not in p.parts]
    assert vcc.find_gfx1250v0_overlay_violations(tmp_path, selected) == []
    # Without the filter, the same corpus does flag it -- confirms the probe
    # file is a real would-be violation and not just inert.
    assert any(
        "outside the gfx1250v0 overlay" in v
        for v in vcc.find_gfx1250v0_overlay_violations(tmp_path)
    )


def test_gfx1250v0_overlay_respects_a_caller_supplied_file_subset_overlay_contents(tmp_path, vcc):
    # Same selection contract, but for a file *inside* the overlay: an
    # excluded Experimental file with a bad header inside the overlay
    # directory must not be flagged either.
    _write_overlay_yaml(
        tmp_path / vcc.GFX1250V0 / "Equality" / "logic.yaml",
        schedule=vcc.GFX1250V0, gfx=vcc.GFX1250,
    )
    _write_overlay_yaml(
        tmp_path / vcc.GFX1250V0 / "Experimental" / "probe.yaml",
        schedule="gfx1250",  # wrong ScheduleName, but excluded from selection
        gfx=vcc.GFX1250,
    )
    selected = [p for p in _all_yaml(tmp_path) if "Experimental" not in p.parts]
    assert vcc.find_gfx1250v0_overlay_violations(tmp_path, selected) == []


# ===========================================================================
# check_corpus_invariants / report_corpus_invariant_violations
# ===========================================================================

def test_check_corpus_invariants_aggregates_sibling_and_overlay_finders(tmp_path, vcc):
    # One violation from each of the two finders check_corpus_invariants()
    # aggregates, planted in the same tmp corpus. (find_chip_id_arch_lock_violations
    # is deliberately not one of them -- see check_corpus_invariants()'s docstring.)
    _write_header_yaml(
        tmp_path / "aldebaran" / "gfx950" / "Equality" / "logic.yaml",
        devices="Device 75a0",
    )
    _write_header_yaml(
        tmp_path / "aldebaran" / "gfx950" / "GridBased" / "logic.yaml",
        devices="Device 75a3",
    )
    (tmp_path / "gfx1250" / "Equality").mkdir(parents=True)
    # An existing-but-empty overlay directory, not merely a missing one, is
    # what actually trips the gfx1250v0-overlay finder when "all" is
    # requested (see test_gfx1250v0_overlay_no_split_at_all_is_not_a_violation_when_not_required).
    (tmp_path / vcc.GFX1250V0).mkdir(parents=True)

    violations = vcc.check_corpus_invariants(tmp_path)
    assert any("Divergent sibling DeviceNames" in v for v in violations)
    assert any("ships no logic" in v for v in violations)


def test_check_corpus_invariants_does_not_require_overlay_from_archs_alone(tmp_path, vcc):
    # Regression for the actual bug this shipped with: requesting
    # architecture gfx1250v0 does not by itself mean the corpus being
    # validated owns a gfx1250/gfx1250v0 split. hipSPARSELt's shared gfx125X
    # CI build invokes TensileLogic with --architecture gfx1250v0 against
    # its own corpus, which never did the split and has no gfx1250v0
    # directory at all -- that must not be a hard failure just because the
    # architecture spelling matched.
    (tmp_path / "gfx1250" / "Equality").mkdir(parents=True)
    assert vcc.check_corpus_invariants(tmp_path, archs=["gfx1250v0"]) == []


def test_check_corpus_invariants_requires_overlay_only_when_caller_opts_in(tmp_path, vcc):
    # hipBLASLt's dedicated gfx1250v0 device-library build opts in
    # explicitly via overlay_required=True (--require-gfx1250v0-overlay);
    # only then is a missing overlay a violation.
    (tmp_path / "gfx1250" / "Equality").mkdir(parents=True)
    violations = vcc.check_corpus_invariants(
        tmp_path, archs=["gfx1250v0"], overlay_required=True
    )
    assert any("required" in v for v in violations)


def test_check_corpus_invariants_skips_overlay_check_for_an_unrelated_architecture(tmp_path, vcc):
    # A gfx942-only build has no reason to care about the gfx1250v0 overlay
    # at all -- an absent (or even empty) overlay must not fail it.
    (tmp_path / vcc.GFX1250V0).mkdir(parents=True)
    _write_header_yaml(tmp_path / "aquavanjaram" / "gfx942" / "Equality" / "a.yaml", gfx="gfx942")
    files = [tmp_path / "aquavanjaram" / "gfx942" / "Equality" / "a.yaml"]
    assert vcc.check_corpus_invariants(tmp_path, files=files, archs=["gfx942"]) == []


def test_check_corpus_invariants_respects_a_caller_supplied_file_subset(tmp_path, vcc):
    # Mirrors Run.py's usage: only the already --architecture-filtered files
    # are compared, not a fresh unfiltered walk of logic_root.
    _write_header_yaml(
        tmp_path / "aldebaran" / "gfx950" / "Equality" / "logic.yaml",
        devices="Device 75a0",
    )
    _write_header_yaml(
        tmp_path / "aldebaran" / "gfx950" / "GridBased" / "logic.yaml",
        devices="Device 75a3",
    )
    _write_header_yaml(tmp_path / "aquavanjaram" / "gfx942" / "Equality" / "a.yaml", gfx="gfx942")
    gfx942_only = [tmp_path / "aquavanjaram" / "gfx942" / "Equality" / "a.yaml"]
    assert vcc.check_corpus_invariants(tmp_path, files=gfx942_only, archs=["gfx942"]) == []


def test_check_corpus_invariants_empty_for_a_clean_corpus(tmp_path, vcc):
    _write_header_yaml(tmp_path / "aquavanjaram" / "gfx942" / "Equality" / "a.yaml")
    _write_overlay_yaml(
        tmp_path / vcc.GFX1250V0 / "Equality" / "logic.yaml",
        schedule=vcc.GFX1250V0, gfx=vcc.GFX1250,
    )
    assert vcc.check_corpus_invariants(tmp_path) == []


def test_check_corpus_invariants_returns_empty_for_a_single_file_path(tmp_path, vcc):
    # LogicPath may be an individual .yaml file rather than a directory --
    # these checks need whole-corpus visibility, so they're inapplicable
    # rather than raising.
    f = _write_header_yaml(tmp_path / "solo.yaml")
    assert vcc.check_corpus_invariants(f) == []


def test_report_corpus_invariant_violations_writes_to_stderr(vcc, capsys):
    vcc.report_corpus_invariant_violations(["something went wrong"])
    err = capsys.readouterr().err
    assert "Error: something went wrong" in err
