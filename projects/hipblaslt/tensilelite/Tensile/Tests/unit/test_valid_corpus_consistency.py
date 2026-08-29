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


def _write_overlay_yaml(path, *, schedule, gfx):
    """gfx1250v0-overlay tests only care about ScheduleName / gfx arch, but
    write the same full header shape ``_write_header_yaml`` does (rather than
    a bare positional list of scalars) so these stay representative of the
    real logic-file dialect that ``load_logic_schedule_name()`` /
    ``load_logic_gfx_arch()`` parse."""
    return _write_header_yaml(path, schedule=schedule, gfx=gfx)


def test_read_device_names_parses_header_line(tmp_path, vcc):
    f = _write_header_yaml(tmp_path / "a.yaml", devices="Device 74a0, Device 74a1")
    assert vcc.read_device_names(f) == ("74a0", "74a1")


def test_read_device_names_parses_mapping_form_header_line(tmp_path, vcc):
    # Origami and similar logic files write DeviceNames as a mapping
    # (``DeviceNames: [Device ...]``) rather than the positional list form;
    # both dialects must parse to the same result. See #11442.
    f = _write_mapping_form_header_yaml(tmp_path / "a.yaml", devices="Device 74a0, Device 74a1")
    assert vcc.read_device_names(f) == ("74a0", "74a1")


def test_read_device_names_returns_none_when_absent(tmp_path, vcc):
    f = tmp_path / "a.yaml"
    f.write_text("- MinimumRequiredVersion: 4.33.0\n- schedule\n- gfx942\n")
    assert vcc.read_device_names(f) is None


def test_read_device_names_returns_none_for_missing_file(tmp_path, vcc):
    assert vcc.read_device_names(tmp_path / "does_not_exist.yaml") is None


def test_iter_arch_dirs_and_all_arch_names(tmp_path, vcc):
    _write_header_yaml(tmp_path / "aquavanjaram" / "gfx942" / "Equality" / "a.yaml")
    _write_header_yaml(tmp_path / "aquavanjaram" / "gfx942_20cu" / "Equality" / "a.yaml")
    _write_header_yaml(tmp_path / "aldebaran" / "gfx950" / "Equality" / "a.yaml")
    # Not a gfx* directory -> excluded.
    (tmp_path / "aquavanjaram" / "notes.txt").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "aquavanjaram" / "notes.txt").write_text("n/a")

    pairs = sorted(vcc.iter_arch_dirs(tmp_path))
    assert [codename for codename, _ in pairs] == ["aldebaran", "aquavanjaram", "aquavanjaram"]
    assert vcc.all_arch_names(tmp_path) == ["gfx942", "gfx942_20cu", "gfx950"]


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
    direct = vcc.find_sibling_device_names_violations(asm_full)
    via_ancestor = vcc.find_sibling_device_names_violations(tmp_path / "library")
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
    assert vcc.find_sibling_device_names_violations(tmp_path) == []


def test_sibling_device_names_flags_mismatched_siblings(tmp_path, vcc):
    # Same basename ("logic.yaml"), same arch dir, divergent DeviceNames --
    # exactly the shape of https://github.com/ROCm/rocm-libraries/issues/11397.
    _write_header_yaml(
        tmp_path / "aldebaran" / "gfx950" / "Equality" / "logic.yaml",
        devices="Device 75a0",
    )
    _write_header_yaml(
        tmp_path / "aldebaran" / "gfx950" / "GridBased" / "logic.yaml",
        devices="Device 75a0, Device 75a3",
    )
    violations = vcc.find_sibling_device_names_violations(tmp_path)
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
    violations = vcc.find_sibling_device_names_violations(tmp_path)
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
    assert vcc.find_sibling_device_names_violations(tmp_path) == []


# ===========================================================================
# find_chip_id_arch_lock_violations
# ===========================================================================

def test_chip_id_arch_lock_clean_for_real_predicate(tmp_path, vcc):
    # gfx950 (chip-ID-aware) and gfx942 (not) both match the real,
    # unpatched supportsChipIdPredicate -- no violations.
    _write_header_yaml(tmp_path / "aldebaran" / "gfx950" / "Equality" / "a.yaml")
    _write_header_yaml(tmp_path / "aquavanjaram" / "gfx942" / "Equality" / "a.yaml")
    assert vcc.find_chip_id_arch_lock_violations(tmp_path) == []


def test_chip_id_arch_lock_flags_a_newly_gated_architecture(tmp_path, vcc, monkeypatch):
    # Simulate a registry edit that makes a non-gfx950 architecture report
    # chip-ID awareness without the corresponding re-audit -- the lock must
    # catch it even though no real logic file changed.
    _write_header_yaml(tmp_path / "codename" / "gfx1200" / "Equality" / "a.yaml")
    monkeypatch.setattr(vcc, "supportsChipIdPredicate", lambda gfx: gfx == "gfx1200")
    violations = vcc.find_chip_id_arch_lock_violations(tmp_path)
    assert len(violations) == 1
    assert "gfx1200" in violations[0]
    assert "expected=False" in violations[0]


def test_chip_id_arch_lock_flags_gfx950_losing_its_gate(tmp_path, vcc, monkeypatch):
    # The lock is symmetric: gfx950 silently losing chip-ID awareness is
    # just as much a violation as another arch silently gaining it.
    _write_header_yaml(tmp_path / "aldebaran" / "gfx950" / "Equality" / "a.yaml")
    monkeypatch.setattr(vcc, "supportsChipIdPredicate", lambda gfx: False)
    violations = vcc.find_chip_id_arch_lock_violations(tmp_path)
    assert len(violations) == 1
    assert "gfx950" in violations[0]
    assert "expected=True" in violations[0]


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


def test_gfx1250v0_overlay_no_split_at_all_is_not_a_violation(tmp_path, vcc):
    # No gfx1250v0 directory anywhere -- this corpus simply hasn't done a
    # v0/v1 split for gfx1250 (e.g. hipSPARSELt's corpus, which ships only a
    # unified gfx1250 tree with no per-revision overlay at all). Not every
    # TensileLogic-checked corpus is hipBLASLt's, so this is inapplicable,
    # not a violation.
    (tmp_path / "gfx1250" / "Equality").mkdir(parents=True)
    assert vcc.find_gfx1250v0_overlay_violations(tmp_path) == []


def test_gfx1250v0_overlay_existing_but_empty_is_a_violation(tmp_path, vcc):
    # The overlay directory exists on disk but ships no logic files -- this
    # is the actually-broken case: something started the v0/v1 split for
    # this corpus but the overlay ended up empty.
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


# ===========================================================================
# check_corpus_invariants / report_corpus_invariant_violations
# ===========================================================================

def test_check_corpus_invariants_aggregates_all_finders(tmp_path, vcc):
    # One violation from each finder, planted in the same tmp corpus.
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
    # what actually trips the gfx1250v0-overlay finder (see
    # test_gfx1250v0_overlay_no_split_at_all_is_not_a_violation).
    (tmp_path / vcc.GFX1250V0).mkdir(parents=True)

    violations = vcc.check_corpus_invariants(tmp_path)
    assert any("Divergent sibling DeviceNames" in v for v in violations)
    assert any("ships no logic" in v for v in violations)


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
