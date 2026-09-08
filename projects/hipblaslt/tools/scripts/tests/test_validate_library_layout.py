#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Install-tree layout validation, focused on the package-upgrade scenario.

Producer-side ``os.unlink`` cleans the *build* tree, but installing a new
package over a prefix already populated by an older build is additive
(``install(DIRECTORY ...)`` never deletes destination files absent from the
source). A stale uncompressed ``.dat`` can therefore survive an upgrade and
shadow the fresh ``.dat.zlib`` at runtime. These tests pin that the post-install
validator *detects* that co-existence rather than relying on deletion.

Pure standard library (no rocisa / ROCm), so it runs in any Python env:
    python3 -m pytest tools/scripts/tests/test_validate_library_layout.py
"""

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SCRIPTS))

import validate_library_layout


def _make_arch_dir(root: Path, arch: str = "gfx942") -> Path:
    arch_dir = root / "lib" / "hipblaslt" / "library" / arch
    arch_dir.mkdir(parents=True)
    (arch_dir / f"hipblasltTransform_{arch}.hsaco").write_bytes(b"x")
    (arch_dir / f"extop_{arch}.co").write_bytes(b"x")
    (arch_dir / f"hipblasltExtOpLibrary_{arch}.dat.zlib").write_bytes(b"x")
    (arch_dir / f"TensileLibrary_{arch}.dat.zlib").write_bytes(b"x")
    return arch_dir


def _coexistence_violations(root: Path):
    return [
        v
        for v in validate_library_layout.validate(root)
        if "both compressed and uncompressed" in v
    ]


def test_clean_install_tree_has_no_coexistence_violation(tmp_path):
    """A freshly installed tree (only .dat.zlib) is accepted."""
    _make_arch_dir(tmp_path)
    assert _coexistence_violations(tmp_path) == []


def test_upgrade_leaves_stale_tensile_dat_is_flagged(tmp_path):
    """Old package's TensileLibrary_<arch>.dat surviving next to the new
    .dat.zlib is reported as a violation (the upgrade-over-prefix scenario)."""
    arch_dir = _make_arch_dir(tmp_path)
    (arch_dir / "TensileLibrary_gfx942.dat").write_bytes(b"stale from old package")

    violations = _coexistence_violations(tmp_path)
    assert len(violations) == 1
    assert "TensileLibrary_gfx942.dat" in violations[0]


def test_upgrade_leaves_stale_extop_dat_is_flagged(tmp_path):
    """The ExtOp orphan that the additive directory-install can leave behind."""
    arch_dir = _make_arch_dir(tmp_path)
    (arch_dir / "hipblasltExtOpLibrary_gfx942.dat").write_bytes(b"stale extop")

    violations = _coexistence_violations(tmp_path)
    assert len(violations) == 1
    assert "hipblasltExtOpLibrary_gfx942.dat" in violations[0]


def test_multiple_stale_dats_each_flagged(tmp_path):
    """Both the Tensile and ExtOp stale .dat are independently reported."""
    arch_dir = _make_arch_dir(tmp_path)
    (arch_dir / "TensileLibrary_gfx942.dat").write_bytes(b"stale")
    (arch_dir / "hipblasltExtOpLibrary_gfx942.dat").write_bytes(b"stale")

    assert len(_coexistence_violations(tmp_path)) == 2


# --------------------------------------------------------------------------- #
# Stepping subtrees. gfx1250 ships as two steppings sharing one ISA, so they get
# library/gfx1250/ and library/gfx1250-strict/. A stepping is a full architecture
# name: it reaches the compiler, it is what the runtime reports for the silicon,
# and every file the runtime opens is spelled with it -- getExtOpLibraryPath
# trims gcnArchName only at ':', so both the directory and the basename it forms
# carry the stepping. So a stepping subtree owes the same files as any other
# subtree, under its own name.
# --------------------------------------------------------------------------- #
def _make_stepping_dir(root: Path, token: str = "gfx1250-strict") -> Path:
    stepping_dir = root / "lib" / "hipblaslt" / "library" / "gfx1250-strict"
    stepping_dir.mkdir(parents=True)
    (stepping_dir / f"hipblasltTransform_{token}.hsaco").write_bytes(b"x")
    (stepping_dir / f"extop_{token}.co").write_bytes(b"x")
    (stepping_dir / f"hipblasltExtOpLibrary_{token}.dat.zlib").write_bytes(b"x")
    (stepping_dir / f"TensileLibrary_lazy_{token}.dat.zlib").write_bytes(b"x")
    (stepping_dir / f"TensileLiteLibrary_lazy_{token}_Mapping.dat").write_bytes(b"x")
    (stepping_dir / f"TensileLibrary_lazy_{token}.co").write_bytes(b"x")
    (stepping_dir / f"Kernels.so-000-{token}.hsaco").write_bytes(b"x")
    return stepping_dir


def test_a_complete_stepping_subtree_is_accepted(tmp_path):
    """What a build actually produces: both subtrees fully populated, each file
    named for the subtree it sits in."""
    _make_arch_dir(tmp_path, "gfx1250")
    _make_stepping_dir(tmp_path)

    assert validate_library_layout.validate(tmp_path) == []


def test_a_stepping_subtree_named_for_the_shared_isa_is_rejected(tmp_path):
    """Spelling the stepping's files gfx1250 puts them under a name the runtime
    never asks for: on strict silicon it opens gfx1250-strict/*_gfx1250-strict.*,
    so these files are unreachable and the ops they back fail on the device."""
    _make_arch_dir(tmp_path, "gfx1250")
    _make_stepping_dir(tmp_path, token="gfx1250")

    violations = validate_library_layout.validate(tmp_path)
    assert any("hipblasltExtOpLibrary_gfx1250-strict" in v for v in violations), violations
    assert any("TensileLibrary" in v and "gfx1250-strict" in v for v in violations), violations


def test_a_target_feature_suffix_still_belongs_to_its_architecture(tmp_path):
    """The spelling a stepping is easily confused with. xnack is a feature of
    gfx942, not a separate architecture, so it stays in gfx942's subtree."""
    arch_dir = _make_arch_dir(tmp_path, "gfx942")
    (arch_dir / "TensileLibrary_gfx942-xnack+.co").write_bytes(b"x")

    assert validate_library_layout.validate(tmp_path) == []


def test_several_target_features_still_belong_to_their_architecture(tmp_path):
    """A name can carry more than one feature. Requiring a sign of only the first
    token reads gfx942-sramecc+-xnack- as a stepping of gfx942-sramecc+ and
    rejects a file the build legitimately produces."""
    arch_dir = _make_arch_dir(tmp_path, "gfx942")
    (arch_dir / "TensileLibrary_gfx942-sramecc+-xnack-.co").write_bytes(b"x")

    assert validate_library_layout.validate(tmp_path) == []


def test_a_stepping_is_told_from_a_feature_by_the_missing_sign(tmp_path):
    """The whole distinction in one place: both spellings hang a hyphenated token
    off gfx1250, and only the sign says which subtree the file belongs in."""
    arch_dir = _make_arch_dir(tmp_path, "gfx1250")
    (arch_dir / "TensileLibrary_gfx1250-xnack-.co").write_bytes(b"x")
    (arch_dir / "TensileLibrary_gfx1250-strict.co").write_bytes(b"x")

    violations = validate_library_layout.validate(tmp_path)
    assert any("TensileLibrary_gfx1250-strict.co" in v for v in violations), violations
    assert not any("gfx1250-xnack-" in v for v in violations), violations


def test_an_unrelated_arch_is_not_treated_as_a_stepping_subtree(tmp_path):
    """A bare name is never a stepping, so an ordinary arch dir still owes its
    ExtOp and Transform files."""
    arch_dir = tmp_path / "lib" / "hipblaslt" / "library" / "gfx942"
    arch_dir.mkdir(parents=True)
    (arch_dir / "TensileLibrary_gfx942.dat.zlib").write_bytes(b"x")

    violations = validate_library_layout.validate(tmp_path)
    assert any("extop_gfx942.co" in v for v in violations), violations


def test_the_subtree_name_regex_accepts_every_real_spelling():
    """Names this validator will meet in an install tree, and names it must refuse.

    The accepted set spans all three shapes a subtree name takes: bare, a
    stepping, and one or more signed target features. The refused set is what
    separates a name from a path -- these become directory components, so a
    spelling carrying a separator or a space has to fail here rather than
    downstream.
    """
    for good in (
        "gfx942",
        "gfx90a",
        "gfx1250",
        "gfx1250-strict",
        "gfx942-xnack+",
        "gfx950-sramecc+-xnack-",
    ):
        assert validate_library_layout._GFX_PREFIX_RE.fullmatch(good), good

    for bad in ("gfx", "notgfx942", "gfx1250/..", "gfx1250 strict", "", "GFX942"):
        assert not validate_library_layout._GFX_PREFIX_RE.fullmatch(bad), bad


def test_a_stepping_is_not_mistaken_for_a_bare_architecture():
    """The two predicates that route a subtree must disagree about a stepping."""
    assert validate_library_layout._arch_dir_name_is_base("gfx1250")
    assert not validate_library_layout._arch_dir_name_is_base("gfx1250-strict")
    assert validate_library_layout._stepping_base("gfx1250-strict") == "gfx1250"
    assert validate_library_layout._stepping_base("gfx1250") is None


def test_the_sign_is_what_tells_a_target_feature_from_a_stepping():
    """Inside a filename both spellings are hyphenated, so only the sign separates them.

    ``_stepping_base`` above may read a hyphen as a stepping because subtree names
    are bare; filenames are not, and gfx942-xnack+ has to stay a gfx942 file
    rather than become a gfx942 stepping called "xnack+".
    """
    feature = validate_library_layout._TARGET_FEATURE_RE
    for suffix in ("xnack+", "xnack-", "sramecc+-xnack-"):
        assert feature.match(suffix), suffix
    for suffix in ("strict", "foo"):
        assert not feature.match(suffix), suffix


def test_a_feature_suffixed_file_belongs_to_its_architecture_but_a_stepping_does_not():
    matches = validate_library_layout._filename_arch_matches_dir
    assert matches("Kernels.so-000-gfx942-xnack-.hsaco", "gfx942")
    assert matches("Kernels.so-000-gfx90a-sramecc+-xnack-.hsaco", "gfx90a")
    assert matches("TensileLibrary_gfx1250-strict.dat", "gfx1250-strict")
    # The stepping's objects must not be accepted into the base architecture's
    # subtree: they carry a different ELF machine code and will not load there.
    assert not matches("TensileLibrary_gfx1250-strict.dat", "gfx1250")
    assert not matches("TensileLibrary_gfx1250.dat", "gfx1250-strict")
