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
# library/gfx1250/ and library/gfx1250-strict/. Only the DIRECTORY carries the
# stepping: Tensile names files from the ISA, so every file inside either subtree
# is named gfx1250. ExtOp and Transform resolve from gcnArchName, which is the
# architecture on both parts, so they never appear in the stepping subtree.
# --------------------------------------------------------------------------- #
def _make_stepping_dir(root: Path) -> Path:
    stepping_dir = root / "lib" / "hipblaslt" / "library" / "gfx1250-strict"
    stepping_dir.mkdir(parents=True)
    (stepping_dir / "TensileLibrary_lazy_gfx1250.dat.zlib").write_bytes(b"x")
    (stepping_dir / "TensileLiteLibrary_lazy_gfx1250_Mapping.dat").write_bytes(b"x")
    (stepping_dir / "TensileLibrary_lazy_gfx1250.co").write_bytes(b"x")
    (stepping_dir / "Kernels.so-000-gfx1250.hsaco").write_bytes(b"x")
    return stepping_dir


def test_a_complete_stepping_subtree_is_accepted(tmp_path):
    """What a build actually produces: the architecture's tree with everything,
    plus a stepping tree holding only Tensile artifacts, all named for the
    architecture."""
    _make_arch_dir(tmp_path, "gfx1250")
    _make_stepping_dir(tmp_path)

    assert validate_library_layout.validate(tmp_path) == []


def test_an_unrelated_arch_is_not_treated_as_a_stepping_subtree(tmp_path):
    """A bare name is never a stepping, so an ordinary arch dir still owes its
    ExtOp and Transform files."""
    arch_dir = tmp_path / "lib" / "hipblaslt" / "library" / "gfx942"
    arch_dir.mkdir(parents=True)
    (arch_dir / "TensileLibrary_gfx942.dat.zlib").write_bytes(b"x")

    violations = validate_library_layout.validate(tmp_path)
    assert any("extop_gfx942.co" in v for v in violations), violations
