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
