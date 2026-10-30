# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The Tilewright directories of hipBLASLt's library logic: where a deploy
writes a subtree's models, and what the checkout ships there."""

from pathlib import Path

import pytest

from lib import tilewright_index as twi
from lib import weights_layout as wl
from lib.correctness import GUARDS_SUFFIX, SIDECAR_SUFFIX

REPO = Path(__file__).resolve().parents[4]
LOGIC = Path("projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic")


@pytest.mark.parametrize(
    "subtree, rel",
    [
        ("gfx950", "asm_full/gfx950/gfx950/Tilewright"),
        ("gfx1250", "asm_full/gfx1250/gfx1250/Tilewright"),
        ("gfx1250-strict", "asm_full/gfx1250-strict/gfx1250-strict/Tilewright"),
        # A build alias ships next to the logic it is built from.
        ("gfx1250v0", "asm_full/gfx1250-strict/gfx1250v0/Tilewright"),
    ],
)
def test_a_subtree_ships_its_models_with_its_logic(subtree, rel):
    assert wl.rel_dir(subtree) == LOGIC / rel
    assert wl.logic_arch(subtree) == Path(rel).parts[1]


def test_shipped_dirs_finds_every_tilewright_directory(tmp_path):
    for rel in ("gfx950/gfx950/Tilewright", "gfx1250/gfx1250/Tilewright"):
        (tmp_path / wl.LOGIC_REL / rel).mkdir(parents=True)
    (tmp_path / wl.LOGIC_REL / "gfx950/gfx950/Equality").mkdir()
    (tmp_path / wl.LOGIC_REL / "gfx942/gfx942/Tilewright").mkdir(parents=True)
    root = tmp_path / wl.LOGIC_REL
    assert [p.relative_to(root) for p in wl.shipped_dirs(tmp_path)] == [
        Path("gfx1250/gfx1250/Tilewright"),
        Path("gfx942/gfx942/Tilewright"),
        Path("gfx950/gfx950/Tilewright"),
    ]
    assert wl.shipped_dirs(tmp_path / "nowhere") == []


def _shipped():
    dirs = wl.shipped_dirs(REPO)
    if not dirs:
        pytest.skip(f"no Tilewright directory under {REPO / wl.LOGIC_REL}")
    return dirs


def test_the_shipped_directories_hold_models_only():
    """TensileCreateLibrary reads every .yaml file of the logic tree as
    library logic, and hipBLASLt copies an index's models and sidecars (the
    blocklists and the guards) to the subtree the directory is named for."""
    for d in _shipped():
        rel = d.relative_to(REPO / wl.LOGIC_REL)
        assert rel == wl.rel_dir(rel.parts[1]).relative_to(wl.LOGIC_REL), d
        names = sorted(p.name for p in d.iterdir())
        assert all(p.is_file() for p in d.iterdir()), d
        assert not [n for n in names if n.endswith((".yaml", ".yml", ".json"))], d
        assert twi.INDEX_NAME in names, d
        index = twi.parse(twi.read_text(d / twi.INDEX_NAME))
        assert index.malformed_rows == [] and index.dropped == [], d
        listed = {f for _, f, _ in index.rows}
        stems = {s for s, _, _ in index.rows}
        for name in names:
            if name == twi.INDEX_NAME:
                continue
            if name.endswith(".tilewright.bin"):
                assert name in listed, f"{d / name} is in no row of the index"
            elif name.endswith((SIDECAR_SUFFIX, GUARDS_SUFFIX)):
                suffix = next(
                    x for x in (SIDECAR_SUFFIX, GUARDS_SUFFIX) if name.endswith(x)
                )
                stem = name[: -len(suffix)]
                assert stem in stems, f"{d / name} names a stem the index lacks"
            else:
                pytest.fail(f"{d / name} is no model, sidecar or index")
        assert listed <= set(names), d
