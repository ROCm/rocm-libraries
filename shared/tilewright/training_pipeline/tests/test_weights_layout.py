# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The weights directories of tilewright's weights root: where a deploy
writes a subtree's models, what the checkout ships there, and how hipBLASLt
installs them."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from lib import mlrec
from lib import tilewright_index as twi
from lib import weights_layout as wl
from lib.correctness import GUARDS_SUFFIX, SIDECAR_SUFFIX

REPO = Path(__file__).resolve().parents[4]
WEIGHTS = Path("shared/tilewright/weights")
INSTALL_SCRIPT = REPO / "projects/hipblaslt/cmake/tilewright_install_models.cmake"
BIN_SUFFIX = ".tilewright.bin"
# The CU-count and chip-ID parts a library build puts before the architecture.
_TAG = re.compile(r"_(?:CU[0-9]+|ID[0-9a-fA-F-]+)(?=_|$)")


@pytest.mark.parametrize(
    "subtree, rel",
    [
        ("gfx950", "gfx950/gfx950"),
        ("gfx1250", "gfx1250/gfx1250"),
        ("gfx1250-strict", "gfx1250-strict/gfx1250-strict"),
    ],
)
def test_a_subtree_ships_its_models_under_its_logic_architecture(subtree, rel):
    assert wl.rel_dir(subtree) == WEIGHTS / rel


def test_shipped_dirs_finds_every_weights_directory(tmp_path):
    root = tmp_path / wl.WEIGHTS_REL
    for rel in ("gfx950/gfx950", "gfx1250/gfx1250", "gfx942/gfx942"):
        (root / rel).mkdir(parents=True)
    (root / ".gitattributes").write_text("*.tilewright.bin binary\n")
    (root / "gfx950" / "stray.tilewright.bin").write_bytes(b"x")
    assert [p.relative_to(root) for p in wl.shipped_dirs(tmp_path)] == [
        Path("gfx1250/gfx1250"),
        Path("gfx942/gfx942"),
        Path("gfx950/gfx950"),
    ]
    assert wl.shipped_dirs(tmp_path / "nowhere") == []


def _shipped():
    dirs = wl.shipped_dirs(REPO)
    if not dirs:
        pytest.skip(f"no weights directory under {REPO / wl.WEIGHTS_REL}")
    return dirs


def test_the_shipped_directories_hold_models_only():
    """hipBLASLt copies an index's models and sidecars (the blocklists and
    the guards) to the subtree the directory is named for."""
    for d in _shipped():
        rel = d.relative_to(REPO / wl.WEIGHTS_REL)
        assert rel == wl.rel_dir(rel.parts[1]).relative_to(wl.WEIGHTS_REL), d
        names = sorted(p.name for p in d.iterdir())
        assert all(p.is_file() for p in d.iterdir()), d
        assert twi.INDEX_NAME in names, d
        index = twi.parse(twi.read_text(d / twi.INDEX_NAME))
        assert index.malformed_rows == [] and index.dropped == [], d
        listed = {f for _, f, _ in index.rows}
        models = {f.split(".", 1)[0] for f in listed}
        for name in names:
            if name == twi.INDEX_NAME:
                continue
            if name.endswith(BIN_SUFFIX):
                assert name in listed, f"{d / name} is in no row of the index"
            elif name.endswith((SIDECAR_SUFFIX, GUARDS_SUFFIX)):
                suffix = next(
                    x for x in (SIDECAR_SUFFIX, GUARDS_SUFFIX) if name.endswith(x)
                )
                assert (
                    name[: -len(suffix)] in models
                ), f"{d / name} is named after no model of the index"
            else:
                pytest.fail(f"{d / name} is no model, sidecar or index")
        assert listed <= set(names), d


def test_the_shipped_models_are_named_after_their_logic_files():
    """A model, and its blocklist and guards, are named exactly after the logic
    file (liblogic) of its library: `<arch>_Cijk_..._UserArgs`, no
    `TensileLibrary_` name and no CU-count or chip-ID tag, and the model records
    that name as its config_id. The index keys are library file stems without
    those parts, and each model has one key."""
    for d in _shipped():
        arch = d.name
        index = twi.parse(twi.read_text(d / twi.INDEX_NAME))
        files = [f for _, f, _ in index.rows]
        assert len(files) == len(set(files)), f"{d}: a model with two rows"
        for key, file, chip in index.rows:
            where = f"{d / twi.INDEX_NAME}: {key} -> {file}"
            model = file.split(".", 1)[0]
            assert file == model + BIN_SUFFIX, where
            assert re.fullmatch(rf"{arch}_Cijk_A(?:lik|ilk)_B(?:ljk|jlk)_\w+_UserArgs", model), where
            assert "TensileLibrary_" not in file and not _TAG.search(file), where
            assert key.startswith("TensileLibrary_") and key.endswith(f"_{arch}"), where
            assert not _TAG.search(key), f"{where}: a key has a CU-count or chip-ID part"
            recorded = mlrec.read_header((d / file).read_bytes())["provenance"]
            assert recorded.get("config_id") == model, where


def _install(src, library):
    """(return code, output) of hipBLASLt's install of the models of the
    weights directory `src` into the library directory `library`
    (target hipblaslt-tilewright-models)."""
    cmake = shutil.which("cmake")
    if cmake is None or not INSTALL_SCRIPT.is_file():
        pytest.skip(f"needs cmake and {INSTALL_SCRIPT}")
    proc = subprocess.run(
        [
            cmake,
            f"-DWEIGHTS_DIR={src}",
            f"-DLIBRARY_DIR={library}",
            "-P",
            str(INSTALL_SCRIPT),
        ],
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout + proc.stderr


def test_a_subtree_installs_its_models_sidecars_and_index(tmp_path):
    src = tmp_path / wl.rel_dir("gfx1250")
    src.mkdir(parents=True)
    (src / twi.INDEX_NAME).write_text(
        "# the subtree's\nA_gfx1250\ta.tilewright.bin\n"
        "A_gfx1250\ta1.tilewright.bin\tchip=0x75a0\n"
        "B_gfx1250\tb.tilewright.bin  # one row\n"
    )
    for name in (
        "a.tilewright.bin",
        "a1.tilewright.bin",
        "b.tilewright.bin",
        "a" + SIDECAR_SUFFIX,
        "b" + GUARDS_SUFFIX,
    ):
        (src / name).write_text(name)
    lib = tmp_path / "library" / "gfx1250"
    lib.mkdir(parents=True)
    for stale in ("gone" + SIDECAR_SUFFIX, "gone" + GUARDS_SUFFIX):
        (lib / stale).write_text("left by an earlier build")
    (lib / "gone.tilewright.bin").write_text("kept")

    rc, out = _install(src, lib)
    assert rc == 0, out
    assert twi.read_text(lib / twi.INDEX_NAME) == twi.read_text(src / twi.INDEX_NAME)
    index = twi.parse(twi.read_text(lib / twi.INDEX_NAME))
    assert index.model_of("B_gfx1250") == "b.tilewright.bin"
    assert sorted(p.name for p in lib.iterdir()) == sorted(
        [
            "a" + SIDECAR_SUFFIX,
            "b" + GUARDS_SUFFIX,
            "a.tilewright.bin",
            "a1.tilewright.bin",
            "b.tilewright.bin",
            "gone.tilewright.bin",
            twi.INDEX_NAME,
        ]
    )
    assert (lib / "b.tilewright.bin").read_text() == "b.tilewright.bin"
    # A sidecar the directory stops shipping leaves the library directory.
    (src / ("b" + GUARDS_SUFFIX)).unlink()
    rc, out = _install(src, lib)
    assert rc == 0, out
    assert not (lib / ("b" + GUARDS_SUFFIX)).exists()


def test_the_install_needs_a_weights_directory_with_an_index(tmp_path):
    src = tmp_path / "weights"
    src.mkdir()
    lib = tmp_path / "library"
    lib.mkdir()
    rc, out = _install(src, lib)
    assert rc != 0 and "holds no tilewright_index" in " ".join(out.split()), out
    rc, out = _install(tmp_path / "nowhere", lib)
    assert rc != 0 and "must name a weights directory" in " ".join(out.split()), out
