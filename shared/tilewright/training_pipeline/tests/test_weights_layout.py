# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The weights directories of tilewright's weights root: where a deploy
writes a subtree's models, what the checkout ships there, and how hipBLASLt
installs them."""

import shutil
import subprocess
from pathlib import Path

import pytest

from lib import tilewright_index as twi
from lib import weights_layout as wl
from lib.correctness import GUARDS_SUFFIX, SIDECAR_SUFFIX

REPO = Path(__file__).resolve().parents[4]
WEIGHTS = Path("shared/tilewright/weights")
INSTALL_SCRIPT = REPO / "projects/hipblaslt/cmake/tilewright_install_models.cmake"


@pytest.mark.parametrize(
    "subtree, rel",
    [
        ("gfx950", "gfx950/gfx950"),
        ("gfx1250", "gfx1250/gfx1250"),
        ("gfx1250-strict", "gfx1250-strict/gfx1250-strict"),
        # A build alias ships under the logic it is built from.
        ("gfx1250v0", "gfx1250-strict/gfx1250v0"),
        # So does a CU-count directory, under its architecture.
        ("gfx1250_96cu", "gfx1250/gfx1250_96cu"),
        ("gfx950_128cu", "gfx950/gfx950_128cu"),
    ],
)
def test_a_subtree_ships_its_models_under_its_logic_architecture(subtree, rel):
    assert wl.rel_dir(subtree) == WEIGHTS / rel
    assert wl.logic_arch(subtree) == Path(rel).parts[0]


@pytest.mark.parametrize(
    "subtree, into",
    [
        ("gfx950", "gfx950"),
        ("gfx1250v0", "gfx1250v0"),
        ("gfx1250_96cu", "gfx1250"),
        ("gfx1250-strict_96cu", "gfx1250-strict"),
        ("gfx950_id75a3", "gfx950_id75a3"),
    ],
)
def test_the_models_of_a_cu_count_directory_install_into_its_architecture(
    subtree, into
):
    assert wl.installed_subtree(subtree) == into
    assert wl.cu_dir_arch(subtree) == (into if into != subtree else None)


def test_installed_with_names_the_directories_one_library_directory_shares(tmp_path):
    for rel in (
        "gfx1250/gfx1250",
        "gfx1250/gfx1250_96cu",
        "gfx1250/gfx950_96cu",
        "gfx950/gfx950",
    ):
        (tmp_path / wl.WEIGHTS_REL / rel).mkdir(parents=True)
    root = tmp_path / wl.WEIGHTS_REL

    def shared(subtree):
        return [p.relative_to(root) for p in wl.installed_with(tmp_path, subtree)]

    assert shared("gfx1250") == [Path("gfx1250/gfx1250_96cu")]
    assert shared("gfx1250_96cu") == [Path("gfx1250/gfx1250")]
    assert shared("gfx1250_192cu") == [
        Path("gfx1250/gfx1250"),
        Path("gfx1250/gfx1250_96cu"),
    ]
    assert shared("gfx950") == []


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


def test_each_library_directory_gets_one_model_per_stem():
    """hipBLASLt installs the directories of a subtree and of its CU-count
    directories into one library directory, and its build fails on a stem
    two of them list or a file two of them ship."""
    by_target = {}
    for d in _shipped():
        sub = d.relative_to(REPO / wl.WEIGHTS_REL).parts[1]
        by_target.setdefault(wl.installed_subtree(sub), []).append(d)
    for dirs in by_target.values():
        stems, files = {}, {}
        for d in dirs:
            index = twi.parse(twi.read_text(d / twi.INDEX_NAME))
            for stem in {s for s, _, _ in index.rows}:
                assert stems.setdefault(stem, d) == d, (stem, stems[stem], d)
            for p in d.iterdir():
                if p.name != twi.INDEX_NAME:
                    assert files.setdefault(p.name, d) == d, (p.name, files[p.name], d)


def _install(dirs, library):
    """(return code, output) of hipBLASLt's install of the models of the
    weights directories `dirs` into the library directory `library`
    (target hipblaslt-tilewright-models)."""
    cmake = shutil.which("cmake")
    if cmake is None or not INSTALL_SCRIPT.is_file():
        pytest.skip(f"needs cmake and {INSTALL_SCRIPT}")
    proc = subprocess.run(
        [
            cmake,
            "-DWEIGHTS_DIRS=" + "|".join(str(d) for d in dirs),
            f"-DLIBRARY_DIR={library}",
            "-P",
            str(INSTALL_SCRIPT),
        ],
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout + proc.stderr


def test_a_subtree_installs_the_models_of_its_cu_count_directories(tmp_path):
    own, cu96 = (tmp_path / wl.rel_dir(s) for s in ("gfx1250", "gfx1250_96cu"))
    for d in (own, cu96):
        d.mkdir(parents=True)
    (own / twi.INDEX_NAME).write_text(
        "# the subtree's\nA\ta.tilewright.bin\nA\ta1.tilewright.bin\tchip=0x75a0\n"
    )
    (cu96 / twi.INDEX_NAME).write_text(
        "# the 96-CU libraries'\nB_CU96\tb.tilewright.bin  # one row\n"
    )
    for d, name in (
        (own, "a.tilewright.bin"),
        (own, "a1.tilewright.bin"),
        (own, "A" + SIDECAR_SUFFIX),
        (cu96, "b.tilewright.bin"),
        (cu96, "B_CU96" + GUARDS_SUFFIX),
    ):
        (d / name).write_text(name)
    lib = tmp_path / "library" / wl.installed_subtree("gfx1250_96cu")
    lib.mkdir(parents=True)
    for stale in ("Gone" + SIDECAR_SUFFIX, "Gone" + GUARDS_SUFFIX):
        (lib / stale).write_text("left by an earlier build")
    (lib / "gone.tilewright.bin").write_text("kept")

    rc, out = _install([own, cu96], lib)
    assert rc == 0, out
    index = twi.parse(twi.read_text(lib / twi.INDEX_NAME))
    assert index.rows == [
        ("A", "a.tilewright.bin", None),
        ("A", "a1.tilewright.bin", 0x75A0),
        ("B_CU96", "b.tilewright.bin", None),
    ]
    assert index.model_of("B_CU96") == "b.tilewright.bin"
    assert sorted(p.name for p in lib.iterdir()) == sorted(
        [
            "A" + SIDECAR_SUFFIX,
            "B_CU96" + GUARDS_SUFFIX,
            "a.tilewright.bin",
            "a1.tilewright.bin",
            "b.tilewright.bin",
            "gone.tilewright.bin",
            twi.INDEX_NAME,
        ]
    )
    assert (lib / "b.tilewright.bin").read_text() == "b.tilewright.bin"
    # One directory installs its index as it is.
    rc, out = _install([own], lib)
    assert rc == 0, out
    assert twi.read_text(lib / twi.INDEX_NAME) == twi.read_text(own / twi.INDEX_NAME)
    assert not (lib / ("B_CU96" + GUARDS_SUFFIX)).exists()


@pytest.mark.parametrize("clash", ["stem", "file"])
def test_one_library_directory_refuses_a_stem_or_a_file_twice(tmp_path, clash):
    own, cu96 = (tmp_path / wl.rel_dir(s) for s in ("gfx1250", "gfx1250_96cu"))
    for d in (own, cu96):
        d.mkdir(parents=True)
    (own / twi.INDEX_NAME).write_text("A\ta.tilewright.bin\n")
    (own / "a.tilewright.bin").write_text("a")
    if clash == "stem":
        (cu96 / twi.INDEX_NAME).write_text("A\tb.tilewright.bin\tchip=0x75a0\n")
        (cu96 / "b.tilewright.bin").write_text("b")
        want = "A is listed in"
    else:
        (cu96 / twi.INDEX_NAME).write_text("B\ta.tilewright.bin\n")
        (cu96 / "a.tilewright.bin").write_text("another a")
        want = "both ship a.tilewright.bin"
    lib = tmp_path / "library"
    lib.mkdir()
    rc, out = _install([own, cu96], lib)
    assert rc != 0 and want in " ".join(out.split()), out
    assert list(lib.iterdir()) == []
