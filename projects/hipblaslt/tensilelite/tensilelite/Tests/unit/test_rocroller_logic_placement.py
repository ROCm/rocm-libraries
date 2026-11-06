# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Every checked-in rocRoller logic file must be reachable from every gfx950.

A logic file's ``DeviceNames`` become its PCI chip id predicate, and a device
matches a predicate on its own id or on the ids it falls back to. Every gfx950
id falls back to 75a0 and nothing falls back to 75a3, so a file that names
only MI355X ids (75a3, 75a2) is invisible on MI350X (75a0): every MX problem
it covers reports no solution there. A ``gfx950_id*`` directory carries the
same restriction, since ValidChipId requires its files to name that id.

The corpus tests read the rocRoller files through ``rocroller_logic_files``;
this also checks that the glob behind it still finds all of them.
"""

import re

import pytest
import yaml

from rocroller_logic_corpus import CORPUS, LOGIC, rocroller_logic_files

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        not LOGIC.is_dir(),
        reason=f"source-tree check: no hipBLASLt logic tree at {LOGIC}; the installed "
        "test payload does not ship it",
    ),
]

_CHIP_ID_DIR = re.compile(r"_id[0-9a-fA-F]+$")


@pytest.fixture(scope="module")
def rocroller_files():
    """Every logic file under LOGIC that names a rocRoller kernel, wherever it is."""
    files = sorted(path for path in LOGIC.rglob("*.yaml") if b"RR_GEMM_" in path.read_bytes())
    assert files, f"no logic file under {LOGIC} names an RR_GEMM_ kernel"
    return files


def test_every_rocroller_logic_file_declares_the_default_chip_id(rocroller_files):
    wrong = []
    for path in rocroller_files:
        doc = yaml.load(path.read_text(), Loader=yaml.CSafeLoader)
        devices = doc.get("DeviceNames") if isinstance(doc, dict) else None
        if devices != ["Device 75a0"]:
            wrong.append(f"{path.relative_to(LOGIC)}: {devices!r}")
    assert not wrong, f"rocRoller logic files must declare DeviceNames [Device 75a0]: {wrong}"


def test_no_rocroller_logic_file_is_in_a_chip_id_directory(rocroller_files):
    pinned = [
        str(path.relative_to(LOGIC))
        for path in rocroller_files
        if any(_CHIP_ID_DIR.search(part) for part in path.relative_to(LOGIC).parent.parts)
    ]
    assert not pinned, f"rocRoller logic files under a chip-id directory: {pinned}"


def test_the_corpus_glob_finds_every_rocroller_logic_file(rocroller_files):
    globbed = rocroller_logic_files()
    missed = sorted(str(path.relative_to(LOGIC)) for path in set(rocroller_files) - set(globbed))
    extra = sorted(str(path.relative_to(LOGIC)) for path in set(globbed) - set(rocroller_files))
    assert not missed and not extra, (
        f"the corpus tests read {CORPUS} through rocroller_logic_files(); it misses "
        f"rocRoller logic files {missed} and picks up files without one {extra}"
    )
