# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Every checked-in rocRoller logic file must be reachable from every gfx950.

A logic file's ``DeviceNames`` become its PCI chip id predicate, and a device
matches a predicate on its own id or on the ids it falls back to. Every gfx950
id falls back to 75a0 and nothing falls back to 75a3, so a file that names
only MI355X ids (75a3, 75a2) is invisible on MI350X (75a0): every MX problem
it covers reports no solution there. The files declare the list LibraryLogic
gives a generated gfx950 library, which names 75a0 next to 75a3 and 75a2. A
``gfx950_id*`` directory carries the same restriction as a 75a3-only list,
since ValidChipId requires its files to name that id.

The corpus tests read the rocRoller files through ``rocroller_logic_files``;
this also checks that the glob behind it still finds all of them.
"""

import re

import pytest
import yaml

from logic_corpus import needs_logic_corpus
from tensilelite.LibraryLogic import libraryDeviceNames
from rocroller_logic_corpus import asm_full, corpus, rocroller_logic_files

pytestmark = [pytest.mark.unit, needs_logic_corpus]

_CHIP_ID_DIR = re.compile(r"_id[0-9a-fA-F]+$")


@pytest.fixture(scope="module")
def rocroller_files():
    """Every logic file under asm_full that names a rocRoller kernel, wherever it is."""
    root = asm_full()
    files = sorted(path for path in root.rglob("*.yaml") if b"RR_GEMM_" in path.read_bytes())
    assert files, f"no logic file under {root} names an RR_GEMM_ kernel"
    return files


def test_every_rocroller_logic_file_declares_the_generated_gfx950_devices(rocroller_files):
    expected = libraryDeviceNames("gfx950", None)
    assert "Device 75a0" in expected, expected
    root = asm_full()
    wrong = []
    for path in rocroller_files:
        doc = yaml.load(path.read_text(), Loader=yaml.CSafeLoader)
        devices = doc.get("DeviceNames") if isinstance(doc, dict) else None
        if devices != expected:
            wrong.append(f"{path.relative_to(root)}: {devices!r}")
    assert not wrong, f"rocRoller logic files must declare DeviceNames {expected}: {wrong}"


def test_no_rocroller_logic_file_is_in_a_chip_id_directory(rocroller_files):
    root = asm_full()
    pinned = [
        str(path.relative_to(root))
        for path in rocroller_files
        if any(_CHIP_ID_DIR.search(part) for part in path.relative_to(root).parent.parts)
    ]
    assert not pinned, f"rocRoller logic files under a chip-id directory: {pinned}"


def test_the_corpus_glob_finds_every_rocroller_logic_file(rocroller_files):
    root = asm_full()
    globbed = rocroller_logic_files()
    missed = sorted(str(path.relative_to(root)) for path in set(rocroller_files) - set(globbed))
    extra = sorted(str(path.relative_to(root)) for path in set(globbed) - set(rocroller_files))
    assert not missed and not extra, (
        f"the corpus tests read {corpus()} through rocroller_logic_files(); it misses "
        f"rocRoller logic files {missed} and picks up files without one {extra}"
    )
