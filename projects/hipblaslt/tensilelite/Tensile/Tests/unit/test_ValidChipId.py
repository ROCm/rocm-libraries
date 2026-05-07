################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

from pathlib import Path

import pytest

from Tensile.TensileLogic.ValidChipId import _validateChipId


def _writeLogicFile(
    path: Path,
    *,
    gfx: str = "gfx950",
    name: str = "gfx950",
    devices: str = "Device 75a0",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "- MinimumRequiredVersion: 4.33.0",
                f"- {name}",
                f"- {gfx}",
                f"- [{devices}]",
                "",
            ]
        )
    )
    return path


def _baseGfx950Path(tmp_path: Path) -> Path:
    return tmp_path / "gfx950" / "gfx950" / "Equality" / "logic.yaml"


def _variantGfx950Path(tmp_path: Path, chip_id: str = "75a3") -> Path:
    return tmp_path / "gfx950" / f"gfx950_id{chip_id}" / "Equality" / "logic.yaml"


def _malformedVariantGfx950Path(tmp_path: Path, chip_id: str = "75a3") -> Path:
    return tmp_path / "gfx950" / f"gfx950_{chip_id}" / "Equality" / "logic.yaml"


def test_validateChipIdAcceptsBaseDefaultChipId(tmp_path):
    logic_file = _writeLogicFile(_baseGfx950Path(tmp_path), devices="Device 75a0")

    assert _validateChipId(logic_file)


def test_validateChipIdAcceptsVariantChipIdWithFallbackFamily(tmp_path):
    logic_file = _writeLogicFile(
        _variantGfx950Path(tmp_path, "75a3"),
        devices="Device 75a3, Device 75a2",
    )

    assert _validateChipId(logic_file)


def test_validateChipIdRejectsDefaultFallbackChipIdInVariantDirectory(tmp_path):
    logic_file = _writeLogicFile(
        _variantGfx950Path(tmp_path, "75a3"),
        devices="Device 75a3, Device 75a2, Device 75a0",
    )

    assert not _validateChipId(logic_file)


@pytest.mark.parametrize(
    "content",
    [
        "\n".join(
            [
                "- MinimumRequiredVersion: 4.33.0",
                "- gfx950",
                "- gfx950",
                "- []",
                "",
            ]
        ),
        "\n".join(
            [
                "- {MinimumRequiredVersion: 4.33.0}",
                "- gfx950",
                "- gfx950",
                "",
            ]
        ),
    ],
)
def test_validateChipIdRejectsMissingOrMalformedGfx950DeviceLine(tmp_path, content):
    logic_file = _baseGfx950Path(tmp_path)
    logic_file.parent.mkdir(parents=True, exist_ok=True)
    logic_file.write_text(content)

    assert not _validateChipId(logic_file)


@pytest.mark.parametrize("devices", ["Device 74a0", "Device ffff"])
def test_validateChipIdRejectsMismatchedOrUnsupportedChipId(tmp_path, devices):
    logic_file = _writeLogicFile(_baseGfx950Path(tmp_path), devices=devices)

    assert not _validateChipId(logic_file)


def test_validateChipIdRejectsNonDefaultChipIdInBaseDirectory(tmp_path):
    logic_file = _writeLogicFile(_baseGfx950Path(tmp_path), devices="Device 75a3")

    assert not _validateChipId(logic_file)


def test_validateChipIdRejectsVariantDirectoryWithoutMatchingChipId(tmp_path):
    logic_file = _writeLogicFile(
        _variantGfx950Path(tmp_path, "75a3"),
        devices="Device 75a8",
    )

    assert not _validateChipId(logic_file)


def test_validateChipIdRejectsDefaultChipIdInVariantDirectory(tmp_path):
    logic_file = _writeLogicFile(
        _variantGfx950Path(tmp_path, "75a0"),
        devices="Device 75a0",
    )

    assert not _validateChipId(logic_file)


def test_validateChipIdRejectsFallbackChipIdInMalformedVariantDirectory(tmp_path):
    logic_file = _writeLogicFile(
        _malformedVariantGfx950Path(tmp_path, "75a3"),
        devices="Device 75a0",
    )

    assert not _validateChipId(logic_file)


def test_validateChipIdDoesNotRequireChipIdDirectoryForNonGatedArch(tmp_path):
    logic_file = _writeLogicFile(
        tmp_path / "aquavanjaram" / "gfx942" / "Equality" / "logic.yaml",
        gfx="gfx942",
        name="aquavanjaram",
        devices="Device 74a0",
    )

    assert _validateChipId(logic_file)


def test_validateChipIdIgnoresUnsupportedChipIdForNonGatedArch(tmp_path):
    logic_file = _writeLogicFile(
        tmp_path / "aquavanjaram" / "gfx942_20cu" / "GridBased" / "logic.yaml",
        gfx="gfx942",
        name="aquavanjaram",
        devices="Device 0050",
    )

    assert _validateChipId(logic_file)
