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

import importlib.util
import sys
import types
from pathlib import Path

import pytest

# When the rocisa C-extension is not importable (e.g. CI lint job), install a
# minimal stub so that `from rocisa import rocIsa` (in Tensile.Common.Utilities)
# and `import rocisa` (in Tensile.Common.Architectures) both succeed. When the
# real rocisa IS available we leave it untouched.
try:  # pragma: no cover - environment-dependent
    import rocisa  # noqa: F401
except ImportError:  # pragma: no cover
    _rocisa_stub = types.ModuleType("rocisa")

    class _RocIsaStub:  # noqa: D401 - test helper
        @staticmethod
        def getInstance():
            return None

    _rocisa_stub.rocIsa = _RocIsaStub
    sys.modules["rocisa"] = _rocisa_stub


# Load ValidChipId.py via importlib to bypass Tensile/TensileLogic/__init__.py,
# which transitively imports joblib / heavy build deps via Run.py.
def _load_validchipid_mod():
    p = Path(__file__).resolve().parents[2] / "TensileLogic" / "ValidChipId.py"
    spec = importlib.util.spec_from_file_location("ValidChipId_under_test", p)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_vci = _load_validchipid_mod()
_validateChipId = _vci._validateChipId
_fallbackFamily = _vci._fallbackFamily

# Architectures.py uses package-relative imports, so spec_from_file_location
# is not viable here. The rocisa stub above is sufficient to import it
# normally without the C-extension.
from Tensile.Common import Architectures as _arch  # noqa: E402

GFX_CHIP_IDS = _arch.GFX_CHIP_IDS
SUPPORTED_CHIP_ID_FALLBACKS = _arch.SUPPORTED_CHIP_ID_FALLBACKS


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


def test_validateChipIdRejectsDefaultFallbackChipIdInVariantDirectory(tmp_path, capsys):
    logic_file = _writeLogicFile(
        _variantGfx950Path(tmp_path, "75a3"),
        devices="Device 75a3, Device 75a2, Device 75a0",
    )

    assert not _validateChipId(logic_file)
    out = capsys.readouterr().out
    assert "may not declare default fallback chip IDs" in out


def test_validateChipIdRejectsEmptyDeviceList(tmp_path, capsys):
    # `- [Device]` (no chip after the keyword) is the only form that yields
    # an empty DeviceIds set without raising in the parser. ValidChipId must
    # surface the specific "must declare at least one Device chip ID" branch.
    logic_file = _baseGfx950Path(tmp_path)
    logic_file.parent.mkdir(parents=True, exist_ok=True)
    logic_file.write_text(
        "\n".join(
            [
                "- MinimumRequiredVersion: 4.33.0",
                "- gfx950",
                "- gfx950",
                "- [Device]",
                "",
            ]
        )
    )

    assert not _validateChipId(logic_file)
    out = capsys.readouterr().out
    assert "must declare at least one Device chip ID" in out


def test_validateChipIdRejectsMissingDeviceLine(tmp_path, capsys):
    # File with no Device line raises LogicFileError in the parser, which
    # ValidChipId must surface via the "Chip ID validation failed" path.
    logic_file = _baseGfx950Path(tmp_path)
    logic_file.parent.mkdir(parents=True, exist_ok=True)
    logic_file.write_text(
        "\n".join(
            [
                "- {MinimumRequiredVersion: 4.33.0}",
                "- gfx950",
                "- gfx950",
                "",
            ]
        )
    )

    assert not _validateChipId(logic_file)
    out = capsys.readouterr().out
    assert "Chip ID validation failed" in out


def test_validateChipIdRejectsMismatchedChipIdReportsPredicateError(tmp_path, capsys):
    # 74a0 belongs to gfx942, not gfx950 — _verifyPredicate must reject it.
    logic_file = _writeLogicFile(_baseGfx950Path(tmp_path), devices="Device 74a0")

    assert not _validateChipId(logic_file)
    out = capsys.readouterr().out
    assert "not associated with gfx950" in out


def test_validateChipIdRejectsUnsupportedChipIdReportsPredicateError(tmp_path, capsys):
    logic_file = _writeLogicFile(_baseGfx950Path(tmp_path), devices="Device ffff")

    assert not _validateChipId(logic_file)
    out = capsys.readouterr().out
    assert "device ID not supported" in out


def test_validateChipIdAcceptsSourceChipIdInBaseDirectory(tmp_path):
    logic_file = _writeLogicFile(_baseGfx950Path(tmp_path), devices="Device 75a3")

    assert _validateChipId(logic_file)


def test_validateChipIdAcceptsMixedSourceAndDefaultChipIdsInPlainArchDirectory(tmp_path):
    logic_file = _writeLogicFile(
        tmp_path / "gfx950" / "GridBased" / "logic.yaml",
        devices="Device 75a3, Device 75a0",
    )

    assert _validateChipId(logic_file)


def test_validateChipIdRejectsVariantDirectoryWithoutMatchingChipId(tmp_path, capsys):
    logic_file = _writeLogicFile(
        _variantGfx950Path(tmp_path, "75a3"),
        devices="Device 75a8",
    )

    assert not _validateChipId(logic_file)
    out = capsys.readouterr().out
    assert "directory must contain id=75a3" in out


def test_validateChipIdRejectsDefaultChipIdInVariantDirectory(tmp_path, capsys):
    logic_file = _writeLogicFile(
        _variantGfx950Path(tmp_path, "75a0"),
        devices="Device 75a0",
    )

    assert not _validateChipId(logic_file)
    out = capsys.readouterr().out
    assert "non-source chip ID" in out


def test_validateChipIdRejectsFallbackChipIdInMalformedVariantDirectory(tmp_path, capsys):
    logic_file = _writeLogicFile(
        _malformedVariantGfx950Path(tmp_path, "75a3"),
        devices="Device 75a0",
    )

    assert not _validateChipId(logic_file)
    out = capsys.readouterr().out
    assert "must use gfx950_id<chip> format" in out
    # Offending directory name must be reported so the user can act on it.
    assert "gfx950_75a3" in out


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


# ---------------------------------------------------------------------------
# Path-walk: chip-ID directory nearest the file wins, regardless of unrelated
# 'gfx950' segments earlier in the path. Regression coverage for the
# last-match-wins bug in _chipIdDirFromPath.
# ---------------------------------------------------------------------------

def test_validateChipIdResolvesNearestChipIdDirWithEnclosingGfxAncestor(tmp_path):
    # Layout: <tmp>/gfx950/checkout/gfx950/gfx950_id75a3/Equality/logic.yaml
    # The inner 'gfx950' must NOT reset the chip-ID dir state and cause the
    # variant directory to be missed.
    logic_file = _writeLogicFile(
        tmp_path / "gfx950" / "checkout" / "gfx950" / "gfx950_id75a3" / "Equality" / "logic.yaml",
        devices="Device 75a3",
    )

    assert _validateChipId(logic_file, display_path=logic_file.relative_to(tmp_path))


def test_validateChipIdAcceptsUppercaseHexInYaml(tmp_path):
    # Regression: uppercase hex in YAML must canonicalize before predicate
    # check; otherwise gets falsely reported as "device ID not supported".
    logic_file = _writeLogicFile(
        _variantGfx950Path(tmp_path, "75a3"),
        devices="Device 75A3",
    )

    assert _validateChipId(logic_file)


# ---------------------------------------------------------------------------
# "Test 5" parametrized matrix: every chip-ID-aware arch in the production
# registries must validate cleanly when placed in a representative directory.
# Derived directly from GFX_CHIP_IDS / SUPPORTED_CHIP_ID_FALLBACKS so this
# test stays in sync with the registry.
# ---------------------------------------------------------------------------

def _gated_archs():
    """Archs currently gated by supportsChipIdPredicate (only gfx950 today)."""
    return [gfx for gfx in GFX_CHIP_IDS if _arch.supportsChipIdPredicate(gfx)]


def _default_chip_ids(gfx):
    arch_keys = {f"id={cid.lower()}" for cid in GFX_CHIP_IDS[gfx]}
    return [
        cid for cid in GFX_CHIP_IDS[gfx]
        if f"id={cid.lower()}" not in SUPPORTED_CHIP_ID_FALLBACKS
        and f"id={cid.lower()}" in arch_keys
    ]


def _source_chip_ids(gfx):
    arch_keys = {f"id={cid.lower()}" for cid in GFX_CHIP_IDS[gfx]}
    return [
        cid for cid in GFX_CHIP_IDS[gfx]
        if f"id={cid.lower()}" in SUPPORTED_CHIP_ID_FALLBACKS
        and f"id={cid.lower()}" in arch_keys
    ]


_BASE_DIR_MATRIX = [
    (gfx, cid)
    for gfx in _gated_archs()
    for cid in GFX_CHIP_IDS[gfx]
]


@pytest.mark.parametrize("gfx,chip_id", _BASE_DIR_MATRIX)
def test_validateChipIdAcceptsAllArchChipIdsInBaseDir(tmp_path, gfx, chip_id):
    logic_file = _writeLogicFile(
        tmp_path / gfx / gfx / "Equality" / "logic.yaml",
        gfx=gfx,
        name=gfx,
        devices=f"Device {chip_id}",
    )

    assert _validateChipId(logic_file), (
        f"chip ID {chip_id} should validate in base {gfx} directory"
    )


_VARIANT_DIR_MATRIX = [
    (gfx, source_id)
    for gfx in _gated_archs()
    for source_id in _source_chip_ids(gfx)
]


@pytest.mark.parametrize("gfx,chip_id", _VARIANT_DIR_MATRIX)
def test_validateChipIdAcceptsEverySourceChipIdInItsVariantDir(tmp_path, gfx, chip_id):
    logic_file = _writeLogicFile(
        tmp_path / gfx / f"{gfx}_id{chip_id}" / "Equality" / "logic.yaml",
        gfx=gfx,
        name=gfx,
        devices=f"Device {chip_id}",
    )

    assert _validateChipId(logic_file), (
        f"source chip ID {chip_id} should validate in {gfx}_id{chip_id} directory"
    )


_NON_GATED_ARCHS = [gfx for gfx in GFX_CHIP_IDS if not _arch.supportsChipIdPredicate(gfx)]


@pytest.mark.parametrize("gfx", _NON_GATED_ARCHS)
def test_validateChipIdSkipsAllNonGatedArchs(tmp_path, gfx):
    # Pick any chip ID for the arch — placement rules don't apply to
    # non-gated archs, so the validator must short-circuit to True.
    chip_id = GFX_CHIP_IDS[gfx][0]
    logic_file = _writeLogicFile(
        tmp_path / gfx / gfx / "Equality" / "logic.yaml",
        gfx=gfx,
        name=gfx,
        devices=f"Device {chip_id}",
    )

    assert _validateChipId(logic_file)


# ---------------------------------------------------------------------------
# Direct unit tests for _fallbackFamily — pin the documented behaviour so
# changes to SUPPORTED_CHIP_ID_FALLBACKS or the family-expansion algorithm
# fail loudly here, not silently in placement validation.
# ---------------------------------------------------------------------------

def test_fallbackFamilyForDefaultChipIdIsSingleton():
    # A default chip ID has no entry in SUPPORTED_CHIP_ID_FALLBACKS; its
    # family must be just itself.
    assert _fallbackFamily("id=75a0", "gfx950") == {"id=75a0"}


def test_fallbackFamilyForSourceChipIdIncludesSiblingsAndFallback():
    # All gfx950 source IDs share id=75a0 as their direct fallback. The
    # family expansion is intentionally permissive so a single variant
    # directory can host logic for siblings sharing a fallback root.
    family = _fallbackFamily("id=75a3", "gfx950")
    assert "id=75a3" in family           # the chip itself
    assert "id=75a0" in family           # its direct fallback
    # All other source IDs whose direct fallback is also id=75a0 are siblings.
    expected_siblings = {
        src
        for src, fallbacks in _arch.SUPPORTED_CHIP_ID_FALLBACKS.items()
        if "id=75a0" in fallbacks and src in {f"id={c.lower()}" for c in GFX_CHIP_IDS["gfx950"]}
    }
    assert expected_siblings.issubset(family)
