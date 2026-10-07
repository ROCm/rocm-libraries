# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for Tensile.ExactLogicSidecar (columnar xz ExactLogic sidecars)."""

from __future__ import annotations

import lzma
import os
import shutil
import types
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

from Tensile import ExactLogicSidecar as ELS
from Tensile import LibraryIO
from Tensile.Common import IsaVersion
from Tensile.Common.Capabilities import makeIsaInfoMap
from Tensile.Common.GlobalParameters import assignGlobalParameters, globalParameters
from Tensile.SolutionLibrary import MatchingLibrary
from Tensile.Toolchain.Assembly import makeAssemblyToolchain
from Tensile.Toolchain.Validators import ToolchainDefaults, validateToolchain

POOL_FILE = os.path.join(os.path.dirname(__file__), "test_data", "solution_pool_gfx950.yaml")

# 4-key GridBased rows: unsorted, a duplicate key with different solutions,
# float and int speeds, and keys up to 2**31.
ROWS = [
    [[256, 128, 1, 64], [0, 812.5]],
    [[128, 256, 1, 64], [0, 0.0]],
    [[128, 256, 1, 64], [0, 3.25]],
    [[1, 1, 1, 1], [0, 0]],
    [[2**31, 2**31, 2**25, 2**31], [0, 0.266945]],
    [[128, 128, 4, 8192], [0, 1.0]],
]


def _sorted(rows: list[Any]) -> list[Any]:
    return [[list(k), [v[0], v[1]]] for k, v in sorted(rows, key=lambda r: tuple(r[0]))]


def _same(got: list[Any], rows: list[Any]) -> bool:
    """True if *got* is *rows* stably sorted by key, with each speed of the same type and value."""
    return got == _sorted(rows) and ELS.rowsKey(got) == ELS.rowsKey(_sorted(rows))


###############################################################################
# Codec
###############################################################################
@pytest.mark.parametrize(
    "rows",
    [
        [],
        [[[0, 0, 0, 0], [0, 0.0]]],
        ROWS,
        # descending keys give negative deltas
        [[[10 - i, 5 * i, 2**31 - i, i % 3], [i, 0.5]] for i in range(10)],
        # many duplicates; stable order of solution indices must be preserved
        [[[64, 64, 1, 64], [i, 0.0]] for i in range(300)],
        [[[-1, -2**31, 3, 0], [8190, 0.0]], [[-1, -2**31, 3, 0], [7, 0.0]]],
        # every speed type, and float values that == alone does not tell apart
        [[[i, 0, 0, 0], [i, sp]] for i, sp in enumerate(
            [0, 0.0, -0.0, 7, -3, 2**70, 1e-300, 5e-324, 1.7976931348623157e308, float("inf"),
             float("-inf"), float("nan"), ".inf", ".nan", "", "x'y,z\u00e9", None, 812.5])],
        # all speeds of one non-float type
        [[[i, 1, 1, 1], [0, 0]] for i in range(5)],
        [[[i, 1, 1, 1], [0, ".inf"]] for i in range(5)],
    ],
)
def test_codec_round_trip(rows: list[Any]) -> None:
    blob = ELS.encodeTable(rows)
    got = ELS.decodeTable(blob)
    assert ELS.rowsKey(got) == ELS.rowsKey(_sorted(rows))


def test_rows_key_is_strict() -> None:
    assert ELS.rowsKey([[[1, 1, 1, 1], [0, 0]]]) != ELS.rowsKey([[[1, 1, 1, 1], [0, 0.0]]])
    assert ELS.rowsKey([[[1, 1, 1, 1], [0, 0.0]]]) != ELS.rowsKey([[[1, 1, 1, 1], [0, -0.0]]])
    assert ELS.rowsKey([[[1, 1, 1, 1], [0, float("nan")]]]) == ELS.rowsKey([[[1, 1, 1, 1], [0, float("nan")]]])


def test_codec_payload_layout() -> None:
    payload = lzma.decompress(ELS.encodeTable(
        [[[3, 1, 1, 200], [5, 9.0]], [[1, 2, 1, 100], [300, 0]], [[4, 2, 1, 100], [1, ".inf"]]]))
    # magic, version, N=3, L=19 (bytes of key and index varints)
    # k0: 1,+2,+1 ; k1: 2,-1,+1 ; k2: 1,0,0 ; k3: 100,+100,-100 ; idx: 300, 5, 1
    # speed types: int, float, str ; float 9.0 ; int 0 ; str ".inf"
    assert payload == (b"TLXL\x02\x03\x13"
                       + b"\x02\x04\x02" + b"\x04\x01\x02" + b"\x02\x00\x00"
                       + b"\xc8\x01\xc8\x01\xc7\x01" + b"\xac\x02\x05\x01"
                       + b"\x01\x00\x02"
                       + b"\x00\x00\x00\x00\x00\x00\x22\x40" + b"\x00" + b"\x04.inf")


def test_codec_large_table() -> None:
    rows = [[[(i * 7919) % 33554432, i % 4096, 1 + i % 3, 2**31 - i], [i % 8191, i / 7.0]]
            for i in range(20000)]
    assert _same(ELS.decodeTable(ELS.encodeTable(rows)), rows)


@pytest.mark.parametrize("speed", [True, False, [1.0], {"a": 1}])
def test_codec_rejects_non_scalar_speed(speed: Any) -> None:
    with pytest.raises(ValueError, match="speed"):
        ELS.encodeTable([[[1, 1, 1, 1], [0, speed]]])


@pytest.mark.parametrize(
    "payload",
    [
        b"XXXX\x02\x00\x00",                          # bad magic
        b"TLXL\x01\x00\x00",                          # old version 1 (no speeds)
        b"TLXL\x03\x00\x00",                          # unknown version
        b"TLXL\x02\x01\x02\x00\x00",                  # truncated: 2 of 5 varints
        b"TLXL\x02\x01\x05\x00\x00\x00\x00\x80",      # dangling continuation byte
        b"TLXL\x02\x01\x09\x00\x00\x00\x00\x00",      # column length past the end
        b"TLXL\x02\x00\x00\x00",                      # trailing bytes after empty table
        b"TLXL\x02\x01\x05\x00\x00\x00\x00\x00",      # missing speed type column
        b"TLXL\x02\x01\x05\x00\x00\x00\x00\x00\x04",  # bad speed type
        b"TLXL\x02\x01\x05\x00\x00\x00\x00\x00\x00\x00",  # truncated float speed
        b"TLXL\x02\x01\x05\x00\x00\x00\x00\x00\x01",  # missing int speed
        b"TLXL\x02\x01\x05\x00\x00\x00\x00\x00\x02\x03ab",  # truncated str speed
        b"TLXL\x02\x01\x05\x00\x00\x00\x00\x00\x02\x01\xff",  # str speed not UTF-8
        b"TLXL\x02\x01\x05\x00\x00\x00\x00\x00\x03\x00",  # trailing bytes after speeds
    ],
)
def test_decode_rejects_corrupt_payload(payload: bytes) -> None:
    with pytest.raises(ELS.ExactLogicSidecarError, match="bad.sidecar"):
        ELS.decodeTable(lzma.compress(payload), "bad.sidecar")


def test_decode_rejects_non_xz() -> None:
    with pytest.raises(ELS.ExactLogicSidecarError, match="not a valid xz"):
        ELS.decodeTable(b"TLXL\x02\x00\x00", "x")


###############################################################################
# YAML fixtures
###############################################################################
def _tableText(rows: list[Any], listFormat: bool) -> str:
    out = []
    for i, (k, v) in enumerate(rows):
        if listFormat:
            out.append(("- - - " if i == 0 else "  - - ") + "[{}]\n".format(", ".join(map(str, k))))
            out.append("    - [{}, {}]\n".format(v[0], v[1]))
        else:
            out.append("- - [{}]\n".format(", ".join(map(str, k))))
            out.append("  - [{}, {}]\n".format(v[0], v[1]))
    return "".join(out)


def _listFormatText(rows: list[Any]) -> str:
    """The gfx950 pool file with its 8-key table replaced by a 4-key table."""
    with open(POOL_FILE) as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("- - - "))
    end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("-"))
    return "".join(lines[:start]) + _tableText(rows, True) + "".join(lines[end:])


def _dictFormatText(rows: list[Any], useKdTree: bool = False) -> str:
    raw = yaml.load(_listFormatText(rows), Loader=yaml.CSafeLoader)
    d = LibraryIO.parseLibraryLogicList(raw, POOL_FILE)
    lib = d.pop("Library")
    d["LibraryType"] = lib["distance"]
    table = d.pop("ExactLogic")
    order = ["MinimumRequiredVersion", "ScheduleName", "ArchitectureName", "CUCount",
             "DeviceNames", "ProblemType", "Solutions", "IndexOrder"]
    head = yaml.dump({k: d[k] for k in order}, default_flow_style=None, sort_keys=False)
    tail = yaml.dump({k: d[k] for k in ["RangeLogic", "TileSelectionIndices", "PerfMetric", "LibraryType"]},
                     default_flow_style=False, sort_keys=False)
    if useKdTree:
        tail += "UseKdTree: true\n"
    assert table is not None
    return head + "ExactLogic:\n" + _tableText(rows, False) + tail


@pytest.fixture(scope="module")
def pool_env() -> types.SimpleNamespace:
    cxxCompiler, _c, bundler = validateToolchain(
        ToolchainDefaults.CXX_COMPILER, ToolchainDefaults.C_COMPILER, ToolchainDefaults.OFFLOAD_BUNDLER)
    isaInfoMap = makeIsaInfoMap([IsaVersion(9, 5, 0)], cxxCompiler)
    assignGlobalParameters({}, isaInfoMap)
    assembler = makeAssemblyToolchain(cxxCompiler, bundler, "default").assembler
    return types.SimpleNamespace(assembler=assembler, isaInfoMap=isaInfoMap)


@pytest.fixture(autouse=True)
def _sequential_cpu_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(globalParameters, "CpuThreads", 1)


def _parse(path: str, env: types.SimpleNamespace):
    return LibraryIO.parseLibraryLogicFile(path, env.assembler, False, False, False, env.isaInfoMap, False)


def _matchingTable(logic) -> list[Any]:
    """``(key, solution index)`` rows of the single MatchingLibrary in *logic*."""
    found = []

    def walk(node):
        if isinstance(node, MatchingLibrary):
            found.append(node)
            return
        for attr in ("library", "rows", "mapping"):
            child = getattr(node, attr, None)
            if isinstance(child, dict):
                for v in child.values():
                    walk(v)
            elif isinstance(child, list):
                for r in child:
                    walk(r.get("library") if isinstance(r, dict) else r)
            elif child is not None:
                walk(child)

    walk(logic.library.library)
    assert len(found) == 1
    m = found[0]
    return [m.distance, m.useKdTree, [(r["key"], r["index"].state(), "speed" in r) for r in m.table]]


###############################################################################
# End to end through parseLibraryLogicFile
###############################################################################
@pytest.mark.parametrize("fmt", ["list", "dict", "dict-kdtree"])
def test_split_then_parse_matches_original(tmp_path: Any, pool_env: types.SimpleNamespace, fmt: str) -> None:
    text = _listFormatText(ROWS) if fmt == "list" else _dictFormatText(ROWS, fmt == "dict-kdtree")
    orig = tmp_path / "orig" / "lib.yaml"
    conv = tmp_path / "conv" / "lib.yaml"
    orig.parent.mkdir()
    conv.parent.mkdir()
    orig.write_text(text)
    conv.write_text(text)

    status, _ = ELS.splitFile(str(conv))
    assert status == "converted"
    assert os.path.isfile(ELS.sidecarPath(str(conv)))
    convText = conv.read_text()
    assert ("ExactLogic: null\n" if fmt != "list" else "\n- null\n") in convText
    assert len(convText) < len(text)

    a = _parse(str(orig), pool_env)
    b = _parse(str(conv), pool_env)
    assert (a.schedule, a.architecture) == (b.schedule, b.architecture)
    assert [s.getAttributes() for s in a.solutions] == [s.getAttributes() for s in b.solutions]
    assert _same(b.exactLogic, a.exactLogic)
    ta, tb = _matchingTable(a), _matchingTable(b)
    assert ta == tb
    assert ta[0] == "GridBased" and ta[1] == (fmt == "dict-kdtree")
    assert not any(hasSpeed for _, _, hasSpeed in ta[2])

    # Splitting again is a no-op.
    assert ELS.splitFile(str(conv))[0] == "already"
    assert ELS.verifyFile(str(conv)) == ("ok", "{} rows".format(len(ROWS)))


@pytest.mark.parametrize("fmt", ["list", "dict"])
def test_surgery_keeps_other_bytes(tmp_path: Any, fmt: str) -> None:
    text = _listFormatText(ROWS) if fmt == "list" else _dictFormatText(ROWS)
    table = _tableText(ROWS, fmt == "list")
    p = tmp_path / "lib.yaml"
    p.write_text(text)
    ELS.splitFile(str(p))
    if fmt == "list":
        expected = text.replace(table, "- null\n")
    else:
        expected = text.replace("ExactLogic:\n" + table, "ExactLogic: null\n")
    assert expected != text
    assert p.read_text() == expected


def test_sidecar_with_inline_table_is_ambiguous(tmp_path: Any) -> None:
    text = _dictFormatText(ROWS)
    p = tmp_path / "lib.yaml"
    p.write_text(text)
    ELS.writeSidecar(ELS.sidecarPath(str(p)), ROWS)
    raw = yaml.load(text, Loader=yaml.CSafeLoader)
    with pytest.raises(ELS.ExactLogicSidecarError, match="present in the YAML and in sidecar"):
        ELS.attachSidecar(raw, str(p))
    with pytest.raises(ELS.ExactLogicSidecarError, match="sidecar exists and table is not null"):
        ELS.splitFile(str(p))


def test_sidecar_next_to_non_gridbased_is_error(tmp_path: Any) -> None:
    p = tmp_path / "lib.yaml"
    raw = {"LibraryType": "Equality", "ExactLogic": None}
    ELS.writeSidecar(ELS.sidecarPath(str(p)), ROWS)
    with pytest.raises(ELS.ExactLogicSidecarError, match="not GridBased"):
        ELS.attachSidecar(raw, str(p))


def test_attach_without_sidecar_is_noop(tmp_path: Any) -> None:
    raw = {"LibraryType": "GridBased", "ExactLogic": None}
    assert ELS.attachSidecar(raw, str(tmp_path / "lib.yaml")) is raw
    assert raw["ExactLogic"] is None
    assert ELS.attachSidecar(raw, None) is raw


def test_corrupt_sidecar_reports_path(tmp_path: Any) -> None:
    p = tmp_path / "lib.yaml"
    sc = ELS.sidecarPath(str(p))
    with open(sc, "wb") as f:
        f.write(b"garbage")
    with pytest.raises(ELS.ExactLogicSidecarError, match=str(sc).replace(".", r"\.")):
        ELS.attachSidecar({"LibraryType": "GridBased", "ExactLogic": None}, str(p))


@pytest.mark.parametrize(
    "raw, reason",
    [
        ({"LibraryType": "Equality", "ExactLogic": ROWS}, "LibraryType"),
        ({"LibraryType": "GridBased", "ExactLogic": []}, "empty"),
        ({"LibraryType": "GridBased", "ExactLogic": None}, "empty"),
        ({"LibraryType": "GridBased", "ExactLogic": [[[1, 2, 3, 4, 5, 6, 7, 8], [0, 0.0]]]}, "key"),
        ({"LibraryType": "GridBased", "ExactLogic": [[[1, 2, 3, 4.0], [0, 0.0]]]}, "key"),
        ({"LibraryType": "GridBased", "ExactLogic": [[[1, 2, 3, True], [0, 0.0]]]}, "key"),
        ({"LibraryType": "GridBased", "ExactLogic": [[[1, 2, 3, 4], [0]]]}, "value"),
        ({"LibraryType": "GridBased", "ExactLogic": [[[1, 2, 3, 4], [-1, 0.0]]]}, "value"),
        ({"LibraryType": "GridBased", "ExactLogic": [[[1, 2, 3, 4], ["0", 0.0]]]}, "value"),
        ({"LibraryType": "GridBased", "ExactLogic": [[[1, 2, 3, 4], [0, True]]]}, "value"),
        ({"LibraryType": "GridBased", "ExactLogic": [[[1, 2, 3, 4], [0, [1.0]]]]}, "value"),
        ({"LibraryType": "GridBased", "ExactLogic": [[[1, 2, 3, 4], [0, [1.0]]]]}, "value"),
    ],
)
def test_ineligible(raw: Any, reason: str) -> None:
    ok, why = ELS.isEligible(raw)
    assert not ok and reason in why


@pytest.mark.parametrize("fmt", ["list", "dict"])
def test_string_speed_is_eligible_and_kept(tmp_path: Any, fmt: str) -> None:
    # Some gfx1250 logic files carry YAML-quoted speeds ('.inf'), which load as str.
    speeds = ["'.inf'", 0.0, "'.nan'", 1.5, "'.inf'", 0.0]  # written verbatim into the YAML
    rows = [[k, [v[0], sp]] for (k, v), sp in zip(ROWS, speeds)]
    text = _listFormatText(rows) if fmt == "list" else _dictFormatText(rows)
    p = tmp_path / "lib.yaml"
    p.write_text(text)
    raw = LibraryIO.read(str(p), True)
    table = raw["ExactLogic"] if fmt == "dict" else raw[ELS.LIST_TABLE_INDEX]
    assert [r[1][1] for r in table] == [".inf", 0.0, ".nan", 1.5, ".inf", 0.0]
    assert ELS.isEligible(raw) == (True, "")
    assert ELS.splitFile(str(p))[0] == "converted"
    attached = ELS.attachSidecar(LibraryIO.read(str(p), True), str(p))
    table = attached["ExactLogic"] if fmt == "dict" else attached[ELS.LIST_TABLE_INDEX]
    assert _same(table, [[k, [v[0], sp.strip("'") if isinstance(sp, str) else sp]]
                         for (k, v), sp in zip(ROWS, speeds)])


def test_cli_split_dump_verify(tmp_path: Any, capsys: pytest.CaptureFixture) -> None:
    d = tmp_path / "Logic"
    d.mkdir()
    (d / "a.yaml").write_text(_listFormatText(ROWS))
    (d / "b.yaml").write_text(_dictFormatText(ROWS))
    shutil.copy(POOL_FILE, d / "c.yaml")  # 8-key table: skipped
    assert ELS.main(["split", str(d)]) == 0
    assert "converted=2" in capsys.readouterr().out
    assert sorted(os.listdir(d)) == sorted(["a.yaml", "b.yaml", "c.yaml",
                                            "a.yaml" + ELS.SIDECAR_SUFFIX, "b.yaml" + ELS.SIDECAR_SUFFIX])
    assert ELS.main(["verify", str(d)]) == 0
    assert "ok=2" in capsys.readouterr().out
    assert ELS.main(["dump", str(d / ("a.yaml" + ELS.SIDECAR_SUFFIX))]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "k0,k1,k2,k3,solutionIdx,speed"
    assert out[1:] == ["{},{},{},{},{},{!r}".format(*k, v[0], v[1]) for k, v in _sorted(ROWS)]
    os.remove(d / "a.yaml")
    assert ELS.main(["verify", str(d)]) == 1


def test_merge_library_load_data_attaches_sidecar(tmp_path: Any) -> None:
    from Tensile import TensileMergeLibrary

    p = tmp_path / "lib.yaml"
    p.write_text(_dictFormatText(ROWS))
    ELS.splitFile(str(p))
    _, data, _ = TensileMergeLibrary.loadData(str(p))
    assert _same(data["ExactLogic"], ROWS)


def test_lib_logic_to_yaml_reader_attaches_sidecar(tmp_path: Any) -> None:
    p = tmp_path / "lib.yaml"
    p.write_text(_listFormatText(ROWS))
    ELS.splitFile(str(p))
    raw = ELS.attachSidecar(LibraryIO.readYAML(str(p)), str(p))
    assert _same(LibraryIO.rawLibraryLogic(raw)[7], ROWS)


def test_update_library_keeps_sidecar_pairing(tmp_path: Any) -> None:
    """UpdateLogic parses with the sidecar table but writes the YAML table back as null."""
    from unittest.mock import Mock, patch
    from Tensile import TensileUpdateLibrary

    inDir, outDir = tmp_path / "in", tmp_path / "out"
    inDir.mkdir()
    p = inDir / "lib.yaml"
    p.write_text(_dictFormatText(ROWS))
    ELS.splitFile(str(p))
    seen = {}

    def fakeParse(data, filename):
        seen["table"] = data["ExactLogic"]
        pt = Mock()
        pt.state = {k: Mock(value=0) for k in [
            "DataType", "MacDataTypeA", "MacDataTypeB", "DataTypeA", "DataTypeB", "DataTypeE",
            "DataTypeAmaxD", "DestDataType", "ComputeDataType", "ActivationComputeDataType",
            "ActivationType", "F32XdlMathOp"]}
        pt.state["BiasDataTypeList"] = []
        return (None, None, pt, [], None, None, None)

    with patch.object(TensileUpdateLibrary.LibraryIO, "parseLibraryLogicData", side_effect=fakeParse):
        TensileUpdateLibrary.UpdateLogic(str(p), str(inDir), str(outDir))
    assert _same(seen["table"], ROWS)
    out = outDir / "lib.yaml"
    assert yaml.safe_load(out.read_text())["ExactLogic"] is None
    assert _same(ELS.readSidecar(ELS.sidecarPath(str(out))), ROWS)


def test_cli_verify_explicit_files(tmp_path: Any, capsys: pytest.CaptureFixture) -> None:
    # Explicit YAML paths must not be mistaken for orphan sidecars.
    p = tmp_path / "a.yaml"
    p.write_text(_listFormatText(ROWS))
    assert ELS.main(["split", str(p)]) == 0
    capsys.readouterr()
    assert ELS.main(["verify", str(p)]) == 0
    cap = capsys.readouterr()
    assert "ok=1" in cap.out and "error" not in cap.out + cap.err
    os.remove(p)
    assert ELS.main(["verify", str(p) + ELS.SIDECAR_SUFFIX]) != 0
    assert "orphan sidecar" in capsys.readouterr().err
