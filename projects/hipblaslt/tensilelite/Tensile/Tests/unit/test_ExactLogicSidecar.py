# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for Tensile.ExactLogicSidecar (csv.xz ExactLogic sidecars)."""

from __future__ import annotations

import lzma
import os
import types

import pytest
import yaml

pytestmark = pytest.mark.unit

from Tensile import ExactLogicSidecar as ELS
from Tensile import LibraryIO
from Tensile.Common import IsaVersion
from Tensile.Common.Capabilities import makeIsaInfoMap
from Tensile.Common.GlobalParameters import assignGlobalParameters, globalParameters
from Tensile.Common.Utilities import state
from Tensile.Toolchain.Assembly import makeAssemblyToolchain
from Tensile.Toolchain.Validators import ToolchainDefaults, validateToolchain

POOL_FILE = os.path.join(os.path.dirname(__file__), "test_data", "solution_pool_gfx950.yaml")
POOL_TABLE = b"- - - [768, 3072, 1, 3840, 768, 768, 3840, 3840]\n    - [0, 0.0]\n"

# Unsorted, with duplicate keys (distinct payloads so order is observable),
# float, int and str speeds, and values at the top of the int32 range.
TABLE = [
    [[256, 256, 1, 256], [0, 1.5]],
    [[128, 128, 1, 128], [0, 0.0]],
    [[2**31, 1, 2**31 - 1, 33554432], [0, 6.57484e-05]],
    [[128, 128, 1, 128], [0, 7.0]],
    [[1, 1, 1, 1], [0, 0]],
    [[0, 0, 0, 0], [0, ".inf"]],
]


def _rows(table):
    return [(k[0], k[1], k[2], k[3], v[0], ELS.formatSpeed(v[1])) for k, v in table]


def _sortedTable(table):
    return sorted(([list(k), [v[0], v[1]]] for k, v in table), key=lambda r: r[0])


def _same(got, table):
    """True if *got* is *table* stably sorted by key, with each speed of the same type and value."""
    return got == _sortedTable(table) and ELS.rowsKey(got) == ELS.rowsKey(_sortedTable(table))


def _listTableText(table) -> bytes:
    out = []
    for i, (k, v) in enumerate(table):
        out.append(("- - - " if i == 0 else "  - - ") + "[%d, %d, %d, %d]\n" % tuple(k))
        out.append("    - [%d, %s]\n" % (v[0], ELS.formatSpeed(v[1])))
    return "".join(out).encode()


def _dictTableText(table) -> bytes:
    out = []
    for k, v in table:
        out.append("- - [%d, %d, %d, %d]\n" % tuple(k))
        out.append("  - [%d, %s]\n" % (v[0], ELS.formatSpeed(v[1])))
    return "".join(out).encode()


###############################################################################
# Codec
###############################################################################
def test_codec_round_trip_sorts_stably():
    blob = ELS.encodeTable(TABLE)
    assert _same(ELS.decodeTable(blob), TABLE)
    # Stable: the two [128,128,1,128] rows keep their relative order.
    rows = ELS.decodeRows(ELS.encodeRows([(5, 5, 5, 5, 9, "1.0"), (1, 1, 1, 1, 0, "0"), (5, 5, 5, 5, 3, "2.0")]))
    assert rows == [(1, 1, 1, 1, 0, "0"), (5, 5, 5, 5, 9, "1.0"), (5, 5, 5, 5, 3, "2.0")]


def test_codec_text_format():
    text = ELS.decodeText(ELS.encodeRows([(3, 2, 1, 4, 7, "98.042"), (2**31, 0, 1, 2, 8190, "'.inf'")]))
    assert text == "M,N,batch,K,solutionIdx,speed\n3,2,1,4,7,98.042\n2147483648,0,1,2,8190,'.inf'\n"
    # Plain .xz container, readable by stock tools.
    assert ELS.encodeRows([(1, 2, 3, 4, 5, "0.0")]).startswith(b"\xfd7zXZ\x00")


@pytest.mark.parametrize(
    "text, value",
    [
        ("0", 0), ("-3", -3), ("10331", 10331), ("0.0", 0.0), ("-0.0", -0.0), ("7.1", 7.1),
        ("2863.", 2863.0), ("6.57484e-05", 6.57484e-05), ("1.0E+16", 1e16),
        ("67.68302816901408", 67.68302816901408), ("'.inf'", ".inf"), ("''", ""), ("'a b#'", "a b#"),
    ],
)
def test_parse_speed_agrees_with_yaml(text, value):
    got = ELS.parseSpeed(text)
    assert ELS.speedKey(got) == ELS.speedKey(value)
    assert ELS.speedKey(yaml.load("[0, %s]" % text, Loader=yaml.CSafeLoader)[1]) == ELS.speedKey(value)


@pytest.mark.parametrize("text", ["", "1e5", ".inf", "007", "+1", "1_000", "'a'b'", "'a,b'", "null", "0x10", "1.0e5"])
def test_parse_speed_rejects_other_forms(text):
    with pytest.raises(ValueError):
        ELS.parseSpeed(text)


@pytest.mark.parametrize("value", [0, -3, 2**70, 0.0, -0.0, 0.875, 1e-05, 1e16, 5e-324, 1.7976931348623157e308,
                                   ".inf", "", "x y"])
def test_format_speed_round_trips(value):
    text = ELS.formatSpeed(value)
    assert ELS.speedKey(ELS.parseSpeed(text)) == ELS.speedKey(value)
    assert ELS.speedKey(yaml.load("[0, %s]" % text, Loader=yaml.CSafeLoader)[1]) == ELS.speedKey(value)


@pytest.mark.parametrize("value", [True, None, float("inf"), float("nan"), "a'b", "a,b", "\u00e9", [1.0]])
def test_format_speed_rejects_values_without_text_form(value):
    with pytest.raises(ValueError):
        ELS.formatSpeed(value)


def test_codec_keeps_speed_type_and_value():
    decoded = ELS.decodeTable(ELS.encodeTable([[[1, 2, 3, 4], [5, 0.875]], [[1, 2, 3, 5], [5, 0]]]))
    assert decoded == [[[1, 2, 3, 4], [5, 0.875]], [[1, 2, 3, 5], [5, 0]]]
    assert isinstance(decoded[0][1][1], float) and isinstance(decoded[1][1][1], int)


def test_codec_empty_and_single():
    assert ELS.decodeTable(ELS.encodeRows([])) == []
    assert ELS.decodeTable(ELS.encodeRows([(9, 8, 7, 6, 5, "0.0")])) == [[[9, 8, 7, 6], [5, 0.0]]]


def test_codec_rejects_bad_rows():
    with pytest.raises(ValueError):
        ELS.encodeTable([[[1, 2, 3], [0, 0.0]]])
    with pytest.raises(ValueError):
        ELS.encodeRows([(1, 2, 3, 4, 5)])
    with pytest.raises(ValueError):
        ELS.encodeRows([(1, 2, 3, 4, 5, "1e5")])


@pytest.mark.parametrize(
    "payload",
    [
        b"not xz at all",
        lzma.compress(b"wrong,header\n1,2,3,4,5,0.0\n"),
        lzma.compress(b"M,N,batch,K,solutionIdx\n1,2,3,4,5\n"),  # old header, no speed
        lzma.compress(b"M,N,batch,K,solutionIdx,speed\n1,2,3,4,5\n"),
        lzma.compress(b"M,N,batch,K,solutionIdx,speed\n1,2,x,4,5,0.0\n"),
        lzma.compress(b"M,N,batch,K,solutionIdx,speed\n1,2,3,4,5,1e5\n"),
        lzma.compress(b"M,N,batch,K,solutionIdx,speed\n1,2,3,4,5,0.0"),
        lzma.compress(b"M,N,batch,K,solutionIdx,speed\n\n1,2,3,4,5,0.0\n"),
        ELS.encodeRows([(1, 2, 3, 4, 5, "0.0")])[:-8],
    ],
)
def test_codec_corrupt_raises_with_source(payload):
    with pytest.raises(ELS.ExactLogicSidecarError, match="some/path"):
        ELS.decodeTable(payload, "some/path")


###############################################################################
# Text surgery
###############################################################################
DICT_HEAD = (b"# Copyright\nMinimumRequiredVersion: 5.0.0\nScheduleName: gfx942\n"
             b"Solutions:\n- SolutionIndex: 0\n  Name: '- - [1, 2, 3, 4]'\n"
             b"IndexOrder: [2, 3, 0, 1]\nExactLogic:\n")
DICT_TAIL = (b"RangeLogic: null\nTileSelectionIndices: null\nPerfMetric: DeviceEfficiency\n"
             b"LibraryType: GridBased\nUseKdTree: true\n")
LIST_HEAD = (b"# leading comment\n- {MinimumRequiredVersion: 5.0.0}\n- gfx942\n"
             b"- {Architecture: gfx942, CUCount: 228, UseKdTree: true}\n- [Device 0049]\n"
             b"- Activation: true\n  DataType: 0\n- - SolutionIndex: 0\n  - SolutionIndex: 1\n"
             b"- [2, 3, 0, 1]\n")
LIST_TAIL = b"- null\n- null\n- DeviceEfficiency\n- GridBased\n- {StaggerU: 32}\n"


def test_split_dict_surgery_is_byte_exact():
    res = ELS.splitText(DICT_HEAD + _dictTableText(TABLE) + DICT_TAIL)
    assert res.status == "split" and res.fmt == "dict"
    assert res.rows == _rows(TABLE)
    assert res.yamlText == DICT_HEAD[:-len(b"ExactLogic:\n")] + b"ExactLogic: null\n" + DICT_TAIL
    assert yaml.safe_load(res.yamlText)["ExactLogic"] is None


def test_split_list_surgery_is_byte_exact():
    res = ELS.splitText(LIST_HEAD + _listTableText(TABLE) + LIST_TAIL)
    assert res.status == "split" and res.fmt == "list"
    assert res.rows == _rows(TABLE)
    assert res.yamlText == LIST_HEAD + b"- null\n" + LIST_TAIL
    parsed = yaml.safe_load(res.yamlText)
    assert parsed[7] is None and parsed[11] == "GridBased"


@pytest.mark.parametrize(
    "text, status",
    [
        (DICT_HEAD + _dictTableText(TABLE) + DICT_TAIL.replace(b"GridBased", b"Equality"),
         "skip: not GridBased"),
        (LIST_HEAD + _listTableText(TABLE) + LIST_TAIL.replace(b"GridBased", b"Equality"),
         "skip: not GridBased"),
        (DICT_HEAD[:-1] + b" null\n" + DICT_TAIL, "skip: table already null"),
        (LIST_HEAD + b"- null\n" + LIST_TAIL, "skip: table already null"),
        (DICT_HEAD[:-1] + b" []\n" + DICT_TAIL, "skip: table not a block sequence (null/empty?)"),
        (LIST_HEAD + b"- []\n" + LIST_TAIL, "skip: table empty or not 4-int keys"),
        # 8-key (Range-style) GridBased tables are out of scope.
        (DICT_HEAD + b"- - [1, 2, 3, 4, 5, 6, 7, 8]\n  - [0, 0.0]\n" + DICT_TAIL,
         "skip: table empty or not 4-int keys"),
        (LIST_HEAD + b"- - - [1, 2, 3, 4, 5, 6, 7, 8]\n    - [0, 0.0]\n" + LIST_TAIL,
         "skip: table empty or not 4-int keys"),
        # Mixed 4/8-key table: first row matches, the rest does not.
        (DICT_HEAD + _dictTableText(TABLE) + b"- - [1, 2, 3, 4, 5, 6, 7, 8]\n  - [0, 0.0]\n" + DICT_TAIL,
         "skip: unrecognized table layout"),
        (LIST_HEAD + _listTableText(TABLE) + b"  - - [1, 2, 3, 4, 5, 6, 7, 8]\n    - [0, 0.0]\n" + LIST_TAIL,
         "skip: unrecognized table layout"),
        # Leading zero would be octal to YAML 1.1; negative keys are not expected.
        (DICT_HEAD + b"- - [1, 2, 3, 017]\n  - [0, 0.0]\n" + DICT_TAIL, "skip: table empty or not 4-int keys"),
        (DICT_HEAD + b"- - [1, 2, 3, -4]\n  - [0, 0.0]\n" + DICT_TAIL, "skip: table empty or not 4-int keys"),
        # Speed text outside the accepted forms keeps the table inline.
        (DICT_HEAD + b"- - [1, 2, 3, 4]\n  - [0, 1e5]\n" + DICT_TAIL, "skip: table empty or not 4-int keys"),
        (DICT_HEAD + _dictTableText(TABLE) + b"- - [1, 2, 3, 4]\n  - [0, .inf]\n" + DICT_TAIL,
         "skip: unrecognized table layout"),
        (LIST_HEAD + _listTableText(TABLE) + b"  - - [1, 2, 3, 4]\n    - [0, \"x\"]\n" + LIST_TAIL,
         "skip: unrecognized table layout"),
        ((DICT_HEAD + _dictTableText(TABLE) + DICT_TAIL).replace(b"\n", b"\r\n"), "skip: CR line endings"),
    ],
)
def test_split_ineligible(text, status):
    res = ELS.splitText(text)
    assert res.status == status
    assert res.rows is None and res.yamlText is None


###############################################################################
# Files: split / attach / verify
###############################################################################
def test_split_file_attach_and_verify(tmp_path):
    p = tmp_path / "a.yaml"
    original = LIST_HEAD + _listTableText(TABLE) + LIST_TAIL
    p.write_bytes(original)
    path, status, n, size = ELS.splitFile(str(p))
    assert (status, n) == ("split", len(TABLE))
    assert os.path.getsize(ELS.sidecarPath(p)) == size
    assert p.read_bytes() == LIST_HEAD + b"- null\n" + LIST_TAIL
    assert ELS.verifyFile(str(p)) == []

    raw = ELS.attachSidecar(yaml.safe_load(p.read_bytes()), p)
    assert _same(raw[7], TABLE)
    # Every speed comes back as the YAML loader read the original table.
    assert ELS.rowsKey(raw[7]) == ELS.rowsKey(sorted(yaml.safe_load(original)[7], key=lambda r: r[0]))
    # Second split is a no-op.
    assert ELS.splitFile(str(p))[1] == "skip: table already null"


def test_attach_without_sidecar_is_noop(tmp_path):
    p = tmp_path / "b.yaml"
    raw = {"ExactLogic": None, "LibraryType": "GridBased"}
    assert ELS.attachSidecar(raw, p) == {"ExactLogic": None, "LibraryType": "GridBased"}
    raw = [None] * 12
    assert ELS.attachSidecar(raw, str(p)) == [None] * 12


def test_attach_dict(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_bytes(DICT_HEAD + _dictTableText(TABLE) + DICT_TAIL)
    ELS.splitFile(str(p))
    raw = ELS.attachSidecar(yaml.safe_load(p.read_bytes()), str(p))
    assert _same(raw["ExactLogic"], TABLE)
    assert raw["UseKdTree"] is True


@pytest.mark.parametrize("fmt", ["dict", "list"])
def test_attach_rejects_inline_table_plus_sidecar(tmp_path, fmt):
    p = tmp_path / "d.yaml"
    text = (DICT_HEAD + _dictTableText(TABLE) + DICT_TAIL) if fmt == "dict" \
        else (LIST_HEAD + _listTableText(TABLE) + LIST_TAIL)
    p.write_bytes(text)
    with open(ELS.sidecarPath(p), "wb") as f:
        f.write(ELS.encodeTable(TABLE))
    with pytest.raises(ELS.ExactLogicSidecarError, match="inline"):
        ELS.attachSidecar(yaml.safe_load(p.read_bytes()), str(p))
    assert any("not null" in m for m in ELS.verifyFile(str(p)))
    # split repairs it: the inline table wins and replaces the sidecar.
    with open(ELS.sidecarPath(p), "wb") as f:
        f.write(ELS.encodeRows([(9, 9, 9, 9, 0, "0.0")]))
    assert ELS.splitFile(str(p))[1] == "split"
    assert _same(ELS.readSidecar(ELS.sidecarPath(p)), TABLE)


def test_attach_corrupt_sidecar_names_path(tmp_path):
    p = tmp_path / "e.yaml"
    p.write_bytes(DICT_HEAD[:-1] + b" null\n" + DICT_TAIL)
    with open(ELS.sidecarPath(p), "wb") as f:
        f.write(b"garbage")
    with pytest.raises(ELS.ExactLogicSidecarError, match="e.yaml.exactlogic.csv.xz"):
        ELS.attachSidecar(yaml.safe_load(p.read_bytes()), str(p))


def test_cli_split_dump_verify(tmp_path, capsys):
    (tmp_path / "sub").mkdir()
    p = tmp_path / "sub" / "f.yaml"
    p.write_bytes(DICT_HEAD + _dictTableText(TABLE) + DICT_TAIL)
    (tmp_path / "eq.yaml").write_bytes(DICT_HEAD + _dictTableText(TABLE) + b"LibraryType: Equality\n")
    assert ELS.main(["split", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "#      1 split" in out and "#      1 skip: not GridBased" in out
    assert ELS.main(["dump", ELS.sidecarPath(p)]) == 0
    assert capsys.readouterr().out.splitlines()[:3] == ["M,N,batch,K,solutionIdx,speed", "0,0,0,0,0,'.inf'",
                                                        "1,1,1,1,0,0"]
    assert ELS.main(["verify", str(tmp_path)]) == 0


###############################################################################
# End to end through LibraryIO.parseLibraryLogicFile
###############################################################################
@pytest.fixture(scope="module")
def pool_env() -> types.SimpleNamespace:
    cxxCompiler, _c_compiler, offload_bundler = validateToolchain(
        ToolchainDefaults.CXX_COMPILER,
        ToolchainDefaults.C_COMPILER,
        ToolchainDefaults.OFFLOAD_BUNDLER,
    )
    isa_info_map = makeIsaInfoMap([IsaVersion(9, 5, 0)], cxxCompiler)
    assignGlobalParameters({}, isa_info_map)
    assembler = makeAssemblyToolchain(cxxCompiler, offload_bundler, "default").assembler
    return types.SimpleNamespace(assembler=assembler, isaInfoMap=isa_info_map)


@pytest.fixture(autouse=True)
def _sequential_cpu_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(globalParameters, "CpuThreads", 1)


def _poolListText(useKdTree: bool) -> bytes:
    text = open(POOL_FILE, "rb").read()
    assert text.count(POOL_TABLE) == 1
    text = text.replace(POOL_TABLE, _listTableText(TABLE))
    if useKdTree:
        text = text.replace(b"\n- gfx950\n- [Device", b"\n- {Architecture: gfx950, UseKdTree: true}\n- [Device", 1)
    return text


def _poolDictText(useKdTree: bool) -> bytes:
    raw = yaml.load(_poolListText(False), Loader=yaml.CSafeLoader)
    d = {
        "MinimumRequiredVersion": raw[0]["MinimumRequiredVersion"],
        "ScheduleName": raw[1],
        "ArchitectureName": raw[2],
        "DeviceNames": raw[3],
        "ProblemType": raw[4],
        "Solutions": raw[5],
        "IndexOrder": raw[6],
        "ExactLogic": raw[7],
        "RangeLogic": raw[8],
        "PerfMetric": raw[10],
        "LibraryType": raw[11],
    }
    if useKdTree:
        d["UseKdTree"] = True
    return yaml.dump(d, Dumper=yaml.CSafeDumper, default_flow_style=None, sort_keys=False).encode()


def _parse(path, env):
    return LibraryIO.parseLibraryLogicFile(str(path), env.assembler, False, False, False, env.isaInfoMap, True)


def _libState(lib):
    return state(lib), {k: state(v) for k, v in lib.lazyLibraries.items()}


@pytest.mark.parametrize("fmt", ["list", "dict"])
@pytest.mark.parametrize("useKdTree", [False, True])
def test_parse_library_logic_file_with_sidecar(tmp_path, pool_env, fmt, useKdTree):
    text = _poolListText(useKdTree) if fmt == "list" else _poolDictText(useKdTree)
    orig = tmp_path / "orig" / "lib.yaml"
    conv = tmp_path / "conv" / "lib.yaml"
    for p in (orig, conv):
        p.parent.mkdir()
        p.write_bytes(text)
    assert ELS.splitFile(str(conv))[1] == "split"
    assert ELS.splitText(conv.read_bytes()).status == "skip: table already null"

    a = _parse(orig, pool_env)
    b = _parse(conv, pool_env)
    assert (a.schedule, a.architecture) == (b.schedule, b.architecture)
    assert a.problemType.state == b.problemType.state
    assert [s._state for s in a.solutions] == [s._state for s in b.solutions]
    assert _same(b.exactLogic, a.exactLogic)
    sa, sb = _libState(a.library), _libState(b.library)
    assert sa == sb
    assert (("useKdTree", True) in _items(sa)) == useKdTree


def _items(obj):
    """All (key, value) pairs of nested dicts, for a coarse membership check."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(v, (dict, list, tuple)):
                out.append((k, v))
            out.extend(_items(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_items(v))
    return out
