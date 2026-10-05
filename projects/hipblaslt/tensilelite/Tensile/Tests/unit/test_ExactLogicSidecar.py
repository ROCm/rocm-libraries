# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for Tensile.ExactLogicSidecar (csv.gz ExactLogic sidecars)."""

from __future__ import annotations

import csv
import gzip
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

# Unsorted, with duplicate keys, non-zero speeds, and values at the top of the
# int32 range. Row order must survive the round trip.
TABLE = [
    [[256, 256, 1, 256], [0, 1.5]],
    [[128, 128, 1, 128], [0, 0.0]],
    [[2**31, 1, 2**31 - 1, 33554432], [0, 0.25]],
    [[128, 128, 1, 128], [0, 7.0]],
    [[1, 1, 1, 1], [0, 0.0]],
    [[0, 0, 0, 0], [0, 0.0]],
]


def _rows(table):
    return [(k[0], k[1], k[2], k[3], v[0]) for k, v in table]


def _noSpeed(table):
    """The table as decoded from a sidecar: file order kept, speed dropped."""
    return [[list(k), [v[0], 0.0]] for k, v in table]


def _listTableText(table) -> bytes:
    out = []
    for i, (k, v) in enumerate(table):
        out.append(("- - - " if i == 0 else "  - - ") + "[%d, %d, %d, %d]\n" % tuple(k))
        out.append("    - [%d, %s]\n" % (v[0], repr(float(v[1]))))
    return "".join(out).encode()


def _dictTableText(table) -> bytes:
    out = []
    for k, v in table:
        out.append("- - [%d, %d, %d, %d]\n" % tuple(k))
        out.append("  - [%d, %s]\n" % (v[0], repr(float(v[1]))))
    return "".join(out).encode()


###############################################################################
# Codec
###############################################################################
def test_codec_round_trip_keeps_order():
    blob = ELS.encodeTable(TABLE)
    assert ELS.decodeTable(blob) == _noSpeed(TABLE)
    rows = [(5, 5, 5, 5, 9), (1, 1, 1, 1, 0), (5, 5, 5, 5, 3)]
    assert ELS.decodeRows(ELS.encodeRows(rows)) == rows


def test_codec_text_format():
    blob = ELS.encodeRows([(3, 2, 1, 4, 7), (2**31, 0, 1, 2, 8190)])
    # csv module default dialect: CRLF line endings, as in PR 12199's sidecars.
    assert ELS.decodeText(blob) == "M,N,batch,K,solutionIdx\r\n3,2,1,4,7\r\n2147483648,0,1,2,8190\r\n"
    # Plain gzip member, level 9, no file name, mtime 0: deterministic bytes.
    assert blob[:4] == b"\x1f\x8b\x08\x00" and blob[4:8] == b"\x00\x00\x00\x00" and blob[8] == 2
    assert blob == ELS.encodeRows([(3, 2, 1, 4, 7), (2**31, 0, 1, 2, 8190)])


def test_codec_readable_by_pr12199_loader(tmp_path):
    """Sidecars decode with the reader from PR 12199 (_loadMeshTableFromCsvGz)."""
    p = tmp_path / "x.yaml.csv.gz"
    p.write_bytes(ELS.encodeTable(TABLE))
    table = []
    with gzip.open(p, "rt") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            table.append([[int(row[0]), int(row[1]), int(row[2]), int(row[3])], [int(row[4]), 0.0]])
    assert table == _noSpeed(TABLE)


def test_codec_reads_pr12199_writer_output(tmp_path):
    """Sidecars written like PR 12199's split_mesh_yaml.py decode here."""
    p = tmp_path / "x.yaml.csv.gz"
    with gzip.open(p, "wt", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["M", "N", "batch", "K", "solutionIdx"])
        for k, v in TABLE:
            writer.writerow([k[0], k[1], k[2], k[3], v[0]])
    assert ELS.readSidecar(str(p)) == _noSpeed(TABLE)
    assert gzip.decompress(p.read_bytes()) == gzip.decompress(ELS.encodeTable(TABLE))


def test_codec_drops_speed_and_decodes_zero():
    decoded = ELS.decodeTable(ELS.encodeTable([[[1, 2, 3, 4], [5, 0.875]]]))
    assert decoded == [[[1, 2, 3, 4], [5, 0.0]]]
    assert isinstance(decoded[0][1][1], float)


def test_codec_empty_and_single():
    assert ELS.decodeTable(ELS.encodeRows([])) == []
    assert ELS.decodeTable(ELS.encodeRows([(9, 8, 7, 6, 5)])) == [[[9, 8, 7, 6], [5, 0.0]]]


def test_codec_rejects_bad_keys():
    with pytest.raises(ValueError):
        ELS.encodeTable([[[1, 2, 3], [0, 0.0]]])
    with pytest.raises(ValueError):
        ELS.encodeRows([(1, 2, 3, 4)])


@pytest.mark.parametrize(
    "payload",
    [
        b"not gzip at all",
        gzip.compress(b""),
        gzip.compress(b"wrong,header\r\n1,2,3,4,5\r\n"),
        gzip.compress(b"M,N,batch,K,solutionIdx\r\n1,2,3,4\r\n"),
        gzip.compress(b"M,N,batch,K,solutionIdx\r\n1,2,x,4,5\r\n"),
        gzip.compress(b"M,N,batch,K,solutionIdx\r\n1,2,3,4,5,6\r\n"),
        gzip.compress(b"M,N,batch,K,solutionIdx\r\n\r\n1,2,3,4,5\r\n"),
        gzip.compress("M,N,batch,K,solutionIdx\r\n1,2,3,4,é\r\n".encode()),
        ELS.encodeRows([(1, 2, 3, 4, 5)])[:-8],
    ],
)
def test_codec_corrupt_raises_with_source(payload):
    with pytest.raises(ELS.ExactLogicSidecarError, match="some/path"):
        ELS.decodeTable(payload, "some/path")


###############################################################################
# Eligibility and text surgery
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


def test_split_keeps_crlf_line_endings():
    text = (DICT_HEAD + _dictTableText(TABLE) + DICT_TAIL).replace(b"\n", b"\r\n")
    res = ELS.splitText(text)
    assert res.status == "split" and res.rows == _rows(TABLE)
    assert res.yamlText == (DICT_HEAD[:-len(b"ExactLogic:\n")] + b"ExactLogic: null\n"
                            + DICT_TAIL).replace(b"\n", b"\r\n")


@pytest.mark.parametrize("fmt", ["list", "dict"])
def test_split_accepts_any_scalar_speed(fmt):
    # Some gfx1250 logic files carry YAML-quoted speeds ('.inf'), which load as str.
    table = b"".join(
        (b"- - - " if fmt == "list" and i == 0 else b"  - - " if fmt == "list" else b"- - ")
        + b"[%d, 1, 1, %d]\n" % (i, i) + (b"    - " if fmt == "list" else b"  - ")
        + b"[%d, %s]\n" % (i, sp)
        for i, sp in enumerate([b"'.inf'", b"0.0", b"'.nan'", b"1.5", b".inf", b"null", b"3"]))
    text = (LIST_HEAD + table + LIST_TAIL) if fmt == "list" else (DICT_HEAD + table + DICT_TAIL)
    res = ELS.splitText(text)
    assert res.status == "split"
    assert res.rows == [(i, 1, 1, i, i) for i in range(7)]


def test_split_accepts_negative_keys():
    res = ELS.splitText(DICT_HEAD + b"- - [1, 2, 3, -4]\n  - [0, 0.0]\n" + DICT_TAIL)
    assert res.status == "split" and res.rows == [(1, 2, 3, -4, 0)]
    assert ELS.decodeRows(ELS.encodeRows(res.rows)) == res.rows


@pytest.mark.parametrize(
    "text, status",
    [
        (DICT_HEAD + _dictTableText(TABLE) + DICT_TAIL.replace(b"GridBased", b"Equality"),
         "skip: LibraryType 'Equality'"),
        (LIST_HEAD + _listTableText(TABLE) + LIST_TAIL.replace(b"GridBased", b"Equality"),
         "skip: LibraryType 'Equality'"),
        (DICT_HEAD[:-1] + b" null\n" + DICT_TAIL, "skip: table already null"),
        (LIST_HEAD + b"- null\n" + LIST_TAIL, "skip: table already null"),
        (DICT_HEAD[:-1] + b" []\n" + DICT_TAIL, "skip: empty table"),
        (LIST_HEAD + b"- []\n" + LIST_TAIL, "skip: empty table"),
        # 8-key (Range-style) GridBased tables are out of scope.
        (DICT_HEAD + b"- - [1, 2, 3, 4, 5, 6, 7, 8]\n  - [0, 0.0]\n" + DICT_TAIL, "skip: key is not 4 ints"),
        (LIST_HEAD + b"- - - [1, 2, 3, 4, 5, 6, 7, 8]\n    - [0, 0.0]\n" + LIST_TAIL, "skip: key is not 4 ints"),
        # Mixed 4/8-key table.
        (DICT_HEAD + _dictTableText(TABLE) + b"- - [1, 2, 3, 4, 5, 6, 7, 8]\n  - [0, 0.0]\n" + DICT_TAIL,
         "skip: key is not 4 ints"),
        (DICT_HEAD + b"- - [1, 2, 3, 4.0]\n  - [0, 0.0]\n" + DICT_TAIL, "skip: key is not 4 ints"),
        (DICT_HEAD + b"- - [1, 2, 3, true]\n  - [0, 0.0]\n" + DICT_TAIL, "skip: key is not 4 ints"),
        (DICT_HEAD + b"- - [1, 2, 3, 4]\n  - [-1, 0.0]\n" + DICT_TAIL, "skip: value is not [index, scalar speed]"),
        (DICT_HEAD + b"- - [1, 2, 3, 4]\n  - ['0', 0.0]\n" + DICT_TAIL, "skip: value is not [index, scalar speed]"),
        (DICT_HEAD + b"- - [1, 2, 3, 4]\n  - [0]\n" + DICT_TAIL, "skip: value is not [index, scalar speed]"),
        (DICT_HEAD + b"- - [1, 2, 3, 4]\n  - [0, [1.0]]\n" + DICT_TAIL, "skip: value is not [index, scalar speed]"),
        (b"- 1\n- 2\n", "skip: not a library logic file"),
        (DICT_TAIL, "skip: no ExactLogic key"),
        (DICT_TAIL.replace(b"GridBased", b"Equality"), "skip: LibraryType 'Equality'"),
        ((DICT_HEAD[:-1] + b" null\n" + DICT_TAIL).replace(b"GridBased", b"Equality"),
         "skip: LibraryType 'Equality'"),
    ],
)
def test_split_ineligible(text, status):
    res = ELS.splitText(text)
    assert res.status == status
    assert res.rows is None and res.yamlText is None


def test_split_rejects_unrecognized_layout():
    # Flow-style table: eligible by content, but the line-based surgery cannot isolate it.
    text = DICT_HEAD[:-1] + b" [[[1, 2, 3, 4], [0, 0.0]]]\n" + DICT_TAIL
    with pytest.raises(ELS.ExactLogicSidecarError):
        ELS.splitText(text)


###############################################################################
# Files: split / attach / verify
###############################################################################
def test_split_file_attach_and_verify(tmp_path):
    p = tmp_path / "a.yaml"
    original = LIST_HEAD + _listTableText(TABLE) + LIST_TAIL
    p.write_bytes(original)
    path, status, n, size = ELS.splitFile(str(p))
    assert (status, n) == ("split", len(TABLE))
    assert ELS.sidecarPath(p) == str(p) + ".csv.gz"
    assert os.path.getsize(ELS.sidecarPath(p)) == size
    assert p.read_bytes() == LIST_HEAD + b"- null\n" + LIST_TAIL
    assert ELS.verifyFile(str(p)) == []

    raw = ELS.attachSidecar(yaml.safe_load(p.read_bytes()), p)
    assert raw[7] == _noSpeed(TABLE)
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
    assert raw["ExactLogic"] == _noSpeed(TABLE)
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
        f.write(ELS.encodeRows([(9, 9, 9, 9, 0)]))
    assert ELS.splitFile(str(p))[1] == "split"
    assert ELS.readSidecar(ELS.sidecarPath(p)) == _noSpeed(TABLE)


def test_attach_corrupt_sidecar_names_path(tmp_path):
    p = tmp_path / "e.yaml"
    p.write_bytes(DICT_HEAD[:-1] + b" null\n" + DICT_TAIL)
    with open(ELS.sidecarPath(p), "wb") as f:
        f.write(b"garbage")
    with pytest.raises(ELS.ExactLogicSidecarError, match=r"e\.yaml\.csv\.gz"):
        ELS.attachSidecar(yaml.safe_load(p.read_bytes()), str(p))


def test_cli_split_dump_verify(tmp_path, capsys):
    (tmp_path / "sub").mkdir()
    p = tmp_path / "sub" / "f.yaml"
    p.write_bytes(DICT_HEAD + _dictTableText(TABLE) + DICT_TAIL)
    (tmp_path / "eq.yaml").write_bytes(DICT_HEAD + _dictTableText(TABLE) + b"LibraryType: Equality\n")
    assert ELS.main(["split", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "#      1 split" in out and "#      1 skip: LibraryType 'Equality'" in out
    assert ELS.main(["dump", ELS.sidecarPath(p)]) == 0
    assert capsys.readouterr().out.splitlines()[:2] == ["M,N,batch,K,solutionIdx", "256,256,1,256,0"]
    assert ELS.main(["verify", str(tmp_path)]) == 0
    os.remove(p)
    assert ELS.main(["verify", str(tmp_path)]) == 1
    assert "orphan sidecar" in capsys.readouterr().out


def test_cli_explicit_files_filtered_by_suffix(tmp_path, capsys):
    p = tmp_path / "a.yaml"
    p.write_bytes(LIST_HEAD + _listTableText(TABLE) + LIST_TAIL)
    (tmp_path / "notes.txt").write_text("x")
    assert ELS.main(["split", str(p), str(tmp_path / "notes.txt")]) == 0
    out = capsys.readouterr().out
    assert "#      1 split" in out and "notes.txt" not in out
    # An explicit YAML is verified, not mistaken for an orphan sidecar.
    assert ELS.main(["verify", str(p)]) == 0
    assert "# verified 1 file(s), 0 problem(s)" in capsys.readouterr().out
    os.remove(p)
    assert ELS.main(["verify", ELS.sidecarPath(p)]) == 1
    assert "orphan sidecar" in capsys.readouterr().out


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
    assert _noSpeed(a.exactLogic) == b.exactLogic
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
