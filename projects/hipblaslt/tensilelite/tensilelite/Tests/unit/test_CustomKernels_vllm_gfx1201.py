# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The vLLM skinny TN custom kernels for gfx1201 (CustomKernels/vllm/*_gfx1201.s)
and the Range logic that routes to them (asm_full/gfx1201/Range/)."""

import itertools
import math
import os
import re
from copy import deepcopy

import pytest
import yaml
from logic_corpus import needs_logic_corpus, require_logic_file

import tensilelite
from tensilelite.CustomKernels import (
    getCustomKernelConfig,
    readCustomKernelConfig,
    validateCustomKernelMetadata,
)
from tensilelite.resources import custom_kernel_names, custom_kernel_text
from tensilelite.SolutionStructs.Problem import ProblemType

pytestmark = pytest.mark.unit

_CK_ROOT = os.path.join(os.path.dirname(tensilelite.__file__), "CustomKernels")

# The shipped kernels, each name read as (family, type of C and D, n, tile);
# wvSplitK takes A and B of that type, wvSplitKQ FP8 A and B. Which ones ship
# follows the measured Range rows, so the set is read from the tree, and
# test_gfx1201_range_logic_uses_every_kernel ties it to the rows.
_NAME = re.compile(
    r"^(?:(wvSplitK)_vllm_(bf16|f16)_tn_n([1-5])_(y\du\d)"
    r"|(wvSplitKQ)_vllm_f8_(bf16|f16)_tn_n([1-4])_(y\du\d))_gfx1201$"
)


def _parse(name):
    m = _NAME.match(name)
    if not m:
        return None
    family, dt, n, tile = next(m.groups()[i : i + 4] for i in (0, 4) if m.group(i + 1))
    return family, dt, int(n), tile


_KERNELS = {name: _parse(name) for name in custom_kernel_names() if _parse(name)}

_ARGS = [
    "SizeSum",
    "SizeFree0",
    "AddressA",
    "AddressB",
    "ComputeUnits",
    "SizeFree1",
    "StrideA0",
    "StrideB0",
    "AddressC",
    "AddressD",
    "Alpha",
    "Beta",
    "StrideC0",
    "StrideD0",
]
_ARG_BYTES = {"int32": 4, "float32": 4, "address": 8}
_UNIT_STRIDE_KEYS = (
    "AssertStrideAEqual",
    "AssertStrideBEqual",
    "AssertStrideCEqual",
    "AssertStrideDEqual",
)


def test_gfx1201_kernel_names():
    """Every gfx1201 vLLM kernel's name gives its family, types, n and tile."""
    gfx1201 = {
        name for name in custom_kernel_names() if name.endswith("_gfx1201") and "_vllm_" in name
    }
    assert gfx1201 and gfx1201 == set(_KERNELS)


@pytest.mark.parametrize("name", sorted(_KERNELS))
def test_gfx1201_kernel_target_and_code_object(name):
    """Built for gfx1201 with 32-lane waves. Library logic references these
    kernels, so TensileCreateLibrary links each into one code object with
    Tensile's kernels, at hipBLASLt's default code object v4; ld.lld rejects
    mixed ABI versions, and the directive in the .s overrides the assembler's
    -mcode-object-version."""
    text = custom_kernel_text(name)
    assert re.search(r'^\s*\.amdgcn_target\s+"amdgcn-amd-amdhsa--gfx1201"\s*$', text, re.M)
    assert re.search(r"^\s*\.amdhsa_code_object_version\s+4\s*$", text, re.M)
    assert re.search(r"^\s*\.amdhsa_wavefront_size32\s+1\s*$", text, re.M)


@pytest.mark.parametrize("name", sorted(_KERNELS))
def test_gfx1201_kernel_config(name):
    """TN with a skinny n, the layout torch.mm(x, w.t()) and F.linear reach
    hipBLASLt with, and a bias of D's type on each row of D: a null bias
    address adds nothing, so the kernels also serve GEMMs without a bias. Rows
    past m read the last row and are not written, so m has no bound, and the
    tokens past the LDS stage are staged in chunks (wvSplitK) or read from
    global memory (wvSplitKQ), so K has none either. Each row of D is reduced
    by one wave in the same order wherever it falls, so the kernels declare
    uniformSummationOrder."""
    family, dt, n, _ = _KERNELS[name]
    valid, msg = validateCustomKernelMetadata(name, _CK_ROOT)
    assert valid, msg

    config = getCustomKernelConfig(name, {}, _CK_ROOT)
    pt = config["ProblemType"]
    d = {"bf16": "b", "f16": "h"}[dt]
    a = "F8" if family == "wvSplitKQ" else d
    assert (pt["DataType"], pt["DestDataType"], pt["ComputeDataType"]) == (a, d, "s")
    assert (pt["TransposeA"], pt["TransposeB"]) == (True, False)
    assert pt.get("UseScaleAB", "") == ("Scalar" if family == "wvSplitKQ" else "")
    assert not pt.get("Activation") and not pt.get("UseScaleAlphaVec")
    assert (pt["UseBias"], pt["BiasSrc"], pt["BiasDataTypeList"]) == (1, "D", [d])
    assert readCustomKernelConfig(name, _CK_ROOT)["Features"]["SupportsBias"] is True

    assert config["AssertSizeEqual"] == {1: n, 2: 1}
    assert "AssertSizeGreaterThan" not in config
    assert "AssertSizeLessThan" not in config
    assert config["AssertSummationElementMultiple"] == {"wvSplitK": 8, "wvSplitKQ": 16}[family]
    assert config["StaggerU"] == 0
    for key in _UNIT_STRIDE_KEYS:
        assert config[key] == {0: 1}, key
    assert config["WavefrontSize"] == 32

    ck = config["CustomKernel"]
    assert ck["grid"] == ["ComputeUnits", "One", "One"]
    assert ck["threads"] == [32, 16, 1]
    scales = ["AddressScaleA", "AddressScaleB"] if family == "wvSplitKQ" else []
    assert [arg["semantic"] for arg in ck["args"]] == [*_ARGS, *scales, "AddressBias"]
    assert ck["uniformSummationOrder"] is True


@pytest.mark.parametrize("name", sorted(_KERNELS))
def test_gfx1201_args_match_kernarg_layout(name):
    """hipBLASLt packs these kernels' arguments from custom.config alone, back to
    back, so the list must match the kernel's own argument metadata: an argument
    missing from custom.config would leave the kernel reading past the buffer."""
    args = getCustomKernelConfig(name, {}, _CK_ROOT)["CustomKernel"]["args"]
    lines = custom_kernel_text(name).splitlines()
    start = lines.index("---") + 1
    kernel = yaml.safe_load("\n".join(lines[start : lines.index("...", start)]))["amdhsa.kernels"][
        0
    ]
    assert len(args) == len(kernel[".args"])
    offset = 0
    for arg, meta in zip(args, kernel[".args"]):
        size = _ARG_BYTES[arg["type"]]
        assert (meta[".offset"], meta[".size"]) == (offset, size), (arg["semantic"], meta)
        assert (meta[".value_kind"] == "global_buffer") == (arg["type"] == "address"), (
            arg["semantic"],
            meta,
        )
        offset += size
    assert kernel[".kernarg_segment_size"] == offset


_HALF, _BFLOAT16, _FLOAT8 = 4, 7, 15
_HEADER = {
    "ScheduleName": "gfx1201",
    "ArchitectureName": "gfx1201",
    # What hipBLASLt reads on the RX 9070 XT (HIP counts workgroup processors).
    "CUCount": 32,
    "DeviceNames": ["Device 73f0"],
    "LibraryType": "Range",
}
_RANGE = "gfx1201/Range/gfx1201_Cijk_Alik_Bljk_{}_UserArgs.yaml"


def _tn_problem_type(dataType, destDataType, useScaleAB=None, bias=False):
    """Plain TN GEMM, with every unlisted field at its TensileLite default."""
    problemType = {
        "OperationType": "GEMM",
        "DataType": dataType,
        "DestDataType": destDataType,
        "ComputeDataType": 0,
        "HighPrecisionAccumulate": True,
        "TransposeA": True,
        "TransposeB": False,
        "UseBeta": True,
        "Batched": True,
        "SupportUserArgs": True,
    }
    if useScaleAB is not None:
        problemType["UseScaleAB"] = useScaleAB
    if bias:
        problemType.update(UseBias=1, BiasSrc="D", BiasDataTypeList=[destDataType])
    return problemType


def _boxes_overlap(a, b):
    """Range keys [m_min, m_max, n_min, n_max, batch_min, batch_max, K_min, K_max]."""
    for dim in range(4):
        loA, hiA, loB, hiB = a[2 * dim], a[2 * dim + 1], b[2 * dim], b[2 * dim + 1]
        if (hiA != -1 and hiA < loB) or (hiB != -1 and hiB < loA):
            return False
    return True


def _box_covered(box, boxes):
    """Whether every problem in Range key box lies in one of boxes. Checks one
    problem per cell of the grid that all their bounds cut each dimension into."""
    span = lambda b, dim: (b[2 * dim], math.inf if b[2 * dim + 1] == -1 else b[2 * dim + 1] + 1)
    corners = []
    for dim in range(4):
        lo, hi = span(box, dim)
        cuts = {x for b in (box, *boxes) for x in span(b, dim)}
        corners.append(sorted(x for x in cuts if lo <= x < hi))
    inside = lambda p, b: all(span(b, dim)[0] <= p[dim] < span(b, dim)[1] for dim in range(4))
    return all(any(inside(p, b) for b in boxes) for p in itertools.product(*corners))


def _kernels(family, dt):
    return {name for name, (f, d, _, _) in _KERNELS.items() if (f, d) == (family, dt)}


_RANGE_FILES = [
    (
        "BBS_BH",
        _tn_problem_type(_BFLOAT16, _BFLOAT16),
        _kernels("wvSplitK", "bf16"),
        None,
    ),
    (
        "HHS_BH",
        _tn_problem_type(_HALF, _HALF),
        _kernels("wvSplitK", "f16"),
        None,
    ),
    (
        "BBS_BH_Bias_B",
        _tn_problem_type(_BFLOAT16, _BFLOAT16, bias=True),
        _kernels("wvSplitK", "bf16"),
        "BBS_BH",
    ),
    (
        "HHS_BH_Bias_H",
        _tn_problem_type(_HALF, _HALF, bias=True),
        _kernels("wvSplitK", "f16"),
        "HHS_BH",
    ),
    (
        "F8BS_BH_SAB",
        _tn_problem_type(_FLOAT8, _BFLOAT16, "Scalar"),
        _kernels("wvSplitKQ", "bf16"),
        None,
    ),
    ("F8HS_BH_SAB", _tn_problem_type(_FLOAT8, _HALF, "Scalar"), _kernels("wvSplitKQ", "f16"), None),
    (
        "F8BS_BH_Bias_B_SAB",
        _tn_problem_type(_FLOAT8, _BFLOAT16, "Scalar", bias=True),
        _kernels("wvSplitKQ", "bf16"),
        "F8BS_BH_SAB",
    ),
    (
        "F8HS_BH_Bias_H_SAB",
        _tn_problem_type(_FLOAT8, _HALF, "Scalar", bias=True),
        _kernels("wvSplitKQ", "f16"),
        "F8HS_BH_SAB",
    ),
]


@needs_logic_corpus
@pytest.mark.parametrize("stem,problemType,kernels,plainStem", _RANGE_FILES)
def test_gfx1201_range_logic(stem, problemType, kernels, plainStem):
    """hipBLASLt reaches the gfx1201 skinny TN kernels through these Range files,
    on devices that report 32 CUs. A custom kernel takes the logic file's
    problem type, and these kernels support no activation or scale vector, so
    the type must be plain TN GEMM in the kernels' data types, with scalar A
    and B scales for FP8; the bias files add a bias of D's type, and the same
    kernels serve both, since a null bias address adds nothing. Types are
    compared after TensileLite fills in defaults. Keys are [m_min, m_max, n_min,
    n_max, batch_min, batch_max, K_min, K_max], -1 unbounded. hipBLASLt takes
    the first row containing the problem, so the rows must not overlap, and
    each must sit inside its kernel's predicates. Each solution lists its
    kernel's arguments, and refuses a user split-K or WGM override: the kernels
    take neither, and hipBLASLt would run a split-K by passing the workspace as
    C and D."""
    with open(require_logic_file(_RANGE.format(stem))) as f:
        doc = yaml.safe_load(f)
    assert {k: doc[k] for k in _HEADER} == _HEADER
    actualType = ProblemType(deepcopy(doc["ProblemType"]), False).state
    expectedType = ProblemType(deepcopy(problemType), False).state
    assert actualType == expectedType

    names = {s["SolutionIndex"]: s["CustomKernel"]["name"] for s in doc["Solutions"]}
    rows = [(key, names[index]) for key, (index, _) in doc["ExactLogic"]]
    assert {name for _, name in rows} == set(names.values())
    assert set(names.values()) <= kernels
    for i, (a, nameA) in enumerate(rows):
        for b, nameB in rows[i + 1 :]:
            assert not _boxes_overlap(a, b), (a, nameA, b, nameB)

    if plainStem:
        # A problem without a bias also matches the bias type. hipBLASLt searches
        # the plain-GEMM file first, so rows inside its rows keep those problems
        # on the plain-GEMM rows.
        with open(require_logic_file(_RANGE.format(plainStem))) as f:
            plainRows = [key for key, _ in yaml.safe_load(f)["ExactLogic"]]
        for key, name in rows:
            assert _box_covered(key, plainRows), (key, name)

    semantics = lambda ck: [a["semantic"] for a in ck["args"]]
    for s in doc["Solutions"]:
        name = s["CustomKernel"]["name"]
        config = getCustomKernelConfig(name, {}, _CK_ROOT)
        assert semantics(s["CustomKernel"]) == semantics(config["CustomKernel"]), name
        for key in ("grid", "threads", "macrotile"):
            assert s["CustomKernel"][key] == config["CustomKernel"][key], (name, key)
        assert (s["ISA"], s["WavefrontSize"]) == ([12, 0, 1], 32), name
        params = s["InternalSupportParams"]
        assert (params["SupportUserGSU"], params["SupportCustomWGM"]) == (False, False), name
    for key, name in rows:
        config = getCustomKernelConfig(name, {}, _CK_ROOT)
        for dim in range(4):
            lo, hi = key[2 * dim], key[2 * dim + 1]
            if dim in config.get("AssertSizeEqual", {}):
                assert lo == hi == config["AssertSizeEqual"][dim], (name, dim)
        assert key[6] >= config["AssertSummationElementMultiple"], name


@needs_logic_corpus
def test_gfx1201_range_logic_uses_every_kernel():
    """Every gfx1201 kernel that ships is in a Range row."""
    used = set()
    for stem, _, _, _ in _RANGE_FILES:
        with open(require_logic_file(_RANGE.format(stem))) as f:
            used |= {s["CustomKernel"]["name"] for s in yaml.safe_load(f)["Solutions"]}
    assert used == set(_KERNELS)
