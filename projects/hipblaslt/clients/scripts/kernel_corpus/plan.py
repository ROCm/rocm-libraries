# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Turn corpus records into hipblaslt-test cases, one small set of problems per kernel.

Each case pins one solution with algo_method 2 and an explicit solution_index, so the library's
own predicate check decides whether the kernel accepts the problem, and the public C API packs
the arguments the way a user's call would. fast_check with integer_exact data is the oracle.

A profile names the problems built for each kernel. Phase 1 has one, `smoke`: shapes derived
from the kernel's own macro tile, DepthU and split-K, honoring its size predicates.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Optional

# TensileLite data type names, as the library files spell them, to hipblaslt-test type names.
TYPE_NAMES = {
    "Float": "f32_r",
    "Half": "f16_r",
    "BFloat16": "bf16_r",
    "Double": "f64_r",
    "ComplexFloat": "f32_c",
    "ComplexDouble": "f64_c",
    "Int8": "i8_r",
    "Int32": "i32_r",
    "Float8": "f8_r",
    "BFloat8": "bf8_r",
    "Float8_fnuz": "f8_fnuz_r",
    "BFloat8_fnuz": "bf8_fnuz_r",
}

# Problem-type features the smoke profile does not exercise yet. A kernel whose problem type
# requires one is planned as deferred, with the reason, rather than silently dropped.
_DEFERRED_FEATURES = (
    ("groupedGemm", "grouped GEMM"),
    ("useGradient", "gradient epilogue"),
    ("useE", "E output"),
    ("useGateResidual", "gate residual"),
    ("outputAmaxD", "amaxD output"),
    ("sparse", "sparse A"),
    ("swizzleTensorA", "swizzled A"),
    ("swizzleTensorB", "swizzled B"),
    ("mxBlockA", "MX block scales"),
    ("mxBlockB", "MX block scales"),
    ("fusedGemmA2A", "fused all-to-all"),
)

# A problem type's useScaleAB to hipblaslt-test's scaleA and scaleB: one value, or one per row
# of A and column of B.
_SCALE_AB = {"Scalar": 1, "Vector": 2}

SHAPE_TAGS = ("exact", "ragged", "ktail", "batch")
PROFILES = ("smoke",)
CASE_PREFIX = "kc"
ALPHA = 2.0
BETA = -2.0


@dataclasses.dataclass(frozen=True)
class Shape:
    tag: str
    m: int
    n: int
    k: int
    batch: int = 1


@dataclasses.dataclass
class Case:
    """One problem for one solution. case_id is stable for a given index and shape tag."""

    index: int
    name: str
    shape: Shape
    types: dict
    trans_a: str
    trans_b: str

    @property
    def case_id(self) -> str:
        return f"{self.index}:{self.shape.tag}"

    @property
    def yaml_name(self) -> str:
        return f"{CASE_PREFIX}_{self.index}_{self.shape.tag}"


def problem_types(record: dict) -> tuple[Optional[dict], Optional[str]]:
    """hipblaslt-test type arguments for a record's problem type, or the reason there are none."""
    p = record["problem"]
    for field, what in _DEFERRED_FEATURES:
        if p.get(field):
            return None, f"deferred: {what}"
    scale_ab = p.get("useScaleAB") or ""
    if scale_ab and scale_ab not in _SCALE_AB:
        return None, f"deferred: scaleA and scaleB ({scale_ab})"
    names = {}
    for key, arg in (
        ("aType", "a_type"),
        ("bType", "b_type"),
        ("cType", "c_type"),
        ("dType", "d_type"),
    ):
        name = TYPE_NAMES.get(p.get(key, ""))
        if name is None:
            return None, f"unmapped data type {p.get(key)!r}"
        names[arg] = name

    compute = p.get("computeType")
    a_type = p.get("aType")
    compute_input = p.get("computeInputTypeA", a_type)
    if compute == "Double":
        names["compute_type"] = "c_f64_r"
        names["scale_type"] = "f64_r"
    elif compute == "Int32":
        names["compute_type"] = "c_i32_r"
        names["scale_type"] = "i32_r"
    elif compute in ("ComplexFloat", "ComplexDouble"):
        return None, "unsupported: fast_check does not check complex data"
    elif compute == "Float":
        names["scale_type"] = "f32_r"
        compute_input_b = p.get("computeInputTypeB", compute_input)
        if p.get("f32XdlMathOp") == "XFloat32" and a_type == "Float":
            names["compute_type"] = "c_xf32_r"
        elif a_type != p.get("bType"):
            # Mixed inputs, such as fp8 A with f16 B: the kernel converts one operand to the
            # compute input type, which hipblaslt-test would otherwise take as f32.
            if compute_input not in TYPE_NAMES or compute_input_b not in TYPE_NAMES:
                return None, f"unmapped compute input type {compute_input!r}"
            names["compute_type"] = "c_f32_r"
            names["compute_input_typeA"] = TYPE_NAMES[compute_input]
            names["compute_input_typeB"] = TYPE_NAMES[compute_input_b]
        elif compute_input == a_type or "8" in str(a_type):
            names["compute_type"] = "c_f32_r"
        elif compute_input == "Half":
            names["compute_type"] = "c_f32_fast_f16_r"
        elif compute_input == "BFloat16":
            names["compute_type"] = "c_f32_fast_bf16_r"
        elif compute_input in TYPE_NAMES and "8" in compute_input:
            # 16-bit inputs converted to fp8 inside the kernel; no compute type names this.
            names["compute_type"] = "c_f32_r"
            names["compute_input_typeA"] = TYPE_NAMES[compute_input]
            names["compute_input_typeB"] = TYPE_NAMES.get(
                p.get("computeInputTypeB", compute_input), TYPE_NAMES[compute_input]
            )
        else:
            return None, f"unmapped compute input type {compute_input!r}"
    else:
        return None, f"unmapped compute type {compute!r}"
    if scale_ab:
        names["scaleA"] = names["scaleB"] = _SCALE_AB[scale_ab]
    if p.get("useScaleCD"):
        names["scaleC"] = names["scaleD"] = 1
    return names, None


def _round_up(x: int, multiple: int) -> int:
    multiple = max(int(multiple or 1), 1)
    return -(-x // multiple) * multiple


def _predicate(record: dict, prefix: str, default: int = 1) -> int:
    """The value of the first size predicate whose key starts with prefix."""
    for key, value in record["predicates"].items():
        if key == prefix or key.startswith(prefix + "["):
            if isinstance(value, (int, float)):
                return int(value)
    return default


def smoke_shapes(record: dict) -> list[Shape]:
    """Exact tile multiple, ragged M and N, a K tail, and a strided batch, for this kernel."""
    sizes = record["sizes"]
    mt0, mt1 = (int(x) for x in sizes.get("macroTile", [16, 16])[:2])
    du = int(sizes.get("depthU", 16))
    gsu = int(sizes.get("globalSplitU", 1))
    gsu = gsu if gsu > 1 else 1

    m_mult = _predicate(record, "Free0SizeMultiple")
    n_mult = _predicate(record, "Free1SizeMultiple")
    k_mult = _predicate(record, "BoundSizeMultiple")
    min_m = _predicate(record, "LeadingFree0SizesGreaterOrEqual")
    min_n = _predicate(record, "LeadingFree1SizesGreaterOrEqual")
    min_k = record["predicates"].get("GlobalSplitUCheckMinK")
    k_base = du * gsu
    if isinstance(min_k, list) and len(min_k) == 2:
        k_base = max(k_base, int(min_k[0]) * max(int(min_k[1]), 1))

    def m_of(x: int) -> int:
        return max(_round_up(x, m_mult), min_m)

    def n_of(x: int) -> int:
        return max(_round_up(x, n_mult), min_n)

    def k_of(x: int) -> int:
        return _round_up(x, k_mult)

    ktail = k_of(2 * k_base + 1) if k_mult <= 1 else 2 * k_base + k_mult
    shapes = [
        Shape("exact", m_of(2 * mt0), n_of(2 * mt1), k_of(2 * k_base)),
        Shape("ragged", m_of(2 * mt0 + 1), n_of(mt1 + 1), k_of(2 * k_base)),
        Shape("ktail", m_of(mt0), n_of(mt1), ktail),
    ]
    if record["problem"].get("stridedBatched", True):
        shapes.append(Shape("batch", m_of(mt0 + 1), n_of(mt1), k_of(k_base), 3))
    return shapes


def plan_kernel(record: dict, profile: str = "smoke") -> tuple[list[Case], Optional[str]]:
    """The cases for one record, or no cases and the reason it is not planned."""
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}")
    types, reason = problem_types(record)
    if types is None:
        return [], reason
    p = record["problem"]
    trans_a = "T" if p.get("transA") else "N"
    trans_b = "T" if p.get("transB") else "N"
    cases = [
        Case(record["index"], record["name"], shape, types, trans_a, trans_b)
        for shape in smoke_shapes(record)
    ]
    return cases, None


def case_yaml_entry(case: Case) -> dict:
    s = case.shape
    a_rows = s.m if case.trans_a == "N" else s.k
    b_rows = s.k if case.trans_b == "N" else s.n
    entry = {
        "name": case.yaml_name,
        "category": "pre_checkin",
        "function": {"matmul": [case.types]},
        "matrix_size": [
            {"M": s.m, "N": s.n, "K": s.k, "lda": a_rows, "ldb": b_rows, "ldc": s.m, "ldd": s.m}
        ],
        "transA": case.trans_a,
        "transB": case.trans_b,
        "alpha": ALPHA,
        "beta": BETA,
        "batch_count": s.batch,
        "initialization": "integer_exact",
        "unit_check": 0,
        "fast_check": 1,
        "algo_method": 2,
        "solution_index": case.index,
    }
    if s.batch > 1:
        entry["stride_a"] = s.m * s.k
        entry["stride_b"] = s.k * s.n
        entry["stride_c"] = s.m * s.n
        entry["stride_d"] = s.m * s.n
    return entry


def write_cases_yaml(cases: list[Case], path: Path, common_yaml: Path) -> None:
    """A hipblaslt-test --yaml file. hipblaslt_gentest.py resolves includes relative to the file,
    so the include is written as an absolute path to the installed hipblaslt_common.yaml."""
    lines = ["---", f"include: {common_yaml.resolve()}", "", "Tests:"]
    for case in cases:
        entry = case_yaml_entry(case)
        lines.append(f"- name: {entry.pop('name')}")
        for key, value in entry.items():
            lines.append(f"  {key}: {json.dumps(value)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


_GTEST_NAME = re.compile(rf"_{CASE_PREFIX}_(\d+)_({'|'.join(SHAPE_TAGS)})_")


def case_id_from_test_name(test_name: str) -> Optional[str]:
    """Maps a gtest name such as matmul/pre_checkin_kc_812_ragged_f32_r..._APIAlgoIndex back to
    its case_id, 812:ragged."""
    match = _GTEST_NAME.search(test_name)
    return f"{match.group(1)}:{match.group(2)}" if match else None


def gtest_filter() -> str:
    return f"*pre_checkin_{CASE_PREFIX}_*"
