# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Turn corpus records into hipblaslt-test cases, one small set of problems per kernel.

Each case pins one solution with algo_method 2 and an explicit solution_index, so the library's
own predicate check decides whether the kernel accepts the problem, and the public C API packs
the arguments the way a user's call would. fast_check with integer_exact data is the oracle.

A profile names the problems built for each kernel. `smoke` builds shapes from the kernel's own
macro tile, DepthU and split-K, honoring its size predicates; a kernel registered for exact sizes
(SizeEqual predicates, from Equality logic) gets its registered problem instead. `edges` builds,
for an exact-size kernel, problems one step outside each of its predicates, which the pinned
solution must refuse.

MX FP6 kernels with host-shuffled 128x4 scales are checked against the CPU reference by default,
or with mx_oracle "fast_check" by fast_check with sparse_k integer_exact data, which needs a
hipblaslt-test whose MX generator supports sparse_k. Other MX kernels are deferred.
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
    ("fusedGemmA2A", "fused all-to-all"),
)

# MX problem types the planner builds: FP6 A and B in TN with E8M0 scales in blocks of 32, A
# pre-tiled (HIPBLASLT_ORDER_COL256_R128_EXT), and scales the host shuffles into 128x4 tiles
# (MXScaleFormat HostPreSwizzle128x4 in TensileLite, scale mode 1018 in hipblaslt-test).
_MX_TYPE_NAMES = {"Float6": "f6_r", "BFloat6": "bf6_r"}
_MX_BLOCK = 32
_MX_SCALE_FORMAT_128X4 = 3
_MX_SCALE_MODE_128X4 = 1018
# Above this K, rand_int E3M2 sums pass 2^24, where fp32 sums depend on their order, so the
# exact unit_check against the CPU reference no longer applies (matmul_gtest.yaml does the same).
_RAND_INT_MAX_K = {"BFloat6": 65536}

# A problem type's useScaleAB to hipblaslt-test's scaleA and scaleB: one value, or one per row
# of A and column of B.
_SCALE_AB = {"Scalar": 1, "Vector": 2}

SMOKE_TAGS = ("exact", "ragged", "ktail", "batch")
REGISTERED_TAG = "registered"
EDGE_TAGS = (
    "kup",
    "kdown",
    "khalf",
    "mup",
    "nup",
    "batch2",
    "alpha2",
    "beta1",
    "ldapad",
    "ldbpad",
    "lddpad",
    "border",
)
SHAPE_TAGS = SMOKE_TAGS + (REGISTERED_TAG,) + EDGE_TAGS
PROFILES = ("smoke", "edges")
# How MX FP6 kernels are checked: the CPU reference, or fast_check with sparse_k data.
MX_ORACLES = ("reference", "fast_check")
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
    lda: Optional[int] = None
    ldb: Optional[int] = None
    ldd: Optional[int] = None


@dataclasses.dataclass
class Case:
    """One problem for one solution. case_id is stable for a given index and shape tag.

    args overrides hipblaslt-test arguments of the default case (alpha, beta, data and checks).
    expect is "refused" for a problem the pinned solution must decline."""

    index: int
    name: str
    shape: Shape
    types: dict
    trans_a: str
    trans_b: str
    args: dict = dataclasses.field(default_factory=dict)
    expect: str = "pass"

    @property
    def case_id(self) -> str:
        return f"{self.index}:{self.shape.tag}"

    @property
    def yaml_name(self) -> str:
        return f"{CASE_PREFIX}_{self.index}_{self.shape.tag}"

    @property
    def cpu_reference(self) -> bool:
        """The case checks D against hipblaslt-test's CPU reference GEMM instead of fast_check."""
        return any(self.args.get(k) for k in ("unit_check", "norm_check", "allclose_check"))


def is_mx(record: dict) -> bool:
    p = record["problem"]
    return bool(p.get("mxBlockA") or p.get("mxBlockB"))


def mx_deferral(record: dict) -> Optional[str]:
    """Why the planner cannot build this MX problem type, or None when it can."""
    p = record["problem"]
    a, b = p.get("aType"), p.get("bType")
    fmt = record["predicates"].get("MXScaleFormatEquals")
    supported = (
        a == b
        and a in _MX_TYPE_NAMES
        and p.get("mxBlockA") == _MX_BLOCK
        and p.get("mxBlockB") == _MX_BLOCK
        and fmt == _MX_SCALE_FORMAT_128X4
        and p.get("swizzleTensorA")
        and p.get("transA")
        and not p.get("transB")
    )
    if supported:
        return None
    return (
        f"deferred: MX block scales ({a} A, {b} B, blocks {p.get('mxBlockA')}x{p.get('mxBlockB')}, "
        f"scale format {fmt}); only FP6 TN with 128x4 scales is planned"
    )


def problem_types(record: dict) -> tuple[Optional[dict], Optional[str]]:
    """hipblaslt-test type arguments for a record's problem type, or the reason there are none."""
    p = record["problem"]
    mx = is_mx(record)
    if mx:
        reason = mx_deferral(record)
        if reason:
            return None, reason
    for field, what in _DEFERRED_FEATURES:
        if p.get(field) and not (mx and field in ("swizzleTensorA", "swizzleTensorB")):
            return None, f"deferred: {what}"
    scale_ab = p.get("useScaleAB") or ""
    if scale_ab and scale_ab not in _SCALE_AB:
        return None, f"deferred: scaleA and scaleB ({scale_ab})"
    type_names = {**TYPE_NAMES, **_MX_TYPE_NAMES} if mx else TYPE_NAMES
    names = {}
    for key, arg in (
        ("aType", "a_type"),
        ("bType", "b_type"),
        ("cType", "c_type"),
        ("dType", "d_type"),
    ):
        name = type_names.get(p.get(key, ""))
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
    if mx:
        names["scaleA"] = names["scaleB"] = _MX_SCALE_MODE_128X4
        names["swizzle_a"] = True
        names["swizzle_b"] = bool(p.get("swizzleTensorB"))
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


def registered_shape(record: dict) -> Optional[Shape]:
    """The one problem an exact-size solution accepts: its SizeEqual predicates (M, N, batch and
    K, by tensor index) and the leading dimensions its StrideEqual predicates pin. None for a
    solution without SizeEqual predicates."""
    pred = record["predicates"]
    sizes = [pred.get(f"SizeEqual[{i}]") for i in range(4)]
    if not all(isinstance(s, int) for s in sizes):
        return None
    m, n, batch, k = sizes
    return Shape(
        REGISTERED_TAG,
        m,
        n,
        k,
        batch,
        lda=pred.get("StrideAEqual[1]"),
        ldb=pred.get("StrideBEqual[1]"),
        ldd=pred.get("StrideDEqual[1]"),
    )


def predicate_args(record: dict) -> dict:
    """Arguments a solution's own predicates demand: the alpha it asserts, beta = 0 when it
    requires it, and the ext API when its problem type has no beta, which the C API never
    sets up."""
    pred = record["predicates"]
    args = {}
    if pred.get("AlphaValue") is not None:
        args["alpha"] = float(pred["AlphaValue"])
    if "BetaZero" in pred:
        args["beta"] = 0.0
    if record["problem"].get("useBeta") is False:
        args["use_ext"] = 1
        args["use_ext_setproblem"] = 1
    return args


def oracle_args(record: dict, shape: Shape, mx_oracle: str = "reference") -> dict:
    """How the case checks D. fast_check with integer_exact data, except for MX kernels. With
    mx_oracle "reference" those use the CPU reference, exactly while fp32 sums stay exact and
    within a norm and elementwise tolerance beyond that. With "fast_check" they use fast_check
    with sparse_k integer_exact data, which keeps a bf16 D exact at any K; that needs a
    hipblaslt-test whose MX generator supports sparse_k (rocm-libraries #16345)."""
    if mx_oracle not in MX_ORACLES:
        raise ValueError(f"unknown MX oracle {mx_oracle!r}")
    if not is_mx(record):
        return {}
    if mx_oracle == "fast_check":
        return {"integer_exact_pattern": "sparse_k"}
    args = {"fast_check": 0, "unit_check": 1, "norm_check": 1, "initialization": "rand_int"}
    if shape.k > _RAND_INT_MAX_K.get(record["problem"].get("aType"), shape.k):
        args.update(initialization="trig_float", unit_check=0, allclose_check=1)
    return args


def edge_cases(record: dict, base: Case) -> list[Case]:
    """Problems one step outside each predicate of an exact-size solution. Each breaks one
    predicate (a larger M also moves the leading dimension of D) and must be refused."""
    pred = record["predicates"]
    s = base.shape
    step_k = int(record["sizes"].get("depthU", 128))
    mt0, mt1 = (int(x) for x in record["sizes"].get("macroTile", [256, 256])[:2])

    def k_shape(tag: str, k: int) -> Shape:
        lda = k if s.lda == s.k else s.lda
        ldb = k if s.ldb == s.k else s.ldb
        return dataclasses.replace(s, tag=tag, k=k, lda=lda, ldb=ldb)

    shapes = [
        k_shape("kup", s.k + step_k),
        k_shape("kdown", s.k - step_k),
        k_shape("khalf", _round_up(s.k // 2, step_k)),
        dataclasses.replace(s, tag="mup", m=s.m + mt0, ldd=(s.ldd or s.m) + mt0),
        dataclasses.replace(s, tag="nup", n=s.n + mt1),
        dataclasses.replace(s, tag="batch2", batch=s.batch + 1),
    ]
    if s.lda:
        shapes.append(dataclasses.replace(s, tag="ldapad", lda=s.lda + step_k))
    if s.ldb:
        shapes.append(dataclasses.replace(s, tag="ldbpad", ldb=s.ldb + step_k))
    if s.ldd:
        shapes.append(dataclasses.replace(s, tag="lddpad", ldd=s.ldd + mt0))
    args = {**base.args, "fast_check": 0, "unit_check": 0, "norm_check": 0, "allclose_check": 0}
    cases = [
        dataclasses.replace(base, shape=shape, args=args, expect="refused") for shape in shapes
    ]
    if pred.get("AlphaValue") is not None:
        cases.append(
            dataclasses.replace(
                cases[0],
                shape=dataclasses.replace(s, tag="alpha2"),
                args={**args, "alpha": float(pred["AlphaValue"]) + 1},
            )
        )
    if "BetaZero" in pred:
        cases.append(
            dataclasses.replace(
                cases[0], shape=dataclasses.replace(s, tag="beta1"), args={**args, "beta": 1.0}
            )
        )
    if "swizzle_b" in base.types:
        types = {**base.types, "swizzle_b": not base.types["swizzle_b"]}
        cases.append(
            dataclasses.replace(
                cases[0], shape=dataclasses.replace(s, tag="border"), types=types, args=args
            )
        )
    return cases


def plan_kernel(
    record: dict, profile: str = "smoke", mx_oracle: str = "reference"
) -> tuple[list[Case], Optional[str]]:
    """The cases for one record, or no cases and the reason it is not planned. mx_oracle picks
    how MX kernels are checked (oracle_args)."""
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}")
    types, reason = problem_types(record)
    if types is None:
        return [], reason
    p = record["problem"]
    trans_a = "T" if p.get("transA") else "N"
    trans_b = "T" if p.get("transB") else "N"
    registered = registered_shape(record)
    if registered is None and (profile == "edges" or is_mx(record)):
        # MX kernels so far come only as exact-size custom kernels; tile-derived shapes and
        # predicate edges for the rest come later.
        return [], f"deferred: the {profile} profile needs an exact-size (SizeEqual) solution"
    shapes = [registered] if registered else smoke_shapes(record)
    base_args = predicate_args(record)
    cases = [
        Case(
            record["index"],
            record["name"],
            shape,
            types,
            trans_a,
            trans_b,
            {**base_args, **oracle_args(record, shape, mx_oracle)},
        )
        for shape in shapes
    ]
    if profile == "edges":
        return edge_cases(record, cases[0]), None
    return cases, None


def case_yaml_entry(case: Case) -> dict:
    s = case.shape
    a_rows = s.m if case.trans_a == "N" else s.k
    b_rows = s.k if case.trans_b == "N" else s.n
    lda, ldb, ldd = s.lda or a_rows, s.ldb or b_rows, s.ldd or s.m
    a_cols = s.k if case.trans_a == "N" else s.m
    b_cols = s.n if case.trans_b == "N" else s.k
    entry = {
        "name": case.yaml_name,
        "category": "pre_checkin",
        "function": {"matmul": [case.types]},
        "matrix_size": [
            {"M": s.m, "N": s.n, "K": s.k, "lda": lda, "ldb": ldb, "ldc": ldd, "ldd": ldd}
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
        entry["stride_a"] = lda * a_cols
        entry["stride_b"] = ldb * b_cols
        entry["stride_c"] = ldd * s.n
        entry["stride_d"] = ldd * s.n
    entry.update(case.args)
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
