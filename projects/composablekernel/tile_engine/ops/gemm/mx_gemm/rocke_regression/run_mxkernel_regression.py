#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Regression benchmark of the workspace rocKE MX GEMM
(dnn-providers/hip-kernel-provider/rocke) the way the TensileLite client
benchmarks hipBLASLt: every kernel of a regression config (default
mxfp4_regression_configs.json; --config mxfp8_regression_configs.json for
MXFP8) on its problem, with the client settings of its hipblaslt_yaml.

From the YAML's GlobalParameters (TensileLite defaults for absent keys, read
from Common/GlobalParameters.py) it takes:

- timing: NumWarmups launches, then SyncsPerBenchmark times: device sync,
  host clock, EnqueuesPerSync launches (raised to MinFlopsPerSync, capped by
  MaxEnqueuesPerSync), device sync, host clock, sleep SleepPercent of that
  time. The result is the total time over all timed launches. KernelTime
  True uses stream events around the launches instead, and NumBenchmarks
  repeats it all (the median is reported).
- rotating buffers: RotatingBufferSize (MB) gives 1 + min(max(warmups,
  syncs * enqueues), ceil(size / (A + B + D [+ C] bytes))) - 1 A/B/D copies,
  cycled by one launch counter across warmups and timed launches. MX scales
  are shared, as in the client.
- inputs: MXDataGenerator False makes A and B the client's initArray trig
  (DataInitTypeA/B, sin / cos of the linear element index, K fastest) with
  DataInitTypeMXSA/B scales (Random: E8M0 124..130). Unset or True (the
  client default) makes them mxDataGenerator's TrigonometricFromFloat: cos
  of a uniform random angle, per-32 block E8M0 scale the rounded mean of the
  element exponents, truncated mantissas, one subnormal forced into every 15
  elements. The client's generator draws from per-thread mt19937 streams, so
  those bytes are equal in distribution, not identical.
  Tensile's A is rocKE's B (the problem is TN with Tensile's free index I as
  rocKE's N), so rocKE A takes DataInitTypeB / MXSB and rocKE B takes
  DataInitTypeA / MXSA. Alpha and beta come from DataInitTypeAlpha / Beta.

The problem sizes, data types and layout of the YAML must match the config.
Before timing, each kernel must be bit-exact on mx_gemm_bench's check inputs
(exact and sparse, beta 0 and 0.5, the config's --verify). Timing inputs are
cached in $ROCKE_MX_TIMING_CACHE (default ~/.cache/rocke_mx_timing); the first
run of a shape builds them (--prepare-inputs does that on the CPU only).

Per kernel it prints the check, VGPRs, LDS, the time per launch, TFLOPS and
TB/s (the client's GFlops and GbpsBW: 2*M*N*K, and A, B, their MX scales and
D, plus C when beta is not 0). Run results and baselines hold absolute
timings, so they stay outside the repo, in --runs-dir
($ROCKE_MX_REGRESSION_DIR, default ~/.cache/rocke_mx_regression). With a
baseline (--save-baseline writes <name>_baseline_gpu<dev>.json there for
--config <name>_configs.json) it compares: a failed check, a resource change
(VGPR, SGPR, spills, scratch, LDS, WMMAs, loop scratch; more spills or
scratch FAIL) or a slowdown beyond --tolerance percent is reported, and the
exit code is 1.

    python3 run_mxkernel_regression.py                    # GPU 3, all kernels
    python3 run_mxkernel_regression.py --only maf,medium
    python3 run_mxkernel_regression.py --set gl2-prefetch-distance=3   # try a parameter on all
    python3 run_mxkernel_regression.py --save-baseline    # record this run
    python3 run_mxkernel_regression.py --host-only        # compile only (CPU): resources
    python3 run_mxkernel_regression.py --prepare-inputs   # build the timing inputs (CPU)
    python3 run_mxkernel_regression.py --dry-run          # print the client settings
    python3 run_mxkernel_regression.py --config mxfp8_regression_configs.json

Runs on the GPU unless --host-only, --prepare-inputs or --dry-run is set. It
re-executes itself with the TensileLite venv python when PyYAML, NumPy or
ml_dtypes is missing.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import math
import os
import statistics
import struct
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "dnn-providers").is_dir())
PLATFORM = REPO / "dnn-providers/hip-kernel-provider/rocke/platform"
TENSILE = REPO / "projects/hipblaslt/tensilelite/tensilelite"
YAML_ROOT = TENSILE / "Utilities/tensile_generator"
GLOBAL_PARAMETERS = TENSILE / "Common/GlobalParameters.py"
PYTHON = REPO / "projects/hipblaslt/tensilelite/.venv/bin/python3"
LLVM_BIN = "/opt/rocm/llvm/bin"
CONFIG = HERE / "mxfp4_regression_configs.json"
RUNS = Path(
    os.environ.get(
        "ROCKE_MX_REGRESSION_DIR", Path.home() / ".cache" / "rocke_mx_regression"
    )
)
CACHE = Path(
    os.environ.get("ROCKE_MX_TIMING_CACHE", Path.home() / ".cache" / "rocke_mx_timing")
)
# Stored with every result; a baseline timed another way is not compared.
METHOD = "tensilelite-client-1"
EXACT = ("vgpr", "sgpr", "spills", "scratch", "lds", "wmma", "loop_scratch")
NO_GROWTH = ("spills", "scratch", "loop_scratch")
KERNEL_OPTIONS = ("m", "n", "k", "verify", "samples")
IN_BYTES = {"fp4": 0.5, "fp8": 1.0, "bf8": 1.0}
OUT_BYTES = {"fp8e4m3": 1, "fp16": 2, "bf16": 2}
TENSILE_IN = {"f4": "fp4", "f8": "fp8", "b8": "bf8"}
TENSILE_OUT = {"f8": "fp8e4m3", "h": "fp16", "b": "bf16"}
SCALE_BLOCK = 32
E8M0_BIAS = 127
MX_SEED = 1713573849  # mxDataGenerator's seed in hipBLASLt's generateMXInput

# TensileLite InitMode values (client/include/DataInitialization.hpp).
ZERO, ONE, TWO, RANDOM, TRIG_SIN, TRIG_COS, NEG_ONE = 0, 1, 2, 3, 12, 13, 17
TRIG_MODES = (12, 13, 14, 15, 23, 24, 25, 26)
SCALAR_INIT = {ZERO: 0.0, ONE: 1.0, TWO: 2.0, NEG_ONE: -1.0}
# Element formats: exponent bits, mantissa bits, largest finite code.
FORMATS = {"fp8": (4, 3, 0x7E), "bf8": (5, 2, 0x7B), "fp4": (2, 1, 0x7)}
SPRINKLE = 15  # mxDataGenerator SPRINKLE_BLOCK_MAX: one subnormal per 15 elements


def ensure_python() -> None:
    try:
        import ml_dtypes  # noqa: F401
        import numpy  # noqa: F401
        import yaml  # noqa: F401
    except ImportError:
        if (
            PYTHON.exists()
            and Path(sys.prefix).resolve() != PYTHON.parents[1].resolve()
        ):
            os.execv(
                str(PYTHON), [str(PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]]
            )
        raise


def setup_rocke(device: str) -> None:
    """Make ``rocke`` importable and pick the device before HIP loads."""
    os.environ["HIP_VISIBLE_DEVICES"] = device
    os.environ["PATH"] = f"{LLVM_BIN}:{os.environ.get('PATH', '')}"
    sys.path.insert(0, str(PLATFORM / "python"))
    os.chdir(PLATFORM)


def bench():
    from rocke.examples.gfx1250.gemm import mx_gemm_bench

    return mx_gemm_bench


# --- TensileLite client settings ------------------------------------------


def tensile_defaults() -> dict:
    """Literal ``globalParameters[...]`` defaults of TensileLite."""
    defaults = {}
    for node in ast.parse(GLOBAL_PARAMETERS.read_text()).body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == "globalParameters"
            and isinstance(target.slice, ast.Constant)
        ):
            try:
                defaults[target.slice.value] = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError):
                pass
    return defaults


@dataclass
class Plan:
    """What the TensileLite client does for one problem."""

    yaml: str
    m: int
    n: int
    k: int
    warmups: int
    enqueues: int
    syncs: int
    benchmarks: int
    kernel_time: bool
    sleep_percent: float
    slots: int
    inputs: str  # "init-array" or "mx-data-generator"
    init_a: int  # InitMode of rocKE A (Tensile B) and B (Tensile A)
    init_b: int
    init_scale_a: int
    init_scale_b: int
    alpha: float
    beta: float

    def describe(self) -> str:
        timer = "events" if self.kernel_time else "host clock"
        return (
            f"{self.yaml}: {self.warmups} warmups, {self.syncs} x {self.enqueues} "
            f"launches ({timer}, sleep {self.sleep_percent:g}%), "
            f"{self.benchmarks} benchmark(s), {self.slots} rotating slot(s), "
            f"{self.inputs} inputs, alpha {self.alpha:g}, beta {self.beta:g}"
        )


def _problem(document: dict) -> tuple[dict, list[int]]:
    problems = document["BenchmarkProblems"]
    if len(problems) != 1:
        raise ValueError(f"expected one benchmark problem, got {len(problems)}")
    problem_type, steps = problems[0][0], problems[0][1:]
    sizes = [
        size["Exact"]
        for step in steps
        for group in (step.get("BenchmarkFinalParameters") or [])
        for size in (group.get("ProblemSizes") or [])
        if isinstance(size, dict) and "Exact" in size
    ]
    if len(sizes) != 1:
        raise ValueError(f"expected one Exact problem size, got {sizes}")
    return problem_type, sizes[0]


def plan_for(name: str, entry: dict, options: dict, defaults: dict) -> Plan:
    import numpy as np
    import yaml

    rel = entry.get("hipblaslt_yaml")
    if not rel:
        raise ValueError(f"{name}: no hipblaslt_yaml")
    document = yaml.safe_load((YAML_ROOT / rel).read_text())
    gp = {**defaults, **(document.get("GlobalParameters") or {})}
    problem_type, exact = _problem(document)

    m, n, k = (int(options[x]) for x in "mnk")
    # Exact is [I, J, batch, L]; Tensile's A (I x L) is rocKE's B.
    tensile_i, tensile_j, batch, tensile_l = (int(v) for v in exact[:4])
    if (tensile_j, tensile_i, batch, tensile_l) != (m, n, 1, k):
        raise ValueError(
            f"{name}: config is M={m} N={n} K={k}, {rel} is Exact {exact} "
            f"(rocKE M=J={tensile_j} N=I={tensile_i} K=L={tensile_l})"
        )
    if not (problem_type.get("TransposeA") and not problem_type.get("TransposeB")):
        raise ValueError(f"{name}: {rel} is not TN")
    a_type = TENSILE_IN.get(
        str(problem_type.get("DataTypeA", problem_type["DataType"])).lower()
    )
    b_type = TENSILE_IN.get(
        str(problem_type.get("DataTypeB", problem_type["DataType"])).lower()
    )
    out_type = TENSILE_OUT.get(str(problem_type["DestDataType"]).lower())
    if (b_type, a_type, out_type) != (
        options["a-dtype"],
        options["b-dtype"],
        options["out-dtype"],
    ):
        raise ValueError(
            f"{name}: config is {options['a-dtype']}/{options['b-dtype']}->{options['out-dtype']}, "
            f"{rel} is {problem_type['DataType']}->{problem_type['DestDataType']}"
        )
    for key in ("MXBlockA", "MXBlockB"):
        if int(problem_type.get(key, 0)) != SCALE_BLOCK:
            raise ValueError(f"{name}: {rel} {key} is not {SCALE_BLOCK}")
    for key in ("DataTypeMXSA", "DataTypeMXSB"):
        if str(problem_type.get(key, "")).lower() != "e8":
            raise ValueError(f"{name}: {rel} {key} is not e8")
    if int(gp["RotatingMode"]) != 0:
        raise ValueError(f"{name}: RotatingMode {gp['RotatingMode']} is not supported")
    for key in ("DataInitTypeAlpha", "DataInitTypeBeta"):
        if int(gp[key]) not in SCALAR_INIT:
            raise ValueError(f"{name}: {key} {gp[key]} is not supported")

    generator = gp.get("MXDataGenerator")
    inputs = (
        "init-array"
        if generator is not None and not int(generator)
        else "mx-data-generator"
    )
    init_a, init_b = int(gp["DataInitTypeB"]), int(gp["DataInitTypeA"])
    init_scale_a, init_scale_b = (
        int(gp["DataInitTypeMXSB"]),
        int(gp["DataInitTypeMXSA"]),
    )
    for side, mode in (("A", init_b), ("B", init_a)):
        allowed = (TRIG_SIN, TRIG_COS) if inputs == "init-array" else TRIG_MODES
        if mode not in allowed:
            raise ValueError(
                f"{name}: DataInitType{side} {mode} with {inputs} is not supported"
            )
    if inputs == "init-array":
        for side, mode in (("A", init_scale_b), ("B", init_scale_a)):
            if mode != RANDOM:
                raise ValueError(
                    f"{name}: DataInitTypeMXS{side} {mode} is not supported"
                )

    alpha = SCALAR_INIT[int(gp["DataInitTypeAlpha"])]
    beta = SCALAR_INIT[int(gp["DataInitTypeBeta"])]
    flops = 2 * m * n * k
    enqueues = int(gp["EnqueuesPerSync"])
    if int(gp["MinFlopsPerSync"]) > 0:
        enqueues = max(enqueues, -(-int(gp["MinFlopsPerSync"]) // flops))
    if int(gp["MaxEnqueuesPerSync"]) >= 0:
        enqueues = min(enqueues, int(gp["MaxEnqueuesPerSync"]))
    warmups, syncs = int(gp["NumWarmups"]), int(gp["SyncsPerBenchmark"])

    slots = 1
    rotating = int(gp["RotatingBufferSize"]) * 1024 * 1024
    if rotating:
        a_bytes, b_bytes, c_bytes, d_bytes = operand_bytes(options)
        per_slot = a_bytes + b_bytes + d_bytes + (c_bytes if beta else 0)
        fit = math.ceil(np.float32(rotating) / np.float32(per_slot))
        slots = 1 + max(0, min(max(warmups, syncs * enqueues), fit) - 1)

    return Plan(
        yaml=rel,
        m=m,
        n=n,
        k=k,
        warmups=warmups,
        enqueues=enqueues,
        syncs=syncs,
        benchmarks=int(gp["NumBenchmarks"]),
        kernel_time=bool(gp["KernelTime"]),
        sleep_percent=float(gp["SleepPercent"]),
        slots=slots,
        inputs=inputs,
        init_a=init_a,
        init_b=init_b,
        init_scale_a=init_scale_a,
        init_scale_b=init_scale_b,
        alpha=alpha,
        beta=beta,
    )


def operand_bytes(options: dict) -> tuple[int, int, int, int]:
    """Bytes of A, B, C and D."""
    m, n, k = (int(options[x]) for x in "mnk")
    out = OUT_BYTES[options["out-dtype"]]
    return (
        int(m * k * IN_BYTES[options["a-dtype"]]),
        int(n * k * IN_BYTES[options["b-dtype"]]),
        m * n * out,
        m * n * out,
    )


def work(options: dict, plan: Plan) -> tuple[float, float]:
    """``(flop, bytes)`` of one launch as the client counts them: A, B and
    their MX scales (K blocks padded to 8), D, and C when beta is not 0."""
    m, n, k = (int(options[x]) for x in "mnk")
    a_bytes, b_bytes, c_bytes, d_bytes = operand_bytes(options)
    k_blocks = math.ceil(math.ceil(k / SCALE_BLOCK) / 8) * 8
    traffic = a_bytes + b_bytes + d_bytes + (m + n) * k_blocks
    if plan.beta:
        traffic += c_bytes
    return 2.0 * m * n * k, float(traffic)


# --- Timing inputs ---------------------------------------------------------


def _format(dtype: str):
    ebits, mbits, max_code = FORMATS[dtype]
    bias = (1 << (ebits - 1)) - 1
    return ebits, mbits, max_code, bias, 1 - bias


def _decode_magnitude(code, mbits, emask, bias, emin):
    import numpy as np

    field = (code >> mbits) & emask
    mantissa = code & ((1 << mbits) - 1)
    return np.where(
        field == 0,
        np.ldexp(mantissa.astype(np.float64), emin - mbits),
        np.ldexp(1.0 + mantissa / float(1 << mbits), field - bias),
    )


def _row_chunks(rows: int, cols: int, multiple: int = 1):
    step = max(multiple, (1 << 23) // cols // multiple * multiple)
    for begin in range(0, rows, step):
        yield begin, min(rows, begin + step)


def _pack(codes, dtype: str):
    import numpy as np

    if dtype == "fp4":
        return codes[:, 0::2] | (codes[:, 1::2] << 4)
    return np.ascontiguousarray(codes)


def trig_operand(rows: int, cols: int, dtype: str, mode: int):
    """The client's initArrayTrig: ``sin`` / ``cos`` (TrigSin / TrigCos) of
    the int32 linear element index, K fastest, to FP32 then to ``dtype``
    rounding to nearest even. Packed as the kernel reads it."""
    import ml_dtypes
    import numpy as np

    target = {
        "fp8": ml_dtypes.float8_e4m3fn,
        "bf8": ml_dtypes.float8_e5m2,
        "fp4": ml_dtypes.float4_e2m1fn,
    }[dtype]
    trig = np.cos if mode == TRIG_COS else np.sin
    width = cols // 2 if dtype == "fp4" else cols
    out = np.empty((rows, width), dtype=np.uint8)
    for begin, end in _row_chunks(rows, cols):
        index = np.arange(begin * cols, end * cols, dtype=np.int64)
        index = (index + (1 << 31)) % (1 << 32) - (1 << 31)
        values = trig(index.astype(np.float64)).astype(np.float32).astype(target)
        out[begin:end] = _pack(values.view(np.uint8).reshape(end - begin, cols), dtype)
    return out


def _mx_trig_chunk(rng, rows: int, cols: int, dtype: str):
    """mxDataGenerator TrigonometricFromFloat with mean block scaling and
    forced subnormals for one ``rows x cols`` chunk (K fastest); the chunk
    starts at a multiple of ``SPRINKLE`` elements. Returns codes and E8M0
    scales ``[rows][cols / 32]``."""
    import numpy as np

    ebits, mbits, max_code, bias, emin = _format(dtype)
    emask, sign_bit = (1 << ebits) - 1, 1 << (ebits + mbits)
    max_value = float(_decode_magnitude(np.array(max_code), mbits, emask, bias, emin))

    bits = np.cos(rng.uniform(0.0, 2.0 * np.pi, (rows, cols))).view(np.uint64)
    sign = (bits >> np.uint64(63)).astype(np.int64) * sign_bit
    exp = ((bits >> np.uint64(52)) & np.uint64(0x7FF)).astype(np.int64) - 1023
    mantissa = bits & np.uint64((1 << 52) - 1)

    # Per element: the scale that puts the value at data exponent 0, and the
    # truncated mantissa; zero below the smallest subnormal at the smallest scale.
    scale = np.clip(exp + E8M0_BIAS, 0, E8M0_BIAS)
    code = ((exp - (scale - E8M0_BIAS) + bias) & emask) << mbits
    code |= (mantissa >> np.uint64(52 - mbits)).astype(np.int64)
    tiny = exp < emin - E8M0_BIAS
    if tiny.any():
        shift = np.clip(emin - E8M0_BIAS - exp - 1, 0, 63)
        low = (mantissa >> np.uint64(53 - mbits)).astype(np.int64) | (1 << (mbits - 1))
        code = np.where(tiny, low >> shift, code)
        scale = np.where(tiny, 0, scale)
    code = np.where(exp >= emin - E8M0_BIAS - mbits, code, 0) | sign
    del bits, exp, mantissa

    # Block scale: the rounded mean of the element scales of non-zero values.
    code = code.reshape(rows, cols // SCALE_BLOCK, SCALE_BLOCK)
    scale = scale.reshape(code.shape)
    nonzero = (code & ~sign_bit) != 0
    count = nonzero.sum(-1)
    mean = (scale * nonzero).sum(-1) / np.maximum(count, 1)
    block = np.floor(mean) + (mean - np.floor(mean) >= 0.5)
    block = np.where(count > 0, block, E8M0_BIAS).astype(np.int64)

    # Re-express every value at the block scale.
    field = (code >> mbits) & emask
    adjusted = field - bias + scale - block[..., None]
    sign = code & sign_bit
    over = (
        _decode_magnitude(code & ~sign_bit, mbits, emask, bias, emin)
        * np.exp2(scale - block[..., None])
        > max_value
    )
    low = (((code & ((1 << mbits) - 1)) >> 1) | (1 << (mbits - 1))) >> np.clip(
        emin - adjusted - 1, 0, 63
    )
    normal = (code & ~(emask << mbits)) | (((adjusted + bias) & emask) << mbits)
    code = np.select(
        [~nonzero, adjusted < emin - mbits, adjusted < emin, ~over],
        [0, 0, sign | low, normal],
        sign | max_code,
    ).reshape(rows, cols)

    # One subnormal in every group of 15 elements that has none.
    flat = code.reshape(-1)
    groups = flat[: flat.size // SPRINKLE * SPRINKLE].reshape(-1, SPRINKLE)
    has = (((groups >> mbits) & emask) == 0) & ((groups & ((1 << mbits) - 1)) != 0)
    need = np.flatnonzero(~has.any(1))
    target = rng.integers(0, SPRINKLE, need.size)
    groups[need, target] = (groups[need, target] & sign_bit) | rng.integers(
        1, 1 << mbits, need.size
    )
    return code.astype(np.uint8), block.astype(np.uint8)


def mx_trig_operand(rows: int, cols: int, dtype: str, stream: int):
    """``(operand, scales [rows][cols / 32])`` from ``_mx_trig_chunk``."""
    import numpy as np

    rng = np.random.default_rng([MX_SEED, stream])
    width = cols // 2 if dtype == "fp4" else cols
    out = np.empty((rows, width), dtype=np.uint8)
    scales = np.empty((rows, cols // SCALE_BLOCK), dtype=np.uint8)
    for begin, end in _row_chunks(rows, cols, SPRINKLE):
        codes, scales[begin:end] = _mx_trig_chunk(rng, end - begin, cols, dtype)
        out[begin:end] = _pack(codes, dtype)
    return out, scales


GENERATORS = (
    _format,
    _decode_magnitude,
    _row_chunks,
    _pack,
    trig_operand,
    _mx_trig_chunk,
    mx_trig_operand,
)


def timing_inputs(spec, plan: Plan):
    """A, B (as the kernel reads them) and E8M0 scales (A ``[M][K/32]``, B
    ``[K/32][N]``) of the timed launches, cached on disk."""
    import numpy as np

    source = "".join(inspect.getsource(f) for f in GENERATORS)
    key = json.dumps(
        [
            METHOD,
            spec.a_dtype,
            spec.b_dtype,
            spec.M,
            spec.N,
            spec.K,
            plan.inputs,
            plan.init_a,
            plan.init_b,
            plan.init_scale_a,
            plan.init_scale_b,
            hashlib.sha256(source.encode()).hexdigest(),
        ]
    )
    path = (
        CACHE
        / f"{spec.a_dtype}_{spec.b_dtype}_{spec.M}x{spec.N}x{spec.K}_{plan.inputs}_{hashlib.sha256(key.encode()).hexdigest()[:16]}.npz"
    )
    if path.exists():
        with np.load(path) as cached:
            return tuple(cached[x] for x in ("a", "b", "a_scale", "b_scale"))

    print(f"  building timing inputs ({plan.inputs}) -> {path}", flush=True)
    begin = time.perf_counter()
    groups = spec.K // SCALE_BLOCK
    if plan.inputs == "init-array":
        rng = np.random.default_rng(0x32F4)
        a = trig_operand(spec.M, spec.K, spec.a_dtype, plan.init_a)
        b = trig_operand(spec.N, spec.K, spec.b_dtype, plan.init_b)
        a_scale = rng.integers(124, 131, (spec.M, groups), dtype=np.uint8)
        b_scale = rng.integers(124, 131, (groups, spec.N), dtype=np.uint8)
    else:
        a, a_scale = mx_trig_operand(spec.M, spec.K, spec.a_dtype, 1)
        b, b_scale = mx_trig_operand(spec.N, spec.K, spec.b_dtype, 0)
        b_scale = np.ascontiguousarray(b_scale.T)
    CACHE.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.npz")
    np.savez(partial, a=a, b=b, a_scale=a_scale, b_scale=b_scale)
    partial.replace(path)
    print(f"  built in {time.perf_counter() - begin:.0f} s", flush=True)
    return a, b, a_scale, b_scale


# --- Kernel ----------------------------------------------------------------


def make_spec(options: dict):
    mb = bench()
    texts = {}
    for option, (_, _, default) in mb._AXES.items():
        texts[option] = str(options.get(option.replace("_", "-"), default))
        if "," in texts[option]:
            raise ValueError(f"--{option.replace('_', '-')} must be one value")
    m, n, k = (int(options[x]) for x in "mnk")
    ((_, spec),) = list(mb.configurations(argparse.Namespace(**texts), m, n, k))
    return spec


def compile_spec(spec, arch: str):
    from rocke.helpers.compile import compile_kernel
    from rocke.instances.gfx1250.mx_gemm import build_mx_gemm, validate

    reason = validate(spec, arch)
    if reason:
        raise ValueError(f"invalid config: {reason}")
    mb = bench()
    artifact = compile_kernel(
        build_mx_gemm(spec, arch=arch), arch=arch, backend="python"
    )
    row = {
        "kernel": artifact.kernel_name,
        "wmma": mb._wmma_count(artifact.llvm_text),
        "loop_scratch": mb._loop_scratch(artifact.hsaco),
        **mb._resources(artifact.hsaco, None, artifact.kernel_name),
    }
    return artifact, row


def check(spec, artifact, options: dict, runtime) -> str:
    """mx_gemm_bench's bit-exact check (exact and sparse data, beta 0 and 0.5)."""
    from rocke.instances.gfx1250.mx_gemm import mx_gemm_grid

    mb = bench()
    args = argparse.Namespace(
        check_data=list(mb.CHECK_DATA),
        check_betas=[0.0, 0.5],
        verify=options.get("verify", "sampled"),
        samples=int(options.get("samples", 4096)),
        check_only=True,
    )
    result, _, _ = mb._run(
        spec,
        artifact,
        mx_gemm_grid(spec),
        spec.launch_cluster,
        args=args,
        runtime=runtime,
        timing_inputs=False,
    )
    return result


class RotatingLauncher:
    """The kernel with ``slots`` copies of A, B, D (and C when beta is not
    0) sharing one copy of the MX scales, like the client's rotating inputs."""

    def __init__(self, runtime, artifact, spec, plan: Plan, inputs, out_bytes: int):
        from rocke.instances.gfx1250.mx_gemm import mx_gemm_grid

        mb = bench()
        self.runtime = runtime
        self.module = runtime.load_module(artifact.hsaco)
        self.function = self.module.get_function(artifact.kernel_name)
        self.grid, self.cluster = mx_gemm_grid(spec), spec.launch_cluster
        self.pointers = []
        a, b, a_scale, b_scale, c = mb.device_inputs(inputs)

        def upload(array=None, nbytes=0):
            pointer = runtime.alloc(array.nbytes if array is not None else nbytes)
            self.pointers.append(pointer)
            if array is not None:
                runtime.memcpy_h2d(pointer, mb._u8_buffer(array), array.nbytes)
            return pointer

        scales = upload(a_scale), upload(b_scale)
        shared_c = None if plan.beta else upload(c)
        self.arguments = []
        for _ in range(plan.slots):
            pa, pb = upload(a), upload(b)
            pc = upload(c) if plan.beta else shared_c
            pd = upload(nbytes=out_bytes)
            self.arguments.append(
                struct.pack(
                    "<QQQQQQiiiff",
                    pa,
                    pb,
                    *scales,
                    pc,
                    pd,
                    spec.M,
                    spec.N,
                    spec.K,
                    plan.alpha,
                    plan.beta,
                )
            )
        runtime.sync()

    def launch(self, index: int) -> None:
        self.runtime.launch(
            self.function,
            self.grid,
            (128, 1, 1),
            self.arguments[index % len(self.arguments)],
            cluster=self.cluster,
        )

    def close(self) -> None:
        self.runtime.sync()
        self.module.unload()
        for pointer in self.pointers:
            self.runtime.free(pointer)


def tensile_time(launcher: RotatingLauncher, plan: Plan) -> list[float]:
    """Microseconds per launch of each benchmark, timed like the client's
    BenchmarkTimer."""
    runtime = launcher.runtime
    results = []
    for _ in range(plan.benchmarks):
        index = 0
        if plan.warmups:
            launcher.launch(index)
            index += 1
            runtime.sync()
            for _ in range(1, plan.warmups):
                launcher.launch(index)
                index += 1
        total_ms, launches = 0.0, 0
        for _ in range(plan.syncs if plan.enqueues else 0):
            if plan.kernel_time:
                start, stop = runtime.event(), runtime.event()
                start.record()
            else:
                runtime.sync()
                begin = time.perf_counter()
            for _ in range(plan.enqueues):
                launcher.launch(index)
                index += 1
            if plan.kernel_time:
                stop.record()
                stop.synchronize()
                batch_ms = start.elapsed_to(stop)
                start.destroy()
                stop.destroy()
            else:
                runtime.sync()
                batch_ms = (time.perf_counter() - begin) * 1.0e3
            total_ms += batch_ms
            launches += plan.enqueues
            if plan.sleep_percent > 0:
                time.sleep(batch_ms * plan.sleep_percent / 100.0 / 1.0e3)
        runtime.sync()
        if launches:
            results.append(total_ms * 1.0e3 / launches)
    return results


def run_kernel(name: str, options: dict, plan: Plan, args, runtime) -> dict:
    import numpy as np

    spec = make_spec(options)
    if args.prepare_inputs:
        timing_inputs(spec, plan)
        return {"check": "HOST", "method": METHOD}
    artifact, row = compile_spec(spec, args.arch)
    row["method"] = METHOD
    row["plan"] = asdict(plan)
    if runtime is None:
        return {"check": "HOST", **row}
    row["check"] = check(spec, artifact, options, runtime)
    if not row["check"].startswith("PASS"):
        return row
    mb = bench()
    a, b, a_scale, b_scale = timing_inputs(spec, plan)
    out_dtype = mb._out_numpy_dtype(spec.out_dtype)
    c = np.zeros((spec.M, spec.N), dtype=np.float32).astype(out_dtype)
    out_bytes = spec.M * spec.N * np.dtype(out_dtype).itemsize
    launcher = RotatingLauncher(
        runtime, artifact, spec, plan, (a, b, a_scale, b_scale, c), out_bytes
    )
    try:
        runs = tensile_time(launcher, plan)
    finally:
        launcher.close()
    row["runs_us"] = runs
    if runs:
        row["us"] = statistics.median(runs)
    return row


# --- Report ----------------------------------------------------------------


def baseline_path(runs: Path, config: Path, device: str) -> Path:
    """``<name>_baseline_gpu<device>.json`` in ``runs`` for ``<name>_configs.json``."""
    stem = config.stem.removesuffix("_configs")
    return runs / f"{stem}_baseline_gpu{device}.json"


def passed(result: str) -> bool:
    return result.startswith(("PASS", "HOST"))


def compare(ref: dict, new: dict, tolerance: float) -> tuple[str, list[str]]:
    if not passed(new["check"]):
        return "FAIL", [f"check {new['check']}"]
    notes = []
    if ref.get("kernel") != new.get("kernel"):
        notes.append(f"kernel name {ref.get('kernel')} -> {new.get('kernel')}")
    result = "OK"
    for key in EXACT:
        if str(ref.get(key)) != str(new.get(key)):
            notes.append(f"{key} {ref.get(key)} -> {new.get(key)}")
            if key in NO_GROWTH and int(new.get(key) or 0) > int(ref.get(key) or 0):
                result = "FAIL"
    if "us" in ref and "us" in new:
        if ref.get("method") != METHOD or ref.get("plan") != new.get("plan"):
            notes.append("baseline timed with other settings; times not compared")
        elif result == "OK":
            delta = 100.0 * (new["us"] / ref["us"] - 1.0)
            if delta > tolerance:
                result = "SLOWER"
            elif delta < -tolerance:
                result = "FASTER"
    return result, notes


def resource(ref: dict | None, new: dict, key: str) -> str:
    if ref and str(ref.get(key)) != str(new.get(key)):
        return f"{ref.get(key)}>{new.get(key)}"
    return str(new.get(key, ""))


def report(title: str, names, options, plans, new, ref, tolerance) -> bool:
    label = f"vs baseline, tolerance {tolerance:g}%" if ref else "no baseline"
    print(f"\n{title} regression: workspace ({label}), TensileLite client timing")
    print(
        f"  {'kernel':7} {'check':6} {'vgpr':>9} {'lds':>13} {'base us':>9} {'us':>9} {'delta':>7}"
        f" {'TFLOPS':>8} {'TB/s':>6}  result"
    )
    ok = True
    for name in names:
        n, r = new[name], (ref or {}).get(name)
        if r:
            result, notes = compare(r, n, tolerance)
        else:
            result, notes = ("OK" if passed(n["check"]) else "FAIL"), []
        ok &= result in ("OK", "FASTER")
        comparable = r and r.get("method") == METHOD and r.get("plan") == n.get("plan")
        base_us = f"{r['us']:.3f}" if comparable and "us" in r else "-"
        if "us" in n:
            flop, traffic = work(options[name], plans[name])
            us = f"{n['us']:.3f}"
            delta = (
                f"{100.0 * (n['us'] / r['us'] - 1.0):+6.2f}%"
                if comparable and "us" in r
                else "-"
            )
            tflops = f"{flop / n['us'] * 1e-6:.1f}"
            tbps = f"{traffic / n['us'] * 1e-6:.2f}"
        else:
            us, delta, tflops, tbps = "-", "-", "-", "-"
        shown = n["check"] if passed(n["check"]) else "FAIL"
        print(
            f"  {name:7} {shown:6} {resource(r, n, 'vgpr'):>9} {resource(r, n, 'lds'):>13} {base_us:>9} {us:>9} {delta:>7}"
            f" {tflops:>8} {tbps:>6}  {result}"
        )
        for note in notes:
            print(f"  {'':7} {note}")
        if not passed(n["check"]) and not notes:
            print(f"  {'':7} {n['check']}")
        if len(n.get("runs_us", [])) > 1:
            print(f"  {'':7} runs " + " ".join(f"{t:.3f}" for t in n["runs_us"]))
        print(f"  {'':7} {n.get('kernel', '')}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="3", help="HIP_VISIBLE_DEVICES value")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=RUNS,
        help="run results and baselines (outside the repo)",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="comma list of kernels (default: all in the config)",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="OPTION=VALUE",
        help="override a kernel option for every kernel, e.g. --set gl2-prefetch-distance=3 (repeatable)",
    )
    parser.add_argument(
        "--tolerance", type=float, default=2.0, help="slowdown in percent before SLOWER"
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="write this run's results as the baseline",
    )
    parser.add_argument(
        "--host-only", action="store_true", help="compile only: resources, no timing"
    )
    parser.add_argument(
        "--prepare-inputs",
        action="store_true",
        help="build and cache the timing inputs on the CPU, then stop",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the client settings of each kernel",
    )
    args = parser.parse_args()
    if args.save_baseline and (args.host_only or args.prepare_inputs or args.dry_run):
        parser.error("--save-baseline needs a timed run")

    config_path = args.config if args.config.is_absolute() else Path.cwd() / args.config
    if not config_path.exists():
        config_path = HERE / args.config.name
    args.runs_dir = args.runs_dir.resolve()
    config = json.loads(config_path.read_text())
    names = args.only.split(",") if args.only else list(config["kernels"])
    for name in names:
        if name not in config["kernels"]:
            parser.error(
                f"unknown kernel {name}; the config has {', '.join(config['kernels'])}"
            )
    overrides = {}
    for item in args.set:
        option, sep, value = item.partition("=")
        if not sep or not option:
            parser.error(f"--set needs OPTION=VALUE, got {item!r}")
        overrides[option.lstrip("-")] = value

    setup_rocke(args.device)
    axes = {o.replace("_", "-") for o in bench()._AXES}
    defaults = tensile_defaults()
    options, plans = {}, {}
    for name in names:
        options[name] = {
            **config.get("common", {}),
            **config["kernels"][name]["args"],
            **overrides,
        }
        unknown = set(options[name]) - axes - set(KERNEL_OPTIONS)
        if unknown:
            parser.error(
                f"{name}: unknown option(s) {', '.join(sorted(unknown))}; timing and "
                "inputs come from hipblaslt_yaml"
            )
        plans[name] = plan_for(name, config["kernels"][name], options[name], defaults)
        print(f"{name}: {plans[name].describe()}")
    if args.dry_run:
        print("dry run, nothing executed")
        return 0

    runtime = None
    args.arch = "gfx1250"
    if not (args.host_only or args.prepare_inputs):
        from rocke.runtime.hip_module import Runtime, get_device_arch

        args.arch = get_device_arch(0)
        if args.arch != "gfx1250":
            raise SystemExit(f"visible HIP device 0 must be gfx1250, got {args.arch!r}")
        runtime = Runtime()

    new = {}
    for name in names:
        print(f"{name}: running", flush=True)
        try:
            new[name] = run_kernel(name, options[name], plans[name], args, runtime)
        except Exception as exc:  # noqa: BLE001 - one row per failed kernel
            new[name] = {"check": f"ERROR: {type(exc).__name__}: {exc}"}
    if args.prepare_inputs:
        print("timing inputs ready")
        return 0 if all(passed(new[n]["check"]) for n in names) else 1

    stem = config_path.stem.removesuffix("_configs")
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    (args.runs_dir / f"{stem}_last_gpu{args.device}.json").write_text(
        json.dumps(new, indent=2) + "\n"
    )
    baseline = baseline_path(args.runs_dir, config_path.resolve(), args.device)
    saved = json.loads(baseline.read_text()) if baseline.exists() else {}
    ref = (
        None
        if args.save_baseline
        else {n: saved[n] for n in names if n in saved} or None
    )
    ok = report(
        stem.removesuffix("_regression").upper(),
        names,
        options,
        plans,
        new,
        ref,
        args.tolerance,
    )
    if args.save_baseline:
        saved.update({n: new[n] for n in names if passed(new[n]["check"])})
        baseline.write_text(json.dumps(saved, indent=2) + "\n")
        print(f"wrote {baseline}")
    return 0 if ok else 1


if __name__ == "__main__":
    ensure_python()
    sys.exit(main())
