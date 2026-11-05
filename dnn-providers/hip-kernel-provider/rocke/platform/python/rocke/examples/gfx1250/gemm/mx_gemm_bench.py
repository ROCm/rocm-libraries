# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Check, time and host-build the gfx1250 MX GEMM (``mx_gemm``).

Every configuration is the cartesian product of the ``--tile``,
``--wave-group``, ``--cluster`` and tuning options (each takes a comma
list). For each one: compile, report code-object resources (VGPR, SGPR,
spills, scratch, LDS) and the WMMA count, require bit-exact output at ``beta = 0``
and ``beta = 0.5`` on two input sets (``--check-data``: FP4 0 / +1 over all
of K, and all E2M1 values in a few K-steps), then time the ``beta = 0`` path,
on hipBLASLt's MAF benchmark data by default (``--timing-inputs``), as
``mxfp4_tdm_gemm_variants_bench`` does (batched enqueues and the median of
event-timed launches). ``--base`` also times the Step 6.14 kernel of
``mxfp4_tdm_gemm`` on the same problem before and after the configurations,
and reports each configuration's speedup over the mean of the two.

``--host-only`` compiles and reports resources without touching a device;
``--check-only`` skips the timing. Absolute timings must not be committed
to the repo; use ``--out`` with a path outside the tree.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import itertools
import re
import statistics
import struct
import sys
import time
from pathlib import Path

import numpy as np

from ....examples.common.scaled_gemm_reference import decode_e8m0, pack_fp4_codes
from ....helpers.compile import compile_kernel
from ....instances.gfx1250.mx_gemm import (
    LdsPad,
    MxGemmSpec,
    build_mx_gemm,
    mx_gemm_grid,
    validate,
)
from ....instances.gfx1250.mxfp4_tdm_gemm import (
    build_mxfp4_tdm_gemm,
    mxfp4_tdm_gemm_grid,
)
from ....instances.gfx1250.mxfp4_tdm_gemm_variants import variant_spec
from .mxfp4_tdm_gemm import (
    YAML_TILE,
    _u8_buffer,
    make_inputs,
    swizzle_mx_scales_gfx1250,
)
from .mxfp4_tdm_gemm_variants_bench import _resources, tensile_timing_inputs

BASE_VARIANT = "step6_14_mxfp4_gemm"
YAML_PROBLEM = (4096, 4096, 65536)
SMOKE_PROBLEM = (1024, 1024, 2048)
_FP8_MAX = 448.0
_E2M1 = np.array(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=np.float32,
)
_WMMA_K = 128
CHECK_DATA = ("exact", "sparse")
# 128-K steps that hold data in the sparse check. Products are multiples of
# 2^-6 (E2M1 0.5, E8M0 2^-2) up to 144 (6 * 6 * 2 * 2), so every FP32
# partial sum is exact while SPARSE_STEPS * 128 * 144 * 2^6 < 2^24.
SPARSE_STEPS = 12
SPARSE_ALPHA = 2.0**-6

FIELDS = [
    "config",
    "kernel",
    "check",
    "vgpr",
    "sgpr",
    "spills",
    "scratch",
    "lds",
    "wmma",
    "loop_scratch",
    "batched_us",
    "median_us",
    "tflops",
    "speedup_vs_base",
]


def _out_numpy_dtype(out_dtype: str):
    import ml_dtypes

    return {
        "fp8e4m3": ml_dtypes.float8_e4m3fn,
        "fp16": np.float16,
        "bf16": ml_dtypes.bfloat16,
    }[out_dtype]


def check_inputs(spec: MxGemmSpec, *, signed: bool = False):
    """Exactly representable inputs (FP4 0 and +1, or +-1 with ``signed``)
    and C in ``spec.out_dtype``, in ``make_inputs``' canonical layout."""
    inputs = list(make_inputs(spec, signed=signed))
    inputs[4] = inputs[4].astype(np.float32).astype(_out_numpy_dtype(spec.out_dtype))
    return tuple(inputs)


def sparse_steps(spec: MxGemmSpec) -> np.ndarray:
    """The 128-K steps of the sparse check: the first and last, both sides
    of the first and last tile boundaries, then random steps up to
    ``SPARSE_STEPS``."""
    steps = -(-spec.K // _WMMA_K)
    per_tile = spec.tile_k // _WMMA_K
    last_tile = (steps - 1) // per_tile * per_tile
    fixed = {0, steps - 1, per_tile - 1, per_tile, last_tile - 1, last_tile}
    chosen = sorted(s for s in fixed if 0 <= s < steps)
    rest = np.setdiff1d(np.arange(steps), chosen)
    extra = max(0, min(SPARSE_STEPS - len(chosen), len(rest)))
    rng = np.random.default_rng(0x5BA5)
    return np.union1d(chosen, rng.choice(rest, size=extra, replace=False))


def sparse_inputs(spec: MxGemmSpec):
    """Random codes of all 16 E2M1 values and E8M0 scales 2^-2..2^1 in the
    ``sparse_steps`` K-steps, zero elsewhere; C uniform in [-2, 2]. The FP32
    sums are exact in any order, so the check stays bit-exact at any K.
    ``make_inputs``' layout: packed A, B, scales, C, then the codes."""
    rng = np.random.default_rng(0x5BA6)
    k_index = np.concatenate(
        [
            np.arange(s * _WMMA_K, min(spec.K, (s + 1) * _WMMA_K))
            for s in sparse_steps(spec)
        ]
    )
    codes = []
    for rows in (spec.M, spec.N):
        full = np.zeros((rows, spec.K), dtype=np.uint8)
        full[:, k_index] = rng.integers(0, 16, (rows, len(k_index)), dtype=np.uint8)
        codes.append(full)
    groups = spec.K // spec.block_k
    a_scale = rng.integers(125, 129, (spec.M, groups), dtype=np.uint8)
    b_scale = rng.integers(125, 129, (groups, spec.N), dtype=np.uint8)
    c = rng.uniform(-2.0, 2.0, (spec.M, spec.N)).astype(np.float32)
    return (
        pack_fp4_codes(codes[0]),
        pack_fp4_codes(codes[1]),
        a_scale,
        b_scale,
        c.astype(_out_numpy_dtype(spec.out_dtype)),
        *codes,
    )


def _pad_scale_groups(scales_mn_k: np.ndarray) -> np.ndarray:
    """Zero-pad ``[MN][K/32]`` scales to whole 128-K steps, as hipBLASLt's
    ``preSwizzleScalesGFX1250`` does when 128 does not divide K."""
    pad = -scales_mn_k.shape[1] % 4
    return np.pad(scales_mn_k, ((0, 0), (0, pad))) if pad else scales_mn_k


def device_inputs(inputs) -> tuple[np.ndarray, ...]:
    """A, B, A_scale, B_scale, C with the scales swizzled ``[ceil(K/128)][MN][4]``."""
    a, b, a_scale, b_scale, c = inputs[:5]
    return (
        a,
        b,
        swizzle_mx_scales_gfx1250(_pad_scale_groups(a_scale)),
        swizzle_mx_scales_gfx1250(_pad_scale_groups(np.ascontiguousarray(b_scale.T))),
        c,
    )


def reference(
    spec: MxGemmSpec, inputs, *, alpha, beta, rows=None, cols=None
) -> np.ndarray:
    """FP32 grouped-scale reference rounded to ``spec.out_dtype`` the way the
    kernel stores it (FP8 saturating, FP16/BF16 round to nearest even). It is
    exact for ``check_inputs`` and ``sparse_inputs``: their FP32 partial sums
    are exact. All-zero K groups of A are skipped.

    ``rows`` and ``cols`` (index arrays) restrict it to that block of D."""
    _, _, a_scale, b_scale, c, a_codes, b_codes = inputs
    if rows is not None:
        a_codes, a_scale, c = a_codes[rows], a_scale[rows], c[rows]
    if cols is not None:
        b_codes, b_scale, c = b_codes[cols], b_scale[:, cols], c[:, cols]
    sa = decode_e8m0(a_scale).astype(np.float32)
    sb = decode_e8m0(b_scale).astype(np.float32)
    out = np.zeros((a_codes.shape[0], b_codes.shape[0]), dtype=np.float32)
    for group in range(spec.K // spec.block_k):
        begin, end = group * spec.block_k, (group + 1) * spec.block_k
        a_group = a_codes[:, begin:end]
        if not a_group.any():
            continue
        out += (_E2M1[a_group] * sa[:, group, None]) @ (
            _E2M1[b_codes[:, begin:end]] * sb[group, :, None]
        ).T
    out = np.float32(alpha) * out + np.float32(beta) * c.astype(np.float32)
    if spec.out_dtype == "fp8e4m3":
        out = np.where(np.isinf(out), np.float32(np.nan), out)
        out = np.clip(out, -_FP8_MAX, _FP8_MAX)
    return out.astype(_out_numpy_dtype(spec.out_dtype))


def block_indices(
    extent: int, tile: int, count: int, rng: np.random.Generator
) -> np.ndarray:
    """The first and last tile plus ``count`` random indices, sorted."""
    edges = np.concatenate(
        [np.arange(min(tile, extent)), np.arange(max(0, extent - tile), extent)]
    )
    rest = rng.choice(extent, size=min(count, extent), replace=False)
    return np.unique(np.concatenate([edges, rest]))


def verify_block(spec: MxGemmSpec, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows and columns of D that ``--verify block`` checks."""
    rng = np.random.default_rng(0xB10C)
    return (
        block_indices(spec.M, spec.tile_m, count, rng),
        block_indices(spec.N, spec.tile_n, count, rng),
    )


def _check(spec, got, expected, *, verify: str, samples: int) -> bool:
    if verify in ("full", "block"):
        return bool(np.array_equal(got, expected))
    rng = np.random.default_rng(0x32F4)
    count = min(samples, spec.M * spec.N)
    flat = rng.choice(spec.M * spec.N, size=count, replace=False)
    rows, cols = np.divmod(flat, spec.N)
    return bool(np.array_equal(got[rows, cols], expected[rows, cols]))


class _Launcher:
    """A compiled kernel with device buffers for A, B, scales, C and D."""

    def __init__(self, runtime, artifact, grid, cluster, arrays, out_bytes: int):
        self.runtime = runtime
        self.module = runtime.load_module(artifact.hsaco)
        self.function = self.module.get_function(artifact.kernel_name)
        self.grid, self.cluster = grid, cluster
        self.out_bytes = out_bytes
        self.pointers = []
        for array in (*arrays, None):
            nbytes = out_bytes if array is None else array.nbytes
            pointer = runtime.alloc(nbytes)
            self.pointers.append(pointer)
            if array is not None:
                runtime.memcpy_h2d(pointer, _u8_buffer(array), nbytes)

    def upload(self, index: int, array: np.ndarray) -> None:
        self.runtime.memcpy_h2d(self.pointers[index], _u8_buffer(array), array.nbytes)

    def arguments(self, m, n, k, alpha, beta) -> bytes:
        return struct.pack(
            "<QQQQQQiiiff", *self.pointers, m, n, k, float(alpha), float(beta)
        )

    def launch(self, arguments: bytes) -> None:
        self.runtime.launch(
            self.function, self.grid, (128, 1, 1), arguments, cluster=self.cluster
        )

    def output(self) -> bytes:
        self.runtime.sync()
        host = (ctypes.c_uint8 * self.out_bytes)()
        self.runtime.memcpy_d2h(host, self.pointers[-1], self.out_bytes)
        return bytes(host)

    def close(self) -> None:
        self.module.unload()
        for pointer in self.pointers:
            self.runtime.free(pointer)


def _time(launcher: _Launcher, arguments: bytes, *, warmup, iters, batch):
    runtime = launcher.runtime
    for _ in range(max(warmup, 1)):
        launcher.launch(arguments)
    runtime.sync()
    batched_us = None
    if batch > 0:
        begin = time.perf_counter()
        for _ in range(batch):
            launcher.launch(arguments)
        runtime.sync()
        batched_us = (time.perf_counter() - begin) * 1.0e6 / batch
    times = []
    for _ in range(iters):
        start, end = runtime.event(), runtime.event()
        start.record()
        launcher.launch(arguments)
        end.record()
        runtime.sync()
        times.append(start.elapsed_to(end) * 1.0e3)
    median_us = statistics.median(times) if times else None
    return batched_us, median_us


def _wmma_count(llvm_text: str | None) -> str:
    if not llvm_text:
        return ""
    return str(len(re.findall(r"call [^\n]*@llvm\.amdgcn\.wmma\.scale", llvm_text)))


_SYMBOL = re.compile(r"^([0-9a-f]+) <[^>]+>:$", re.MULTILINE)
_ADDRESS = re.compile(r"//\s*([0-9A-F]+):")
_TARGET = re.compile(r"<[^+>]+\+0x([0-9a-f]+)>")


def loop_scratch_ops(objdump_text: str) -> int | None:
    """Scratch instructions inside the longest backward branch (the main
    loop) of a disassembly, or ``None`` if it has no backward branch."""
    symbol = _SYMBOL.search(objdump_text)
    if not symbol:
        return None
    base = int(symbol.group(1), 16)
    loop = None
    addressed = []
    for line in objdump_text.splitlines():
        address = _ADDRESS.search(line)
        if not address:
            continue
        pc = int(address.group(1), 16)
        addressed.append((pc, line))
        target = _TARGET.search(line)
        if target and "branch" in line:
            begin = base + int(target.group(1), 16)
            if begin <= pc and (loop is None or pc - begin > loop[1] - loop[0]):
                loop = (begin, pc)
    if loop is None:
        return None
    return sum(
        1 for pc, line in addressed if loop[0] <= pc <= loop[1] and "scratch_" in line
    )


def _loop_scratch(hsaco: bytes) -> str:
    import tempfile

    from ....analysis.isa import analyze_hsaco

    try:
        with tempfile.NamedTemporaryFile(suffix=".hsaco") as handle:
            handle.write(hsaco)
            handle.flush()
            text = analyze_hsaco(Path(handle.name), keep_text=True).objdump_text
    except Exception as exc:  # noqa: BLE001 - resources are best effort
        print(f"warning: disassembly failed: {exc}", file=sys.stderr)
        return ""
    count = loop_scratch_ops(text)
    return "" if count is None else str(count)


def _run(spec, artifact, grid, cluster, *, args, runtime, timing_inputs: bool):
    """Check at every ``--check-betas`` value, then time. Returns (check, batched, median)."""
    m, n, k = spec.M, spec.N, spec.K
    alpha = np.float32(1.0 / max(k, 256))
    inputs = check_inputs(spec)
    out_dtype = _out_numpy_dtype(spec.out_dtype)
    out_bytes = m * n * np.dtype(out_dtype).itemsize
    launcher = _Launcher(
        runtime, artifact, grid, cluster, device_inputs(inputs), out_bytes
    )
    rows = cols = None
    if args.verify == "block":
        rows, cols = verify_block(spec, args.samples)
    try:
        failed = []
        for kind in args.check_data:
            data, data_alpha = inputs, alpha
            if kind == "sparse":
                data, data_alpha = sparse_inputs(spec), np.float32(SPARSE_ALPHA)
                for index, array in enumerate(device_inputs(data)):
                    launcher.upload(index, array)
            for beta in args.check_betas:
                launcher.launch(launcher.arguments(m, n, k, data_alpha, beta))
                got = np.frombuffer(launcher.output(), dtype=np.uint8)
                got = got.reshape(m, n, -1)
                expected = reference(
                    spec, data, alpha=data_alpha, beta=beta, rows=rows, cols=cols
                )
                if rows is not None:
                    got = got[np.ix_(rows, cols)]
                expected = expected.view(np.uint8).reshape(got.shape)
                if not _check(
                    spec, got, expected, verify=args.verify, samples=args.samples
                ):
                    failed.append(f"{kind} beta={beta:g}")
            if kind == "sparse":
                for index, array in enumerate(device_inputs(inputs)):
                    launcher.upload(index, array)
        if failed:
            return "FAIL:" + ",".join(failed), None, None
        if args.check_only:
            return "PASS", None, None
        timed = launcher.arguments(m, n, k, alpha, 0.0)
        if timing_inputs:
            timing = device_inputs((*tensile_timing_inputs(spec), inputs[4]))
            for index, array in enumerate(timing[:4]):
                launcher.upload(index, array)
            timed = launcher.arguments(m, n, k, 1.0, 0.0)
        batched, median = _time(
            launcher, timed, warmup=args.warmup, iters=args.iters, batch=args.batch
        )
        return "PASS", batched, median
    finally:
        launcher.close()


def run_case(spec: MxGemmSpec, *, betas=(0.0, 0.5)) -> str:
    """Compile and launch ``spec`` on visible device 0 (gfx1250) and require
    bit-exact output on both ``--check-data`` input sets at each beta. Returns the kernel name.
    """
    from ....runtime.hip_module import Runtime, get_device_arch

    reason = validate(spec)
    if reason:
        raise ValueError(reason)
    arch = get_device_arch(0)
    if arch != "gfx1250":
        raise RuntimeError(f"visible HIP device 0 must be gfx1250, got {arch!r}")
    artifact = compile_kernel(build_mx_gemm(spec), arch=arch, backend="python")
    args = argparse.Namespace(
        check_betas=list(betas),
        check_data=CHECK_DATA,
        verify="full",
        samples=0,
        check_only=True,
    )
    check, _, _ = _run(
        spec,
        artifact,
        mx_gemm_grid(spec),
        spec.launch_cluster,
        args=args,
        runtime=Runtime(),
        timing_inputs=False,
    )
    if check != "PASS":
        raise AssertionError(f"{artifact.kernel_name}: {check}")
    return artifact.kernel_name


def _pair(text: str) -> tuple[int, int]:
    x, y = (int(v) for v in text.lower().split("x"))
    return x, y


def _pad(text: str) -> LdsPad | None:
    """``derived``, ``none`` or ``<pad bytes>x<block bytes>``."""
    if text == "derived":
        return None
    if text == "none":
        return LdsPad()
    pad, block = _pair(text)
    return LdsPad(pad, block)


def _optional_int(text: str) -> int | None:
    return None if text == "derived" else int(text)


def _optional_str(text: str) -> str | None:
    return None if text == "derived" else text


def _bool(text: str) -> bool:
    if text not in ("0", "1"):
        raise argparse.ArgumentTypeError("expected 0 or 1")
    return text == "1"


_AXES = {
    # option: (spec field, parser, default)
    "tile": (
        "tile",
        lambda t: tuple(int(v) for v in t.lower().split("x")),
        "256x256x512",
    ),
    "wave_group": ("wave_group", _pair, "2x2"),
    "cluster": ("cluster_dim", lambda t: (*_pair(t), 1), "4x4"),
    "out_dtype": ("out_dtype", str, "fp8e4m3"),
    "gl2_prefetch_distance": ("gl2_prefetch_distance", int, "4"),
    "cluster_signal_wmmas": ("cluster_signal_wmmas", _optional_int, "derived"),
    "cluster_signal": ("cluster_signal", _optional_str, "derived"),
    "init_c_peel": ("init_c_peel", _bool, "1"),
    "cluster_order": ("cluster_order", str, "hw"),
    "local_prefetch": ("local_prefetch", int, "1"),
    "half_plr": ("half_plr", str, "b"),
    "mma": ("mma", str, "32x16"),
    "a_lds_pad": ("a_lds_pad", _pad, "derived"),
    "b_lds_pad": ("b_lds_pad", _pad, "derived"),
    "a_scale_lds_pad": ("a_scale_lds_pad", _pad, "derived"),
    "b_scale_lds_pad": ("b_scale_lds_pad", _pad, "derived"),
}


def configurations(args, m: int, n: int, k: int):
    """``(label, spec)`` for every combination of the axis options."""
    axes = []
    for option, (fieldname, parse, _) in _AXES.items():
        texts = [t.strip() for t in getattr(args, option).split(",") if t.strip()]
        axes.append([(option, t, fieldname, parse(t)) for t in texts])
    for combo in itertools.product(*axes):
        fields = {}
        label = []
        for option, text, fieldname, value in combo:
            if fieldname == "tile":
                fields.update(tile_m=value[0], tile_n=value[1], tile_k=value[2])
            else:
                fields[fieldname] = value
            if text != _AXES[option][2]:
                label.append(f"{option}={text}")
        yield "|".join(label) or "(defaults)", MxGemmSpec(
            name="mxfp4_gemm", M=m, N=n, K=k, **fields
        )


def _row(config: str, **values) -> dict:
    row = {f: "" for f in FIELDS}
    row["config"] = config
    row.update(values)
    return row


def _fmt(value) -> str:
    return "" if value is None else f"{value:.3f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=("smoke", "yaml"), default="smoke")
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    for option, (_, _, default) in _AXES.items():
        parser.add_argument(
            "--" + option.replace("_", "-"),
            dest=option,
            default=default,
            help=f"comma list (default {default})",
        )
    parser.add_argument("--base", action="store_true", help=f"time {BASE_VARIANT} too")
    parser.add_argument("--check-betas", default="0,0.5")
    parser.add_argument(
        "--check-data",
        default=",".join(CHECK_DATA),
        help="comma list: exact (FP4 0 / +1 over all of K), sparse (all E2M1 "
        f"values, E8M0 2^-2..2^1, in up to {SPARSE_STEPS} 128-K steps); both "
        "bit-exact",
    )
    parser.add_argument(
        "--verify",
        choices=("full", "sampled", "block"),
        default=None,
        help="block: rows x cols of the first and last tile plus --samples "
        "random rows and columns",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=4096,
        help="sampled: elements checked; block: random rows and columns",
    )
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--batch", type=int, default=2000)
    parser.add_argument(
        "--timing-inputs",
        choices=("tensile", "check"),
        default="tensile",
        help="data of the timed launches (see mxfp4_tdm_gemm_variants_bench)",
    )
    parser.add_argument("--check-only", action="store_true", help="check, do not time")
    parser.add_argument(
        "--host-only", action="store_true", help="compile only; no device"
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--emit-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    m, n, k = YAML_PROBLEM if args.shape == "yaml" else SMOKE_PROBLEM
    m, n, k = args.m or m, args.n or n, args.k or k
    args.verify = args.verify or ("full" if args.shape == "smoke" else "sampled")
    args.check_betas = [float(x) for x in args.check_betas.split(",") if x.strip()]
    args.check_data = [x.strip() for x in args.check_data.split(",") if x.strip()]
    if not args.check_data or set(args.check_data) - set(CHECK_DATA):
        parser.error(f"--check-data must be a comma list of {CHECK_DATA}")

    runtime = None
    arch = "gfx1250"
    if not args.host_only:
        from ....runtime.hip_module import Runtime, get_device_arch

        arch = get_device_arch(0)
        if arch != "gfx1250":
            raise SystemExit(f"visible HIP device 0 must be gfx1250, got {arch!r}")
        runtime = Runtime()

    def compile_and_run(config, kernel, grid, cluster, run_spec):
        artifact = compile_kernel(kernel, arch=arch, backend="python")
        if args.emit_dir is not None:
            args.emit_dir.mkdir(parents=True, exist_ok=True)
            (args.emit_dir / f"{artifact.kernel_name}.hsaco").write_bytes(
                artifact.hsaco
            )
            if artifact.llvm_text:
                (args.emit_dir / f"{artifact.kernel_name}.ll").write_text(
                    artifact.llvm_text
                )
        resources = _resources(artifact.hsaco, args.emit_dir, artifact.kernel_name)
        values = dict(
            kernel=artifact.kernel_name,
            wmma=_wmma_count(artifact.llvm_text),
            loop_scratch=_loop_scratch(artifact.hsaco),
            **resources,
        )
        if args.host_only:
            return _row(config, check="HOST", **values), None
        check, batched, median = _run(
            run_spec,
            artifact,
            grid,
            cluster,
            args=args,
            runtime=runtime,
            timing_inputs=args.timing_inputs == "tensile",
        )
        best = batched if batched is not None else median
        tflops = 2.0 * m * n * k / (best * 1.0e-6) / 1.0e12 if best else None
        row = _row(
            config,
            check=check,
            batched_us=_fmt(batched),
            median_us=_fmt(median),
            tflops=_fmt(tflops),
            **values,
        )
        return row, best

    def run_base(label):
        spec = variant_spec(BASE_VARIANT, m, n, k, **YAML_TILE)
        run_spec = MxGemmSpec(name="base", M=m, N=n, K=k)
        return compile_and_run(
            label,
            build_mxfp4_tdm_gemm(spec, arch=arch),
            mxfp4_tdm_gemm_grid(spec),
            spec.cluster_dim,
            run_spec,
        )

    rows, base_times = [], []
    if args.base:
        row, best = run_base(f"{BASE_VARIANT} (before)")
        rows.append(row)
        base_times.append(best)
        print("\t".join(str(row[f]) for f in FIELDS), flush=True)
    timed = []
    for config, spec in configurations(args, m, n, k):
        reason = validate(spec, arch)
        if reason:
            row, best = _row(config, check=f"INVALID:{reason}"), None
        else:
            try:
                row, best = compile_and_run(
                    config,
                    build_mx_gemm(spec, arch=arch),
                    mx_gemm_grid(spec),
                    spec.launch_cluster,
                    spec,
                )
            except Exception as exc:  # noqa: BLE001 - one row per failed config
                row, best = (
                    _row(config, check=f"ERROR:{type(exc).__name__}: {exc}"),
                    None,
                )
        rows.append(row)
        timed.append((row, best))
        print("\t".join(str(row[f]) for f in FIELDS), flush=True)
    if args.base:
        row, best = run_base(f"{BASE_VARIANT} (after)")
        rows.append(row)
        base_times.append(best)
        print("\t".join(str(row[f]) for f in FIELDS), flush=True)
    base = [t for t in base_times if t]
    if base:
        base_mean = sum(base) / len(base)
        for row, best in timed:
            if best:
                row["speedup_vs_base"] = f"{base_mean / best:.3f}"

    print("\t".join(FIELDS))
    for row in rows:
        print("\t".join(str(row[f]) for f in FIELDS))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.out}", flush=True)
    bad = [r for r in rows if not r["check"].startswith(("PASS", "HOST", "INVALID"))]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
