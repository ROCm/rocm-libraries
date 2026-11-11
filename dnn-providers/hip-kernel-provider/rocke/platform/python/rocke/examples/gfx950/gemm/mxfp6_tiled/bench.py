# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Evaluate one tiled MXFP6 GEMM configuration on a gfx950 device.

One process per configuration, so a faulting or hanging kernel cannot take the
tuner down with it. ``--mode eval`` compiles (with an HSACO cache), checks a
small shape exactly against a float64 reference built from dyadic inputs,
times the target shape and checks a sampled block of its output, then writes
one JSON record. ``--mode profile`` only launches the cached kernel a few
times, for a profiler wrapped around this process.

Measured numbers are written to ``--out``, which must live outside the
repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from .....instances.gfx950.block_scaled_gemm_tiled import (
    STAGE_K,
    TiledBlockScaledGemmSpec,
    build_tiled_block_scaled_gemm,
    is_valid_spec,
    tiled_block_scaled_gemm_grid,
    tiled_block_scaled_gemm_signature,
)
from ....common.scaled_gemm_reference import decode_fp6, pack_fp6_codes

# Dense FP6 MFMA work per clock per CU on gfx950 (4 SIMDs, 32x32x64 in 32 cycles).
FLOP_PER_CLK_CU = 16384


def _spec(shape, knobs: dict) -> TiledBlockScaledGemmSpec:
    m, n, k = shape
    return TiledBlockScaledGemmSpec(M=m, N=n, K=k, **knobs)


def _tool(name: str) -> str | None:
    from .....tests.llc import find_llvm_tool

    return find_llvm_tool(name)


def code_object_facts(hsaco: Path) -> dict:
    facts: dict = {}
    readelf, objdump = _tool("llvm-readelf"), _tool("llvm-objdump")
    if readelf:
        notes = subprocess.run([readelf, "--notes", str(hsaco)], capture_output=True, text=True, check=False).stdout
        for key in (
            "vgpr_count",
            "agpr_count",
            "sgpr_count",
            "vgpr_spill_count",
            "sgpr_spill_count",
            "private_segment_fixed_size",
            "group_segment_fixed_size",
        ):
            m = re.search(r"\." + key + r":\s*(\d+)", notes)
            if m:
                facts[key] = int(m.group(1))
    if objdump:
        asm = subprocess.run(
            [objdump, "-d", "--mcpu=gfx950", str(hsaco)], capture_output=True, text=True, check=False
        ).stdout
        hsaco.with_suffix(".s").write_text(asm)
        # Instruction text only: the symbol encodes the knobs, the code may not.
        body = [m.group(1) for m in (re.match(r"^\s+([a-z][a-z0-9_]*\b[^/]*)", l) for l in asm.splitlines()) if m]
        facts["isa_sha"] = hashlib.sha256("\n".join(s.rstrip() for s in body).encode()).hexdigest()[:16]
        ops: dict[str, int] = {}
        for line in asm.splitlines():
            m = re.match(r"^\s+((?:s|v|ds|global|buffer|scratch)_[a-z0-9_]+)\b", line)
            if m:
                ops[m.group(1)] = ops.get(m.group(1), 0) + 1

        def count(prefix: str) -> int:
            return sum(v for k, v in ops.items() if k.startswith(prefix))

        facts["isa"] = {
            "instructions": sum(ops.values()),
            "mfma": count("v_mfma"),
            "ds_read_b128": ops.get("ds_read_b128", 0),
            "ds_read_b64": ops.get("ds_read_b64", 0),
            "ds_read_other": count("ds_read") - ops.get("ds_read_b128", 0) - ops.get("ds_read_b64", 0),
            "buffer_load": count("buffer_load"),
            "global_load": count("global_load"),
            "s_waitcnt": ops.get("s_waitcnt", 0),
            "s_barrier": ops.get("s_barrier", 0),
            "s_nop": ops.get("s_nop", 0),
            "v_accvgpr": count("v_accvgpr"),
            "scratch": count("scratch_"),
        }
    return facts


def _direct_compiler() -> Path:
    """The direct compiler's `target_compile` tool (ROCKE_DIRECT_TARGET_COMPILE)."""
    import os

    path = os.environ.get("ROCKE_DIRECT_TARGET_COMPILE")
    if not path:
        raise RuntimeError("backend='direct' needs ROCKE_DIRECT_TARGET_COMPILE: the direct compiler's target_compile")
    return Path(path)


def compile_cached(spec: TiledBlockScaledGemmSpec, cache: Path) -> tuple[Path, dict]:
    from .....core.ir_serialize import serialize
    from .....core.lower_llvm import lower_kernel_to_llvm
    from .....helpers import compile_kernel

    cache.mkdir(parents=True, exist_ok=True)
    kernel = build_tiled_block_scaled_gemm(spec, arch="gfx950")
    # Key on the IR (and the direct compiler binary), not only the name: a
    # builder or compiler change keeps the name.
    if spec.backend == "direct":
        text, tool = serialize(kernel), _direct_compiler()
        stat = tool.stat()
        key = f"{text}\n{tool}:{stat.st_size}:{stat.st_mtime_ns}"
    else:
        key = lower_kernel_to_llvm(kernel, arch="gfx950")
    ir_sha = hashlib.sha256(key.encode()).hexdigest()[:12]
    # The IR hash keys the file; the name prefix keeps it under 255 bytes.
    hsaco = cache / f"{spec.kernel_name()[:200]}_{ir_sha}.hsaco"
    meta = hsaco.with_suffix(".json")
    if hsaco.exists() and meta.exists():
        return hsaco, json.loads(meta.read_text())
    t0 = time.perf_counter()
    if spec.backend == "direct":
        out = cache / f"direct_{ir_sha}"
        out.mkdir(exist_ok=True)
        ir_path = out / f"{spec.kernel_name()}.ir"
        ir_path.write_text(text)
        proc = subprocess.run(
            [str(tool), "gfx950", str(out), str(ir_path)], capture_output=True, text=True, check=False
        )
        produced = out / f"{spec.kernel_name()}.hsaco"
        if proc.returncode or not produced.exists():
            raise RuntimeError(f"direct compile failed: {(proc.stdout + proc.stderr)[-800:]}")
        hsaco.write_bytes(produced.read_bytes())
        facts = {"compile_s": time.perf_counter() - t0, "direct": proc.stdout.strip().splitlines()[-1]}
    else:
        art = compile_kernel(kernel, arch="gfx950", capture_ir_text=False)
        facts = {"compile_s": time.perf_counter() - t0}
        hsaco.write_bytes(art.hsaco)
    facts.update(code_object_facts(hsaco))
    meta.write_text(json.dumps(facts, indent=2))
    return hsaco, facts


# ---------------------------------------------------------------- device I/O


class Device:
    def __init__(self):
        from .....runtime.hip_module import Runtime

        self.rt = Runtime()
        self.allocs: list[int] = []

    def upload(self, arr: np.ndarray) -> int:
        flat = np.ascontiguousarray(arr).reshape(-1).view(np.uint8)
        if not flat.flags.writeable:  # ctypes views need a writable buffer
            flat = flat.copy()
        ptr = self.rt.alloc(flat.nbytes)
        self.rt.memcpy_h2d(ptr, np.ctypeslib.as_ctypes(flat), flat.nbytes)
        self.allocs.append(ptr)
        return ptr

    def alloc(self, nbytes: int, fill: int = 0xFF) -> int:
        ptr = self.rt.alloc(nbytes)
        self.rt.memset(ptr, fill, nbytes)
        self.allocs.append(ptr)
        return ptr

    def download_16bit(self, ptr: int, shape, dtype: str) -> np.ndarray:
        bits = np.empty(int(np.prod(shape)), dtype=np.uint16)
        self.rt.memcpy_d2h(np.ctypeslib.as_ctypes(bits), ptr, bits.nbytes)
        if dtype == "bf16":
            return (bits.astype(np.uint32) << 16).view(np.float32).reshape(shape)
        return bits.view(np.float16).astype(np.float32).reshape(shape)

    def close(self):
        for ptr in self.allocs:
            self.rt.free(ptr)
        self.allocs.clear()


def kernel_args(spec, ptrs) -> bytes:
    """The kernel's argument bytes in its signature's order: the pointers
    `ptrs` (A, B, A_scale, B_scale, C), then any i32 shape argument."""
    shape = {"M": spec.M, "N": spec.N, "K": spec.K}
    sig = tiled_block_scaled_gemm_signature(spec)
    is_ptr = [p["type"].startswith("ptr<") for p in sig]
    if sum(is_ptr) != len(ptrs):
        raise ValueError(f"the kernel takes {sum(is_ptr)} pointers, not {len(ptrs)}")
    it = iter(ptrs)
    vals = [next(it) if ptr else shape[p["name"]] for p, ptr in zip(sig, is_ptr)]
    return struct.pack("<" + "".join("Q" if ptr else "i" for ptr in is_ptr), *vals)


def _launcher(dev: Device, hsaco: Path, spec, ptrs):
    module = dev.rt.load_module(hsaco.read_bytes())
    fn = module.get_function(spec.kernel_name())
    packed = kernel_args(spec, ptrs)
    grid, block = tiled_block_scaled_gemm_grid(spec), (spec.threads, 1, 1)

    def launch():
        dev.rt.launch(fn, grid, block, packed)

    return module, launch


# ---------------------------------------------------------------- inputs


def random_inputs(spec, data: str, cache: Path):
    """The inputs for `--data`, cached per data kind and shape: random FP6
    bytes (every code is finite) or zero codes, with E8M0 scales near unity,
    or Gaussian operands (`gaussian_operand`, A on stream 1, B on stream 2)."""
    m, n, k = spec.M, spec.N, spec.K
    kind = f"{data}_{spec.dtype_a}_{spec.dtype_b}" if data == "gaussian" else data
    path = cache / f"inputs_{kind}_m{m}n{n}k{k}.npz"
    if path.exists():
        z = np.load(path)
        return z["a"], z["b"], z["sa"], z["sb"]
    if data == "gaussian":
        a, sa = gaussian_operand(m, k, spec.dtype_a, stream=1)
        b, sb = gaussian_operand(n, k, spec.dtype_b, stream=2)
        sb = np.ascontiguousarray(sb.T)
    else:
        rng = np.random.default_rng(0x6E6)
        if data == "zero":
            a = np.zeros((m, k * 3 // 4), np.uint8)
            b = np.zeros((n, k * 3 // 4), np.uint8)
        else:
            a = rng.integers(0, 256, size=(m, k * 3 // 4), dtype=np.uint8)
            b = rng.integers(0, 256, size=(n, k * 3 // 4), dtype=np.uint8)
        sa = rng.integers(124, 131, size=(m, k // 32), dtype=np.uint8)
        sb = rng.integers(124, 131, size=(k // 32, n), dtype=np.uint8)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez(path, a=a, b=b, sa=sa, sb=sb)
    return a, b, sa, sb


# Seed of the Gaussian operands; streams tell the operands apart.
GAUSS_SEED = 42
# The exponent of each FP6 format's largest normal value (7.5 and 28).
_FP6_EMAX = {"fp6e2m3": 2, "fp6e3m2": 4}


def gaussian_operand(rows: int, k: int, dtype: str, stream: int, seed: int = GAUSS_SEED):
    """Packed FP6 codes [rows][k*3/4] and E8M0 scales [rows][k/32] of
    `gaussian_samples` quantized by `_mx_quantize`, 32 rows at a time."""
    codes = np.empty((rows, k * 3 // 4), np.uint8)
    scales = np.empty((rows, k // 32), np.uint8)
    for r in range(0, rows, 32):
        c, s = _mx_quantize(gaussian_samples(np.arange(r, min(rows, r + 32)), k, stream, seed), dtype)
        codes[r : r + len(c)] = _pack_fp6(c.reshape(len(c), k))
        scales[r : r + len(c)] = s
    return codes, scales


def gaussian_samples(rows, k: int, stream: int, seed: int = GAUSS_SEED) -> np.ndarray:
    """float32 N(0, 1) samples [len(rows)][k/32][32] of the given rows, so a
    row does not depend on the row count. Element i of K32 block j of row r
    is sqrt(-2 ln u1) cos(2 pi u2) in float64, with u = ((h >> 11) + 1/2) / 2^53
    for h = splitmix64(2 (c + i) + 1) and splitmix64(2 (c + i) + 2), and the
    block's counter c = splitmix64(seed ^ stream << 56 ^ (r k/32 + j) * 0x100000001B3)."""
    nblk = k // 32
    blk = (np.asarray(rows, dtype=np.int64)[:, None] * nblk + np.arange(nblk)).astype(np.uint64)
    with np.errstate(over="ignore"):
        c = _splitmix64(np.uint64(seed) ^ (np.uint64(stream) << np.uint64(56)) ^ (blk * np.uint64(0x100000001B3)))
        h = (c[..., None] + np.arange(32, dtype=np.uint64)) * np.uint64(2)
    u1, u2 = (((_splitmix64(h + np.uint64(j)) >> np.uint64(11)).astype(np.float64) + 0.5) / 2.0**53 for j in (1, 2))
    return (np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)).astype(np.float32)


def _mx_quantize(x: np.ndarray, dtype: str) -> tuple[np.ndarray, np.ndarray]:
    """OCP MX quantization of float32 K32 blocks x [...][32], each with a
    nonzero element, to FP6 `dtype`: (codes [...][32], E8M0 scales [...]).
    A block's scale is 2^(floor(log2 amax) - emax), log2 rounded to float32;
    each element over the scale rounds to the nearest FP6 value, ties to the
    even code, saturating at the largest."""
    mag = decode_fp6(pack_fp6_codes(np.arange(32, dtype=np.uint8)[None]), dtype)[0].astype(np.float32)
    amax = np.abs(x).max(axis=-1)
    se = np.floor(np.log2(amax.astype(np.float64)).astype(np.float32)).astype(np.int64) - _FP6_EMAX[dtype]
    v = np.ldexp(x, (-se)[..., None].astype(np.int32))
    a = np.abs(v)
    lo = (np.searchsorted(mag, a, side="right") - 1).clip(0, 31)
    hi = np.minimum(lo + 1, 31)
    dl, dh = a - mag[lo], mag[hi] - a
    c = np.where(dl < dh, lo, np.where(dh < dl, hi, np.where(lo & 1, hi, lo)))
    c = np.where(a < mag[31], c, 31)
    return (c | np.where(np.signbit(v), 32, 0)).astype(np.uint8), (se + 127).astype(np.uint8)


def _pack_fp6(codes: np.ndarray) -> np.ndarray:
    """`pack_fp6_codes`, vectorized: code k of a row at bits [6k, 6k + 6)."""
    q = codes.reshape(codes.shape[0], -1, 4).astype(np.uint32)
    w = q[..., 0] | q[..., 1] << 6 | q[..., 2] << 12 | q[..., 3] << 18
    return np.stack([w, w >> 8, w >> 16], axis=-1).astype(np.uint8).reshape(codes.shape[0], -1)


def _splitmix64(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def dyadic_inputs(spec, seed=0xB10C):
    """Small dyadic codes and scales in {1/2, 1, 2}: every FP32 partial sum is exact."""
    rng = np.random.default_rng(seed)

    def codes(rows, dtype):
        top = 9 if dtype == "fp6e2m3" else 13
        c = rng.integers(0, top, size=(rows, spec.K), dtype=np.uint8)
        return pack_fp6_codes(c | (rng.integers(0, 2, size=c.shape, dtype=np.uint8) << 5))

    a, b = codes(spec.M, spec.dtype_a), codes(spec.N, spec.dtype_b)
    sa = rng.integers(126, 129, size=(spec.M, spec.K // 32), dtype=np.uint8)
    sb = rng.integers(126, 129, size=(spec.K // 32, spec.N), dtype=np.uint8)
    return a, b, sa, sb


def reference(spec, a, b, sa, sb, rows=None, cols=None):
    rows = np.arange(spec.M) if rows is None else rows
    cols = np.arange(spec.N) if cols is None else cols

    def scaled(packed, scales, dtype):
        return decode_fp6(packed, dtype) * np.repeat(np.ldexp(1.0, scales.astype(np.int32) - 127), 32, axis=1)

    A = scaled(a[rows], sa[rows], spec.dtype_a)
    B = scaled(b[cols], sb[:, cols].T, spec.dtype_b)
    return A @ B.T, np.abs(A) @ np.abs(B).T


def _round_out(x: np.ndarray, dtype: str) -> np.ndarray:
    import ml_dtypes

    t = ml_dtypes.bfloat16 if dtype == "bf16" else np.float16
    return x.astype(t).astype(np.float32)


def device_inputs(spec, inputs):
    """The arrays the kernel reads: A and B repacked to the spec's layouts."""
    a, b, sa, sb = inputs
    if spec.atom16:
        return a, b_c0c1p(b, spec.wave_tile_n), sa, scales16_pairs(sb, spec.wave_tile_n)

    def pack(x: np.ndarray, layout: str) -> np.ndarray:
        if layout == "row":
            return x
        return np.ascontiguousarray(x.reshape(x.shape[0], -1, STAGE_K * 3 // 4).transpose(1, 0, 2))

    kt, per = spec.K // STAGE_K, STAGE_K // 32
    if spec.b_vgpr:
        sa = np.ascontiguousarray(sa.reshape(spec.M, kt, per).transpose(1, 0, 2)) if spec.a_scale_layout == "kblock" else sa
        return pack(a, spec.a_layout), b_lanes(b), sa, scales_lanes(sb)
    if spec.a_scale_layout == "kblock":
        sa = np.ascontiguousarray(sa.reshape(spec.M, kt, per).transpose(1, 0, 2))
    if spec.b_scale_layout == "kblock":
        sb = np.ascontiguousarray(sb.reshape(kt, per, spec.N).transpose(0, 2, 1))
    elif spec.b_scale_layout == "kblock4":
        sb = scales_kblock4(sb, spec.wave_tile_n, T10_BLOCKS if spec.k_t10 else ROT_BLOCKS)
        sb = scales_pairs(sb) if spec.b_scale_pairs else sb
    b = pack(b, "kblock" if spec.b_layout == "lanes" else spec.b_layout)
    if spec.b_split:
        b = split_halves(b, T10_BLOCKS if spec.k_t10 else ROT_BLOCKS)
        b = half_lanes(b) if spec.b_lanes else b
    return pack(a, spec.a_layout), b, sa, sb


# The K32 block lane half h takes in step s ([h][s]): k_order "rot" and "t10".
ROT_BLOCKS = ((0, 3), (2, 1))
T10_BLOCKS = ((0, 1), (2, 3))


def scales_kblock4(sb: np.ndarray, wave_n: int, blocks=ROT_BLOCKS) -> np.ndarray:
    """B scales ([K/32][N] bytes) in the `kblock4` order: per stage, column
    group of `wave_n` and lane (h, r), dword s byte j is the scale of column
    32j + r in K32 block `blocks[h][s]`; unused bytes zero."""
    kt, n = sb.shape[0] // 4, sb.shape[1]
    cols = wave_n // 32
    s = np.zeros((kt, 4, n // wave_n, 4, 32), dtype=np.uint8)  # [kt][blk][g][j][r]
    s[:, :, :, :cols] = sb.reshape(kt, 4, n // wave_n, cols, 32)
    blk = np.array(blocks)  # [h][s]
    return np.ascontiguousarray(s[:, blk].transpose(0, 3, 1, 5, 2, 4))  # [kt][g][h][r][s][j]


def scales_pairs(sb: np.ndarray) -> np.ndarray:
    """`kblock4` B scales ([kt][g][h][r][s][j]) for `b_scale_fetch="pair"`: stages
    2v and 2v + 1 side by side per lane ([kt/2][g][h][r][2][s][j]); an odd stage
    count gets a zero stage."""
    if sb.shape[0] % 2:
        sb = np.concatenate([sb, np.zeros_like(sb[:1])])
    x = sb.reshape(sb.shape[0] // 2, 2, *sb.shape[1:])
    return np.ascontiguousarray(x.transpose(0, 2, 3, 4, 1, 5, 6))


def b_lanes(b: np.ndarray) -> np.ndarray:
    """Row-major B ([N][K*3/4]) in the `b_path="vgpr"` order: per stage, 32-column
    block and K64 step s the 64 lanes' bytes [0:16) of their K32 block, lane
    (h, r) column 32c + r and block 2h (s = 0) or 3 - 2h (s = 1); then all the
    bytes [16:24) in the same order."""
    n = b.shape[0]
    blk = np.array([[0, 3], [2, 1]])  # [h][s]
    x = b.reshape(n // 32, 32, -1, 4, 24)[:, :, :, blk]  # [c][r][t][h][s][24]
    x = x.transpose(2, 0, 4, 3, 1, 5)  # [t][c][s][h][r][24]
    return np.concatenate([x[..., :16].reshape(-1), x[..., 16:].reshape(-1)])


def scales_lanes(sb: np.ndarray) -> np.ndarray:
    """B scales ([K/32][N] bytes) in the `b_path="vgpr"` order: per stage,
    64-column group and lane (h, r), byte 2s + j of its dword the scale of
    column 32j + r in K32 block 2h (s = 0) or 3 - 2h (s = 1)."""
    kt, n = sb.shape[0] // 4, sb.shape[1]
    blk = np.array([[0, 3], [2, 1]])  # [h][s]
    s = sb.reshape(kt, 4, n // 64, 2, 32)[:, blk]  # [kt][h][s][g][j][r]
    return np.ascontiguousarray(s.transpose(0, 3, 1, 5, 2, 4))  # [kt][g][h][r][s][j]


def split_halves(b: np.ndarray, blocks=ROT_BLOCKS) -> np.ndarray:
    """kblock B ([K/128][N][96 B]) in the `b_release="half"` order
    [K/128][2][N][48 B]: half s holds the K32 blocks (x, y) = (blocks[0][s],
    blocks[1][s]) as x[0:16] x[16:24] y[16:24] y[0:16]."""
    blk = b.reshape(b.shape[0], b.shape[1], 4, 24)
    halves = [
        np.concatenate([blk[:, :, x, :16], blk[:, :, x, 16:], blk[:, :, y, 16:], blk[:, :, y, :16]], axis=2)
        for x, y in zip(*blocks)
    ]
    return np.ascontiguousarray(np.stack(halves, axis=1))


def b_c0c1p(b: np.ndarray, wave_n: int = 128) -> np.ndarray:
    """Row-major B ([N][K*3/4]) for `mfma_atom="16x16x128"`: per stage, column
    half h, group g of `wave_n` columns, column-block pair p (16-column blocks
    4h + 2p and 4h + 2p + 1 of the group) and plane k, the 64 lanes' bytes
    [16k, 16k + 16) of the 48 their two operands make side by side, lane
    l = 16 kg + c taking column 16 (4h + 2p + q) + c's bytes 24 kg .. 24 kg + 24
    of the stage ([K/128][2][N/wave_n][2][3][64][16 B])."""
    n, kt = b.shape[0], b.shape[1] // 96
    x = b.reshape(n // wave_n, 2, 2, 2, 16, kt, 4, 24)  # [g][h][p][q][c][t][kg][24]
    x = x.transpose(5, 1, 0, 2, 6, 4, 3, 7).reshape(kt, 2, n // wave_n, 2, 64, 3, 16)  # [t][h][g][p][l][k][16]
    return np.ascontiguousarray(x.transpose(0, 1, 2, 3, 5, 4, 6)).reshape(-1)


def b_c0c1p_unpack(x: np.ndarray, n: int, k: int, wave_n: int = 128) -> np.ndarray:
    """The row-major B that `b_c0c1p` packed into `x`."""
    kt = k // 128
    y = x.reshape(kt, 2, n // wave_n, 2, 3, 64, 16).transpose(0, 1, 2, 3, 5, 4, 6).reshape(kt, 2, n // wave_n, 2, 4, 16, 2, 24)
    return np.ascontiguousarray(y.transpose(2, 1, 3, 6, 5, 0, 4, 7)).reshape(n, kt * 96)


def scales16_pairs(sb: np.ndarray, wave_n: int = 128) -> np.ndarray:
    """B scales ([K/32][N] bytes) for `mfma_atom="16x16x128"`: per pair of
    stages v, group g of `wave_n` columns and lane l = 16 kg + c, four dwords,
    dword 2q + h's byte j the scale of column 16 (4h + j) + c of the group in
    K32 block kg of stage 2v + q ([K/256][N/wave_n][64][2][2][4] bytes)."""
    kt, n = sb.shape[0] // 4, sb.shape[1]
    s = sb.reshape(kt // 2, 2, 4, n // wave_n, 2, 4, 16)  # [v][q][kg][g][h][j][c]
    return np.ascontiguousarray(s.transpose(0, 3, 2, 6, 1, 4, 5)).reshape(-1)


def half_lanes(h: np.ndarray) -> np.ndarray:
    """`split_halves` B ([K/128][2][N][48 B]) in the `b_layout="lanes"` order:
    per stage half and 32-column block, lane (h, r) = column 32c + r's 16-byte
    piece of its block (x[0:16] for h = 0, y[0:16] for h = 1) at 16 l, then
    its 8-byte piece (x[16:24], y[16:24]) at 1024 + 8 l."""
    kt, _, n, _ = h.shape
    x = h.reshape(kt, 2, n // 32, 32, 48)
    wide = np.stack([x[..., 0:16], x[..., 32:48]], axis=3)  # [kt][2][c][h][r][16]
    narrow = np.stack([x[..., 16:24], x[..., 24:32]], axis=3)  # [kt][2][c][h][r][8]
    return np.ascontiguousarray(np.concatenate(
        [wide.reshape(kt, 2, n // 32, -1), narrow.reshape(kt, 2, n // 32, -1)], axis=3))


def run_case(dev: Device, hsaco: Path, spec, inputs):
    ptrs = [dev.upload(x) for x in device_inputs(spec, inputs)]
    c_ptr = dev.alloc(spec.M * spec.N * 2)
    module, launch = _launcher(dev, hsaco, spec, ptrs + [c_ptr])
    return module, launch, c_ptr


# ---------------------------------------------------------------- checks


def exact_check(spec, cache: Path) -> dict:
    """Run the same knobs on a small multi-tile shape; demand exact equality."""
    # mfma_atom="16x16x128" runs whole trips of eight stages only.
    small = replace(spec, M=2 * spec.block_m, N=2 * spec.block_n, K=8 * STAGE_K if spec.atom16 else 1152)
    ok, why = is_valid_spec(small)
    override = {}
    if not ok and spec.xcd_remap != "none":
        # The remap only permutes workgroups over tiles and needs a full grid;
        # check the kernel body without it (the sampled check covers the remap).
        small, override = replace(small, xcd_remap="none"), {"xcd_remap": "none"}
        ok, why = is_valid_spec(small)
    if not ok:
        return {"status": "skipped", "reason": why}
    hsaco, _ = compile_cached(small, cache)
    inputs = dyadic_inputs(small)
    dev = Device()
    try:
        module, launch, c_ptr = run_case(dev, hsaco, small, inputs)
        launch()
        dev.rt.sync()
        got = dev.download_16bit(c_ptr, (small.M, small.N), small.dtype_c)
        module.unload()
    finally:
        dev.close()
    ref = _round_out(reference(small, *inputs)[0], small.dtype_c)
    bad = int(np.count_nonzero(got != ref))
    return {
        "status": "pass" if bad == 0 else "fail", "bad": bad, "n": int(ref.size),
        "shape": [small.M, small.N, small.K], **({"override": override} if override else {}),
    }


def sampled_check(spec, got_fn, inputs) -> dict:
    """Sampled outputs against the float64 reference, within 2^-7 |ref| (twice
    BF16's rounding bound) plus 2^-16 sum |a b| (FP32 accumulation): both terms
    scale with the operands, so random and Gaussian data share the tolerance."""
    rng = np.random.default_rng(7)
    edges = np.array([0, 1, 31, 32, spec.block_m - 1, spec.block_m, spec.M - 1])
    rows = np.unique(np.concatenate([edges[edges < spec.M], rng.choice(spec.M, 40, replace=False)]))
    edges = np.array([0, 1, 31, 32, spec.block_n - 1, spec.block_n, spec.N - 1])
    cols = np.unique(np.concatenate([edges[edges < spec.N], rng.choice(spec.N, 40, replace=False)]))
    ref, absref = reference(spec, *inputs, rows=rows, cols=cols)
    got = got_fn()[np.ix_(rows, cols)].astype(np.float64)
    err = np.abs(got - ref)
    bad = int(np.count_nonzero(err > 2.0**-7 * np.abs(ref) + 2.0**-16 * absref))
    nonfinite = int(np.count_nonzero(~np.isfinite(got)))
    return {"status": "pass" if bad == 0 and nonfinite == 0 else "fail", "bad": bad, "nonfinite": nonfinite, "n": int(err.size)}


# ---------------------------------------------------------------- modes


def evaluate(args) -> dict:
    from .....runtime.launcher import time_launches

    shape = tuple(int(v) for v in args.shape.split(","))
    knobs = json.loads(args.config)
    spec = _spec(shape, knobs)
    record: dict = {"config": knobs, "shape": list(shape), "data": args.data, "kernel": spec.kernel_name()}
    ok, why = is_valid_spec(spec)
    if not ok:
        return {**record, "status": "invalid", "reason": why}
    cache = Path(args.cache)
    hsaco, facts = compile_cached(spec, cache)
    record["code_object"] = facts
    record["lds_bytes"] = spec.lds_bytes
    if facts.get("vgpr_spill_count") or facts.get("private_segment_fixed_size"):
        record["spills"] = True
    ablation = spec.ablate != "none" or spec.dma_probe != "none"
    unchecked = ablation or args.skip_checks
    record["exact_check"] = {"status": "skipped"} if unchecked else exact_check(spec, cache)
    if record["exact_check"]["status"] == "fail":
        return {**record, "status": "wrong"}

    inputs = random_inputs(spec, args.data, cache)
    dev = Device()
    try:
        module, launch, c_ptr = run_case(dev, hsaco, spec, inputs)
        # Under the board power limit the clock settles only after sustained
        # load; time after it has.
        t_end = time.perf_counter() + args.sustain_ms / 1e3
        while time.perf_counter() < t_end:
            for _ in range(8):
                launch()
            dev.rt.sync()
        samples = [time_launches(launch, warmup=args.warmup, iters=args.iters) for _ in range(args.repeats)]
        record["sustain_ms"] = args.sustain_ms
        record["sampled_check"] = (
            sampled_check(spec, lambda: dev.download_16bit(c_ptr, (spec.M, spec.N), spec.dtype_c), inputs)
            if args.data in ("random", "gaussian") and not unchecked
            else {"status": "skipped"}
        )
        module.unload()
    finally:
        dev.close()
    us = sorted(1e3 * s for s in samples)
    flops = 2.0 * spec.M * spec.N * spec.K
    record["us_median"] = us[len(us) // 2]
    record["us_min"] = us[0]
    record["tflops"] = flops / (record["us_median"] * 1e-6) / 1e12
    checks = (record["exact_check"]["status"], record["sampled_check"]["status"])
    if ablation:
        record["status"] = "ablation"
    elif args.skip_checks:
        record["status"] = "timing"
    else:
        record["status"] = "ok" if "fail" not in checks else "wrong"
    return record


def profile(args) -> None:
    shape = tuple(int(v) for v in args.shape.split(","))
    spec = _spec(shape, json.loads(args.config))
    cache = Path(args.cache)
    hsaco, _ = compile_cached(spec, cache)
    dev = Device()
    try:
        module, launch, _ = run_case(dev, hsaco, spec, random_inputs(spec, args.data, cache))
        t_end = time.perf_counter() + args.sustain_ms / 1e3
        while time.perf_counter() < t_end:
            launch()
            dev.rt.sync()
        for _ in range(args.iters):
            launch()
        dev.rt.sync()
        module.unload()
    finally:
        dev.close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--mode", choices=("eval", "profile"), default="eval")
    p.add_argument("--config", default="{}", help="JSON object of TiledBlockScaledGemmSpec knobs")
    p.add_argument("--shape", default="4096,4096,65536")
    p.add_argument("--data", choices=("random", "zero", "gaussian"), default="random",
                   help="uniform random FP6 bytes, zero codes, or OCP-quantized N(0, 1) per K32 block")
    p.add_argument("--cache", required=True, help="HSACO and input cache directory")
    p.add_argument("--out", help="JSON record path (eval mode)")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--sustain-ms", type=int, default=0, help="back-to-back launches before timing or profiling")
    p.add_argument("--skip-checks", action="store_true", help="timing only; status 'timing' (eval mode)")
    args = p.parse_args(argv)
    if args.mode == "profile":
        profile(args)
        return 0
    try:
        record = evaluate(args)
    except Exception as exc:  # noqa: BLE001 - the record carries the failure
        record = {"config": json.loads(args.config), "status": "error", "reason": f"{type(exc).__name__}: {exc}"}
    text = json.dumps(record, indent=2, default=str)
    if args.out:
        Path(args.out).write_text(text)
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
