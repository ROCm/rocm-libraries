#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Chunkwise KDA prefill sweep on gfx942: fused vs split, over B / H / T / DK / DV.

``dispatch/kda/common.py`` defers the fused/split routing decision because
"there is no measured crossover, so there is no honest threshold to encode".
This harness is the instrument that measures it. It walks token lengths, head
dimensions and batch dimensions, times both paths on every shape, and reports
the parallelism at which the split path starts winning -- the number a future
ranker would need.

Both paths compute the same recurrence and trade HBM traffic against
parallelism:

* **fused** -- one workgroup per (batch, head, V partition) walks that head's
  chunks with the six per-chunk tiles resident in LDS. Minimum traffic, and
  parallelism capped at ``batch * heads * v_partitions``.
* **split** -- ``chunk_prep`` builds those tiles one workgroup per *chunk*,
  fully parallel over the sequence, then ``chunk_scan`` walks them. Pays a tile
  round trip through HBM to buy that parallelism.

So the answer should have the shape "fused wins wherever
``batch * heads * v_partitions`` already fills the device, split wins below
that", with sequence length setting how much work prep can spread.

Torch-free by construction: inputs, packing and the float64 oracle come from
``builders/gfx942/kda/hostpack.py`` (numpy), kernels are selected through
``dispatch.kda`` so this exercises the registered path rather than a private
copy of it, and launches go straight at ``rocke.runtime.hip_module`` with
HIP-event timing. That lets the sweep run on a bare cluster node, and since it
is the same pack the manifest runner binds, the two lanes cannot drift.

Run (needs a gfx942 device)::

    PYTHONPATH=library:platform/python \\
      python library/benchmarks/gfx942/kda/benchmark_chunkwise.py

    # one axis at a time around a different base shape (the default mode)
    ... --base 8,32,4096,128,128

    # full cartesian product instead of per-axis sweeps
    ... --grid --batch 1,8,32 --seqlen 1024,4096

    # explicit shapes, with a correctness gate on the small ones
    ... --shapes 2x4x256,8x8x1024 --verify
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_RK = os.path.abspath(os.path.join(_HERE, "../../../.."))
for _p in (os.path.join(_RK, "platform", "python"), os.path.join(_RK, "library")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from builders.gfx942.kda.hostpack import (  # noqa: E402
    bf16_bits_to_f32,
    make_inputs,
    pack_initial_state,
    pack_v_partitions,
    ref_token_serial,
    unpack_outputs,
)
from dispatch.kda import KDA_PARTITION_HEAD_V, KdaRequest, dispatch_kda  # noqa: E402
from rocke.helpers.compile import compile_kernel  # noqa: E402
from rocke.runtime.host_buffers import as_u8_buffer, nbytes  # noqa: E402
from rocke.runtime.hip_module import (  # noqa: E402
    HipError,
    Runtime,
    get_device_arch,
    get_device_name,
    get_device_num_cus,
)
from rocke.runtime.packing import pack_args  # noqa: E402

ARCH = "gfx942"
GIB = float(2**30)


# ---------------------------------------------------------------------
# shapes
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Shape:
    batch: int
    heads: int
    seqlen: int
    head_k: int
    head_v: int
    chunk: int

    @property
    def label(self) -> str:
        return (
            f"B{self.batch:<3d} H{self.heads:<3d} T{self.seqlen:<5d} "
            f"DK{self.head_k:<4d} DV{self.head_v:<4d} C{self.chunk}"
        )

    @property
    def parts(self) -> int:
        """V partitions per logical head -- gfx942's 64-channel LDS split."""
        return self.head_v // KDA_PARTITION_HEAD_V

    @property
    def workgroups(self) -> int:
        """Independent scan streams, i.e. the whole of the fused path's parallelism."""
        return self.batch * self.heads * self.parts

    @property
    def num_chunks(self) -> int:
        return self.seqlen // self.chunk

    @property
    def tiles(self) -> int:
        """Chunk tiles across every partitioned stream -- the prep grid."""
        return self.workgroups * self.num_chunks

    def request(self, *, has_initial_state: bool, algorithm: str) -> KdaRequest:
        return KdaRequest(
            batch=self.batch,
            num_heads=self.heads,
            seqlen=self.seqlen,
            arch=ARCH,
            head_k=self.head_k,
            head_v=self.head_v,
            chunk_size=self.chunk,
            algorithm=algorithm,
            has_initial_state=has_initial_state,
        )


def _ints(s: str) -> List[int]:
    return [int(x) for x in s.replace(",", " ").split()]


def _dedupe(shapes: Sequence[Shape]) -> List[Shape]:
    seen, out = set(), []
    for shape in shapes:
        if shape not in seen:
            seen.add(shape)
            out.append(shape)
    return out


def build_shapes(args) -> List[Shape]:
    """Per-axis sweeps around a base shape, a full cartesian, or explicit triples.

    Per-axis is the default because it answers the question the crossover needs
    -- how one dimension moves the fused/split balance with the others pinned --
    and a cartesian over the same lists is hundreds of compile-and-time cycles
    for a table nobody reads.
    """
    batch, heads, seqlen, head_k, head_v = args.base
    base = Shape(batch, heads, seqlen, head_k, head_v, args.chunk)

    if args.shapes:
        explicit = []
        for triple in args.shapes.replace(",", " ").split():
            b, h, t = (int(x) for x in triple.lower().split("x"))
            explicit.append(replace(base, batch=b, heads=h, seqlen=t))
        return _dedupe(explicit)

    axes = {
        "seqlen": args.seqlen,
        "batch": args.batch,
        "heads": args.heads,
        "head_k": args.head_k,
        "head_v": args.head_v,
    }
    if args.grid:
        return _dedupe(
            [
                Shape(b, h, t, dk, dv, args.chunk)
                for b in axes["batch"]
                for h in axes["heads"]
                for t in axes["seqlen"]
                for dk in axes["head_k"]
                for dv in axes["head_v"]
            ]
        )

    swept = [base]
    for field, values in axes.items():
        swept.extend(replace(base, **{field: value}) for value in values)
    return _dedupe(swept)


# ---------------------------------------------------------------------
# traffic and work models
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Traffic:
    """Bytes crossing HBM, split into what the problem needs and what is paid.

    ``essential`` is a property of the problem alone, so it is the denominator
    that makes the two paths comparable. ``fused`` differs from it only by the
    duplicated q/k/g/beta reads the V partition forces, and ``split`` adds the
    tile round trip it exists to pay for.
    """

    essential: float
    fused: float
    tile: float

    @property
    def split(self) -> float:
        return self.fused + 2.0 * self.tile


def traffic_of(shape: Shape, *, has_initial_state: bool) -> Traffic:
    tokens = float(shape.batch * shape.heads * shape.seqlen)
    dk, dv, chunk = shape.head_k, shape.head_v, shape.chunk
    qk = 2.0 * tokens * dk * 2  # q and k, bf16
    gate = tokens * dk * 4  # per-channel log decay, f32
    beta = tokens * 4
    v = tokens * dv * 2
    o = tokens * dv * 2
    state = float(shape.batch * shape.heads * dk * dv * 4)  # ht, f32
    h0 = state if has_initial_state else 0.0

    essential = qk + gate + beta + v + o + state + h0
    # q/k/g/beta are re-read once per V partition; v and o are carved up between
    # the partitions, so their totals do not change.
    fused = shape.parts * (qk + gate + beta) + v + o + state + h0

    # Per chunk: a and aqk (C x C), gk and gq (C x DK), kt (DK x C) in bf16,
    # plus the f32 DK decay row. Written once by prep, read once by scan.
    tile_elems = 2 * chunk * chunk + 2 * chunk * dk + dk * chunk
    tile = float(shape.tiles) * (tile_elems * 2 + dk * 4)
    return Traffic(essential=essential, fused=fused, tile=tile)


def flops_of(shape: Shape) -> float:
    """The two dominant ``DK x DV`` products per token, at 2 flops each.

    Matches the count the KDA manifest reports, so a number from this harness
    and one from ``rocke.run_manifest`` mean the same thing. It undercounts the
    gate and the rank-1 write, deliberately: this recurrence is bandwidth and
    latency bound, so the flop rate is context rather than the headline.
    """
    return 4.0 * shape.batch * shape.heads * shape.seqlen * shape.head_k * shape.head_v


# ---------------------------------------------------------------------
# compile + launch
# ---------------------------------------------------------------------


@dataclass
class Compiled:
    kernel_name: str
    fn: object
    grid: Tuple[int, int, int]
    block: Tuple[int, int, int]
    signature: Tuple[dict, ...]


class KernelCache:
    """Compile each distinct spec once, and keep every module loaded.

    The spec depends only on head_k, chunk and the state flags, so a whole
    batch/seqlen sweep reuses one code object per path. The grid does depend on
    the request, so only the function handle is cached. Modules are retained
    for the process lifetime because unloading one invalidates the handles
    taken from it.
    """

    def __init__(self, rt: Runtime) -> None:
        self._rt = rt
        self._modules: List[object] = []
        self._fns: Dict[str, object] = {}

    def get(self, shape: Shape, algorithm: str, *, has_initial_state: bool) -> Compiled:
        req = shape.request(has_initial_state=has_initial_state, algorithm=algorithm)
        result = dispatch_kda(req)
        name = result.spec.kernel_name()
        if name not in self._fns:
            artifact = compile_kernel(
                result.build(),
                arch=ARCH,
                backend="python",
                capture_ir_text=False,
            )
            module = self._rt.load_module(artifact.hsaco)
            self._modules.append(module)
            self._fns[name] = module.get_function(artifact.kernel_name)
        return Compiled(
            kernel_name=name,
            fn=self._fns[name],
            grid=result.grid,
            block=result.block,
            signature=result.signature,
        )


def time_launches(
    rt: Runtime, fire: Callable[[], None], *, warmup: int, iters: int
) -> float:
    """Mean ms per iteration, measured with HIP events on the default stream.

    Deliberately not ``rocke.runtime.launcher.time_launches``: that one needs
    torch. This is the same loop with the same shape -- warm, drain, then
    bracket the timed region with two events -- so the measurement never
    charges a host round trip per iteration, which at the short sequences this
    sweep exists to probe would be most of the number.
    """
    for _ in range(warmup):
        fire()
    rt.sync()
    start, stop = rt.event(), rt.event()
    start.record()
    for _ in range(iters):
        fire()
    stop.record()
    stop.synchronize()
    ms = start.elapsed_to(stop) / float(iters)
    start.destroy()
    stop.destroy()
    rt.sync()
    return ms


# ---------------------------------------------------------------------
# device-resident problem
# ---------------------------------------------------------------------


class Problem:
    """One shape's inputs packed into V partitions and resident on the device.

    Allocated once and shared by both paths, so the fused and split timings see
    byte-identical inputs and any difference between them cannot be an artifact
    of the data. Outputs are re-zeroed before a verification run and never
    between timed iterations, since the kernels overwrite them.
    """

    def __init__(
        self, rt: Runtime, shape: Shape, *, has_initial_state: bool, seed: int
    ) -> None:
        self._rt = rt
        self.shape = shape
        self.ptrs: List[int] = []
        self._oracle: Optional[Tuple[np.ndarray, np.ndarray]] = None
        try:
            self._allocate(shape, has_initial_state=has_initial_state, seed=seed)
        except Exception:
            # A big shape can exhaust device memory partway through. Hand the
            # already-acquired buffers back before propagating, so the sweep can
            # record the failure and keep going instead of bleeding the device.
            self.free()
            raise

    def _allocate(self, shape: Shape, *, has_initial_state: bool, seed: int) -> None:
        self._dense = make_inputs(
            shape.batch,
            shape.heads,
            shape.seqlen,
            shape.head_k,
            shape.head_v,
            seed=seed,
        )
        q, k, v, g, beta = self._dense
        packed = pack_v_partitions(shape.chunk, KDA_PARTITION_HEAD_V, q, k, v, g, beta)
        self.packed = packed

        num_tiles = packed.bh * packed.parts * packed.nc
        self.o_host = np.zeros(
            (num_tiles, shape.chunk * KDA_PARTITION_HEAD_V), dtype=np.uint16
        )
        self.ht_host = np.zeros(
            (packed.bh * packed.parts, KDA_PARTITION_HEAD_V, shape.head_k),
            dtype=np.float32,
        )

        self.h0_dense = None
        h0_packed = None
        if has_initial_state:
            rng = np.random.default_rng(seed + 7)
            self.h0_dense = np.float32(0.1) * rng.standard_normal(
                (shape.batch, shape.heads, shape.head_k, shape.head_v),
                dtype=np.float32,
            )
            h0_packed = pack_initial_state(
                KDA_PARTITION_HEAD_V, self.h0_dense, packed.parts
            )

        self.dev = {
            "q": self._upload(packed.q),
            "k": self._upload(packed.k),
            "g": self._upload(packed.g),
            "beta": self._upload(packed.beta),
            "v": self._upload(packed.v),
            "o": self._upload(self.o_host),
            "ht": self._upload(self.ht_host),
        }
        # A kernel built without an initial state never reads h0_ptr, but the
        # ABI still has the slot; aim it at ht the way the host builders do.
        self.dev["h0"] = (
            self.dev["ht"] if h0_packed is None else self._upload(h0_packed)
        )

        # The six per-chunk tiles. Only the split path touches them, but they
        # are allocated up front so allocation never lands inside a timed loop.
        chunk, dk = shape.chunk, shape.head_k
        self.tiles = {
            "a": self._alloc(num_tiles * chunk * chunk * 2),
            "gk": self._alloc(num_tiles * chunk * dk * 2),
            "gq": self._alloc(num_tiles * chunk * dk * 2),
            "aqk": self._alloc(num_tiles * chunk * chunk * 2),
            "kt": self._alloc(num_tiles * dk * chunk * 2),
            "dec": self._alloc(num_tiles * dk * 4),
        }

    def _alloc(self, size: int) -> int:
        ptr = self._rt.alloc(size)
        self.ptrs.append(ptr)
        return ptr

    def _upload(self, host: np.ndarray) -> int:
        ptr = self._alloc(nbytes(host))
        self._rt.memcpy_h2d(ptr, as_u8_buffer(host), nbytes(host))
        return ptr

    @property
    def scale(self) -> float:
        return float(self.shape.head_k) ** -0.5

    def free(self) -> None:
        for ptr in self.ptrs:
            self._rt.free(ptr)
        self.ptrs.clear()

    def clear_outputs(self) -> None:
        self._rt.memset(self.dev["o"], 0, nbytes(self.o_host))
        self._rt.memset(self.dev["ht"], 0, nbytes(self.ht_host))

    def oracle(self) -> Tuple[np.ndarray, np.ndarray]:
        """The float64 token-serial reference, computed once and reused."""
        if self._oracle is None:
            q, k, v, g, beta = self._dense
            self._oracle = ref_token_serial(
                bf16_bits_to_f32(q),
                bf16_bits_to_f32(k),
                bf16_bits_to_f32(v),
                g,
                beta,
                self.scale,
                h0=self.h0_dense,
            )
        return self._oracle

    def worst_relative_error(self) -> float:
        """Read back ``o`` and ``ht`` and score both against the oracle.

        Same criterion as the host builders: worst absolute error over the
        reference's own magnitude, so both lanes agree on what passing means
        for a bf16 kernel.
        """
        rt, shape, packed = self._rt, self.shape, self.packed
        rt.memcpy_d2h(as_u8_buffer(self.o_host), self.dev["o"], nbytes(self.o_host))
        rt.memcpy_d2h(as_u8_buffer(self.ht_host), self.dev["ht"], nbytes(self.ht_host))
        o_bits, ht_got = unpack_outputs(
            chunk=shape.chunk,
            head_k=shape.head_k,
            head_v=KDA_PARTITION_HEAD_V,
            o=self.o_host,
            ht=self.ht_host,
            B=shape.batch,
            H=shape.heads,
            T=shape.seqlen,
            DV=shape.head_v,
            NC=packed.nc,
            parts=packed.parts,
        )
        o_ref, s_ref = self.oracle()
        worst = 0.0
        for got, ref in (
            (bf16_bits_to_f32(o_bits).astype(np.float64), o_ref),
            (ht_got.astype(np.float64), s_ref),
        ):
            err = float(np.abs(got - ref).max()) if got.size else 0.0
            den = max(float(np.abs(ref).max()) if ref.size else 0.0, 1e-30)
            worst = max(worst, err / den)
        return worst


# ---------------------------------------------------------------------
# the two paths
# ---------------------------------------------------------------------


def fused_fire(rt: Runtime, problem: Problem, kern: Compiled) -> Callable[[], None]:
    args = pack_args(
        kern.signature,
        {
            "q_ptr": problem.dev["q"],
            "k_ptr": problem.dev["k"],
            "g_ptr": problem.dev["g"],
            "beta_ptr": problem.dev["beta"],
            "v_ptr": problem.dev["v"],
            "o_ptr": problem.dev["o"],
            "h0_ptr": problem.dev["h0"],
            "ht_ptr": problem.dev["ht"],
            "scale": problem.scale,
            "nc": problem.packed.nc,
        },
    )

    def fire() -> None:
        rt.launch(kern.fn, kern.grid, kern.block, args)

    return fire


def split_fires(
    rt: Runtime, problem: Problem, prep: Compiled, scan: Compiled
) -> Tuple[Callable[[], None], Callable[[], None]]:
    """Prep and scan as two closures over one stream.

    No fence between them: stream FIFO order already makes the scan see the
    tiles prep wrote, which is how the pair runs in a pipeline.
    """
    tiles = problem.tiles
    prep_args = pack_args(
        prep.signature,
        {
            "q_ptr": problem.dev["q"],
            "k_ptr": problem.dev["k"],
            "g_ptr": problem.dev["g"],
            "beta_ptr": problem.dev["beta"],
            "a_ptr": tiles["a"],
            "gk_ptr": tiles["gk"],
            "gq_ptr": tiles["gq"],
            "aqk_ptr": tiles["aqk"],
            "kt_ptr": tiles["kt"],
            "dec_ptr": tiles["dec"],
            "scale": problem.scale,
        },
    )
    scan_args = pack_args(
        scan.signature,
        {
            "a_ptr": tiles["a"],
            "gk_ptr": tiles["gk"],
            "gq_ptr": tiles["gq"],
            "aqk_ptr": tiles["aqk"],
            "kt_ptr": tiles["kt"],
            "dec_ptr": tiles["dec"],
            "v_ptr": problem.dev["v"],
            "o_ptr": problem.dev["o"],
            "h0_ptr": problem.dev["h0"],
            "ht_ptr": problem.dev["ht"],
            "nc": problem.packed.nc,
        },
    )

    def fire_prep() -> None:
        rt.launch(prep.fn, prep.grid, prep.block, prep_args)

    def fire_scan() -> None:
        rt.launch(scan.fn, scan.grid, scan.block, scan_args)

    return fire_prep, fire_scan


# ---------------------------------------------------------------------
# per-shape measurement
# ---------------------------------------------------------------------


def _rate(bytes_moved: float, ms: float) -> float:
    return bytes_moved / (ms * 1e-3) / GIB if ms > 0 else 0.0


def _tflops(shape: Shape, ms: float) -> float:
    return flops_of(shape) / (ms * 1e-3) / 1e12 if ms > 0 else 0.0


def measure(rt: Runtime, cache: KernelCache, shape: Shape, args) -> dict:
    """Time the requested paths on one shape.

    Returns a record; a shape the dispatcher refuses comes back carrying an
    ``error`` instead of timings, because a validator saying no is a result
    about coverage and not a reason to abandon the sweep.
    """
    traffic = traffic_of(shape, has_initial_state=args.initial_state)
    rec: dict = {
        "shape": {
            "batch": shape.batch,
            "heads": shape.heads,
            "seqlen": shape.seqlen,
            "head_k": shape.head_k,
            "head_v": shape.head_v,
            "chunk": shape.chunk,
        },
        "workgroups": shape.workgroups,
        "chunks_per_stream": shape.num_chunks,
        "tiles": shape.tiles,
        "bytes": {
            "essential": traffic.essential,
            "fused_actual": traffic.fused,
            "split_actual": traffic.split,
            "tile_round_trip": 2.0 * traffic.tile,
        },
    }

    want_fused = args.only in ("both", "fused")
    want_split = args.only in ("both", "split")

    kernels: Dict[str, Compiled] = {}
    try:
        if want_fused:
            kernels["fused"] = cache.get(
                shape, "chunk_fused", has_initial_state=args.initial_state
            )
        if want_split:
            kernels["prep"] = cache.get(
                shape, "chunk_prep", has_initial_state=args.initial_state
            )
            kernels["scan"] = cache.get(
                shape, "chunk_scan", has_initial_state=args.initial_state
            )
    except (ValueError, NotImplementedError) as exc:
        rec["error"] = str(exc)
        return rec

    # The oracle is a python-level token loop, so it costs far more than the
    # kernel it checks. Gate it on the work it implies rather than on shape.
    oracle_work = float(
        shape.batch * shape.heads * shape.seqlen * shape.head_k * shape.head_v
    )
    do_verify = args.verify and oracle_work <= args.verify_budget
    if args.verify and not do_verify:
        rec["verify_skipped"] = "oracle cost over --verify-budget"

    try:
        problem = Problem(
            rt, shape, has_initial_state=args.initial_state, seed=args.seed
        )
    except (HipError, MemoryError) as exc:
        rec["error"] = f"could not stage the problem: {exc}"
        return rec

    try:
        if want_fused:
            fire = fused_fire(rt, problem, kernels["fused"])
            if do_verify:
                problem.clear_outputs()
                fire()
                rt.sync()
                rec["fused_rel_err"] = problem.worst_relative_error()
            ms = time_launches(rt, fire, warmup=args.warmup, iters=args.iters)
            rec["fused"] = {
                "ms": ms,
                "gibps_essential": _rate(traffic.essential, ms),
                "gibps_actual": _rate(traffic.fused, ms),
                "tflops": _tflops(shape, ms),
                "kernel": kernels["fused"].kernel_name,
            }

        if want_split:
            fire_prep, fire_scan = split_fires(
                rt, problem, kernels["prep"], kernels["scan"]
            )

            def fire_both() -> None:
                fire_prep()
                fire_scan()

            if do_verify:
                problem.clear_outputs()
                fire_both()
                rt.sync()
                rec["split_rel_err"] = problem.worst_relative_error()
            ms = time_launches(rt, fire_both, warmup=args.warmup, iters=args.iters)
            # The halves are timed alone as well because they answer different
            # questions: prep is bandwidth against a roofline, scan is a serial
            # chain whose only lever is occupancy. Timing prep first leaves the
            # tiles valid for the scan-only loop.
            prep_ms = time_launches(rt, fire_prep, warmup=args.warmup, iters=args.iters)
            scan_ms = time_launches(rt, fire_scan, warmup=args.warmup, iters=args.iters)
            rec["split"] = {
                "ms": ms,
                "prep_ms": prep_ms,
                "scan_ms": scan_ms,
                "gibps_essential": _rate(traffic.essential, ms),
                "gibps_actual": _rate(traffic.split, ms),
                "tflops": _tflops(shape, ms),
                "prep_kernel": kernels["prep"].kernel_name,
                "scan_kernel": kernels["scan"].kernel_name,
            }
    finally:
        problem.free()
        rt.sync()

    fused_ms = rec.get("fused", {}).get("ms")
    split_ms = rec.get("split", {}).get("ms")
    if fused_ms and split_ms:
        rec["split_speedup"] = fused_ms / split_ms
        rec["winner"] = "split" if split_ms < fused_ms else "fused"
    return rec


# ---------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------


def _gm(values: Sequence[float]) -> float:
    vals = [v for v in values if v and v > 0]
    if not vals:
        return float("nan")
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def print_record(rec: dict, shape: Shape, tol: float) -> None:
    if "error" in rec:
        print(f"  {shape.label}  SKIP  {rec['error']}")
        return

    cells = []
    for path in ("fused", "split"):
        info = rec.get(path)
        if not info:
            continue
        cell = (
            f"{path}={info['ms']:.4f}ms "
            f"{info['gibps_essential']:6.1f}GiB/s {info['tflops']:5.1f}TF"
        )
        err = rec.get(f"{path}_rel_err")
        if err is not None:
            cell += f" [{'ok' if err <= tol else f'REL {err:.1e}'}]"
        cells.append(cell)

    verdict = ""
    if "split_speedup" in rec:
        factor = max(rec["split_speedup"], 1.0 / rec["split_speedup"])
        verdict = f"  -> {rec['winner']} by {factor:.2f}x"
    print(f"  {shape.label}  wg={rec['workgroups']:<5d} " + "  ".join(cells) + verdict)

    split = rec.get("split")
    if split:
        print(
            f"      split halves: prep={split['prep_ms']:.4f}ms "
            f"scan={split['scan_ms']:.4f}ms  "
            f"tile round trip {rec['bytes']['tile_round_trip'] / GIB:.2f} GiB"
        )


def print_summary(records: List[dict], num_cus: Optional[int]) -> None:
    timed = [r for r in records if "split_speedup" in r]
    if not timed:
        print("\nNo shape ran both paths, so there is no crossover to report.")
        return

    cu_note = f", device has {num_cus} CUs" if num_cus else ""
    print("\n=== fused vs split by parallelism ===")
    print(f"  workgroups = batch * heads * v_partitions{cu_note}")
    header = (
        f"  {'wg':>6}  {'chunks':>7}  {'fused ms':>9}  "
        f"{'split ms':>9}  {'split/fused':>11}  winner"
    )
    print(header)
    for rec in sorted(timed, key=lambda r: (r["workgroups"], r["chunks_per_stream"])):
        ratio = rec["split"]["ms"] / rec["fused"]["ms"]
        print(
            f"  {rec['workgroups']:>6}  {rec['chunks_per_stream']:>7}  "
            f"{rec['fused']['ms']:>9.4f}  {rec['split']['ms']:>9.4f}  "
            f"{ratio:>11.2f}  {rec['winner']}"
        )

    ratios = [r["split"]["ms"] / r["fused"]["ms"] for r in timed]
    wins = [r for r in timed if r["winner"] == "split"]
    print(
        f"\n  split wins on {len(wins)}/{len(timed)} shapes; "
        f"geomean split/fused = {_gm(ratios):.2f}x"
    )
    if not wins:
        print(
            "  fused wins everywhere measured -- the tile round trip is never "
            "repaid at these shapes, which is why fused is the dispatch default."
        )
    elif len(wins) == len(timed):
        print(
            "  split wins everywhere measured -- widen the sweep upward before "
            "reading a threshold out of this."
        )
    else:
        boundary = max(r["workgroups"] for r in wins)
        print(
            f"  crossover: split wins at up to {boundary} workgroups and loses "
            "above it. That is the threshold dispatch/kda currently declines to "
            "encode; check it holds across seqlen before encoding it."
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--base",
        type=_ints,
        default=[4, 16, 2048, 128, 128],
        help="B,H,T,DK,DV held fixed while each axis is swept "
        "(default 4,16,2048,128,128)",
    )
    ap.add_argument("--seqlen", type=_ints, default=[512, 1024, 2048, 4096, 8192])
    ap.add_argument("--batch", type=_ints, default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--heads", type=_ints, default=[4, 8, 16, 32])
    ap.add_argument("--head-k", type=_ints, default=[64, 128])
    ap.add_argument("--head-v", type=_ints, default=[64, 128, 256])
    ap.add_argument("--chunk", type=int, default=16, choices=(16, 32))
    ap.add_argument(
        "--grid",
        action="store_true",
        help="cartesian product of every axis instead of one axis at a time",
    )
    ap.add_argument(
        "--shapes",
        default=None,
        help="explicit BxHxT triples, e.g. 2x4x256,8x8x1024 (DK/DV come from --base)",
    )
    ap.add_argument(
        "--only",
        choices=("both", "fused", "split"),
        default="both",
        help="restrict to one path; the default times both, which is what "
        "yields a crossover",
    )
    ap.add_argument(
        "--initial-state",
        action="store_true",
        help="pass a nonzero h0, exercising the has_initial_state spec variant",
    )
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--verify",
        action="store_true",
        help="check each path against the float64 token-serial oracle first",
    )
    ap.add_argument(
        "--verify-budget",
        type=float,
        default=2e8,
        help="skip verification when B*H*T*DK*DV exceeds this; the oracle is a "
        "python-level token loop, far slower than the kernel (default 2e8)",
    )
    ap.add_argument("--tol", type=float, default=3e-2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output-json", type=Path, default=None)
    args = ap.parse_args()

    if len(args.base) != 5:
        print("--base expects five ints: B,H,T,DK,DV", file=sys.stderr)
        return 2

    device_arch = get_device_arch()
    if device_arch is None:
        print("no HIP device found", file=sys.stderr)
        return 1
    if not device_arch.startswith(ARCH):
        print(
            f"these kernels are {ARCH}-only and this device is {device_arch}; "
            "a code object built for another arch fails to load at launch",
            file=sys.stderr,
        )
        return 1

    num_cus = get_device_num_cus()
    print(f"device: {get_device_name()}  arch: {device_arch}  CUs: {num_cus}")

    shapes = build_shapes(args)
    if args.limit is not None:
        shapes = shapes[: args.limit]
    print(
        f"shapes: {len(shapes)}  paths: {args.only}  chunk: {args.chunk}  "
        f"warmup/iters: {args.warmup}/{args.iters}"
    )

    rt = Runtime()
    cache = KernelCache(rt)
    records: List[dict] = []
    failures = 0
    for i, shape in enumerate(shapes, 1):
        print(f"[{i}/{len(shapes)}]")
        rec = measure(rt, cache, shape, args)
        records.append(rec)
        print_record(rec, shape, args.tol)
        for path in ("fused", "split"):
            err = rec.get(f"{path}_rel_err")
            if err is not None and err > args.tol:
                failures += 1

    print_summary(records, num_cus)

    if args.output_json:
        args.output_json.write_text(json.dumps(records, indent=2, default=str))
        print(f"\nwrote {args.output_json}  ({len(records)} shapes)")

    if failures:
        print(f"\n{failures} path/shape pairs exceeded tol={args.tol:.1e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
