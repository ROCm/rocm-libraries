# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Quantum-circuit tensor-network monokernel (gfx950, complex64 state, f32 MFMA).

EXPERIMENTAL: not reached by any dispatcher; build it explicitly.

One persistent launch executes a whole contraction path (a gate circuit: every
gate is one extent-2-mode contraction ``psi' = U . psi``). The path is cut into
*phases* (:func:`.circuit_plan.plan_phases`); per phase every workgroup streams
its share of state tiles:

1. gather-load a tile of ``2^t`` floats (re/im + ``t-1`` qubits: the ``c-1``
   lowest qubits as a contiguous ``2^c``-float run, plus ``t-c`` arbitrary
   "extra" qubits addressed by their strides - the permute of a TTGT path
   folded into the load) into LDS;
2. apply every block of the phase on the LDS tile as an f32 MFMA GEMM
   ``out[n_out, g] = sum_kk R[n_out, kk] X[g, kk]`` (``R`` = real embedding of
   the block's ``2^k x 2^k`` complex unitary, ``kk`` = (re/im, block qubits),
   ``g`` = the other tile bits) - intermediates between contractions never leave
   the CU;
3. scatter-store the tile in place.

Between phases a grid barrier (one counter per phase boundary, monotone across
launches via an ``epoch`` argument, so no reset kernel). All workgroups must be
co-resident: size ``grid`` to the resident capacity (:class:`CircuitRunner`
refuses a larger grid when given the CU count). ``mono=False`` builds the same
body as one kernel per phase (no grid barrier).

Knobs (:class:`CircuitSpec`): ``t`` (tile bits), ``c`` (contiguous low bits),
``swizzle`` (LDS XOR swizzle), ``grid`` (persistent workgroups), ``mono``,
``prefetch`` (issue the next tile's global loads before the current tile's MFMA
work; default off), ``rhoist`` (matrices readonly / noalias so LLVM may keep the
fragments in VGPRs across tiles; default off), ``lazy`` (per-block lane
constants recomputed at the block; default on), ``wpe`` (waves-per-EU
attribute), ``bar`` (grid barrier variant), ``nt`` (streaming tile stores).

Kernel ABI: ``(state: f32*, mats: f32*, bar: i32*, epoch: i32)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from rocke.core.ir import F32, I32, IRBuilder, KernelDef, PtrType, TemporalHint

MFMA = "mfma_f32_32x32x2_f32"


@dataclass(frozen=True)
class BlockSpec:
    qubits: Tuple[int, ...]  # k qubits, kk bit (1+i) <-> qubits[i]
    mat_off: int  # offset (in float4 units / 64 lanes) of the packed fragment


@dataclass(frozen=True)
class PhaseSpec:
    extra: Tuple[int, ...]
    blocks: Tuple[BlockSpec, ...]


@dataclass(frozen=True)
class CircuitSpec:
    """One circuit monokernel. ``n`` qubits and ``phases`` are the problem."""

    n: int
    phases: Tuple[PhaseSpec, ...]
    t: int = 13
    c: int = 5
    grid: int = 512
    swizzle: str = "x4"  # none | x4
    mono: bool = True
    prefetch: bool = False
    rhoist: bool = (
        False  # matrices noalias/readonly: LLVM may keep them in VGPRs across tiles
    )
    wpe: int = (
        0  # amdgpu-waves-per-eu lower bound (0 = compiler default); 2 caps VGPRs at 256, 4 at 128
    )
    lazy: bool = (
        True  # per-block lane constants computed at the block (short live ranges) vs hoisted
    )
    # grid barrier: "full" (every workgroup: agent release, atomic arrive, spin, agent acquire),
    # "last" (the last arriver sets a flag that every workgroup spins on), or "xcd" (opt-in):
    # workgroups arrive at a per-XCD counter (XCD = block id % nxcd, the round-robin dispatch),
    # only the last arriver of each XCD issues the agent release, the last XCD sets a flag
    # replicated over `nrep` 4 KiB-strided slots, each workgroup spins on its replica. "xcd"
    # relies on that dispatch mapping and on the other workgroups' stores having reached the
    # XCD's L2 before the release; it is not a guarantee of the memory model, so it is off by
    # default and every use must be checked numerically.
    bar: str = "full"
    nxcd: int = 8
    nrep: int = 8
    nt: bool = False  # streaming (nontemporal) tile stores
    name: str = "rk_qc_mono"

    def __post_init__(self) -> None:
        if self.bar not in ("full", "last", "xcd"):
            raise ValueError(f"bar must be full | last | xcd, got {self.bar!r}")
        if self.swizzle not in ("none", "x4"):
            raise ValueError(f"swizzle must be none | x4, got {self.swizzle!r}")

    @property
    def nthreads(self) -> int:
        return 64 << (self.t - 10)

    @property
    def tiles(self) -> int:
        return 1 << (self.n + 1 - self.t)

    def kernel_name(self, phase: int = -1) -> str:
        tag = "mono" if self.mono else f"ph{phase}"
        return (
            f"{self.name}_n{self.n}_t{self.t}c{self.c}_g{self.grid}_{self.swizzle}"
            f"_p{len(self.phases)}{'_pf' if self.prefetch else ''}{'' if self.rhoist else '_nh'}"
            f"{f'_w{self.wpe}' if self.wpe else ''}{'' if self.lazy else '_hz'}"
            f"{'' if self.bar == 'full' else '_b' + self.bar}{'_nt' if self.nt else ''}_{tag}"
        )


# --------------------------------------------------------------------------- host side
def kk_of(s: int, kin: int) -> int:
    """MFMA (step s, k-in-atom) -> semantic kk; re/im pairs land in steps 2j, 2j+1."""
    return (s & 1) | (kin << 1) | ((s >> 1) << 2)


def pack_block_matrix(R: np.ndarray) -> np.ndarray:
    """Fragment-order packing of the real embedding ``R`` (NK x NK, NK = 32|64).

    Layout ``[mt][q][lane][e]`` (float), ``s = 4q + e``: lane holds
    ``R[mt*32 + lane%32, kk_of(s, lane//32)]``.
    """
    NK = R.shape[0]
    MT, S = NK // 32, NK // 2
    out = np.zeros((MT, S // 4, 64, 4), dtype=np.float32)
    for mt in range(MT):
        for s in range(S):
            for lane in range(64):
                out[mt, s // 4, lane, s % 4] = R[
                    mt * 32 + lane % 32, kk_of(s, lane // 32)
                ]
    return out


def local_bits(spec: CircuitSpec, ph: PhaseSpec) -> List[int]:
    """Local float bit j <-> global float bit L[j]."""
    L = list(range(spec.c)) + [q + 1 for q in ph.extra]
    assert len(L) == spec.t and len(set(L)) == spec.t, (L, spec.t)
    assert all(L[i] < L[i + 1] for i in range(len(L) - 1))
    return L


def validate(spec: CircuitSpec) -> None:
    if spec.t < 11 or spec.t > 14:
        raise ValueError("t must be in 11..14")
    if spec.tiles % spec.grid:
        raise ValueError(f"grid {spec.grid} must divide tiles {spec.tiles}")
    for ph in spec.phases:
        L = local_bits(spec, ph)
        for bl in ph.blocks:
            if len(bl.qubits) not in (4, 5):
                raise ValueError(f"block {bl.qubits}: k must be 4 or 5")
            for q in bl.qubits:
                if q + 1 not in L:
                    raise ValueError(f"block qubit {q} not local in phase {ph.extra}")


# --------------------------------------------------------------------------- device side
def _swz(b: IRBuilder, a, mode: str):
    if mode == "none":
        return a
    # XOR float bits 2..5 with bits 6..9: float4 / float2 runs stay contiguous.
    hi = b.land(b.lshr(a, b.const_i32(6)), b.const_i32(15))
    return b.xor(a, b.shl(hi, b.const_i32(2)))


def _deposit(b: IRBuilder, v, positions: Sequence[int]):
    """sum_i ((v >> i) & 1) << positions[i] (runtime v), merging runs."""
    acc = None
    i = 0
    while i < len(positions):
        j = i
        while (
            j + 1 < len(positions)
            and positions[j + 1] == positions[j] + 1
            and positions[j + 1] - positions[i] == j + 1 - i
        ):
            j += 1
        width = j - i + 1
        field_ = b.land(b.lshr(v, b.const_i32(i)), b.const_i32((1 << width) - 1))
        term = (
            b.shl(field_, b.const_i32(positions[i] - i + i)) if positions[i] else field_
        )
        acc = term if acc is None else b.lor(acc, term)
        i = j + 1
    return acc if acc is not None else b.const_i32(0)


def _deposit_const(v: int, positions: Sequence[int]) -> int:
    r = 0
    for i, p in enumerate(positions):
        r |= ((v >> i) & 1) << p
    return r


def _emit_phase(
    b: IRBuilder,
    spec: CircuitSpec,
    ph: PhaseSpec,
    pidx: int,
    smem,
    state,
    mats,
    wg,
    tid,
    lane,
    wave,
):
    t = spec.t
    L = local_bits(spec, ph)
    NT = spec.nthreads
    nch = 1 << (t - 2)
    cpt = nch // NT  # float4 chunks per thread
    # ---- per-thread chunk offsets (constant across the phase's tiles)
    goffs, loffs = [], []
    for i in range(cpt):
        ch = b.add(tid, b.const_i32(i * NT))
        # local float bits 2..t-1 <- chunk bits 0..t-3
        go = _deposit(b, ch, [L[j] for j in range(2, t)])
        goffs.append(go)
        loffs.append(_swz(b, b.shl(ch, b.const_i32(2)), spec.swizzle))
    # ---- tile base: tile-id bits -> global float bits not in L
    rest = [p for p in range(spec.n + 1) if p not in set(L)]
    tiles_per_wg = spec.tiles // spec.grid

    # ---- per-block lane constants (hoisted before the tile loop, or recomputed per block)
    def block_consts(bl, lazy=False):
        # lazy: an opaque (side-effecting, empty) copy of the lane id per block, so LLVM's LICM
        # cannot hoist the block's LDS addresses out of the tile loop (each hoisted block holds
        # its VGPRs for the whole loop and lowers occupancy). rhoist keeps the R-fragment
        # loads on the plain lane id (hoistable into VGPRs across tiles).
        ln = b.inline_asm("", "=v,0", [lane], I32, sideeffect=True) if lazy else lane
        k = len(bl.qubits)
        NK = 1 << (k + 1)
        MT = NK // 32
        kpos = [0] + [L.index(q + 1) for q in bl.qubits]  # kk bit -> local bit
        gpos = [j for j in range(t) if j not in set(kpos)]  # g bit -> local bit
        mt = b.land(wave, b.const_i32(MT - 1)) if MT > 1 else b.const_i32(0)
        gt = b.lshr(wave, b.const_i32(MT.bit_length() - 1)) if MT > 1 else wave
        g = b.lor(b.shl(gt, b.const_i32(5)), b.land(ln, b.const_i32(31)))
        kin = b.lshr(ln, b.const_i32(5))
        gdep = _deposit(b, g, gpos)
        # B (state) read base: g rows + kk bit 1 (= k_in)
        rbase = b.lor(gdep, b.shl(kin, b.const_i32(kpos[1])))
        # output base: g rows + n_out bit 2 (= lane//32) + bit 5 (= mt)
        obase = b.lor(gdep, b.shl(kin, b.const_i32(kpos[2])))
        if MT > 1:
            obase = b.lor(obase, b.shl(mt, b.const_i32(kpos[5])))
        # R fragment base (float index): ((mat_off + mt*S/4) * 64 + lane) * 4
        S = NK // 2
        rf = b.mul(
            b.add(b.const_i32(bl.mat_off), b.mul(mt, b.const_i32(S // 4))),
            b.const_i32(64),
        )
        rf = b.shl(b.add(rf, lane if spec.rhoist else ln), b.const_i32(2))
        return (k, NK, MT, S, kpos, rbase, obase, rf)

    blk_consts = None if spec.lazy else [block_consts(bl) for bl in ph.blocks]

    def load_tile(tb):
        return [
            b.global_load_vN(state, b.add(tb, goffs[i]), F32, 4) for i in range(cpt)
        ]

    def tile_base(tau):
        return _deposit(b, tau, rest)

    def compute_and_store(vals, tb):
        for i in range(cpt):
            b.smem_store_vN(smem, [loffs[i]], vals[i], 4)
        b.sync()
        for bi, bl in enumerate(ph.blocks):
            if bi:
                b.sched_barrier(
                    0
                )  # keep each block's loads / MFMAs inside its own region
            k, NK, MT, S, kpos, rbase, obase, rf = (
                block_consts(bl, True) if spec.lazy else blk_consts[bi]
            )
            rq = [
                b.global_load_vN(mats, b.add(rf, b.const_i32(q * 256)), F32, 4)
                for q in range(S // 4)
            ]
            acc = b.zero_vec_f32(16)
            for j in range(S // 2):
                off = _deposit_const(j << 2, kpos)
                a2 = _swz(b, b.add(rbase, b.const_i32(off)), spec.swizzle)
                x2 = b.smem_load_vN(smem, a2, dtype=F32, n=2)
                for cc in (0, 1):
                    s = 2 * j + cc
                    av = b.vec_extract(rq[s // 4], s % 4)
                    acc = b.mma(MFMA, av, b.vec_extract(x2, cc), acc)
            if MT > 1:
                b.sync()  # waves sharing rows: all reads done before any write
            for r in range(0, 16, 2):
                noc = 8 * (r // 4) + (r % 4)
                off = _deposit_const(noc, kpos)
                a2 = _swz(b, b.add(obase, b.const_i32(off)), spec.swizzle)
                b.smem_store_vN(
                    smem,
                    [a2],
                    b.vec_pack([b.vec_extract(acc, r), b.vec_extract(acc, r + 1)], F32),
                    2,
                )
            b.sync()
        for i in range(cpt):
            v = b.smem_load_vN(smem, loffs[i], dtype=F32, n=4)
            if spec.nt:
                b.global_store_vN(
                    state,
                    b.add(tb, goffs[i]),
                    v,
                    4,
                    temporal_hint=TemporalHint.STREAMING,
                )
            else:
                b.global_store_vN(state, b.add(tb, goffs[i]), v, 4)
        # next tile overwrites LDS: every wave must have read its chunks
        b.sync()

    c0, c1 = b.const_i32(0), b.const_i32(1)
    if not spec.prefetch:
        with b.scf_for(c0, b.const_i32(tiles_per_wg), c1, iv_name=f"jt{pidx}") as j:
            tau = b.add(wg, b.mul(j, b.const_i32(spec.grid)))
            tb = tile_base(tau)
            vals = load_tile(tb)
            compute_and_store(vals, tb)
    else:
        tb0 = tile_base(wg)
        v0 = load_tile(tb0)
        loop = b.scf_for_iter(
            c0,
            b.const_i32(tiles_per_wg),
            c1,
            [(f"pv{i}_{pidx}", v0[i]) for i in range(cpt)],
            iv_name=f"jp{pidx}",
        )
        with loop as (j, cur):
            tau = b.add(wg, b.mul(j, b.const_i32(spec.grid)))
            tb = tile_base(tau)
            # prefetch the next tile (clamped to the last one: harmless re-read)
            jn = b.add(j, c1)
            jn = b.select(b.cmp_lt(jn, b.const_i32(tiles_per_wg)), jn, j)
            tbn = tile_base(b.add(wg, b.mul(jn, b.const_i32(spec.grid))))
            nxt = load_tile(tbn)
            compute_and_store(list(cur), tb)
            b.scf_yield(*nxt)


def build_circuit_kernels(
    spec: CircuitSpec, *, arch: str = "gfx950"
) -> List[KernelDef]:
    """One KernelDef (``mono``) or one per phase. Launch each with grid
    ``spec.grid`` x ``spec.nthreads`` threads; ``bar`` holds
    ``4096 + nrep * 1024`` i32 (zero before the first launch); ``epoch`` grows
    by one per launch."""
    if arch != "gfx950":
        raise ValueError(f"the circuit monokernel is gfx950-only, got {arch}")
    validate(spec)
    kernels = []
    groups = (
        [list(range(len(spec.phases)))]
        if spec.mono
        else [[p] for p in range(len(spec.phases))]
    )
    for grp in groups:
        b = IRBuilder(spec.kernel_name(grp[0]))
        b.kernel.attrs["max_workgroup_size"] = spec.nthreads
        if spec.wpe:
            b.kernel.attrs["waves_per_eu"] = (spec.wpe, 8)
        state = b.param("state", PtrType(F32, "global"), noalias=spec.rhoist, align=16)
        if spec.rhoist:
            mats = b.param(
                "mats", PtrType(F32, "global"), noalias=True, readonly=True, align=16
            )
        else:
            mats = b.param("mats", PtrType(F32, "global"), align=16)
        bar = b.param("bar", PtrType(I32, "global"), align=4)
        epoch = b.param("epoch", I32)
        smem = b.smem_alloc(F32, [1 << spec.t], name_hint="tile")
        tid = b.thread_id_x()
        wg = b.block_id_x()
        lane = b.land(tid, b.const_i32(63))
        wave = b.readfirstlane(b.lshr(tid, b.const_i32(6)))
        expect = b.mul(b.const_i32(spec.grid), b.add(epoch, b.const_i32(1)))
        nph = len(spec.phases)
        ep1 = b.add(epoch, b.const_i32(1))
        for gi, p in enumerate(grp):
            if gi > 0:
                # grid barrier between phases p-1 and p (counter slot p-1; flag slot P + p-1)
                b.sync()
                with b.scf_if(b.cmp_eq(tid, b.const_i32(0))):
                    if spec.bar != "xcd":
                        b.fence(scope="agent", ordering="release")
                    if spec.bar == "xcd":
                        nx = spec.nxcd
                        per = spec.grid // nx
                        xs = b.add(
                            b.const_i32((p - 1) * nx), b.mod(wg, b.const_i32(nx))
                        )
                        o1 = b.global_atomic_add(bar, xs, b.const_i32(1))
                        with b.scf_if(
                            b.cmp_eq(
                                o1, b.sub(b.mul(b.const_i32(per), ep1), b.const_i32(1))
                            )
                        ):
                            b.fence(scope="agent", ordering="release")
                            o2 = b.global_atomic_add(
                                bar, b.const_i32(nph * nx + p - 1), b.const_i32(1)
                            )
                            with b.scf_if(
                                b.cmp_eq(
                                    o2,
                                    b.sub(b.mul(b.const_i32(nx), ep1), b.const_i32(1)),
                                )
                            ):
                                for r in range(spec.nrep):
                                    b.global_flag_store(
                                        bar,
                                        b.const_i32(4096 + r * 1024 + p - 1),
                                        ep1,
                                        ordering="monotonic",
                                    )
                        rs = b.add(
                            b.const_i32(4096 + p - 1),
                            b.mul(b.mod(wg, b.const_i32(spec.nrep)), b.const_i32(1024)),
                        )
                        b.global_flag_wait_eq(bar, rs, ep1, ordering="monotonic")
                    elif spec.bar == "last":
                        old = b.global_atomic_add(
                            bar, b.const_i32(p - 1), b.const_i32(1)
                        )
                        with b.scf_if(b.cmp_eq(old, b.sub(expect, b.const_i32(1)))):
                            b.global_flag_store(
                                bar,
                                b.const_i32(nph + p - 1),
                                b.add(epoch, b.const_i32(1)),
                                ordering="monotonic",
                            )
                        b.global_flag_wait_eq(
                            bar,
                            b.const_i32(nph + p - 1),
                            b.add(epoch, b.const_i32(1)),
                            ordering="monotonic",
                        )
                    else:
                        b.global_atomic_add(bar, b.const_i32(p - 1), b.const_i32(1))
                        b.global_flag_wait_eq(
                            bar, b.const_i32(p - 1), expect, ordering="monotonic"
                        )
                    b.fence(scope="agent", ordering="acquire")
                b.sync()
                if spec.bar != "xcd":
                    b.fence(scope="agent", ordering="acquire")
            _emit_phase(
                b, spec, spec.phases[p], p, smem, state, mats, wg, tid, lane, wave
            )
        b.ret()
        kernels.append(b.kernel)
    return kernels


# --------------------------------------------------------------------------- host runner
def spec_from_plan(
    n: int,
    phases,
    *,
    t: int,
    c: int,
    grid: int,
    swizzle: str = "x4",
    mono: bool = True,
    prefetch: bool = False,
    rhoist: bool = False,
    wpe: int = 0,
    lazy: bool = True,
    bar: str = "full",
    nt: bool = False,
    name: str = "rk_qc_mono",
):
    """(CircuitSpec, packed f32 matrices) from ``circuit_plan.Phase`` objects."""
    from .circuit_plan import real_embedding

    slabs = []
    off = 0
    pspecs = []
    for ph in phases:
        bspecs = []
        for bl in ph.blocks:
            P = pack_block_matrix(real_embedding(bl.U))
            bspecs.append(BlockSpec(tuple(bl.qubits), off))
            off += P.shape[0] * P.shape[1]  # (mt, q) slabs of 64 lanes x 4 floats
            slabs.append(P.reshape(-1))
        pspecs.append(PhaseSpec(tuple(ph.extra), tuple(bspecs)))
    mats = np.concatenate(slabs) if slabs else np.zeros(256, np.float32)
    spec = CircuitSpec(
        n=n,
        phases=tuple(pspecs),
        t=t,
        c=c,
        grid=grid,
        swizzle=swizzle,
        mono=mono,
        prefetch=prefetch,
        rhoist=rhoist,
        wpe=wpe,
        lazy=lazy,
        bar=bar,
        nt=nt,
        name=name,
    )
    return spec, mats


def occupancy(launcher, nthreads: int) -> int:
    """hipModuleOccupancyMaxActiveBlocksPerMultiprocessor for a loaded kernel."""
    import ctypes

    from rocke.runtime import hip_module

    lib = hip_module._resolve_hip()
    fn = lib.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
    ]
    fn.restype = ctypes.c_int
    nb = ctypes.c_int(0)
    rc = fn(ctypes.byref(nb), launcher._fn.p, nthreads, 0)
    if rc != 0:
        raise RuntimeError(f"occupancy query rc={rc}")
    return nb.value


class CircuitRunner:
    """Compiled kernels + device matrices + barrier counters for one spec."""

    SIG = [
        {"name": "state", "type": "ptr<f32, global>"},
        {"name": "mats", "type": "ptr<f32, global>"},
        {"name": "bar", "type": "ptr<i32, global>"},
        {"name": "epoch", "type": "i32"},
    ]

    def __init__(
        self,
        spec: CircuitSpec,
        mats: np.ndarray,
        device="cuda",
        arch="gfx950",
        n_cu: int = 0,
    ):
        import time

        import torch

        from rocke.helpers.compile import compile_kernel
        from rocke.runtime.launcher import KernelLauncher

        self.spec = spec
        self.mats = torch.as_tensor(mats, device=device)
        assert len(spec.phases) * (spec.nxcd + 1) < 4096 and len(spec.phases) < 1024
        self.bar = torch.zeros(
            4096 + spec.nrep * 1024, dtype=torch.int32, device=device
        )
        self.epoch = 0
        self.launchers = []
        self.compile_s = 0.0
        self.isa_stats = []
        for k in build_circuit_kernels(spec, arch=arch):
            t0 = time.time()
            art = compile_kernel(k, arch=arch)
            self.compile_s += time.time() - t0
            self.launchers.append(
                KernelLauncher(hsaco=art.hsaco, kernel_name=k.name, signature=self.SIG)
            )
        self.occ = occupancy(self.launchers[0], spec.nthreads)
        if spec.mono and n_cu and spec.grid > n_cu * self.occ:
            raise RuntimeError(
                f"grid {spec.grid} > resident capacity {n_cu} x {self.occ}: barrier would deadlock"
            )

    def run(self, state) -> None:
        from rocke.runtime.launcher import LaunchConfig

        cfg = LaunchConfig(
            grid=(self.spec.grid, 1, 1), block=(self.spec.nthreads, 1, 1), fence=False
        )
        for ln in self.launchers:
            ln(
                {
                    "state": state,
                    "mats": self.mats,
                    "bar": self.bar,
                    "epoch": self.epoch,
                },
                config=cfg,
            )
        self.epoch += 1
