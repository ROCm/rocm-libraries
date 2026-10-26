# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests of the backward stage-scheduler table (``_attention_bwd_sched``).

CPU part (stage-table counts):

* MMA instructions per GEMM per wave equal a brute-force enumeration of each
  wave's 16x16 output tiles times its instruction K steps, for the start-point
  tiles of every arch and a grid of legal geometries; G1/G3 come from their own
  atom (not G0's), and gfx942 counts a K = 32 step as two instructions.
* A synthetic tile step is emitted operation by operation from that
  enumeration (one MMA per tile and K step, one LDS read per fragment piece,
  the XT writer of the LDS planner, resident K / V fragments read from LDS,
  LDS dK / dV accumulators read and written around their MMA chains, ...),
  with the stage table's hints
  between the stages. Per stage, the IR op counts by class equal the table's
  counts, and the ``sched_group_barrier`` groups sum to the same counts.

Compile part (instruction counts): the synthetic tile steps are lowered and
compiled with comgr for every arch, and the ``v_mfma`` / ``v_wmma``
instructions of the code object equal the model, per GEMM and per step; the
LLVM IR carries exactly the modelled number of ``sched.group.barrier`` and
``sched.barrier`` calls. These need comgr (and ``llvm-objdump`` for the
disassembly) and skip otherwise.
"""

from __future__ import annotations

import dataclasses
import glob
import itertools
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest
from rocke.core.ir import F16, F32, IRBuilder, PtrType
from rocke.helpers.compile import COMGR_FRESH_PROCESS_ATTR
from rocke.helpers.schedule import DS_READ, DS_WRITE, MFMA, VMEM_READ, VMEM_WRITE

from kernels.common._attention_bwd_lds import default_xt_writer
from kernels.common._attention_bwd_sched import (
    STAGES,
    StageTable,
    instruction_k,
    interleave_groups,
    native_atom_ks,
)

# Per-arch start-point tiles, as stage-table parameters.
_942 = {"arch": "gfx942", "waves": 4}
_950 = {"arch": "gfx950", "waves": 4}
CONFIGS: list[tuple[str, dict]] = [
    (
        "942-d32",
        dict(
            _942,
            head_size=32,
            k_m0=32,
            k_n0=128,
            k_k4=64,
            grid_g4=(2, 2),
            atom_g02=32,
            atom_g4=32,
        ),
    ),
    (
        "942-d64",
        dict(_942, head_size=64, k_m0=32, k_n0=128, k_k4=32, atom_g02=32, atom_g4=32),
    ),
    (
        "942-d128",
        dict(_942, head_size=128, k_m0=16, k_n0=128, k_k4=32, atom_g02=32, atom_g4=32),
    ),
    (
        "942-sv1",
        dict(
            _942,
            head_size=128,
            k_m0=16,
            k_n0=64,
            k_k4=32,
            stage_vec=1,
            sched="fences_only",
        ),
    ),
    (
        "942-debug",
        {
            "arch": "gfx942",
            "waves": 1,
            "head_size": 64,
            "k_m0": 16,
            "k_n0": 16,
            "k_k4": 16,
            "transpose_source": "lds_plain",
            "kt_source": "lds",
            "sched": "none",
        },
    ),
    (
        "950-d32",
        dict(
            _950,
            head_size=32,
            k_m0=32,
            k_n0=128,
            k_k4=64,
            grid_g4=(2, 2),
            atom_g02=32,
            atom_g4=32,
        ),
    ),
    (
        "950-d64",
        dict(
            _950,
            head_size=64,
            k_m0=32,
            k_n0=256,
            k_k4=32,
            atom_g02=32,
            atom_g13=32,
            atom_g4=32,
            transpose_source="tr_read",
            kt_source="lds",
            global_path="dma",
        ),
    ),
    (
        "950-d128",
        dict(
            _950,
            head_size=128,
            k_m0=16,
            k_n0=192,
            k_k4=32,
            atom_g02=32,
            atom_g4=32,
            transpose_source="tr_read",
            kt_source="lds",
            global_path="dma",
            sched="stage_table_atomics",
            setprio=True,
        ),
    ),
    (
        "950-d128-short",
        dict(
            _950,
            head_size=128,
            k_m0=32,
            k_n0=128,
            k_k4=32,
            atom_g02=32,
            atom_g13=32,
            atom_g4=32,
            transpose_source="tr_read",
            kt_source="lds",
            global_path="dma",
        ),
    ),
    (
        "950-sv1",
        dict(
            _950,
            head_size=128,
            k_m0=16,
            k_n0=64,
            k_k4=32,
            atom_g02=32,
            atom_g4=32,
            stage_vec=1,
            transpose_source="lds_plain",
            sched="fences_only",
        ),
    ),
    *[
        (
            f"{arch[3:]}-debug-d{d}",
            {
                "arch": arch,
                "waves": 1,
                "head_size": d,
                "k_m0": 16,
                "k_n0": 16,
                "k_k4": 16,
                "transpose_source": "wmma_lds",
                "kt_source": "lds",
                "sched": "none",
            },
        )
        for arch in ("gfx1151", "gfx1201")
        for d in (32, 64, 128)
    ],
    # Resident K / V in LDS and LDS dK / dV accumulators under a stage table.
    (
        "942-d128-kvlds-acclds",
        dict(
            _942,
            head_size=128,
            k_m0=16,
            k_n0=128,
            k_k4=32,
            atom_g02=32,
            atom_g4=32,
            kt_source="lds",
            kv_residency="lds",
            acc_in_lds=True,
        ),
    ),
    (
        "942-d64-kvlds",
        dict(
            _942,
            head_size=64,
            k_m0=32,
            k_n0=128,
            k_k4=32,
            kt_source="lds",
            kv_residency="lds",
            sched="stage_table_atomics",
        ),
    ),
    (
        "950-d128-acclds",
        dict(
            _950,
            head_size=128,
            k_m0=16,
            k_n0=192,
            k_k4=32,
            atom_g02=32,
            atom_g4=32,
            transpose_source="tr_read",
            kt_source="lds",
            global_path="dma",
            acc_in_lds=True,
        ),
    ),
    *[
        (
            f"{arch[3:]}-kvlds-acclds",
            {
                "arch": arch,
                "waves": 2,
                "head_size": 64,
                "k_m0": 16,
                "k_n0": 64,
                "k_k4": 32,
                "transpose_source": "wmma_lds",
                "kt_source": "lds",
                "kv_residency": "lds",
                "acc_in_lds": True,
            },
        )
        for arch in ("gfx1151", "gfx1201")
    ],
]
CONFIG_IDS = [c[0] for c in CONFIGS]


def _table(kw: dict) -> StageTable:
    """The reference (CK) model: the synthetic step emits the reference
    pipeline's work."""
    return StageTable(**{"pipeline": "ck", **kw})


# ---------------------------------------------------------------------------
# Brute-force enumeration of the tile step
# ---------------------------------------------------------------------------


def _wave_tiles(
    rows: int, cols: int, grid: tuple[int, int], wave: int
) -> list[tuple[int, int]]:
    """16x16 output tiles owned by ``wave`` (row-major wave index in the grid)."""
    gr, gc = grid
    wr, wc = divmod(wave, gc)
    rb, cb = rows // gr, cols // gc
    return [
        (r, c)
        for r in range(wr * rb, (wr + 1) * rb, 16)
        for c in range(wc * cb, (wc + 1) * cb, 16)
    ]


@dataclass
class _Plan:
    """Per-GEMM tile and fragment enumeration of one wave."""

    tiles: dict[str, list[tuple[int, int]]]
    ksteps: dict[str, int]


def _enumerate(t: StageTable, wave: int) -> _Plan:
    w, d, m0, n0 = t.waves, t.head_size, t.k_m0, t.k_n0
    ik = t.instr_k
    tiles = {
        "g0": _wave_tiles(m0, n0, (1, w), wave),
        "g2": _wave_tiles(m0, n0, (1, w), wave),
        "g1": _wave_tiles(n0, d, (w, 1), wave),
        "g3": _wave_tiles(n0, d, (w, 1), wave),
        "g4": _wave_tiles(m0, d, t.grid_g4, wave) if t.dq_mode == "atomic" else [],
    }
    ksteps = {
        "g0": d // ik["g0"],
        "g2": d // ik["g2"],
        "g1": m0 // ik["g1"],
        "g3": m0 // ik["g3"],
        "g4": n0 // ik["g4"],
    }
    return _Plan(tiles, ksteps)


def _brute_mfma(t: StageTable, wave: int) -> dict[str, int]:
    p = _enumerate(t, wave)
    return {g: len(p.tiles[g]) * p.ksteps[g] for g in ("g0", "g1", "g2", "g3", "g4")}


# ---------------------------------------------------------------------------
# Synthetic tile step: every op emitted from the enumeration
# ---------------------------------------------------------------------------

_CLASS_OF = (
    ("tile.mma", MFMA),
    ("memref.global_load", VMEM_READ),
    ("memref.global_atomic", VMEM_WRITE),
    ("memref.global_store", VMEM_WRITE),
    ("tile.smem_load", DS_READ),
    ("tile.ds_read_tr", DS_READ),
    ("tile.smem_store", DS_WRITE),
)


def _op_class(name: str) -> int | None:
    for prefix, cls in _CLASS_OF:
        if name.startswith(prefix):
            return cls
    return None


class _SynthStep:
    """Emit one q step of the backward body as plain IR, from the enumeration.

    Operands that live in registers in the real body are loaded from global
    memory in a prologue (outside the stages); operands the table says come
    from LDS are read inside the stage that the stage table assigns them to.
    ``only`` restricts the MMA work to one GEMM (all memory work is kept).
    """

    LDS_HALFS = 16384

    def __init__(
        self,
        t: StageTable,
        *,
        prefetch: bool = True,
        prev_atomics: bool = True,
        only: str | None = None,
    ):
        self.t = t
        self.prefetch = prefetch
        self.prev_atomics = prev_atomics
        self.only = only
        self.ws = t.wave_size
        self.threads = t.waves * self.ws
        self.segments: dict[str, tuple[int, int]] = {}
        self._lds_cursor = 0

    # -- helpers --
    def _c(self, v: int):
        return self.b.const_i32(int(v))

    def _lds_idx(self, nelem: int):
        """A distinct, aligned LDS element index per read/write site."""
        base = (self._lds_cursor * 8) % (self.LDS_HALFS - 1024)
        self._lds_cursor += 1
        align = max(nelem, 1)
        return self.b.add(
            self._c(base - base % align), self.b.mul(self.lane, self._c(align))
        )

    def _reg(self, n: int):
        """A distinct register operand (prologue global load, splat)."""
        b = self.b
        self._reg_cursor += 1
        s = b.global_load_f16(self.src, b.add(self.tid, self._c(self._reg_cursor * 64)))
        return b.vector_splat(s, n)

    def _lds32_idx(self):
        """A distinct fp32 LDS index per accumulator access site."""
        base = (self._lds32_cursor * 4) % (1024 - self.ws)
        self._lds32_cursor += 1
        return self.b.add(self._c(base), self.lane)

    def _acc_load(self, n: int):
        """Read one lane's ``n``-element fp32 accumulator, one b32 per element."""
        b = self.b
        vals = [
            b.vec_extract(
                b.smem_load_vN(self.lds32, self._lds32_idx(), dtype=F32, n=1), 0
            )
            for _ in range(n)
        ]
        return b.vec_pack(vals, F32)

    def _acc_store(self, c, n: int) -> None:
        b = self.b
        for i in range(n):
            b.smem_store_vN(self.lds32, [self._lds32_idx()], b.vec_extract(c, i), 1)

    def _lds_frag(self, n: int, how: str):
        """Read one ``n``-element operand fragment from LDS (``how`` = model source)."""
        b = self.b
        if how == "tr_read":
            pieces = [
                b.ds_read_tr16_b64(self.lds, self._lds_idx(4)) for _ in range(n // 4)
            ]
        elif how == "lds_plain":
            scalars = [
                b.vec_extract(
                    b.smem_load_vN(self.lds, self._lds_idx(1), dtype=F16, n=1), 0
                )
                for _ in range(n)
            ]
            return b.vec_pack(scalars, F16)
        else:  # contiguous: one read per 16 bytes
            per = min(n, 8)
            pieces = [
                b.smem_load_vN(self.lds, self._lds_idx(per), dtype=F16, n=per)
                for _ in range(n // per)
            ]
        v = pieces[0]
        for p in pieces[1:]:
            v = b.vec_concat(v, p)
        return v

    def _mma(self, gemm: str, a, bb, c):
        return self.b.mma(self.t.mma_op_id(gemm), a, bb, c)

    def _begin(self, name: str) -> None:
        self.t.emit_stage_begin(self.b, name)
        self._seg_start = len(self.b.kernel.body.ops)

    def _end(self, name: str) -> None:
        self.segments[name] = (self._seg_start, len(self.b.kernel.body.ops))
        self.t.emit_stage_end(
            self.b, name, prefetch=self.prefetch, prev_atomics=self.prev_atomics
        )

    def _on(self, gemm: str) -> bool:
        return self.only is None or self.only == gemm

    # -- the step --
    def build(self, name: str = "bwd_sched_synth"):
        t = self.t
        self.b = b = IRBuilder(name)
        b.kernel.attrs["max_workgroup_size"] = max(64, self.threads)
        self.src = b.param("src", PtrType(F16, "global"), readonly=True)
        self.stats = b.param("stats", PtrType(F32, "global"), readonly=True)
        self.out = b.param("out", PtrType(F32, "global"))
        self.dq = b.param("dq", PtrType(F32, "global"))
        self.lds = b.smem_alloc(F16, [self.LDS_HALFS])
        self.lds32 = b.smem_alloc(F32, [1024])
        self.tid = b.thread_id_x()
        self.lane = b.mod(self.tid, self._c(self.ws))
        self._reg_cursor = 0
        self._lds32_cursor = 0
        plan = _enumerate(t, 0)
        ik = t.instr_k
        fl = {g: t.frag_len(g) for g in ("g0", "g1", "g2", "g3", "g4")}
        cl = t.c_frag_len
        tsrc = {"xt_lds": "contig", "wmma_lds": "contig"}.get(
            t.transpose_source, t.transpose_source
        )

        # Prologue: register-resident operands.
        q_reg = {
            (mt, ks): self._reg(fl["g0"])
            for mt in range(0, t.k_m0, 16)
            for ks in range(plan.ksteps["g0"])
        }
        do_reg = {
            (mt, ks): self._reg(fl["g2"])
            for mt in range(0, t.k_m0, 16)
            for ks in range(plan.ksteps["g2"])
        }
        kv_reg = t.kv_residency == "reg"
        k_cols = sorted({nt for _, nt in plan.tiles["g0"]})
        v_cols = sorted({nt for _, nt in plan.tiles["g2"]})
        k_reg = {
            (nt, ks): self._reg(fl["g0"])
            for nt in (k_cols if kv_reg else ())
            for ks in range(plan.ksteps["g0"])
        }
        v_reg = {
            (nt, ks): self._reg(fl["g2"])
            for nt in (v_cols if kv_reg else ())
            for ks in range(plan.ksteps["g2"])
        }
        pt_reg = {
            (mt, ks): self._reg(fl["g1"])
            for mt, _ in plan.tiles["g1"]
            for ks in range(plan.ksteps["g1"])
        }
        dst_reg = {
            (mt, ks): self._reg(fl["g3"])
            for mt, _ in plan.tiles["g3"]
            for ks in range(plan.ksteps["g3"])
        }
        kt_reg = {}
        if t.kt_source == "reg":
            kt_reg = {
                (nt, ks): self._reg(fl["g4"])
                for _, nt in plan.tiles["g4"]
                for ks in range(plan.ksteps["g4"])
            }
        zero = b.zero_vec(F32, cl)
        acc = {}

        # ---- stage A: G0 + next-tile global reads + dO^T fragment reads ----
        t.emit_loop_head(b)
        self._begin("A")
        prefetched = []
        if self.prefetch:
            vec = 8 if t.global_path == "dma" else t.stage_vec
            per_lane = -(-t.k_m0 * t.head_size // (self.threads * vec))
            for tensor in range(2):
                for i in range(per_lane):
                    idx = b.add(self.tid, self._c((tensor * 64 + i) * self.threads))
                    if vec == 1:
                        prefetched.append(b.global_load_f16(self.src, idx))
                    else:
                        prefetched.append(
                            b.global_load_vN(
                                self.src, b.mul(idx, self._c(vec)), F16, vec
                            )
                        )
            for _ in range(2 * (-(-t.k_m0 // self.ws))):
                b.global_load_f32(self.stats, self.lane)
        if t.sched == "stage_table_atomics" and self.prev_atomics:
            for i in range(t.dq_atomics_per_lane):
                b.global_atomic_add_f32(
                    self.dq,
                    b.add(self.tid, self._c(i * self.threads)),
                    b.const_f32(0.0),
                )
        if not kv_reg:  # resident K: B fragments of G0 from LDS
            k_reg = {
                (nt, ks): self._lds_frag(fl["g0"], "contig")
                for nt in k_cols
                for ks in range(plan.ksteps["g0"])
            }
        if self._on("g0"):
            for mt, nt in plan.tiles["g0"]:
                c = zero
                for ks in range(plan.ksteps["g0"]):
                    c = self._mma("g0", q_reg[(mt, ks)], k_reg[(nt, ks)], c)
                acc[("s", mt, nt)] = c
        # B fragments of G1 (dO^T), distinct (n tile, k step) pairs.
        dot_lds = {
            (nt, ks): self._lds_frag(fl["g1"], tsrc)
            for nt in sorted({nt for _, nt in plan.tiles["g1"]})
            for ks in range(plan.ksteps["g1"])
        }
        self._end("A")

        # ---- stage B: G1 + G2, Q^T reads, next-tile LDS writes ----
        self._begin("B")
        if not kv_reg:  # resident V: B fragments of G2 from LDS
            v_reg = {
                (nt, ks): self._lds_frag(fl["g2"], "contig")
                for nt in v_cols
                for ks in range(plan.ksteps["g2"])
            }
        for mt, nt in plan.tiles["g1"]:
            c = self._acc_load(cl) if t.acc_in_lds else zero
            if self._on("g1"):
                for ks in range(plan.ksteps["g1"]):
                    c = self._mma("g1", pt_reg[(mt, ks)], dot_lds[(nt, ks)], c)
                acc[("dv", mt, nt)] = c
            if t.acc_in_lds:
                self._acc_store(c, cl)
        if self._on("g2"):
            for mt, nt in plan.tiles["g2"]:
                c = zero
                for ks in range(plan.ksteps["g2"]):
                    c = self._mma("g2", do_reg[(mt, ks)], v_reg[(nt, ks)], c)
                acc[("dp", mt, nt)] = c
        qt_lds = {
            (nt, ks): self._lds_frag(fl["g3"], tsrc)
            for nt in sorted({nt for _, nt in plan.tiles["g3"]})
            for ks in range(plan.ksteps["g3"])
        }
        if self.prefetch and t.global_path == "vgpr":
            vec = t.stage_vec
            for v in prefetched:  # Q and dO into their normal LDS images
                b.smem_store_vN(self.lds, [self._lds_idx(vec)], v, vec)
            if t.transpose_source == "xt_lds":
                w = default_xt_writer(t.k_m0, t.head_size, self.threads)
                for _ in range(2):  # Q^T and dO^T
                    for _it in range(w.iterations):
                        for _j in range(w.vec_d):
                            val = self._xt_value(w.k_per_thread)
                            b.smem_store_vN(
                                self.lds,
                                [self._lds_idx(w.k_per_thread)],
                                val,
                                w.k_per_thread,
                            )
            for _ in range(2 * (-(-t.k_m0 // self.threads))):
                b.smem_store_vN(self.lds32, [self.lane], b.const_f32(0.0), 1)
        self._end("B")

        # ---- stage C: G3, dS writes, first dS slice, next Q / lse2 reads ----
        self._begin("C")
        for mt, nt in plan.tiles["g3"]:
            c = self._acc_load(cl) if t.acc_in_lds else zero
            if self._on("g3"):
                for ks in range(plan.ksteps["g3"]):
                    c = self._mma("g3", dst_reg[(mt, ks)], qt_lds[(nt, ks)], c)
                acc[("dk", mt, nt)] = c
            if t.acc_in_lds:
                self._acc_store(c, cl)
        if t.dq_mode == "atomic" or t.pt_route == "lds":
            for mt, nt in plan.tiles["g0"]:  # dS tiles in the G0/G2 C layout
                src_acc = acc.get(("dp", mt, nt), zero)
                h = b.vec_trunc_f32_to_f16(src_acc)
                if self.ws == 64:
                    b.smem_store_vN(self.lds, [self._lds_idx(cl)], h, cl)
                else:
                    for i in range(cl):
                        b.smem_store_vN(
                            self.lds, [self._lds_idx(1)], b.vec_extract(h, i), 1
                        )
        g4_rows = sorted({mt for mt, _ in plan.tiles["g4"]})
        per_slice = t.k_k4 // ik["g4"]
        n_slices = t.k_n0 // t.k_k4 if plan.tiles["g4"] else 0
        ds_lds = {}

        def read_slice(sl: int) -> None:
            for mt in g4_rows:
                for k in range(per_slice):
                    ds_lds[(mt, sl * per_slice + k)] = self._lds_frag(
                        fl["g4"], "contig"
                    )

        # the first dS slice is read under G3 only for g4_slicing="slice_db_g3"
        in_c = 1 if n_slices and t.g4_slicing == "slice_db_g3" else 0
        if in_c:
            read_slice(0)
        if self.prefetch:
            for _ in range((t.k_m0 // 16) * plan.ksteps["g0"]):
                self._lds_frag(fl["g0"], "contig")
            for _ in range(t.k_m0 // 16):
                b.smem_load_vN(self.lds32, self.lane, dtype=F32, n=4)
        self._end("C")

        # ---- stage D: G4, remaining dS slices (+ K^T reads), next dO / Dsum ----
        self._begin("D")
        for sl in range(in_c, n_slices):
            read_slice(sl)
        kt_lds = {}
        if plan.tiles["g4"] and t.kt_source == "lds":
            kt_lds = {
                (nt, ks): self._lds_frag(fl["g4"], tsrc)
                for nt in sorted({nt for _, nt in plan.tiles["g4"]})
                for ks in range(plan.ksteps["g4"])
            }
        if self._on("g4"):
            for mt, nt in plan.tiles["g4"]:
                c = zero
                for ks in range(plan.ksteps["g4"]):
                    bop = kt_reg[(nt, ks)] if t.kt_source == "reg" else kt_lds[(nt, ks)]
                    c = self._mma("g4", ds_lds[(mt, ks)], bop, c)
                acc[("dq", mt, nt)] = c
        if self.prefetch:
            for _ in range((t.k_m0 // 16) * plan.ksteps["g2"]):
                self._lds_frag(fl["g2"], "contig")
            for _ in range(t.k_m0 // 16):
                b.smem_load_vN(self.lds32, self.lane, dtype=F32, n=4)
        self._end("D")

        # Epilogue: dQ atomics (one per element) and stores that keep every
        # accumulator live through the backend.
        n_out = 0
        for key, c in acc.items():
            if key[0] == "dq":
                for i in range(cl):
                    b.global_atomic_add_f32(
                        self.dq,
                        b.add(self.tid, self._c(n_out * self.threads)),
                        b.vec_extract(c, i),
                    )
                    n_out += 1
            else:
                b.global_store_vN(
                    self.out,
                    b.mul(b.add(self.tid, self._c(n_out * self.threads)), self._c(cl)),
                    c,
                    cl,
                )
                n_out += 1
        return b.kernel

    def _xt_value(self, n: int):
        b = self.b
        s = b.vec_extract(b.vec_trunc_f32_to_f16(b.zero_vec(F32, 2)), 0)
        return b.vector_splat(s, n)

    # -- counting --
    def segment_counts(self, kernel) -> dict[str, Counter]:
        out = {}
        ops = kernel.body.ops
        for name, (lo, hi) in self.segments.items():
            cnt: Counter = Counter()
            for op in ops[lo:hi]:
                cls = _op_class(op.name)
                if cls is not None:
                    cnt[cls] += 1
            out[name] = cnt
        return out


def _group_sums(kernel) -> dict[str, Counter]:
    """Per stage: summed ``sched_group_barrier`` counts by mask (stages end at fences)."""
    sums: dict[str, Counter] = {s: Counter() for s in STAGES}
    stage = 0
    for op in kernel.body.ops:
        if op.name == "tile.sched_group_barrier":
            sums[STAGES[stage]][op.attrs["mask"]] += op.attrs["count"]
        elif op.name == "tile.sched_barrier":
            stage += 1
    return sums


# ---------------------------------------------------------------------------
# CPU tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cid,kw", CONFIGS, ids=CONFIG_IDS)
def test_mfma_counts_equal_brute_force_per_wave(cid, kw):
    t = _table(kw)
    model = t.mfma.as_dict()
    for wave in range(t.waves):
        assert _brute_mfma(t, wave) == model, f"wave {wave}"


def _legal_grid():
    for arch in ("gfx942", "gfx950"):
        for d in (32, 64, 128):
            for m0 in (16, 32, 64):
                for n0 in (64, 128, 256):
                    for w in (1, 2, 4, 8):
                        if n0 % (16 * w):
                            continue
                        for g4 in ((1, w), (2, w // 2)):
                            if g4[1] == 0 or d % (16 * g4[1]) or m0 % (16 * g4[0]):
                                continue
                            for k02, k13 in ((16, 16), (32, 16), (32, 32)):
                                if k13 == 32 and m0 < 32:
                                    continue
                                yield {
                                    "arch": arch,
                                    "head_size": d,
                                    "k_m0": m0,
                                    "k_n0": n0,
                                    "k_k4": 32,
                                    "waves": w,
                                    "grid_g4": g4,
                                    "atom_g02": k02,
                                    "atom_g13": k13,
                                    "atom_g4": k02,
                                }


def test_mfma_counts_equal_brute_force_over_legal_grid():
    n = 0
    for kw in _legal_grid():
        t = StageTable(**kw)
        for wave in range(t.waves):
            assert _brute_mfma(t, wave) == t.mfma.as_dict(), kw
        n += 1
    assert n > 500


def test_g13_counts_come_from_their_own_atom():
    # gfx950 d128 start point: G0 on 16x16x32, G1 on 16x16x16. Deriving G1 from
    # G0's warp tile (as the reference pipeline's scheduler does) halves it.
    t = StageTable(
        arch="gfx950",
        head_size=128,
        k_m0=16,
        k_n0=192,
        k_k4=32,
        waves=4,
        atom_g02=32,
        atom_g4=32,
    )
    assert t.instr_k == {"g0": 32, "g1": 16, "g2": 32, "g3": 16, "g4": 32}
    assert t.mfma.g1 == 2 * t.mfma.g0
    # gfx942: a K = 32 step is two 16x16x16 instructions.
    t942 = StageTable(
        arch="gfx942",
        head_size=128,
        k_m0=16,
        k_n0=128,
        k_k4=32,
        waves=4,
        atom_g02=32,
        atom_g4=32,
    )
    t942_16 = StageTable(
        arch="gfx942", head_size=128, k_m0=16, k_n0=128, k_k4=32, waves=4
    )
    assert t942.instr_k["g0"] == 16
    assert t942.mfma == t942_16.mfma
    assert instruction_k("gfx942", 32) == 16 and instruction_k("gfx950", 32) == 32
    assert native_atom_ks("gfx942") == (16,) and native_atom_ks("gfx950") == (16, 32)


def test_dq_atomics_per_lane():
    t = StageTable(arch="gfx942", head_size=128, k_m0=16, k_n0=128, k_k4=32, waves=4)
    assert t.dq_atomics_per_lane == 16 * 128 // (4 * 64)
    split = StageTable(
        arch="gfx942",
        head_size=128,
        k_m0=16,
        k_n0=128,
        k_k4=32,
        waves=4,
        dq_mode="split",
    )
    assert split.dq_atomics_per_lane == 0 and split.mfma.g4 == 0
    w = StageTable(
        arch="gfx1151",
        head_size=64,
        k_m0=16,
        k_n0=16,
        k_k4=16,
        waves=1,
        transpose_source="wmma_lds",
        sched="none",
    )
    assert w.dq_atomics_per_lane == 16 * 64 // 32


@pytest.mark.parametrize("g4_slicing", ["step", "slice", "slice_db", "slice_db_g3"])
@pytest.mark.parametrize("cid,kw", CONFIGS, ids=CONFIG_IDS)
@pytest.mark.parametrize("prefetch,prev", [(True, True), (False, True), (True, False)])
def test_stage_counts_equal_brute_force_ir(cid, kw, prefetch, prev, g4_slicing):
    t = _table(dict(kw, g4_slicing=g4_slicing))
    syn = _SynthStep(t, prefetch=prefetch, prev_atomics=prev)
    kernel = syn.build()
    seg = syn.segment_counts(kernel)
    for sc in t.stages(prefetch=prefetch, prev_atomics=prev):
        got = seg[sc.stage]
        assert got[MFMA] == sc.mfma, (sc.stage, "mfma")
        assert got[VMEM_READ] == sc.vmem_read, (sc.stage, "vmem_read")
        assert got[VMEM_WRITE] == sc.vmem_write, (sc.stage, "vmem_write")
        assert got[DS_READ] == sc.ds_read, (sc.stage, "ds_read")
        assert got[DS_WRITE] == sc.ds_write, (sc.stage, "ds_write")
    # The emitted hints: per stage the group sums equal the stage counts.
    names = Counter(op.name for op in kernel.body.ops)
    model = t.ir_count_model(prefetch=prefetch, prev_atomics=prev)
    for key in (
        "tile.sched_group_barrier",
        "tile.sched_barrier",
        "tile.s_setprio",
        "tile.iglp_opt",
    ):
        assert names.get(key, 0) == model[key], key
    if t.sched in ("stage_table", "stage_table_atomics"):
        sums = _group_sums(kernel)
        for sc in t.stages(prefetch=prefetch, prev_atomics=prev):
            want = {m: c for m, c in sc.by_mask().items() if c}
            assert dict(sums[sc.stage]) == want, sc.stage


@pytest.mark.parametrize("m", [0, 1, 3, 8, 16, 37])
def test_interleave_preserves_counts(m):
    mem = [(VMEM_READ, 5), (DS_READ, 11), (DS_WRITE, 0), (VMEM_WRITE, 7)]
    groups = interleave_groups(m, mem)
    sums = Counter()
    for mask, c in groups:
        assert c > 0
        sums[mask] += c
    want = Counter({k: v for k, v in mem if v})
    if m:
        want[MFMA] = m
    assert sums == want
    # No two adjacent groups of one class (they are merged).
    assert all(a[0] != b[0] for a, b in itertools.pairwise(groups))
    if m:
        # VMEM reads are issued no later than the first MFMA group allows.
        assert groups[0][0] in (VMEM_READ, MFMA)


@pytest.mark.parametrize(
    "sched",
    ["none", "iglp0", "iglp1", "fences_only", "stage_table", "stage_table_atomics"],
)
def test_sched_variants_emit_the_modelled_ops(sched):
    t = StageTable(
        arch="gfx942", head_size=64, k_m0=32, k_n0=128, k_k4=32, waves=4, sched=sched
    )
    b = IRBuilder("variants")
    t.emit_loop_head(b)
    for s in STAGES:
        t.emit_stage_begin(b, s)
        before_end = len(b.kernel.body.ops)
        t.emit_stage_end(b, s)
    after_d = [op.name for op in b.kernel.body.ops[before_end:]]
    names = Counter(op.name for op in b.kernel.body.ops)
    model = t.ir_count_model()
    for key in (
        "tile.sched_group_barrier",
        "tile.sched_barrier",
        "tile.s_setprio",
        "tile.iglp_opt",
    ):
        assert names.get(key, 0) == model[key], key
    if sched in ("iglp0", "iglp1"):
        assert (
            names["tile.sched_group_barrier"] == 0 and names["tile.sched_barrier"] == 0
        )
    # Fences are sched_barrier(0) and never follow stage D.
    fences = [op for op in b.kernel.body.ops if op.name == "tile.sched_barrier"]
    assert all(op.attrs["mask"] == 0 for op in fences)
    assert "tile.sched_barrier" not in after_d
    # iglp_opt(1) (only) marks the kernel to compile in a fresh process
    assert b.kernel.attrs.get(COMGR_FRESH_PROCESS_ATTR, False) is (sched == "iglp1")


def test_the_fresh_process_mark_leaves_the_llvm_ir_unchanged():
    """A whole iglp1 main kernel carries the mark; its LLVM IR is the same
    with the mark removed (no lowerer reads it), and a kernel of another
    schedule does not carry it."""
    from rocke.core.lower_llvm import lower_kernel_to_llvm

    from kernels.common.attention_bwd import (
        AttnBwdSpec,
        build_attn_bwd_main,
        validate_attn_bwd_spec,
    )

    for arch in ("gfx942", "gfx950"):
        s = validate_attn_bwd_spec(
            AttnBwdSpec(head_size=64, dtype="fp16", sched="iglp1", edge_tiles=True),
            arch,
        )
        k = build_attn_bwd_main(s, arch=arch)
        assert k.attrs.get(COMGR_FRESH_PROCESS_ATTR) is True
        ll = lower_kernel_to_llvm(k, arch=arch)
        del k.attrs[COMGR_FRESH_PROCESS_ATTR]
        assert lower_kernel_to_llvm(k, arch=arch) == ll
        assert "llvm.amdgcn.iglp.opt(i32 1)" in ll
        for sched in ("iglp0", "none"):
            other = build_attn_bwd_main(
                validate_attn_bwd_spec(AttnBwdSpec(head_size=64, sched=sched), arch),
                arch=arch,
            )
            assert COMGR_FRESH_PROCESS_ATTR not in other.attrs


def test_atomics_variant_moves_dq_atomics_into_stage_a():
    kw = {
        "arch": "gfx942",
        "head_size": 128,
        "k_m0": 16,
        "k_n0": 128,
        "k_k4": 32,
        "waves": 4,
    }
    plain = StageTable(**kw).stage("A")
    atom = StageTable(sched="stage_table_atomics", **kw)
    assert plain.vmem_write == 0
    assert atom.stage("A").vmem_write == atom.dq_atomics_per_lane
    assert atom.stage("A", prev_atomics=False).vmem_write == 0


def test_last_step_drops_next_tile_work():
    t = StageTable(
        arch="gfx942", head_size=128, k_m0=16, k_n0=128, k_k4=32, waves=4,
        pipeline="ck",
    )  # fmt: skip
    last = {s.stage: s for s in t.stages(prefetch=False)}
    full = {s.stage: s for s in t.stages()}
    assert last["A"].vmem_read == 0 and full["A"].vmem_read > 0
    assert last["B"].ds_write == 0 and full["B"].ds_write > 0
    assert last["C"].ds_read < full["C"].ds_read
    assert all(last[s].mfma == full[s].mfma for s in STAGES)


@pytest.mark.parametrize(
    "kw,match",
    [
        ({"sched": "iglp0", "setprio": True}, "setprio"),
        ({"sched": "bogus"}, "sched"),
        ({"k_n0": 96}, "kN0"),
        ({"grid_g4": (2, 1)}, "warp grid"),
        ({"atom_g13": 32}, "kM0 16"),
        ({"k_k4": 48}, "kN0"),
    ],
)
def test_illegal_tables_raise(kw, match):
    base = {
        "arch": "gfx942",
        "head_size": 128,
        "k_m0": 16,
        "k_n0": 128,
        "k_k4": 32,
        "waves": 4,
    }
    base.update(kw)
    with pytest.raises(ValueError, match=match):
        StageTable(**base)


def test_from_geometry_reads_a_geometry_like_object():
    # Local stand-in for BwdTileGeometry (owned by the body module).
    @dataclass(frozen=True)
    class _Geom:
        waves: int = 4
        k_m0: int = 16
        k_n0: int = 192
        k_k4: int = 32
        head_size: int = 128
        grid_g4: tuple = (1, 4)
        atoms: tuple = (("g02", "16x16x32"), ("g13", "16x16x16"), ("g4", "16x16x32"))
        transpose_source: str = "tr_read"
        kt_source: str = "lds"
        global_path: str = "dma"
        pt_route: str = "relabel"
        sched: str = "stage_table"
        setprio: bool = True

    g = _Geom()
    g = _Geom(atoms=dict(g.atoms))
    t = StageTable.from_geometry(g, "gfx950")
    assert t.instr_k["g0"] == 32 and t.instr_k["g1"] == 16 and t.setprio
    assert t.kv_residency == "reg" and not t.acc_in_lds
    desc = t.describe()
    assert desc["mfma"] == t.mfma.as_dict()
    assert set(desc["stages"]) == set(STAGES)

    @dataclass(frozen=True)
    class _GeomLds(_Geom):
        kv_residency: str = "lds"

    tl = StageTable.from_geometry(
        _GeomLds(atoms=dict(g.atoms)), "gfx950", acc_in_lds=True
    )
    assert tl.kv_residency == "lds" and tl.acc_in_lds
    assert tl.describe()["kv_residency"] == "lds"


def _counts(t: StageTable) -> dict[str, tuple[int, int, int]]:
    return {s.stage: (s.mfma, s.ds_read, s.ds_write) for s in t.stages()}


@pytest.mark.parametrize("arch", ["gfx942", "gfx950", "gfx1151", "gfx1201"])
def test_kv_residency_and_acc_in_lds_add_their_lds_traffic(arch):
    # Each residency choice adds exactly its own DS traffic to its own stage:
    # K fragments to A, V fragments to B, the dV read-modify-write to B and
    # the dK read-modify-write to C; MMA counts are unchanged.
    w, d, m0, n0 = 4, 128, 16, 128
    base = {
        "arch": arch,
        "head_size": d,
        "k_m0": m0,
        "k_n0": n0,
        "k_k4": 32,
        "waves": w,
        "kt_source": "lds",
        "transpose_source": "wmma_lds" if arch.startswith("gfx1") else "xt_lds",
        "pipeline": "ck",
    }
    variants = {
        (kv, al): _counts(StageTable(kv_residency=kv, acc_in_lds=al, **base))
        for kv in ("reg", "lds")
        for al in (False, True)
    }
    assert len({tuple(sorted(v.items())) for v in variants.values()}) == 4
    ref = StageTable(**base)
    frag = ref._reads_per_frag("g0", "contig")
    kv_reads = (n0 // (16 * w)) * (d // ref.instr_k["g0"]) * frag
    rmw = (n0 // (16 * w)) * (d // 16) * ref.c_frag_len
    r = variants[("reg", False)]
    for (kv, al), got in variants.items():
        dk = kv_reads if kv == "lds" else 0
        da = rmw if al else 0
        assert got["A"] == (r["A"][0], r["A"][1] + dk, r["A"][2])
        assert got["B"] == (r["B"][0], r["B"][1] + dk + da, r["B"][2] + da)
        assert got["C"] == (r["C"][0], r["C"][1] + da, r["C"][2] + da)
        assert got["D"] == r["D"]


def test_from_geometry_counts_follow_kv_residency_and_acc_in_lds():
    # A gfx942 d128 16x128 W4 geometry with K^T from LDS: every residency
    # choice gives a distinct stage table.
    @dataclass(frozen=True)
    class _Geom:
        waves: int = 4
        k_m0: int = 16
        k_n0: int = 128
        k_k4: int = 32
        head_size: int = 128
        grid_g4: tuple = (1, 4)
        kt_source: str = "lds"
        kv_residency: str = "reg"

    seen = set()
    for kv in ("reg", "lds"):
        for al in (False, True):
            t = StageTable.from_geometry(
                _Geom(kv_residency=kv), "gfx942", acc_in_lds=al
            )
            seen.add(tuple(sorted(_counts(t).items())))
    assert len(seen) == 4


@pytest.mark.parametrize("cid,kw", CONFIGS, ids=CONFIG_IDS)
def test_g4_slicing_moves_only_the_first_slice(cid, kw):
    """Every ``g4_slicing`` value gives the same per-step totals; only
    ``slice_db_g3`` reads the first dS slice in stage C (under G3), the others
    read every slice in stage D."""
    tabs = {v: _table(dict(kw, g4_slicing=v)) for v in ("step", "slice", "slice_db")}
    tabs["slice_db_g3"] = _table(dict(kw, g4_slicing="slice_db_g3"))
    ref = {s.stage: s for s in tabs["step"].stages()}
    for v, t in tabs.items():
        got = {s.stage: s for s in t.stages()}
        for f in ("mfma", "ds_read", "ds_write", "vmem_read", "vmem_write"):
            assert sum(getattr(s, f) for s in got.values()) == sum(
                getattr(s, f) for s in ref.values()
            ), (v, f)
        moved = got["C"].ds_read - ref["C"].ds_read
        assert moved == ref["D"].ds_read - got["D"].ds_read
        assert (moved > 0) == (v == "slice_db_g3" and t.mfma.g4 > 0), v
        assert t.describe()["g4_slicing"] == v
    with pytest.raises(ValueError, match="g4_slicing"):
        _table(dict(kw, g4_slicing="db"))


def test_from_geometry_reads_g4_slicing():
    @dataclass(frozen=True)
    class _Geom:
        waves: int = 4
        k_m0: int = 16
        k_n0: int = 128
        k_k4: int = 32
        head_size: int = 128
        grid_g4: tuple = (1, 4)
        kt_source: str = "lds"
        g4_slicing: str = "slice_db_g3"

    assert StageTable.from_geometry(_Geom(), "gfx942").g4_slicing == "slice_db_g3"

    @dataclass(frozen=True)
    class _Plain:  # a geometry-like object without the field: step emission
        waves: int = 4
        k_m0: int = 16
        k_n0: int = 128
        k_k4: int = 32
        head_size: int = 128

    assert StageTable.from_geometry(_Plain(), "gfx942").g4_slicing == "step"


def test_kt_from_registers_needs_registers_for_k():
    with pytest.raises(ValueError, match="kv_residency"):
        StageTable(
            arch="gfx942",
            head_size=128,
            k_m0=16,
            k_n0=128,
            k_k4=32,
            waves=4,
            kt_source="reg",
            kv_residency="lds",
        )


# ---------------------------------------------------------------------------
# The synchronous model (the built body; the IR census is in
# test_attention_bwd_sched_body.py)
# ---------------------------------------------------------------------------

_SYNC = {"arch": "gfx942", "head_size": 64, "k_m0": 32, "k_n0": 128, "k_k4": 32}


def test_interleave_lead_and_spans():
    mem = [(VMEM_READ, 6), (DS_READ, 13), (DS_WRITE, 8), (VMEM_WRITE, 5)]
    g = interleave_groups(
        10, mem, lead={VMEM_READ: 6, DS_READ: 3}, spans={DS_WRITE: 4, VMEM_WRITE: 2}
    )
    sums = Counter()
    for mask, c in g:
        sums[mask] += c
    assert sums == Counter({VMEM_READ: 6, DS_READ: 13, DS_WRITE: 8, VMEM_WRITE: 5,
                            MFMA: 10})  # fmt: skip
    first = g.index((MFMA, 1))
    assert g[:first] == [(VMEM_READ, 6), (DS_READ, 3)]
    # a spanned class follows only the last `span` MFMAs
    mf_pos = [i for i, x in enumerate(g) if x[0] == MFMA]
    for mask, span in ((DS_WRITE, 4), (VMEM_WRITE, 2)):
        pos = [i for i, x in enumerate(g) if x[0] == mask]
        assert min(pos) > mf_pos[-span - 1]
    # every VMEM read precedes the first VMEM write
    assert max(i for i, x in enumerate(g) if x[0] == VMEM_READ) < min(
        i for i, x in enumerate(g) if x[0] == VMEM_WRITE
    )
    # the defaults keep the plain spread
    assert interleave_groups(10, mem) == interleave_groups(10, mem, lead={}, spans={})
    with pytest.raises(ValueError):
        interleave_groups(4, mem, lead={DS_READ: 14})
    with pytest.raises(ValueError):
        interleave_groups(4, mem, spans={DS_WRITE: 5})


def test_sync_stage_counts_by_hand():
    """gfx942 d64 32x128 on 8 waves (16x16x16, grids (1, 8) / (8, 1), G4 (2, 4)),
    register K / V / K^T, relabel route, lds_plain transposes, stage_table."""
    t = StageTable(**_SYNC, waves=8, grid_g4=(2, 4), transpose_source="lds_plain")
    a, b, c, d = t.stages()
    # G0 / G2: fm 2, fn 1, 4 K steps; Q / dO frags of 4 elements, one read
    # each; G1 / G3: fm 1, fn 4, 2 K steps; dO^T / Q^T element reads (4 each)
    assert (a.mfma, b.mfma, c.mfma, d.mfma) == (8, 16, 8, 8)
    assert a.vmem_read == 2 * 2 * 4  # lse2 + Dsum per (row frag, C slot)
    assert a.ds_read == 4 * 2 + 4 * 2 * 4 and a.vmem_write == 0
    assert dict(a.lead) == {VMEM_READ: 16, DS_READ: 2}
    assert b.ds_read == 4 * 2 + 32 and b.ds_write == 2 * 1 * 4
    assert dict(b.spans) == {DS_WRITE: 8}
    assert (c.ds_read, c.ds_write, dict(c.lead)) == (0, 0, {DS_READ: 0})
    # G4: fm 1, fn 1, 8 K steps of one dS read; 32 * 64 / (8 * 64) atomics
    assert d.ds_read == 8 and d.vmem_write == 4 and dict(d.spans) == {VMEM_WRITE: 1}
    atom = StageTable(**_SYNC, waves=8, grid_g4=(2, 4), sched="stage_table_atomics")
    assert atom.stage("A").vmem_write == 4 and atom.stage("D").vmem_write == 0
    assert atom.stage("A", prev_atomics=False).vmem_write == 0
    plain = StageTable(
        **_SYNC, waves=8, grid_g4=(2, 4), sched="none", transpose_source="lds_plain"
    )
    # no fenced placement: the stats with the softmax, operands at their use
    assert plain.stage("A").vmem_read == 0 and plain.stage("B").vmem_read == 16
    assert plain.stage("A").ds_read == 8 and plain.stage("C").ds_read == 32


def test_sync_model_ignores_block_k4_and_follows_grids_and_pads():
    base = dict(_SYNC, waves=4, kt_source="lds", transpose_source="lds_plain")
    t = StageTable(**base)
    assert t.stages() == StageTable(**dict(base, k_k4=64)).stages()
    alt = StageTable(**base, pt_route="lds", grid_g02=(2, 2), grid_g13=(2, 2))
    lds = StageTable(**base, pt_route="lds")
    assert [s.mfma for s in alt.stages()] == [s.mfma for s in t.stages()]
    assert alt.stages() != lds.stages()
    pad = StageTable(**dict(base, atom_g02=32, arch="gfx950"), lds_swizzle="q=pad8")
    wide = StageTable(**dict(base, atom_g02=32, arch="gfx950"))
    # an 8-byte row pad splits each 8-element Q read in two
    assert pad.stage("A").ds_read - wide.stage("A").ds_read == wide._frags("g0")[0] * 2
    assert pad._align("q") == 4 and pad._align("do") == 8
    assert StageTable(**base, lds_swizzle="q=pad16")._align("q") == 8


def test_from_geometry_picks_the_pipeline():
    @dataclass(frozen=True)
    class _G:
        waves: int = 4
        k_m0: int = 32
        k_n0: int = 128
        k_k4: int = 32
        head_size: int = 64
        ring_depth: int = 1
        sched: str = "fences_only"

    assert StageTable.from_geometry(_G(), "gfx942").pipeline == "sync"
    assert StageTable.from_geometry(_G(ring_depth=2), "gfx942").pipeline == "ring"
    assert StageTable.from_geometry(_G(ring_depth=3), "gfx950").pipeline == "ring"
    dma = dataclasses.make_dataclass(
        "_D", [("global_path", str, "dma")], bases=(_G,), frozen=True
    )
    assert StageTable.from_geometry(dma(), "gfx950").pipeline == "ring"
    merged = dataclasses.make_dataclass(
        "_M", [("barrier_merge", str, "on")], bases=(_G,), frozen=True
    )
    assert StageTable.from_geometry(merged(ring_depth=2), "gfx942").pipeline == "merged"
    with pytest.raises(ValueError, match="pipeline"):
        StageTable(**_SYNC, waves=4, pipeline="pipelined")
    with pytest.raises(ValueError, match="does not model"):
        StageTable(**_SYNC, waves=4, pipeline="ring")  # (vgpr, 1) is synchronous
    with pytest.raises(ValueError, match="does not model"):
        StageTable(**_SYNC, waves=4, ring_depth=2)  # a ring is not "sync"
    with pytest.raises(ValueError, match="does not divide the waves"):
        StageTable(**_SYNC, waves=4, grid_g02=(3, 1))


# ---------------------------------------------------------------------------
# Compile tests: model vs instructions in the code object
# ---------------------------------------------------------------------------


def _objdump_candidates() -> list[str]:
    cands = [
        os.environ.get("LLVM_OBJDUMP"),
        shutil.which("llvm-objdump"),
        *sorted(glob.glob("/opt/rocm*/llvm/bin/llvm-objdump"), reverse=True),
    ]
    try:
        import importlib.util

        spec = importlib.util.find_spec("_rocm_sdk_core")
        if spec and spec.submodule_search_locations:
            root = Path(next(iter(spec.submodule_search_locations)))
            cands += [
                str(root / "lib" / "llvm" / "bin" / "llvm-objdump.exe"),
                str(root / "lib" / "llvm" / "bin" / "llvm-objdump"),
            ]
    except (ImportError, ValueError):  # optional discovery only
        spec = None
    out: list[str] = []
    for c in cands:
        if c and Path(c).is_file() and c not in out:
            out.append(c)
    return out


def _disassemble(hsaco: bytes, arch: str) -> str:
    """Disassembly text from the first ``llvm-objdump`` that understands ``arch``.

    A tool that exits non-zero or prints no ``s_endpgm`` (an older LLVM that
    does not know the target) is skipped; with none left the test skips.
    """
    errors = []
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "k.hsaco"
        path.write_bytes(hsaco)
        for tool in _objdump_candidates():
            proc = subprocess.run(
                [tool, "-d", f"--mcpu={arch}", str(path)],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            if proc.returncode == 0 and "s_endpgm" in proc.stdout:
                return proc.stdout
            errors.append(f"{tool}: rc={proc.returncode} {proc.stderr.strip()[:200]}")
    pytest.skip(f"no llvm-objdump disassembles {arch}: {errors}")
    return ""


def _compile(kernel, arch: str):
    try:
        from rocke.helpers.compile import compile_kernel

        return compile_kernel(kernel, arch=arch, backend="python")
    except (OSError, ImportError) as exc:  # comgr missing on this host
        pytest.skip(f"comgr unavailable: {exc}")


_MMA_RE = re.compile(r"^\s*(v_mfma_\S+|v_wmma_\S+)", re.MULTILINE)


def _isa_mma_count(hsaco: bytes, arch: str) -> int:
    return len(_MMA_RE.findall(_disassemble(hsaco, arch)))


@pytest.mark.parametrize("cid,kw", CONFIGS, ids=CONFIG_IDS)
def test_model_equals_code_object_mma_count_per_step(cid, kw):
    t = _table(kw)
    syn = _SynthStep(t)
    kernel = syn.build(f"bwd_sched_synth_{cid.replace('-', '_')}")
    art = _compile(kernel, t.arch)
    model = t.ir_count_model()
    assert _isa_mma_count(art.hsaco, t.arch) == model["mfma_total"]
    llvm = art.llvm_text
    assert (
        llvm.count("call void @llvm.amdgcn.sched.group.barrier(")
        == model["tile.sched_group_barrier"]
    )
    assert (
        llvm.count("call void @llvm.amdgcn.sched.barrier(")
        == model["tile.sched_barrier"]
    )


@pytest.mark.parametrize(
    "cid,kw",
    [
        c
        for c in CONFIGS
        if c[0]
        in (
            "942-d32",
            "942-d128",
            "950-d64",
            "950-d128",
            "1151-debug-d64",
            "1201-debug-d64",
            "942-d128-kvlds-acclds",
            "950-d128-acclds",
            "1201-kvlds-acclds",
        )
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
@pytest.mark.parametrize("gemm", ["g0", "g1", "g2", "g3", "g4"])
def test_model_equals_code_object_mma_count_per_gemm(cid, kw, gemm):
    t = _table(kw)
    syn = _SynthStep(t, only=gemm)
    kernel = syn.build(f"bwd_sched_synth_{cid.replace('-', '_')}_{gemm}")
    art = _compile(kernel, t.arch)
    assert _isa_mma_count(art.hsaco, t.arch) == t.mfma.as_dict()[gemm]
