# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Quantum-circuit tensor network: the rocKE circuit monokernel vs a torch TTGT path.

Every gate of a random 2-D lattice circuit is one contraction of the network
(extent-2 modes, complex64). Arms (same process, interleaved):

* ``vendor_k<kv>``: dense-block gate fusion into ``kv``-qubit blocks, each
  block = permute (torch, merged dims; layout tracked so there is ONE permute
  per block, none when the block qubits are already the fastest modes) + GEMM
  (``torch.matmul``; with ``PYTORCH_TUNABLEOP_ENABLED=1`` TunableOp picks the
  fastest solution it enumerates; real-embedding f32 GEMM or complex64 GEMM,
  whichever is faster per block size), the whole circuit captured in ONE HIP
  graph. Sub-graphs time the GEMM-only and permute-only parts.
* ``vendor_k<kv>_idealperm`` (a timing bound only, not an implementable arm):
  the same graph with every permute replaced by a contiguous copy of the same
  bytes.
* ``rk_mono_*``: ONE persistent rocKE launch for the whole circuit (phases of
  gates applied on LDS tiles, grid barrier between phases).
* ``rk_multi_*``: the same phases, one launch per phase.
* ``rk_gett_k<kv>``: the fused blocks of ``vendor_k<kv>``, one phase per block
  (permute folded into the loads, no phase fusion).

Every arm runs from the same random state and is checked against an f64
gate-by-gate reference (torch complex128, independent of every plan) before
and after its timing; the run exits non-zero if any arm fails. Results go to a
JSONL file outside the repository.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time

import torch
from kernels.gfx950.tensor_network_mono import circuit_kernel as M
from kernels.gfx950.tensor_network_mono import circuit_plan as P

U32 = 2.0**-24


# --------------------------------------------------------------------------- helpers
def merged_permute(old_order, new_order):
    return P.merged_permute(old_order, new_order, tail=2)


class VendorArm:
    def __init__(self, gates, n, kv, dev, log):
        self.n, self.kv, self.dev = n, kv, dev
        t0 = time.time()
        self.blocks = P.fuse_blocks(gates, kv)
        self.fuse_s = time.time() - t0
        N = 2 << n
        self.buf = [torch.empty(N, dtype=torch.float32, device=dev) for _ in range(2)]
        order = list(range(n - 1, -1, -1))
        self.ops = (
            []
        )  # ("perm", old_shape, perm, new_shape) | ("gemm", k, R^T f32, U^T c64)
        nperm = 0
        for bl in self.blocks:
            k = len(bl.qubits)
            S = set(bl.qubits)
            if set(order[-k:]) != S:
                new = [q for q in order if q not in S] + [q for q in order if q in S]
                self.ops.append(("perm",) + merged_permute(order, new))
                order = new
                nperm += 1
            qorder = [order[-1 - i] for i in range(k)]
            U = P.embed(bl.U, bl.qubits, qorder)
            RT = torch.as_tensor(
                P.real_embedding(U).T.copy(), dtype=torch.float32, device=dev
            )
            UT = torch.as_tensor(U.T.copy(), dtype=torch.complex64, device=dev)
            self.ops.append(("gemm", k, RT, UT))
        self.final_order = order
        self.nperm = nperm
        self.mode = {}
        log(
            f"vendor kv={kv}: blocks={len(self.blocks)} permutes={nperm} "
            f"sizes={sorted(len(b.qubits) for b in self.blocks)}"
        )

    # --- one op on the current buffer
    def _gemm(self, src, dst, k, RT, UT, mode):
        if mode == "real":
            torch.matmul(src.view(-1, 2 << k), RT, out=dst.view(-1, 2 << k))
        else:
            torch.matmul(
                src.view(torch.complex64).view(-1, 1 << k),
                UT,
                out=dst.view(torch.complex64).view(-1, 1 << k),
            )

    def calibrate(self, log):
        """Pick real-embedding f32 vs complex64 GEMM per block size."""
        ks = sorted({op[1] for op in self.ops if op[0] == "gemm"})
        for k in ks:
            op = next(o for o in self.ops if o[0] == "gemm" and o[1] == k)
            best = None
            for mode in ("real", "cplx"):
                for _ in range(3):  # TunableOp tunes on first use
                    self._gemm(self.buf[0], self.buf[1], k, op[2], op[3], mode)
                torch.cuda.synchronize()
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record()
                for _ in range(5):
                    self._gemm(self.buf[0], self.buf[1], k, op[2], op[3], mode)
                e1.record()
                e1.synchronize()
                ms = e0.elapsed_time(e1) / 5
                log(f"  vendor gemm k={k} {mode}: {ms*1e3:.1f} us")
                if best is None or ms < best[1]:
                    best = (mode, ms)
            self.mode[k] = best[0]

    def _seq(self, what="all", ideal=False):
        cur = 0
        for op in self.ops:
            if op[0] == "perm":
                if what in ("all", "perm"):
                    src, dst = self.buf[cur], self.buf[1 - cur]
                    if ideal:
                        dst.copy_(src)
                    else:
                        _, old_shape, perm, new_shape = op
                        dst.view(new_shape).copy_(src.view(old_shape).permute(perm))
                    cur = 1 - cur
            else:
                if what in ("all", "gemm"):
                    _, k, RT, UT = op
                    self._gemm(
                        self.buf[cur], self.buf[1 - cur], k, RT, UT, self.mode[k]
                    )
                    cur = 1 - cur
        return cur

    def capture(self):
        self.graphs = {}
        for key, what, ideal in (
            ("all", "all", False),
            ("gemm", "gemm", False),
            ("perm", "perm", False),
            ("ideal", "all", True),
        ):
            self._seq(what, ideal)  # warm (TunableOp already tuned)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                with torch.cuda.graph(g):
                    out = self._seq(what, ideal)
            torch.cuda.current_stream().wait_stream(s)
            self.graphs[key] = g
            if key == "all":
                self.out_idx = out

    def result(self, psi0_f32):
        self.buf[0].copy_(psi0_f32)
        self.graphs["all"].replay()
        torch.cuda.synchronize()
        # untimed un-permute to the canonical layout by index gather (any rank)
        n = self.n
        i = torch.arange(1 << n, device=self.dev, dtype=torch.int64)
        idx = torch.zeros_like(i)
        for q in range(n):
            mpos = (
                n - 1 - self.final_order.index(q)
            )  # bit of q in the memory amplitude index
            idx |= ((i >> q) & 1) << mpos
        x = self.buf[self.out_idx].view(-1, 2)
        return x.index_select(0, idx).reshape(-1)


def to_c128(f32):
    v = f32.view(-1, 2).to(torch.float64)
    return torch.complex(v[:, 0], v[:, 1])


def rel_err(f32, ref):
    d = to_c128(f32) - ref
    return float(torch.linalg.vector_norm(d) / torch.linalg.vector_norm(ref))


def time_fn(fn, reps):
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(reps):
        fn()
    e1.record()
    e1.synchronize()
    return e0.elapsed_time(e1) / reps


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--kv", default="3,4,5,6")
    ap.add_argument(
        "--fused",
        default="13:5:4:x4:0:0,13:5:5:x4:0:0",
        help="comma list t:c:kf:swizzle:prefetch:rhoist[:bar[:nt[:wpc]]]",
    )
    ap.add_argument(
        "--multi", type=int, default=1, help="also time one-launch-per-phase"
    )
    ap.add_argument("--gett", default="5", help="kv list for rk_gett arms")
    ap.add_argument("--wgpercu", type=int, default=0, help="0 = occupancy query")
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="qc_bench.jsonl")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    def log(*x):
        print(*x, flush=True)

    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    props = torch.cuda.get_device_properties(0)
    n_cu = props.multi_processor_count
    arch = props.gcnArchName.split(":")[0]
    n = a.rows * a.cols
    case = f"qc_{a.rows}x{a.cols}_d{a.depth}_s{a.seed}"
    log(
        f"== {case} n={n} state={(8 << n) / 2**30:.3f} GiB device={props.name} cu={n_cu} "
        f"tunableop={os.environ.get('PYTORCH_TUNABLEOP_ENABLED')}"
    )
    gates = P.sycamore_like(a.rows, a.cols, a.depth, seed=a.seed)
    log(f"gates={len(gates)} (2q={sum(1 for g in gates if len(g[0]) == 2)})")

    # random normalized initial state + f64 reference (independent of all plans)
    g = torch.Generator(device=dev).manual_seed(1234 + a.seed)
    psi = torch.complex(
        torch.randn(1 << n, dtype=torch.float64, device=dev, generator=g),
        torch.randn(1 << n, dtype=torch.float64, device=dev, generator=g),
    )
    psi /= torch.linalg.vector_norm(psi)
    t0 = time.time()
    ref = P.apply_reference(psi, gates, n)
    torch.cuda.synchronize()
    log(f"reference f64 {time.time() - t0:.1f}s")
    psi0 = torch.view_as_real(psi.to(torch.complex64)).reshape(-1).contiguous()
    del psi
    torch.cuda.empty_cache()

    arms = {}  # name -> dict(run=callable, check=callable, meta=dict)
    st = torch.empty_like(psi0)

    # ---------------- vendor arms
    for kv in [int(x) for x in a.kv.strip("\"'").split(",") if x]:
        va = VendorArm(gates, n, kv, dev, log)
        t0 = time.time()
        va.calibrate(log)
        va.capture()
        log(f"  vendor kv={kv} tune+capture {time.time() - t0:.1f}s modes={va.mode}")
        nb = len(va.blocks)
        meta = dict(
            blocks=nb,
            permutes=va.nperm,
            modes=str(va.mode),
            passes=nb + va.nperm,
            kmax=kv,
        )
        arms[f"vendor_k{kv}"] = dict(
            run=va.graphs["all"].replay,
            check=(lambda va=va: va.result(psi0)),
            meta=meta,
            tol=nb * (2 << kv) * U32 * 4,
        )
        arms[f"vendor_k{kv}_gemmonly"] = dict(
            run=va.graphs["gemm"].replay, check=None, meta=dict(part="gemm", kmax=kv)
        )
        arms[f"vendor_k{kv}_permonly"] = dict(
            run=va.graphs["perm"].replay, check=None, meta=dict(part="perm", kmax=kv)
        )
        arms[f"vendor_k{kv}_idealperm"] = dict(
            run=va.graphs["ideal"].replay, check=None, meta=dict(part="bound", kmax=kv)
        )

    # ---------------- rocKE arms
    def add_rk(*args, **kw):
        try:
            _add_rk(*args, **kw)
        except RuntimeError as e:  # e.g. forced grid above the resident capacity
            log(f"SKIP {args[0]}: {e}")

    def _add_rk(
        name,
        phases,
        t,
        c,
        sw,
        pf,
        rh,
        mono,
        kmax,
        bar="full",
        nt=False,
        wpc_force=0,
        wpe=0,
    ):
        tiles = 1 << (n + 1 - t)
        grid = min(tiles, 1 << int(math.log2(n_cu)))
        spec, mats = M.spec_from_plan(
            n,
            phases,
            t=t,
            c=c,
            grid=grid,
            swizzle=sw,
            mono=mono,
            prefetch=pf,
            rhoist=rh,
            bar=bar,
            nt=nt,
            wpe=wpe,
        )
        r = M.CircuitRunner(spec, mats, n_cu=0, arch=arch)
        occ = r.occ
        wpc = wpc_force or a.wgpercu or max(1, min(occ, 4))
        grid = min(tiles, 1 << int(math.log2(n_cu * wpc)))
        if grid != spec.grid:
            spec, mats = M.spec_from_plan(
                n,
                phases,
                t=t,
                c=c,
                grid=grid,
                swizzle=sw,
                mono=mono,
                prefetch=pf,
                rhoist=rh,
                bar=bar,
                nt=nt,
                wpe=wpe,
            )
            r = M.CircuitRunner(spec, mats, n_cu=n_cu, arch=arch)
        nb = sum(len(p.blocks) for p in phases)

        def run(r=r):
            r.run(st)

        def check(r=r):
            st.copy_(psi0)
            r.run(st)
            torch.cuda.synchronize()
            return st

        log(
            f"rk {name}: phases={len(phases)} blocks={nb} grid={spec.grid} occ={occ} "
            f"compile={r.compile_s:.1f}s"
        )
        if mono and bar == "xcd" and spec.grid % 8:
            raise RuntimeError("xcd barrier needs grid % 8 == 0")
        arms[name] = dict(
            run=run,
            check=check,
            tol=nb * (2 << kmax) * U32 * 4,
            meta=dict(
                phases=len(phases),
                blocks=nb,
                grid=spec.grid,
                occ=occ,
                passes=len(phases),
                t=t,
                c=c,
                swizzle=sw,
                prefetch=pf,
                rhoist=rh,
                mono=mono,
                kmax=kmax,
                bar=bar,
                nt=nt,
                wpe=wpe,
                launches=1 if mono else len(phases),
            ),
        )

    for fs in [x for x in a.fused.strip("\"'").split(",") if x]:
        parts = fs.split(":")
        t, c, kf, sw, pf, rh = parts[:6]
        bar = parts[6] if len(parts) > 6 else "full"
        nt = bool(int(parts[7])) if len(parts) > 7 else False
        wpc_force = int(parts[8]) if len(parts) > 8 else 0
        wpe = int(parts[9]) if len(parts) > 9 else 0
        t, c, kf, pf, rh = int(t), int(c), int(kf), bool(int(pf)), bool(int(rh))
        t0 = time.time()
        phases = P.plan_phases(gates, n, t=t, c=c, kf=kf)
        log(f"plan t={t} c={c} kf={kf}: {len(phases)} phases {time.time() - t0:.1f}s")
        tag = (
            f"t{t}c{c}k{kf}{sw}{'pf' if pf else ''}{'' if rh else 'nh'}"
            f"{'' if bar == 'full' else '_' + bar}{'_nt' if nt else ''}"
            f"{f'_w{wpc_force}' if wpc_force else ''}{f'_e{wpe}' if wpe else ''}"
        )
        add_rk(
            f"rk_mono_{tag}",
            phases,
            t,
            c,
            sw,
            pf,
            rh,
            True,
            kf,
            bar,
            nt,
            wpc_force,
            wpe,
        )
        mtag = (
            tag.replace("_" + bar, "") if bar != "full" else tag
        )  # multi: no grid barrier
        if a.multi and f"rk_multi_{mtag}" not in arms:
            add_rk(
                f"rk_multi_{mtag}",
                phases,
                t,
                c,
                sw,
                pf,
                rh,
                False,
                kf,
                "full",
                nt,
                wpc_force,
                wpe,
            )
    for kv in [int(x) for x in a.gett.strip("\"'").split(",") if x]:
        t, c = 13, 5
        low = list(range(c - 1))
        phases = []
        for bl in P.fuse_blocks(gates, kv):
            extra = sorted(set(bl.qubits) - set(low))
            for q in range(n):
                if len(extra) >= t - c:
                    break
                if q not in low and q not in extra:
                    extra.append(q)
            extra = sorted(extra)
            phases.append(
                P.Phase(
                    tuple(extra), [P.pad_block(bl, max(4, len(bl.qubits)), low + extra)]
                )
            )
        add_rk(f"rk_gett_k{kv}", phases, t, c, "x4", False, False, True, max(kv, 4))

    # ---------------- numerics before timing
    res = {}
    for name, arm in arms.items():
        if arm["check"] is None:
            continue
        out = arm["check"]()
        e = rel_err(out, ref)
        ok = e <= arm["tol"]
        res[name] = {"err_pre": e, "tol": arm["tol"], "pass_pre": ok}
        log(
            f"check {name}: rel_l2_err={e:.3e} tol={arm['tol']:.1e} {'PASS' if ok else 'FAIL'}"
        )

    # ---------------- interleaved timing
    names = list(arms)
    for name in names:  # warm
        arms[name]["run"]()
    torch.cuda.synchronize()
    times = {nm: [] for nm in names}
    rng = random.Random(a.seed)
    for rd in range(a.rounds):
        order = names[:]
        rng.shuffle(order)
        for nm in order:
            times[nm].append(time_fn(arms[nm]["run"], a.reps))
        from rocke.runtime.launcher import wait_stream_and_release

        wait_stream_and_release()

    # ---------------- numerics after timing
    for name, arm in arms.items():
        if arm["check"] is None:
            continue
        out = arm["check"]()
        e = rel_err(out, ref)
        res[name]["err_post"] = e
        res[name]["pass_post"] = e <= arm["tol"]
        log(
            f"recheck {name}: rel_l2_err={e:.3e} {'PASS' if e <= arm['tol'] else 'FAIL'}"
        )

    state_bytes = 8 << n
    with open(a.out, "a") as f:
        for nm in names:
            ms = statistics.median(times[nm])
            rec = dict(
                case=case,
                n=n,
                depth=a.depth,
                arm=nm,
                ms_median=ms,
                ms_min=min(times[nm]),
                ms_all=times[nm],
                tag=a.tag,
                device=props.name,
                slurm_job=os.environ.get("SLURM_JOB_ID", ""),
                **arms[nm]["meta"],
                **res.get(nm, {}),
            )
            if "passes" in rec:
                rec["eff_GBps"] = 2 * state_bytes * rec["passes"] / (ms * 1e-3) / 1e9
            f.write(json.dumps(rec) + "\n")
            log(
                f"TIME {nm:34s} {ms:9.3f} ms  min {min(times[nm]):9.3f}  "
                f"{json.dumps(arms[nm]['meta'])}"
            )
    vv = [
        (statistics.median(times[nm]), nm)
        for nm in names
        if nm.startswith("vendor_k") and nm.count("_") == 1
    ]
    rr = [(statistics.median(times[nm]), nm) for nm in names if nm.startswith("rk_")]
    if vv and rr:
        best_v, best_r = min(vv), min(rr)
        log(
            f"SUMMARY {case}: best vendor {best_v[1]} {best_v[0]:.3f} ms | best rocKE {best_r[1]} "
            f"{best_r[0]:.3f} ms | speedup {best_v[0] / best_r[0]:.2f}x"
        )
    bad = [
        nm for nm, r in res.items() if not (r.get("pass_pre") and r.get("pass_post"))
    ]
    if bad:
        log(f"NUMERIC FAIL: {bad}")
        sys.exit(3)


if __name__ == "__main__":
    main()
