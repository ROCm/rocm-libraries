# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CCL-LM (large-message) benchmark, phase diagnostics and xGMI link probe of
the in-kernel all-reduce (gfx950). One process per GPU under torchrun::

    python -m torch.distributed.run --nproc_per_node N \
        -m rocke.examples.gfx950.ccl.bench_ccl_lm bench --out DIR [--extra-arms mod:fn]
    ... bench_ccl_lm diag   --out DIR        # per-CTA phase stamps of the grid-synced chain
    ... bench_ccl_lm linkbw --out DIR        # push / pull bandwidth between the ranks

``bench``: the layer-shaped (grid-synced) all-reduce chain of every CCL-LM
lever arm (``ARMS``: lamport, unit_kib, publish, self_local, striped grid
sync, the cache / slot / pause / wave levers, ``ar_dtype``) against the
W3 forms, RCCL (best protocol per size) and external arms (the aiter custom
all-reduce plugin), interleaved in a new shuffled order every round, numerics
first (every rocKE arm bitwise against the CPU reference on every rank).
The numerics gate runs the timed chain with non-contracting per-rank scales
(``--chain-growth``, default ``ref.LM_GROWTH``: the chain's sum grows every
collective, so a build that reads a previous collective's partial cannot
match; with W3's contracting scales the chain reaches a bitwise fixed point
after 2-3 collectives and such a build passed) and, at every size, the
must-fail stale-read twins (``MUSTFAIL``: the pull ``slot`` and lamport
``noreset`` mutants of the selection's builds) right after the real arms:
a twin the gate does not reject is reported (``[lm] MUSTFAIL MISSED``) and
fails the run. The arm order of every round is shuffled with ``--seed``
(default: fresh per run, recorded). Controls: ``line_os#b`` / ``lam_os_st#b`` (same binary again) and
``line_os#rc`` / ``lam_os_st#rc`` (the same IR compiled again under another
symbol). rocKE arms report the device-stamp latency per collective inside
one persistent launch (and wall time per collective including the launch);
host-launched arms report wall time per call. Per-round times are kept, so the
DESIGN 9.1 win rule is applied offline (``ratios_lm.py``).

``diag``: ``diag=True`` builds of the chosen arms (thread 0 of every CTA
stamps 8 phases of collectives 8-11: start, pushes issued, drained, published,
consumed, grid-sync arrival, release, end). Diagnostic builds are never timed
for ratios (stamps perturb).

``linkbw``: :mod:`kernels.gfx950.qwen38_mono.ccl.lm_probe` push / pull of
``nbytes`` into / from the peer arena, both directions at once or one.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import socket
import sys
import time

import numpy as np

from .bench_ccl import _init, _rccl_groups
from .harness import RF, C, L, LI, Rank, code_digest, first_mismatch, wait_launch_done

K = L.CclKnobs


def _gs(**kw):
    return K(header=False, sync="grid", **kw)


# W3 forms (the base) and the CCL-LM lever arms, all layer-shaped (grid sync after every collective)
BASE = {
    "line_os": _gs(),
    "line_ts": _gs(ar="twoshot"),
    "pair_os": _gs(form="pair"),
    "pair_ts": _gs(form="pair", ar="twoshot"),
}
LEVERS = {
    "line_os_st": _gs(gsync="stripe"),
    "line_ts_st": _gs(ar="twoshot", gsync="stripe"),
    "lam_os": _gs(form="lamport"),
    "lam_ts": _gs(form="lamport", ar="twoshot"),
    "lam_os_st": _gs(form="lamport", gsync="stripe"),
    "lam_ts_st": _gs(form="lamport", ar="twoshot", gsync="stripe"),
    "lam_os_st_u2": _gs(form="lamport", gsync="stripe", unit_kib=2),
    "lam_os_st_u4": _gs(form="lamport", gsync="stripe", unit_kib=4),
    "lam_ts_st_u2": _gs(form="lamport", ar="twoshot", gsync="stripe", unit_kib=2),
    "lam_ts_st_u4": _gs(form="lamport", ar="twoshot", gsync="stripe", unit_kib=4),
    "lam_os_st_sl": _gs(form="lamport", gsync="stripe", self_local=True),
    "lam_os_st_sl_u2": _gs(form="lamport", gsync="stripe", self_local=True, unit_kib=2),
    "lam_os_st_sl_u4": _gs(form="lamport", gsync="stripe", self_local=True, unit_kib=4),
    "line_os_st_u2": _gs(gsync="stripe", unit_kib=2),
    "line_os_st_u4": _gs(gsync="stripe", unit_kib=4),
    "line_ts_st_u2": _gs(ar="twoshot", gsync="stripe", unit_kib=2),
    "line_ts_st_u4": _gs(ar="twoshot", gsync="stripe", unit_kib=4),
    "line_os_st_cnt": _gs(gsync="stripe", publish="counted"),
    "line_os_st_cw": _gs(gsync="stripe", publish="comm_wave"),
    "line_os_st_nt": _gs(gsync="stripe", push_cpol="sc0sc1nt"),
    "lam_os_st_nt": _gs(form="lamport", gsync="stripe", push_cpol="sc0sc1nt"),
    "line_os_st_hd": _gs(gsync="stripe", homing="dual"),
    "line_os_hd": _gs(homing="dual"),
    "lam_os_st_s3": _gs(form="lamport", gsync="stripe", slots=3),
    "line_os_st_s3": _gs(gsync="stripe", slots=3),
    "lam_os_st_ps": _gs(form="lamport", gsync="stripe", pause="s_sleep1"),
    "lam_os_st_pn": _gs(form="lamport", gsync="stripe", pause="none"),
    "line_os_st_ps": _gs(gsync="stripe", pause="s_sleep1"),
    "lam_os_st_w16": _gs(form="lamport", gsync="stripe", waves=16),
    "lam_os_st_w4": _gs(form="lamport", gsync="stripe", waves=4),
    "line_os_st_w16": _gs(gsync="stripe", waves=16),
    "lam_os_st_u2_w16": _gs(form="lamport", gsync="stripe", unit_kib=2, waves=16),
    "pair_os_st": _gs(form="pair", gsync="stripe"),
    "pair_ts_st": _gs(form="pair", ar="twoshot", gsync="stripe"),
    # XCD-weighted unit placement (calibrated per size and GPU pair before numerics / timing)
    "lam_os_st_pt": _gs(form="lamport", gsync="stripe", place="table"),
    "lam_os_pt": _gs(form="lamport", place="table"),
    "lam_os_st_u2_pt": _gs(form="lamport", gsync="stripe", unit_kib=2, place="table"),
    "lam_os_st_u4_pt": _gs(form="lamport", gsync="stripe", unit_kib=4, place="table"),
    "lam_os_st_sl_pt": _gs(form="lamport", gsync="stripe", self_local=True, place="table"),
    "lam_ts_st_pt": _gs(form="lamport", ar="twoshot", gsync="stripe", place="table"),
    "line_os_st_pt": _gs(gsync="stripe", place="table"),
    "line_ts_st_u4_pt": _gs(ar="twoshot", gsync="stripe", unit_kib=4, place="table"),
    # pull (aiter's large-message scheme: stage locally + flag, remote reads)
    "pull_os": _gs(form="pull"),
    "pull_os_st": _gs(form="pull", gsync="stripe"),
    "pull_os_st_u2": _gs(form="pull", gsync="stripe", unit_kib=2),
    "pull_os_st_u4": _gs(form="pull", gsync="stripe", unit_kib=4),
    "pull_os_st_pt": _gs(form="pull", gsync="stripe", place="table"),
    "pull_os_st_u4_pt": _gs(form="pull", gsync="stripe", unit_kib=4, place="table"),
    "pull_os_st_cnt": _gs(form="pull", gsync="stripe", publish="counted"),
    "pull_os_st_u2_cnt": _gs(form="pull", gsync="stripe", unit_kib=2, publish="counted"),
    "pull_os_st_u4_cnt": _gs(form="pull", gsync="stripe", unit_kib=4, publish="counted"),
    "pull_os_u4": _gs(form="pull", unit_kib=4),
    "pull_os_st_u4_w16": _gs(form="pull", gsync="stripe", unit_kib=4, waves=16),
    "pull_os_st_u4_w4": _gs(form="pull", gsync="stripe", unit_kib=4, waves=4),
    # capped-grid pull with the one-word grid sync (the word sync tied or beat the stripe for pull on the default grid)
    "pull_os_u2_g32": _gs(form="pull", unit_kib=2, ctas=32),
    "pull_os_u2_g64": _gs(form="pull", unit_kib=2, ctas=64),
    "pull_os_u4_g32": _gs(form="pull", unit_kib=4, ctas=32),
    "pull_os_u4_g40": _gs(form="pull", unit_kib=4, ctas=40),
    "pull_os_u4_g48": _gs(form="pull", unit_kib=4, ctas=48),
    "pull_os_u4_g56": _gs(form="pull", unit_kib=4, ctas=56),
    "pull_os_u4_g64": _gs(form="pull", unit_kib=4, ctas=64),
    "pull_os_u4_g96": _gs(form="pull", unit_kib=4, ctas=96),
    "pull_os_u4_g128": _gs(form="pull", unit_kib=4, ctas=128),
    "pull_os_u2_g48": _gs(form="pull", unit_kib=2, ctas=48),
    "pull_os_u2_g96": _gs(form="pull", unit_kib=2, ctas=96),
    "pull_os_u2_g128": _gs(form="pull", unit_kib=2, ctas=128),
    "pull_os_st_u4_g40": _gs(form="pull", gsync="stripe", unit_kib=4, ctas=40),
    "pull_os_st_u4_g56": _gs(form="pull", gsync="stripe", unit_kib=4, ctas=56),
    "pull_os_st_u4_g64": _gs(form="pull", gsync="stripe", unit_kib=4, ctas=64),
    # the earlier declared selection's grid caps as knobs (also usable by diag)
    "pull_os_st_u2_g32": _gs(form="pull", gsync="stripe", unit_kib=2, ctas=32),
    "pull_os_st_u2_g64": _gs(form="pull", gsync="stripe", unit_kib=2, ctas=64),
    "pull_os_st_u4_g32": _gs(form="pull", gsync="stripe", unit_kib=4, ctas=32),
    "pull_os_st_u4_g48": _gs(form="pull", gsync="stripe", unit_kib=4, ctas=48),
    # longer poll pauses (less poll traffic on the receive arena)
    "lam_os_st_s4": _gs(form="lamport", gsync="stripe", pause="s_sleep4"),
    "lam_os_st_s16": _gs(form="lamport", gsync="stripe", pause="s_sleep16"),
    "lam_os_st_u4_s4": _gs(form="lamport", gsync="stripe", unit_kib=4, pause="s_sleep4"),
    "lam_os_st_u4_s16": _gs(form="lamport", gsync="stripe", unit_kib=4, pause="s_sleep16"),
    # tolerance class (bf16 partials of f32 data): f32 sizes only, bitwise against its own reference
    "lam_os_st_wbf16": _gs(form="lamport", gsync="stripe", ar_dtype="bf16"),
    "lam_ts_st_wbf16": _gs(form="lamport", ar="twoshot", gsync="stripe", ar_dtype="bf16"),
}
# the same binaries launched on fewer CTAs (every wave loops over more units: fewer concurrent pushers)
CTAS = {
    "lam_os_st_c64": ("lam_os_st", 64),
    "lam_os_st_c32": ("lam_os_st", 32),
    "lam_os_st_u2_c64": ("lam_os_st_u2", 64),
    "lam_os_st_u4_c40": ("lam_os_st_u4", 40),
    "lam_os_st_u4_c32": ("lam_os_st_u4", 32),
    "lam_os_st_u4_c16": ("lam_os_st_u4", 16),
    "lam_ts_st_u4_c32": ("lam_ts_st_u4", 32),
    "pull_os_st_u4_c32": ("pull_os_st_u4", 32),
    "pull_os_st_c64": ("pull_os_st", 64),
    "line_os_st_cnt_c32": ("line_os_st_cnt", 32),
    "line_os_st_c32": ("line_os_st", 32),
    "line_os_st_u4_c32": ("line_os_st_u4", 32),
}
CONTROLS = {"line_os#b": "line_os", "line_os#rc": "line_os", "lam_os_st#b": "lam_os_st", "lam_os_st#rc": "lam_os_st"}
# must-fail stale-read twins, checked by the numerics gate at every size and never timed: the selection's pull build
# reading the other slot and its lamport build without the consumer reset (plus the twin of ``lm_auto`` itself)
MUSTFAIL = {
    "mut_pull_slot": _gs(form="pull", gsync="stripe", unit_kib=4, ctas=48, mutant="slot"),
    "mut_lam_noreset": _gs(form="lamport", gsync="stripe", mutant="noreset"),
}
DIAG_ARMS = ("line_os", "line_ts", "pair_os", "line_os_st", "lam_os", "lam_os_st", "lam_ts_st", "lam_os_st_u4", "lam_os_st_sl", "pull_os_st", "pull_os_st_u4")


def _spec(N, rows, cols, dt, kn):
    try:
        return L.AllReduceSpec(N, rows, cols, dt, kn)
    except ValueError:
        return None


def _arms_for(sel):
    arms = {**BASE, **LEVERS}
    if sel:
        keep = set(sel.split(","))
        arms = {a: k for a, k in arms.items() if a in keep or a in BASE}
    return arms


def _ctas_for(spec_str: str) -> dict:
    """``--ctas base:n/n/n,base:n`` -> extra CTA-count arms ``<base>_c<n>``."""
    out = {}
    for item in filter(None, spec_str.split(",")):
        base, ns = item.split(":")
        for n in ns.split("/"):
            out[f"{base}_c{n}"] = (base, int(n))
    return out


def _mustfail_twin(kn):
    """The stale-read must-fail twin of a build (pull: ``slot``, lamport:
    ``noreset``); None for the tagged forms (their tags reject stale lines)."""
    from dataclasses import replace

    if kn.form == "pull":
        return replace(kn, mutant="slot")
    if kn.form == "lamport":
        return replace(kn, mutant="noreset")
    return None


def mustfail_check(rk: Rank, specs: dict, iters: int) -> dict:
    """Run every must-fail twin once at the timed chain length (short
    deadlines: a twin may also time out) and report per twin whether the
    numerics gate rejected it on some rank (True = detected, as required)."""
    det = {}
    for a, sp in specs.items():
        b = rk.ar_prepare(sp, 4242)
        rk.reset()  # a clean group and (lamport) zeroed payload regions, as the real arms had
        rk.quiesce_for(b)
        rk.launch(b, rk.ar_args(b, iters, dl_first_s=2.0, dl_s=0.5))
        if not wait_launch_done(rk, 60):
            print(f"[rank {rk.rank}] HANG in must-fail twin {sp.name()}: exiting", flush=True)
            os._exit(3)
        bad = first_mismatch(rk.ar_result(sp), rk.ar_ref(sp, 4242, iters)) is not None or rk.group.record().status != 0
        det[a] = any(rk.all_gather(bool(bad)))
        rk.reset()
    return det


def bench_lm(rk: Rank, out: str, rounds: int, iters: int, sizes, protos, extra=None, sel="", ctas="", auto=False, seed=None, mustfail=True):
    import torch
    import torch.distributed as dist

    N = rk.ranks
    if seed is None:
        seed = rk.all_gather(int.from_bytes(os.urandom(4), "little"))[0]  # fresh per run, the same on every rank
    tdt = {"bf16": torch.bfloat16, "f32": torch.float32}
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    arms_kn = _arms_for(sel)
    groups = _rccl_groups(rk, protos) if protos else {}
    ext = None
    if extra:
        mod, fn = extra.split(":")
        ext = getattr(importlib.import_module(mod), fn)(rk, iters=iters)
    hd = [a for a, kn in arms_kn.items() if kn.homing == "dual"]
    pages = set()
    for a in hd:
        for r, c, d in sizes:
            sp = _spec(N, r, c, d, arms_kn[a])
            if sp is not None:
                lay = L.make_layout([sp])
                pages |= {lay.flag_page(e.name) for e in lay.exchanges if e.flags >= 0}
    flips = rk.calibrate_flips(pages) if pages else 0
    if rk.rank == 0:
        print(f"[lm] code={code_digest()} node={socket.gethostname()} buses={rk.group.buses} protos={list(protos)} extra={extra} "
              f"arms={len(arms_kn)} flips={flips:#x} seed={seed} chain_growth={rk.chain_growth} mustfail={mustfail}", flush=True)
    rows_all = []
    missed = []
    for rows, cols, dt in sizes:
        specs = {}
        for a, kn in arms_kn.items():
            if kn.ar_dtype != "wire" and dt != "f32":
                continue
            sp = _spec(N, rows, cols, dt, kn)
            if sp is not None:
                specs[a] = sp
        if auto:  # the declared CCL-LM selection (layout.lm_select) in the layer shape, plus its own controls
            from dataclasses import replace

            kn = replace(L.lm_select(N, rows * cols * (2 if dt == "bf16" else 4), dt, layer=True), header=False)
            specs["lm_auto"] = L.AllReduceSpec(N, rows, cols, dt, kn)
            CONTROLS.update({"lm_auto#b": "lm_auto", "lm_auto#rc": "lm_auto"})
        mf_specs = {}
        if mustfail:
            for a, kn in MUSTFAIL.items():
                sp = _spec(N, rows, cols, dt, kn)
                if sp is not None:
                    mf_specs[a] = sp
            if "lm_auto" in specs and _mustfail_twin(specs["lm_auto"].knobs) is not None:
                sp = _spec(N, rows, cols, dt, _mustfail_twin(specs["lm_auto"].knobs))
                if sp is not None:
                    mf_specs["mut_lm_auto"] = sp
        built, numer, shares = {}, {}, {}
        for a, sp in specs.items():
            if sp.knobs.place == "table":
                shares[a] = calibrate_xcd(rk, sp)
            b = rk.ar_prepare(sp, 42)
            if sp.knobs.place == "table":
                rk.set_utab(b, shares[a]["shares"])
            rk.launch(b, rk.ar_args(b, iters, flips=flips if sp.knobs.homing == "dual" else 0))
            rk.sync()
            ok = first_mismatch(rk.ar_result(sp), rk.ar_ref(sp, 42, iters)) is None and rk.group.record().status == 0
            numer[a] = all(rk.all_gather(ok))
            built[a] = b
            if rk.group.record().status:
                rk.reset()
        # must-fail twins right after the real arms (the gate must reject every one of them)
        mf = mustfail_check(rk, mf_specs, iters) if mf_specs else {}
        for a, d in mf.items():
            if not d:
                missed.append(f"{rows}x{cols}x{dt}:{a}")
        for cname, (base, n) in {**CTAS, **_ctas_for(ctas)}.items():
            if base not in built or not numer.get(base) or n >= built[base].nctas or built[base].spec.knobs.place != "rr":
                continue
            sp = specs[base]
            rk.ar_prepare(sp, 42)
            bc = rk.with_ctas(built[base], n)
            rk.launch(bc, rk.ar_args(bc, iters))
            rk.sync()
            ok = first_mismatch(rk.ar_result(sp), rk.ar_ref(sp, 42, iters)) is None and rk.group.record().status == 0
            numer[cname] = all(rk.all_gather(ok))
            built[cname] = bc
            if rk.group.record().status:
                rk.reset()
        for cname, base in CONTROLS.items():
            if base not in specs:
                continue
            if cname.endswith("#b"):
                built[cname] = built[base]
                numer[cname] = numer[base]
            else:
                sp = specs[base]
                b = rk.build(sp, C.allreduce_kernel_in, rename="_rc")
                rk.ar_prepare(sp, 42)
                rk.quiesce_for(b)
                rk.launch(b, rk.ar_args(b, iters))
                rk.sync()
                ok = first_mismatch(rk.ar_result(sp), rk.ar_ref(sp, 42, iters)) is None and rk.group.record().status == 0
                numer[cname] = all(rk.all_gather(ok))
                built[cname] = b
        # RCCL + external arms
        x = torch.randn(rows, cols, device=f"cuda:{rk.device}").to(tdt[dt])
        allx = [None] * N
        dist.all_gather_object(allx, x.float().cpu().numpy(), group=rk.gloo)
        ref = np.sum(np.stack(allx).astype(np.float64), axis=0)
        tol = 2e-2 if dt == "bf16" else 1e-5
        xr = x.clone()
        graphs = {}
        for p, g in groups.items():
            y = x.clone()
            dist.all_reduce(y, group=g)
            torch.cuda.synchronize()
            err = float(np.max(np.abs(y.float().cpu().numpy() - ref)) / (np.max(np.abs(ref)) + 1e-30))
            numer[f"rccl_graph_{p}"] = all(rk.all_gather(err < tol))
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    dist.all_reduce(xr, group=g)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                for _ in range(iters):
                    dist.all_reduce(xr, group=g)
            graphs[f"rccl_graph_{p}"] = gr
        ext_arms, ext_info = {}, {}
        if ext is not None:
            ed = ext.prepare(rows, cols, dt, x, ref, tol)
            ext_arms = ed.get("arms", {})
            for a in ext_arms:
                numer[a] = bool(ed.get("numerics_ok", {}).get(a, False))
            ext_info = ed.get("info", {})
        # an arm whose numerics failed is never timed (reported with numerics FAIL)
        dropped = {a: "numerics" for a in built if not numer.get(a, False)}
        rk_arms = [a for a in built if a not in dropped]
        arms = rk_arms + [a for a in graphs if numer.get(a)] + [a for a in ext_arms if numer.get(a)]
        wall = {a: [] for a in arms}
        dev = {a: [] for a in rk_arms}
        round_t = []  # UTC epoch seconds (start, end) of every round: the clock sampler's samples map onto rounds
        drop_records = {}
        for rnd in range(rounds):
            t_start = time.time()
            order = arms[:]
            random.Random(seed * 1000003 + rnd * 7919 + rows * 31 + cols + (dt == "f32")).shuffle(order)
            for a in order:
                if a in dropped:
                    continue
                if a in built:
                    b = built[a]
                    rk.quiesce_for(b)  # sync + barrier on a layout change (lamport: zero its region), outside the timing
                rk.barrier()
                torch.cuda.synchronize()
                if a in built:
                    b = built[a]
                    args = rk.ar_args(b, iters, flips=flips if b.spec.knobs.homing == "dual" else 0)
                    ev0.record()
                    rk.launch(b, args)
                    ev1.record()
                    torch.cuda.synchronize()
                    wall[a].append(ev0.elapsed_time(ev1) * 1e3 / iters)
                    st = rk.stamps(b)
                    dev[a].append(float(st[:, 2].max() - st[:, 1].max()) / (iters - 1) * 10.0 / 1e3)  # us per collective
                elif a in graphs:
                    ev0.record()
                    graphs[a].replay()
                    ev1.record()
                    torch.cuda.synchronize()
                    wall[a].append(ev0.elapsed_time(ev1) * 1e3 / iters)
                else:
                    kind, fn = ext_arms[a]
                    k = iters if kind == "graph" else 64
                    ev0.record()
                    if kind == "graph":
                        fn()
                    else:
                        for _ in range(64):
                            fn()
                    ev1.record()
                    torch.cuda.synchronize()
                    wall[a].append(ev0.elapsed_time(ev1) * 1e3 / k)
                if a in built and any(rk.all_gather(rk.group.record().status != 0)):
                    # a CCL error on any rank: drop the arm everywhere (every rank sees the same gather), reset the group;
                    # every rank's error record is kept (the originating rank's code / stage / use / tags classify it)
                    rec = rk.group.record()
                    recs = rk.all_gather({k: getattr(rec, k) for k in ("status", "code", "stage", "use", "peer", "tag_expected", "tag_seen", "cta",
                                                                       "rank")} | {"name": rec.name})
                    prev = order[order.index(a) - 1] if order.index(a) else None
                    dropped[a] = f"error {rec.name if rec.status else 'on a peer'}"
                    drop_records[a] = {"round": rnd, "prev_arm": prev, "records": recs}
                    if rk.rank == 0:
                        print(f"[lm] dropped {a} at round {rnd} (after {prev}): {dropped[a]}; records {json.dumps(recs)}", flush=True)
                    rk.reset()
            round_t.append((t_start, time.time()))
        for a in dropped:
            wall.pop(a, None)
            dev.pop(a, None)
        allr = rk.all_gather({"wall_us": wall, "dev_us": dev})
        row = {
            "rows": rows, "cols": cols, "dtype": dt, "ranks": N, "iters": iters, "rounds": rounds, "kib_per_rank": rows * cols * (2 if dt == "bf16" else 4) / 1024,
            "numerics_ok": numer, "per_rank_rounds": allr, "bus": rk.group.buses, "mem": rk.mem, "node": socket.gethostname(), "code": code_digest(),
            "specs": {a: built[a].spec.name() for a in built}, "nctas": {a: built[a].nctas for a in built}, "ext_info": ext_info, "xcd_shares": shares,
            "dropped": dropped, "seed": seed, "chain_growth": rk.chain_growth, "mustfail": mf, "round_t": round_t, "drop_records": drop_records,
            "mustfail_specs": {a: s.name() for a, s in mf_specs.items()},
        }  # fmt: skip
        rows_all.append(row)
        if rk.rank == 0:
            def rmax(kind, a):  # per-round rank-max, then median
                per = np.max(np.array([r[kind][a] for r in allr]), axis=0)
                return float(np.median(per))

            base = min((rmax("dev_us", a), a) for a in BASE if a in dev)
            lev = sorted((rmax("dev_us", a), a) for a in dev if a not in BASE and "#" not in a and "wbf16" not in a)
            msg = f"[lm] N={N} {rows}x{cols} {dt} ({row['kib_per_rank']:.0f} KiB/rank): base best {base[1]}"
            if lev:
                msg += f"; best lever {lev[0][1]} = {lev[0][0] / base[0]:.3f} of base"
            ea = [a for a in ext_arms if "graph" in a]
            if ea:
                e = min((rmax("wall_us", a), a) for a in ea)
                msg += f"; base/{e[1]} = {base[0] / e[0]:.3f}"
                if lev:
                    msg += f", best lever/{e[1]} = {lev[0][0] / e[0]:.3f}"
            ctl = {c: rmax("dev_us", c) / rmax("dev_us", CONTROLS[c]) for c in CONTROLS if c in dev and CONTROLS[c] in dev}
            msg += f"; controls {json.dumps({k: round(v, 3) for k, v in ctl.items()})}; numerics {all(numer.values())}"
            print(msg, flush=True)
            if not all(numer.values()):
                print(f"[lm] NUMERICS FAILED: {[a for a, v in numer.items() if not v]}", flush=True)
            if mf:
                print(f"[lm] MUSTFAIL {'ok' if all(mf.values()) else 'MISSED'} {rows}x{cols} {dt}: {json.dumps(mf)}", flush=True)
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, f"lm_bench_n{N}{os.environ.get('CCL_TAG', '')}.json"), "w") as fh:
                json.dump(rows_all, fh)
        del graphs
        if ext is not None:
            ext.release()
        torch.cuda.synchronize()
    if missed and rk.rank == 0:
        print(f"[lm] MUSTFAIL MISSED (the numerics gate accepted a stale-read twin): {missed}", flush=True)
    return rows_all, missed


def calibrate_xcd(rk: Rank, spec, passes: int = 4, iters: int = 16) -> dict:
    """``place="table"``: per-XCD unit shares that make the XCDs finish
    together. Runs the ``diag`` twin of ``spec`` with the current shares,
    takes per XCD the latest CTA arrival at the grid sync (phase 5, the
    CTA's slowest wave done) of each stamped collective (median over the 4
    collectives, max over ranks), and moves the shares toward the measured
    per-XCD rates (``share / time``), damped by half, ``passes`` times.
    Both ranks use the same shares. Returns the shares and the history."""
    from dataclasses import replace

    sp_d = replace(spec, knobs=replace(spec.knobs, diag=True))
    sh = np.full(8, 1.0 / 8)
    hist = []
    b = rk.ar_prepare(sp_d, 42)
    for _ in range(passes):
        tab = rk.set_utab(b, sh)
        rk.tensor("stamps", b.nctas * L.stamp_cols(sp_d.knobs) * 8).zero_()
        rk.sync()
        rk.barrier()
        rk.launch(b, rk.ar_args(b, iters))
        if not wait_launch_done(rk, 60):
            os._exit(3)
        ok = first_mismatch(rk.ar_result(sp_d), rk.ar_ref(sp_d, 42, iters)) is None and rk.group.record().status == 0
        st = rk.stamps(b).astype(np.float64)
        own = np.diff(np.asarray(tab)).reshape(b.nctas, -1).sum(axis=1)  # units per CTA
        tx = np.full((L.DIAG_N, 8), np.nan)
        for j in range(L.DIAG_N):
            blk = st[:, 3 + j * L.DIAG_PH: 3 + (j + 1) * L.DIAG_PH].copy()
            blk[blk == 0] = np.nan
            t0 = np.nanmin(blk[:, 0])
            for x in range(8):
                sel = [c for c in range(x, b.nctas, 8) if own[c] > 0]
                if sel:
                    tx[j, x] = np.nanmax(blk[sel, 5] - t0) / 100.0
        t_rank = np.nanmedian(tx, axis=0)
        g = rk.all_gather({"t": t_rank.tolist(), "ok": bool(ok)})
        t = np.nanmax(np.array([x["t"] for x in g], dtype=np.float64), axis=0)
        hist.append({"shares": sh.tolist(), "t_us": t.tolist(), "ok": all(x["ok"] for x in g)})
        live = np.isfinite(t) & (sh > 0)
        if not live.any():
            break
        rate = np.where(live, sh / np.where(live, t, 1.0), 0.0)
        new = rate / rate.sum()
        sh = 0.5 * sh + 0.5 * new
        sh = sh / sh.sum()
    rk.reset()  # the diag twin shares the arena map; start the timed build from a clean group
    return {"shares": sh.tolist(), "history": hist}


def diag_lm(rk: Rank, out: str, sizes, launches: int = 5, iters: int = 16, sel=""):
    """Phase stamps (``diag=True`` builds) of the chosen arms."""
    from dataclasses import replace

    N = rk.ranks
    arms = [a for a in (sel.split(",") if sel else DIAG_ARMS)]
    allk = {**BASE, **LEVERS}
    res = []
    for rows, cols, dt in sizes:
        for a in arms:
            kn = replace(allk[a], diag=True)
            sp = _spec(N, rows, cols, dt, kn)
            if sp is None:
                continue
            b = rk.ar_prepare(sp, 42)
            st_all, ok_all = [], True
            for li in range(launches):
                rk.quiesce_for(b)
                rk.tensor("stamps", b.nctas * L.stamp_cols(kn) * 8).zero_()
                rk.sync()
                rk.barrier()
                rk.launch(b, rk.ar_args(b, iters))
                if not wait_launch_done(rk, 60):
                    os._exit(3)
                ok_all &= first_mismatch(rk.ar_result(sp), rk.ar_ref(sp, 42, iters)) is None and rk.group.record().status == 0
                st_all.append(rk.stamps(b).tolist())
            g = rk.all_gather({"stamps": st_all, "ok": ok_all})
            res.append({"arm": a, "spec": sp.name(), "rows": rows, "cols": cols, "dtype": dt, "nctas": b.nctas, "ranks": N,
                        "per_rank": g, "node": socket.gethostname(), "bus": rk.group.buses, "code": code_digest()})
            if rk.rank == 0:
                print(f"[diag] {a} {rows}x{cols} {dt}: {_diag_summary(g, L.DIAG_N, L.DIAG_PH)} numerics {all(x['ok'] for x in g)}", flush=True)
    if rk.rank == 0:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, f"lm_diag_n{N}{os.environ.get('CCL_TAG', '')}.json"), "w") as fh:
            json.dump(res, fh)
    return res


def _diag_summary(per_rank, dn, dp):
    """Median over (rank, launch, collective, CTA) of each phase's offset from
    the collective's earliest CTA start, in ticks of 10 ns; plus the span."""
    offs = {p: [] for p in range(1, dp)}
    spans = []
    for r in per_rank:
        for st in r["stamps"]:
            s = np.array(st, dtype=np.int64)
            for j in range(dn):
                blk = s[:, 3 + j * dp: 3 + (j + 1) * dp].astype(np.float64)
                blk[blk == 0] = np.nan
                t0 = np.nanmin(blk[:, 0])
                for p in range(1, dp):
                    offs[p] += list(blk[:, p] - t0)
                spans.append(np.nanmax(blk[:, dp - 1]) - t0)
    med = {p: (round(float(np.nanmedian(v)) / 100, 2) if np.isfinite(np.nanmedian(v)) else None) for p, v in offs.items() if v}
    return {"phase_us_median": med, "span_us_median": round(float(np.median(spans)) / 100, 2)}


def _xrank(mask: int):
    r, k = 0, 0
    for x in range(8):
        if (mask >> x) & 1:
            r |= k << (4 * x)
            k += 1
    return r, k


def linkbw(rk: Rank, out: str, reps: int = 7, full: bool = True):
    """Push / pull bandwidth probe (ranks drive peer (rank + 1) mod N), all
    XCDs and XCD subsets (``xmask``: which XCDs' CTAs move the bytes)."""
    from rocke.helpers.compile import compile_kernel

    from kernels.gfx950.qwen38_mono.ccl import lm_probe as LP

    N = rk.ranks
    off0 = 128 << 20
    res = []
    cfgs = []
    for mode, nt in (("push", False), ("push", True), ("pull", False)):
        for vec in ((1, 4) if full else (4,)):
            for nbytes in (5 << 19, 32 << 20):
                for nctas in (80, 256):
                    cfgs.append((mode, nt, vec, nbytes, nctas, 0xFF))
    for mode in ("push", "pull"):
        for mask in (0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x03, 0x0C, 0x30, 0xC0, 0x0F, 0xF0, 0x3C, 0xC3):
            cfgs.append((mode, False, 4, 1 << 20, 256, mask))
    mods = {}
    for mode, nt, vec, nbytes, nctas, mask in cfgs:
        sp = LP.LinkBwSpec(N, mode, nt, 8, vec)
        if sp.name() not in mods:
            k = LP.build_linkbw_kernel(sp)
            LI.check(k)
            art = compile_kernel(k, arch="gfx950", backend="python")
            mod = rk.rt.load_module(art.hsaco)
            rk._mods.append(mod)
            mods[sp.name()] = (k, mod.get_function(art.kernel_name))
        k, fn = mods[sp.name()]
        xr, na = _xrank(mask)
        for direction in ("bi", "uni"):
            ts = []
            for _ in range(reps):
                active = 1 if (direction == "bi" or rk.rank == 0) else 0
                v = dict(rk.group.kernargs())
                v.update(sink=rk.tensor("lbwsink", 256 * 16 * 64 * 4).data_ptr(), stamps=rk.tensor("lbwst", 256 * 2 * 8).data_ptr(),
                         rank=rk.rank, nctas=nctas, nbytes=nbytes, off0=off0, active=active, xmask=mask, xrank=xr, nact=na)
                args = C.pack_kernargs(k, v)
                rk.sync()
                rk.barrier()
                rk.rt.launch(fn, (nctas, 1, 1), (sp.threads, 1, 1), args, stream=rk.stream)
                rk.sync()
                st = rk.get("lbwst", nctas * 2 * 8, np.int64).reshape(nctas, 2)
                ts.append(float(st[:, 1].max() - st[:, 0].min()) * 10.0 / 1e3)  # us
            g = rk.all_gather(ts)
            res.append({"mode": mode, "nt": nt, "vec": vec, "nbytes": nbytes, "nctas": nctas, "xmask": mask, "dir": direction, "us_per_rank": g,
                        "node": socket.gethostname(), "bus": rk.group.buses})
            if rk.rank == 0:
                m0 = float(np.median(g[0]))
                print(f"[linkbw] {mode}{'nt' if nt else ''} v{vec} {nbytes >> 10} KiB ctas {nctas} xmask {mask:#04x} {direction}: rank0 median {m0:.2f} us "
                      f"-> {nbytes / m0 / 1e3:.1f} GB/s", flush=True)
    if rk.rank == 0:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, f"lm_linkbw_n{N}{os.environ.get('CCL_TAG', '')}.json"), "w") as fh:
            json.dump(res, fh)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bench", "diag", "linkbw"])
    ap.add_argument("--out", default=os.environ.get("Q38_OUT", "ccl_out"))
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--iters", type=int, default=96)
    ap.add_argument("--sizes", default="16x2560xbf16,32x2560xbf16,64x2560xbf16,128x2560xbf16,8x10240xbf16,32x10240xbf16,64x10240xbf16,128x10240xbf16")
    ap.add_argument("--rccl-protos", default="LL,LL128,Simple")
    ap.add_argument("--extra-arms", default="")
    ap.add_argument("--arms", default="", help="comma list of lever arms (the base forms always run)")
    ap.add_argument("--ctas", default="", help="extra CTA-count arms: base:n/n,base:n (the base must be in the arm set)")
    ap.add_argument("--no-ctas-default", action="store_true", help="skip the built-in CTAS arms")
    ap.add_argument("--auto", action="store_true", help="add the declared selection arm lm_auto (layout.lm_select) and its controls")
    ap.add_argument("--chain-growth", type=float, default=RF.LM_GROWTH,
                    help="per-collective growth of the all-reduce chain (ref.chain_scales); 1 = W3's contracting chain (blind to stale reads)")
    ap.add_argument("--seed", type=int, default=None, help="arm-order shuffle seed (default: fresh per run, recorded)")
    ap.add_argument("--no-mustfail", action="store_true", help="skip the must-fail stale-read twins")
    a = ap.parse_args(argv)
    dev, gloo = _init(backend_nccl=a.cmd == "bench")
    mem = os.environ.get("CCL_MEM", "uncached")
    rk = Rank(device=dev, mem=mem, group=gloo)
    rk.gloo = gloo
    rk.chain_growth = a.chain_growth
    sizes = []
    for t in a.sizes.split(","):
        r, c, d = t.split("x")
        sizes.append((int(r), int(c), d))
    if rk.rank == 0:
        print(f"[ccl-lm] ranks={rk.ranks} mem={mem} buses={rk.group.buses} code={code_digest()} node={socket.gethostname()} cmd={a.cmd}", flush=True)
    if a.cmd == "bench":
        if a.no_ctas_default:
            CTAS.clear()
        _, missed = bench_lm(rk, a.out, a.rounds, a.iters, sizes, tuple(p for p in a.rccl_protos.split(",") if p), extra=a.extra_arms or None, sel=a.arms,
                             ctas=a.ctas, auto=a.auto, seed=a.seed, mustfail=not a.no_mustfail)
        rc = 4 if missed else 0
    elif a.cmd == "diag":
        rc = 0
        diag_lm(rk, a.out, sizes, sel=a.arms)
    else:
        rc = 0
        linkbw(rk, a.out)
    rk.close()
    import torch.distributed as dist

    dist.destroy_process_group()
    sys.exit(rc)


if __name__ == "__main__":
    main()
