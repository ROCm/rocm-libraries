# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Multi-process GPU suite and microbenchmark of the in-kernel CCL (gfx950).

Run one process per GPU under torchrun (the rocKE ``library/`` root on
``sys.path``)::

    python -m torch.distributed.run --nproc_per_node N \
        -m rocke.examples.gfx950.ccl.bench_ccl test  --out DIR
    python -m torch.distributed.run --nproc_per_node N \
        -m rocke.examples.gfx950.ccl.bench_ccl bench --out DIR [--rounds 12]

``test``: bitwise rank agreement against the CPU references (every
collective, form and size), consecutive launches without reset under host
skew, the u32 tag wrap, delay-injection stress, and the negative controls
(dropped peer, dropped use, aborting peer, stale / ahead tags, launch-header
mismatch, acceptance mutants, failure containment, coherence bits). Each test
prints one PASS / FAIL line per collective outcome; every must-fail control
asserts its expected failure (a vacuous control fails the test);
``DIR/test_n{N}.json`` holds the per-rank details.

``bench``: per-collective latency of the rocKE all-reduce against RCCL
``all_reduce`` through ``torch.distributed``, interleaved in a shuffled order
per round, numerics first. rocKE arms: a persistent chain of ``--iters``
collectives (device stamps) in which each wave waits only for its own unit
(the best case), the same chain with a grid-wide barrier after every
collective (``sync="grid"``: a consumer that needs the whole vector), that
grid-synced chain on at most 32 CTAs, and one collective per launch (header
off, and the default header-on model build). RCCL arms: a graph-captured
chain and eager calls on one communicator per ``--rccl-protos`` entry
(``NCCL_PROTO`` set while that communicator is created; ``default`` = RCCL's
own choice), "RCCL best" = the fastest of them. ``--extra-arms mod:fn`` adds
external arms (e.g. a custom all-reduce library). ``DIR/bench_n{N}.json``
holds raw device times; the console prints ratios.

Environment: ``CCL_MEM`` (uncached | fine | coarse), ``CCL_DEVICE_SHIFT``
(rank r drives GPU (r + shift) mod N: the GPU-permutation run), ``CCL_TAG``
(output file suffix).
"""

from __future__ import annotations

import argparse
import importlib
import json
import socket
import os
import random
import sys
import time
import traceback

import numpy as np

from .harness import C, L, LI, RF, Rank, code_digest, digest, first_mismatch, wait_launch_done

TAG_MOD = (1 << 32) - 1


def _init(backend_nccl: bool):
    import torch
    import torch.distributed as dist

    local = int(os.environ.get("LOCAL_RANK", "0"))
    n = int(os.environ.get("WORLD_SIZE", "1"))
    shift = int(os.environ.get("CCL_DEVICE_SHIFT", "0"))
    dev = (local + shift) % max(n, 1)
    torch.cuda.set_device(dev)
    if backend_nccl:
        dist.init_process_group(backend="nccl", device_id=torch.device("cuda", dev))
        gloo = dist.new_group(backend="gloo")
    else:
        dist.init_process_group(backend="gloo")
        gloo = None
    return dev, gloo


# ===================================================================== tests


class Suite:
    def __init__(self, rk: Rank, out: str, quick: bool):
        self.rk, self.out, self.quick = rk, out, quick
        self.rows = []

    def log(self, test, ok, **info):
        info = {k: (v if isinstance(v, (int, float, str, bool, type(None), list, dict)) else str(v)) for k, v in info.items()}
        self.rows.append({"test": test, "rank": self.rk.rank, "ok": bool(ok), **info})

    def summary(self):
        allrows = self.rk.all_gather(self.rows)
        if self.rk.rank == 0:
            flat = [r for rows in allrows for r in rows]
            by = {}
            for r in flat:
                by.setdefault(r["test"], []).append(r)
            npass = 0
            for t, rs in by.items():
                ok = all(r["ok"] for r in rs)
                npass += ok
                bad = [r for r in rs if not r["ok"]][:2]
                print(f"{'PASS' if ok else 'FAIL'} {t} ({len(rs)} rank-checks){'  ' + json.dumps(bad)[:600] if bad else ''}", flush=True)
            print(f"SUMMARY {npass}/{len(by)} tests passed (ranks={self.rk.ranks}, mem={self.rk.mem}, code={code_digest()}, node={socket.gethostname()}, "
                  f"buses={self.rk.group.buses})", flush=True)
            os.makedirs(self.out, exist_ok=True)
            tag = os.environ.get("CCL_TAG", "")
            with open(os.path.join(self.out, f"test_n{self.rk.ranks}{tag}.json"), "w") as fh:
                json.dump(flat, fh, indent=1)
            return npass == len(by)
        return True

    # -------------------------------------------------------------- helpers
    def run_ar(self, spec, seed, iters, launches=1, dl_first_s=10.0, dl_s=1.0, hang_s=60.0, pre_launch=None):
        rk = self.rk
        b = rk.ar_prepare(spec, seed)
        res = []
        for li in range(launches):
            if pre_launch:
                pre_launch(li)
            rk.launch(b, rk.ar_args(b, iters, dl_first_s=dl_first_s, dl_s=dl_s))
            if not wait_launch_done(rk, hang_s):
                print(f"[rank {rk.rank}] HANG in {spec.name()}: exiting", flush=True)
                os._exit(3)
            res.append(rk.ar_result(spec))
        return b, res

    def check_ar(self, test, spec, seed, iters, res, **info):
        rk = self.rk
        want = rk.ar_ref(spec, seed, iters)
        rec = rk.group.record()
        ok = rec.status == 0
        mism = None
        for got in res:
            mm = first_mismatch(got, want)
            if mm is not None:
                ok, mism = False, mm
        dg = rk.all_gather(digest(res[-1]))
        agree = len(set(dg)) == 1
        err = None
        if rec.status:
            err = {"code": rec.name, "stage": rec.stage, "use": rec.use, "peer": rec.peer, "tag_expected": rec.tag_expected,
                   "tag_seen": rec.tag_seen, "cta": rec.cta}
        self.log(test, ok and agree, spec=spec.name(), mismatch_byte=mism, agree=agree, err=err, **info)
        if any(rk.all_gather(rec.status != 0)):
            self.reset()  # an error record is sticky until the collective reset
        return ok and agree

    def reset(self, epoch=1):
        self.rk.reset(epoch=epoch)

    # ------------------------------------------------------------- the tests
    def t_single_rank(self):
        """N = 1 group per process: the whole exchange path (push to self,
        poll, sum of one) equals the input."""
        rk = self.rk
        from rocke.runtime import peer as P

        saved = rk.group
        rk.group = P.PeerGroup(0, 1, 16 << 20, mem=rk.mem, device=rk.device)
        try:
            for kn in [L.CclKnobs(), L.CclKnobs(form="pair"), L.CclKnobs(ar="twoshot"), L.CclKnobs(form="pair", ar="twoshot")]:
                for dt in ("bf16", "f32"):
                    sp = L.AllReduceSpec(1, 3, 2560, dt, kn)
                    b = rk.ar_prepare(sp, 11)
                    rk.launch(b, rk.ar_args(b, 3))
                    rk.sync()
                    got = rk.ar_result(sp)
                    from .harness import src_wire

                    want, _ = RF.allreduce_chain([src_wire(sp, rk.rank, 11)], dt, 3, [1.0], twoshot=kn.ar == "twoshot")
                    self.log("single_rank", first_mismatch(got, want) is None and rk.group.record().status == 0, spec=sp.name())
        finally:
            rk.group.close()
            rk.group = saved
            rk._cache = {k: v for k, v in rk._cache.items() if "_n1_" not in k}

    def t_bitwise(self):
        N = self.rk.ranks
        rows = [1, 5, 128] if self.quick else [1, 3, 5, 17, 127, 128]
        for kn in [L.CclKnobs(), L.CclKnobs(form="pair"), L.CclKnobs(ar="twoshot"), L.CclKnobs(form="pair", ar="twoshot")]:
            for cols, dt in [(2560, "bf16"), (2560, "f32"), (10240, "bf16"), (10240, "f32")]:
                for r in rows:
                    sp = L.AllReduceSpec(N, r, cols, dt, kn)
                    _, res = self.run_ar(sp, 1000 + r, 3, launches=2)
                    self.check_ar("bitwise_allreduce", sp, 1000 + r, 3, res)
        # the four default forms above run the launch header (default on); the header-off and grid-sync builds here
        for kn in [L.CclKnobs(homing="dual"), L.CclKnobs(slots=3), L.CclKnobs(push_cpol="sc0sc1nt", pause="s_sleep1"), L.CclKnobs(header=False),
                   L.CclKnobs(carry="mem"), L.CclKnobs(form="pair", carry="mem"), L.CclKnobs(ar="twoshot", carry="mem"), L.CclKnobs(waves=4),
                   L.CclKnobs(sync="grid"), L.CclKnobs(form="pair", ar="twoshot", sync="grid"), L.CclKnobs(form="pair", sync="grid", header=False)]:
            sp = L.AllReduceSpec(N, 17, 2560, "bf16", kn)
            _, res = self.run_ar(sp, 77, 4, launches=3)
            self.check_ar("bitwise_allreduce_knobs", sp, 77, 4, res)

    def t_allgather(self):
        rk, N = self.rk, self.rk.ranks
        for kn in [L.CclKnobs(), L.CclKnobs(form="pair"), L.CclKnobs(header=False)]:
            for rows, cols, dt in [(1, 2560, "bf16"), (16, 1280, "f32"), (128, 2560, "bf16"), (5, 512, "f32")]:
                sp = L.AllGatherSpec(N, rows, cols, dt, kn)
                b = rk.ag_prepare(sp, 5)
                for li in range(2):
                    rk.launch(b, rk.ag_args(b, 3))
                    rk.sync()
                got = rk.ag_result(sp)
                want = rk.ag_ref(sp, 5, 3)
                dg = rk.all_gather(digest(got))
                self.log("bitwise_allgather", first_mismatch(got, want) is None and len(set(dg)) == 1 and rk.group.record().status == 0, spec=sp.name())

    def t_argmax(self):
        rk, N = self.rk, self.rk.ranks
        for rows, hdr in ((1, True), (16, True), (100, True), (128, True), (100, False)):
            sp = L.ArgmaxSpec(N, rows, L.CclKnobs(form="pair", waves=2, header=hdr))
            ov, oi, wv, wi = rk.amax_run(sp, 3 + rows, 5)
            ties = 0
            dg = rk.all_gather(digest(np.concatenate([ov.view(np.int32), oi])))
            ok = np.array_equal(ov.view(np.int32), wv.view(np.int32)) and np.array_equal(oi, wi) and len(set(dg)) == 1
            self.log("argmax_merge_ties", ok and rk.group.record().status == 0, spec=sp.name(), ties=ties)

    def t_consecutive(self):
        rk = self.rk
        n = 60 if self.quick else 1000
        rnd = random.Random(1234 + rk.rank)
        sp = L.AllReduceSpec(rk.ranks, 8, 2560, "bf16")
        _, res = self.run_ar(sp, 9, 2, launches=n, pre_launch=lambda li: time.sleep(rnd.random() * 0.02))
        self.check_ar("consecutive_launches_skewed", sp, 9, 2, res, launches=n, control=rk.control())

    def t_wrap(self):
        rk = self.rk
        start = TAG_MOD - 300
        for kn in [L.CclKnobs(), L.CclKnobs(form="pair"), L.CclKnobs(ar="twoshot", carry="mem")]:
            self.reset(epoch=start)  # every build crosses the wrap (200 uses from 2^32 - 301)
            sp = L.AllReduceSpec(rk.ranks, 4, 2560, "bf16", kn)
            _, res = self.run_ar(sp, 21, 8, launches=50)
            ctl = rk.control()
            self.check_ar("tag_wrap", sp, 21, 8, res, epoch_after=ctl["epoch"], wrapped=ctl["epoch"] < start)
        self.reset()

    def t_stress(self):
        rk = self.rk
        n = 50 if self.quick else 2000
        smax = 400  # up to 400 x s_sleep 1 (64 clocks each) at every push and poll point
        sp = L.AllReduceSpec(rk.ranks, 16, 2560, "bf16", L.CclKnobs(hooks=True))
        b = rk.ar_prepare(sp, 31)
        rnd = random.Random(99)
        res = []
        for li in range(n):
            rk.set_hooks(salt=rnd.randrange(1, 1 << 30), smax=smax)
            rk.launch(b, rk.ar_args(b, 4))
            if not wait_launch_done(rk, 60):
                os._exit(3)
            res.append(rk.ar_result(sp))
        rk.set_hooks()
        self.check_ar("delay_injection_stress", sp, 31, 4, res, launches=n)
        sp2 = L.AllReduceSpec(rk.ranks, 16, 2560, "bf16", L.CclKnobs(hooks=True, form="pair", ar="twoshot"))
        b = rk.ar_prepare(sp2, 32)
        res = []
        for li in range(n // 2):
            rk.set_hooks(salt=rnd.randrange(1, 1 << 30), smax=smax)
            rk.launch(b, rk.ar_args(b, 4))
            if not wait_launch_done(rk, 60):
                os._exit(3)
            res.append(rk.ar_result(sp2))
        rk.set_hooks()
        self.check_ar("delay_injection_stress", sp2, 32, 4, res, launches=n // 2)

    def t_dropped_peer(self):
        """The last rank joins the reset but never launches: the others record
        PEER_TIMEOUT (or PEER_ABORT pushed by another timed-out rank)."""
        rk = self.rk
        N = rk.ranks
        sp = L.AllReduceSpec(N, 4, 2560, "bf16")
        b = rk.ar_prepare(sp, 3)
        self.reset()
        t0 = time.monotonic()
        if rk.rank != N - 1:
            rk.launch(b, rk.ar_args(b, 2, dl_first_s=0.5, dl_s=0.5))
            done = wait_launch_done(rk, 30)
        else:
            done = True
        dt = time.monotonic() - t0
        rec = rk.group.record()
        codes = rk.all_gather(rec.name if rec.status else "OK")
        if rk.rank != N - 1:
            ok = done and rec.status == 1 and rec.name in ("PEER_TIMEOUT", "PEER_ABORT") and "PEER_TIMEOUT" in codes
            self.log("dropped_peer_times_out", ok, err=rec.name, peer=rec.peer, codes=codes, wall_s=round(dt, 3))
        self.reset()
        _, res = self.run_ar(sp, 3, 2)
        self.check_ar("recovery_after_reset", sp, 3, 2, res)

    def t_drop_use(self):
        rk = self.rk
        sp = L.AllReduceSpec(rk.ranks, 4, 2560, "bf16", L.CclKnobs(hooks=True))
        b = rk.ar_prepare(sp, 4)
        self.reset()
        rk.set_hooks(drop_use=2 if rk.rank == 1 else -1)
        rk.launch(b, rk.ar_args(b, 5, dl_first_s=2.0, dl_s=0.3))
        done = wait_launch_done(rk, 30)
        rec = rk.group.record()
        rk.set_hooks()
        codes = rk.all_gather(rec.name if rec.status else "OK")
        ok = done and rec.status == 1 and rec.name in ("PEER_TIMEOUT", "PEER_ABORT") and "PEER_TIMEOUT" in codes
        if rec.name == "PEER_TIMEOUT":
            ok = ok and rec.use == 2
        self.log("dropped_use_times_out", ok, err=rec.name, use=rec.use, stage=rec.stage, codes=codes)
        self.reset()

    def t_abort_peer(self):
        rk = self.rk
        sp = L.AllReduceSpec(rk.ranks, 4, 2560, "bf16", L.CclKnobs(hooks=True))
        b = rk.ar_prepare(sp, 5)
        self.reset()
        rk.set_hooks(abort_use=1 if rk.rank == 0 else -1)
        t0 = time.monotonic()
        rk.launch(b, rk.ar_args(b, 6, dl_first_s=5.0, dl_s=5.0))
        done = wait_launch_done(rk, 30)
        dt = time.monotonic() - t0
        rec = rk.group.record()
        rk.set_hooks()
        if rk.rank == 0:
            ok = done and rec.status == 1 and rec.stage == L.STAGES["hook"]
        else:
            ok = done and rec.status == 1 and rec.name == "PEER_ABORT" and dt < 2.5
        self.log("aborting_peer_propagates", ok, err=rec.name, stage=rec.stage, wall_s=round(dt, 3))
        self.reset()

    def t_mismatch(self):
        """Ranks launched with different shapes: the default (header-on)
        build of every collective records CCL_MISMATCH on every rank."""
        rk = self.rk
        rows = 4 if rk.rank == 0 else 5
        for kind in ("allreduce", "allgather", "argmax"):
            self.reset()
            if kind == "allreduce":
                sp = L.AllReduceSpec(rk.ranks, rows, 2560, "bf16")
                b = rk.ar_prepare(sp, 6)
                self.reset()
                rk.launch(b, rk.ar_args(b, 3, dl_first_s=2.0, dl_s=0.5))
            elif kind == "allgather":
                sp = L.AllGatherSpec(rk.ranks, rows, 2560, "bf16")  # rows 4 and 5 are both whole units
                b = rk.ag_prepare(sp, 6)
                self.reset()
                rk.launch(b, rk.ag_args(b, 3, dl_first_s=2.0, dl_s=0.5))
            else:
                sp = L.ArgmaxSpec(rk.ranks, 60 + rows)
                b = rk.build(sp, C.argmax_kernel_in)
                rk.quiesce_for(b)
                self.reset()
                rk.amax_launch(b, 6, 2, dl_first_s=2.0, dl_s=0.5)
            done = wait_launch_done(rk, 30)
            rec = rk.group.record()
            assert sp.knobs.header
            self.log("launch_mismatch_detected", done and rec.status == 1 and rec.name == "CCL_MISMATCH", kind=kind, err=rec.name, stage=rec.stage)
        self.reset()

    def _prefill(self, spec, lay, src: int, tag: int, garbage: int):
        """On this rank, fill (slot of the next launch's use 0, src) of every
        unit with ``tag`` and garbage payload."""
        rk = self.rk
        ctl = rk.control()
        slot = ctl["slotbase"] % spec.knobs.slots
        ex = lay.exchanges[0]
        for u in range(spec.units):
            off = ex.payload_off(slot, src, u)
            if spec.knobs.form == "line":
                rk.write_arena(off, np.full(256, garbage, np.uint32))
                for cp in range(ex.copies):
                    rk.write_arena(ex.flag_off(slot, src, u, cp), np.array([tag], np.uint32))
            else:
                pairs = np.empty(256, np.uint32)
                pairs[0::2] = garbage
                pairs[1::2] = tag
                rk.write_arena(off, pairs)

    def t_stale(self):
        """Receiver rank 0's slot for (next use, src 1) is pre-filled with a
        stale or ahead tag and garbage; rank 1 launches 0.3 s late. The real
        build never accepts a stale tag (u - 1, u - S: correct result) and
        raises TAG_AHEAD on u + S. Acceptance mutants prove each case
        discriminates, each with ONE allowed outcome (accepted garbage, no
        error recorded): ``ge`` (accept >= u, no ahead check) on u + S, ``any``
        (accept any non-zero tag) on u - 1, u - S and u + S. Header off: the
        tag arithmetic below counts the collective as use 0."""
        rk = self.rk
        for form in ("line", "pair"):
            for mut in ("none", "ge", "any"):
                for which in ("minus1", "minusS", "plusS"):
                    sp = L.AllReduceSpec(rk.ranks, 4, 2560, "bf16", L.CclKnobs(form=form, mutant=mut, header=False))
                    b = rk.ar_prepare(sp, 8)
                    self.reset(epoch=1000)
                    # one normal launch first (slots hold real older tags)
                    rk.launch(b, rk.ar_args(b, 1))
                    rk.sync()
                    ctl = rk.control()
                    S = sp.knobs.slots
                    u0 = RF_tag(ctl["epoch"], 0)
                    tag = {"minus1": RF_tag(ctl["epoch"], -1), "minusS": RF_tag(ctl["epoch"], -S), "plusS": RF_tag(ctl["epoch"], S)}[which]
                    rk.barrier()
                    if rk.rank == 0:
                        self._prefill(sp, b.lay, 1, tag, 0x3F803F80)
                    rk.sync()
                    rk.barrier()
                    if rk.rank == 1:
                        time.sleep(0.3)
                    rk.launch(b, rk.ar_args(b, 1, dl_first_s=3.0, dl_s=1.0))
                    done = wait_launch_done(rk, 30)
                    got = rk.ar_result(sp)
                    want = rk.ar_ref(sp, 8, 1)
                    rec = rk.group.record()
                    correct = first_mismatch(got, want) is None and rec.status == 0
                    if rk.rank == 0:
                        accepted_garbage = rec.status == 0 and first_mismatch(got, want) is not None
                        if which == "plusS" and mut == "none":
                            ok, expect = rec.status == 1 and rec.name == "TAG_AHEAD", "TAG_AHEAD"
                        elif mut == "any" or (mut == "ge" and which == "plusS"):
                            ok, expect = accepted_garbage, "must-fail: garbage accepted, no error"
                        else:
                            ok, expect = correct, "correct"
                        self.log("stale_tags", ok and done, form=form, mutant=mut, which=which, expect=expect, err=rec.name if rec.status else None, correct=correct,
                                 accepted_garbage=accepted_garbage, tag0=u0, tag=tag)
                    self.reset()

    def t_mutants(self):
        rk = self.rk
        for mut in ("revsum", "slot"):
            sp = L.AllReduceSpec(rk.ranks, 8, 2560, "f32", L.CclKnobs(mutant=mut))
            b = rk.ar_prepare(sp, 12)
            self.reset()
            rk.launch(b, rk.ar_args(b, 3))
            done = wait_launch_done(rk, 30)
            got = rk.ar_result(sp)
            want = rk.ar_ref(sp, 12, 3)
            bad = first_mismatch(got, want) is not None
            dg = rk.all_gather(digest(got))
            detected = rk.all_gather(bad)
            discriminating = not (mut == "revsum" and rk.ranks < 3)
            ok = done and (any(detected) or not discriminating)
            self.log("mutant_must_fail", ok, mutant=mut, detected_on=[i for i, d in enumerate(detected) if d], discriminating=discriminating, agree=len(set(dg)) == 1)
            self.reset()

    def t_coherence(self):
        """Mailbox-load cache bits on this arena memory type, 5 launches each:
        ``sc0sc1`` (default) and ``sc1`` (lever) must be correct on every rank;
        ``none`` (plain loads, the negative control) must fail on some rank
        (discriminating). The lint must reject the ``sc1`` / ``none`` builds
        unless waived (they are built here with the explicit waiver)."""
        rk = self.rk
        for pc in ("sc0sc1", "sc1", "none"):
            for form in ("line", "pair"):
                sp = L.AllReduceSpec(rk.ranks, 8, 2560, "bf16", L.CclKnobs(form=form, poll_cpol=pc))
                b = rk.ar_prepare(sp, 13)
                self.reset()
                ok_runs = 0
                for li in range(5):
                    rk.launch(b, rk.ar_args(b, 4, dl_first_s=1.0, dl_s=0.3))
                    if not wait_launch_done(rk, 30):
                        os._exit(3)
                    good = first_mismatch(rk.ar_result(sp), rk.ar_ref(sp, 13, 4)) is None and rk.group.record().status == 0
                    ok_runs += good
                    if rk.group.record().status:
                        break
                rec = rk.group.record()
                allok = all(rk.all_gather(ok_runs == 5))
                lint_rejects = bool(LI.lint_peer_access(b.kernel))
                if pc == "none":
                    ok, expect = (not allok) and lint_rejects, "must-fail on some rank; lint rejects"
                elif pc == "sc1":
                    ok, expect = allok and lint_rejects, "correct on every rank; lint rejects without the waiver"
                else:
                    ok, expect = allok and not lint_rejects, "correct on every rank; lint clean"
                self.log("coherence_cache_bits", ok, mem=rk.mem, poll_cpol=pc, form=form, expect=expect, correct_runs=ok_runs, all_ranks_correct=allok,
                         discriminating=not allok, lint_rejects=lint_rejects, err=rec.name if rec.status else None)
                self.reset()

    def t_silent(self):
        """Failure containment: a wave that recorded an error never publishes
        again. One-shot: rank 0's wave 0 / CTA 0 aborts at use 1 (hook) before
        its push. Two-shot: rank 1 drops its partials of use 1, so the owner
        of unit 0 (rank 0) times out in its reduce-scatter poll. In both, no
        rank's arena may hold rank 0's tag of use 1 (or 2) for unit 0; the
        ``deadpush`` mutant (no guard) must put it there."""
        rk = self.rk
        N = rk.ranks
        for form in ("line", "pair"):
            for ar in ("oneshot", "twoshot"):
                for mut in ("none", "deadpush"):
                    kn = L.CclKnobs(form=form, ar=ar, hooks=True, header=False, mutant=mut)
                    sp = L.AllReduceSpec(N, 4, 2560, "bf16", kn)
                    b = rk.ar_prepare(sp, 14)
                    self.reset()
                    if ar == "oneshot":
                        rk.set_hooks(abort_use=1 if rk.rank == 0 else -1)
                    else:
                        rk.set_hooks(drop_use=1 if rk.rank == 1 else -1)
                    rk.launch(b, rk.ar_args(b, 3, dl_first_s=2.0, dl_s=0.3))
                    done = wait_launch_done(rk, 30)
                    rk.set_hooks()
                    rec = rk.group.record()
                    rk.barrier()
                    ex = b.lay.exchanges[0] if ar == "oneshot" else b.lay.ex("0.ag")
                    seen = {}
                    for use in (1, 2):
                        slot = use % sp.knobs.slots  # slot base 0 after the reset
                        if form == "line":
                            w = rk.read_arena(ex.flag_off(slot, 0, 0, 0), 4).view(np.uint32)[0]
                        else:
                            w = rk.read_arena(ex.payload_off(slot, 0, 0) + 4, 4).view(np.uint32)[0]
                        seen[use] = int(w)
                    tags = {use: RF_tag(1, use) for use in (1, 2)}
                    published = any(seen[u] == tags[u] for u in (1, 2))
                    pub_all = rk.all_gather(published)
                    if mut == "none":
                        ok, expect = (not any(pub_all)) and rec.status == 1, "silent: no rank holds the failed wave's tag"
                    else:
                        ok, expect = any(pub_all), "must-fail: the unguarded wave published"
                    self.log("failed_wave_goes_silent", ok and done, form=form, ar=ar, mutant=mut, expect=expect, words=seen, tags=tags,
                             published_on=[i for i, p in enumerate(pub_all) if p], err=rec.name if rec.status else None)
                    self.reset()


    # ------------------------------------------------------- CCL-LM tests
    # The CCL-LM tests run the chain with non-contracting scales (ref.LM_GROWTH): with W3's contracting scales the
    # chain reaches a bitwise fixed point after 2-3 collectives, so a build that reads a previous collective's partial
    # (wrong slot, read-before-publish) matched the reference. Every build gets its own seed (from its name) and every
    # launch its own input, so neither a previous build's nor a previous launch's leftovers can match either.
    @staticmethod
    def _lm_seed(sp, salt=0):
        import zlib

        return (zlib.crc32(sp.name().encode()) + 7 * salt) % 100003 + 1

    def lm_run(self, test, sp, iters, launches, *, nctas=None, shares=None, salt=0, **info):
        """Prepare ``sp``, run ``launches`` launches of ``iters`` chained
        collectives without a reset, each from a fresh input (seed per build
        and launch), check each launch bitwise on every rank. ``nctas``: the
        same binary on fewer CTAs; ``shares``: ``place="table"`` shares."""
        from .harness import src_wire

        rk = self.rk
        seed = self._lm_seed(sp, salt)
        b = rk.ar_prepare(sp, seed)
        bl = rk.with_ctas(b, nctas) if nctas else b
        if shares is not None:
            tab = rk.set_utab(bl, shares)
            info["place_units"] = [int(x) for x in np.diff(tab).reshape(bl.nctas, -1).sum(axis=1)[:8]]
        ok, mism, res = True, None, None
        for li in range(launches):
            s = seed + 7919 * li
            if li:
                rk.put("src", src_wire(sp, rk.rank, s))
            rk.launch(bl, rk.ar_args(bl, iters))
            if not wait_launch_done(rk, 60):
                print(f"[rank {rk.rank}] HANG in {sp.name()}: exiting", flush=True)
                os._exit(3)
            res = rk.ar_result(sp)
            mm = first_mismatch(res, rk.ar_ref(sp, s, iters))
            if mm is not None and mism is None:
                ok, mism = False, (li, mm)
        rec = rk.group.record()
        ok = ok and rec.status == 0
        dg = rk.all_gather(digest(res))
        agree = len(set(dg)) == 1
        err = {"code": rec.name, "stage": rec.stage, "use": rec.use, "peer": rec.peer} if rec.status else None
        self.log(test, ok and agree, spec=sp.name(), seed=seed, launches=launches, iters=iters, growth=rk.chain_growth, mismatch=mism, agree=agree,
                 err=err, **info)
        if any(rk.all_gather(rec.status != 0)):
            self.reset()
        return ok and agree

    def _lm_growth(fn):  # noqa: N805 - decorator inside the class body
        def wrapped(self):
            g = self.rk.chain_growth
            self.rk.chain_growth = RF.LM_GROWTH
            try:
                return fn(self)
            finally:
                self.rk.chain_growth = g

        wrapped.__doc__ = fn.__doc__
        return wrapped

    @_lm_growth
    def t_lm_bitwise(self):
        """CCL-LM builds (lamport one- / two-shot, self_local, unit_kib 2 / 4,
        counted and comm-wave publish, striped grid sync, ar_dtype bf16,
        diag stamps, the levers on lamport) bitwise against the CPU reference
        on every rank, chained (non-contracting scales), several launches
        without a reset, a fresh input per build and launch."""
        rk, N = self.rk, self.rk.ranks
        K = L.CclKnobs
        rows = [1, 5, 128] if self.quick else [1, 5, 16, 17, 128]
        forms = [K(form="lamport"), K(form="lamport", ar="twoshot"), K(form="lamport", unit_kib=2), K(form="lamport", ar="twoshot", unit_kib=4),
                 K(unit_kib=2), K(unit_kib=4, ar="twoshot"), K(form="lamport", self_local=True), K(form="lamport", self_local=True, unit_kib=4),
                 K(form="pull"), K(form="pull", unit_kib=4)]
        for kn in forms:
            for cols, dt in [(2560, "bf16"), (10240, "f32")]:
                for r in rows:
                    try:
                        sp = L.AllReduceSpec(N, r, cols, dt, kn)
                    except ValueError:
                        continue  # the payload is not a whole number of units
                    self.lm_run("lm_bitwise", sp, 3, 2)
        knobs = [("bf16", K(publish="counted", carry="mem")), ("bf16", K(publish="comm_wave")), ("bf16", K(sync="grid", gsync="stripe")),
                 ("bf16", K(form="lamport", sync="grid", gsync="stripe")), ("bf16", K(form="lamport", sync="grid")),
                 ("bf16", K(form="lamport", ar="twoshot", sync="grid", gsync="stripe", unit_kib=2)), ("bf16", K(sync="grid", diag=True)),
                 ("bf16", K(form="lamport", sync="grid", gsync="stripe", diag=True)), ("bf16", K(form="lamport", waves=16)),
                 ("bf16", K(form="lamport", slots=3)), ("bf16", K(form="lamport", pause="s_sleep1")), ("bf16", K(form="lamport", push_cpol="sc0sc1nt")),
                 ("bf16", K(form="lamport", carry="mem")), ("bf16", K(form="lamport", header=False)), ("bf16", K(form="lamport", self_local=True, sync="grid", gsync="stripe")),
                 ("bf16", K(ar="twoshot", sync="grid", gsync="stripe", unit_kib=4)), ("bf16", K(publish="counted", sync="grid", gsync="stripe")),
                 ("f32", K(ar_dtype="bf16")), ("f32", K(ar_dtype="bf16", form="lamport")), ("f32", K(ar_dtype="bf16", form="lamport", ar="twoshot")),
                 ("f32", K(ar_dtype="bf16", ar="twoshot", unit_kib=2)), ("f32", K(ar_dtype="bf16", form="lamport", sync="grid", gsync="stripe")),
                 ("bf16", K(form="pull", sync="grid", gsync="stripe")), ("bf16", K(form="pull", sync="grid", unit_kib=2, place="table")),
                 ("f32", K(form="pull", ar_dtype="bf16")), ("bf16", K(form="pull", publish="counted", carry="mem"))]
        for dt, kn in knobs:
            for r in (16, 128):
                try:
                    sp = L.AllReduceSpec(N, r, 2560, dt, kn)
                except ValueError:
                    continue
                self.lm_run("lm_bitwise", sp, 4, 3)
        # the declared CCL-LM selection (layout.lm_select, default header on) at the measured sizes, 2 launches x 3 chained
        for r, cols, dt in ((16, 2560, "bf16"), (64, 2560, "bf16"), (128, 2560, "bf16"), (64, 10240, "bf16"), (128, 10240, "bf16"),
                            (32, 2560, "f32"), (64, 2560, "f32"), (32, 10240, "f32"), (128, 10240, "f32")):
            sp = L.auto_allreduce_spec(N, r, cols, dt, lm=True)
            self.lm_run("lm_bitwise", sp, 3, 2, selection="lm_select")
        # XCD-weighted placement (place="table"): a different random table on every rank, one XCD idle
        rng = np.random.default_rng(1234 + rk.rank)
        for kn in (K(form="lamport", sync="grid", gsync="stripe", place="table"), K(form="lamport", ar="twoshot", place="table", carry="mem"),
                   K(place="table", sync="grid"), K(form="lamport", unit_kib=4, place="table", sync="grid", gsync="stripe")):
            for r, cols, dt in ((128, 2560, "bf16"), (128, 10240, "bf16"), (32, 10240, "f32")):
                try:
                    sp = L.AllReduceSpec(N, r, cols, dt, kn)
                except ValueError:
                    continue
                sh = rng.random(8)
                sh[int(rng.integers(0, 8))] = 0.0
                self.lm_run("lm_bitwise", sp, 3, 2, shares=sh)
        # grid-synced CCL-LM builds on 32 CTAs (every wave loops over its units)
        for kn in (K(form="lamport", sync="grid", gsync="stripe"), K(form="lamport", ar="twoshot", sync="grid", gsync="stripe"),
                   K(sync="grid", gsync="stripe", unit_kib=2), K(publish="counted", sync="grid", gsync="stripe"),
                   K(form="lamport", sync="grid", gsync="stripe", place="table"), K(form="pull", sync="grid", gsync="stripe", unit_kib=4, publish="counted"),
                   K(form="pull", sync="grid", gsync="stripe", unit_kib=2)):
            sp = L.AllReduceSpec(N, 128, 10240, "bf16", kn)
            self.lm_run("lm_bitwise", sp, 3, 2, nctas=32, salt=32, ctas=32)
        # the timed layer-shaped selection builds (lm_select(layer=True) + header off) at the timed chain length
        if N == 2:
            from dataclasses import replace

            for r, cols, dt in ((16, 2560, "bf16"), (32, 2560, "bf16"), (128, 2560, "bf16"), (64, 10240, "bf16"), (128, 10240, "bf16"),
                                (32, 2560, "f32"), (64, 2560, "f32"), (32, 10240, "f32"), (64, 10240, "f32"), (128, 10240, "f32")):
                kn = replace(L.lm_select(N, r * cols * L.DTYPES[dt], dt, layer=True), header=False)
                self.lm_run("lm_bitwise", L.AllReduceSpec(N, r, cols, dt, kn), 96, 2, selection="lm_auto (timed)")

    @_lm_growth
    def t_lm_canon(self):
        """Lamport canonicalisation: rank 0's input carries the empty-looking
        pattern (bf16 0xFFFF / f32 0xFFFFFFFF) every 97th element. The real
        build must finish with the reference's result (NaN where the
        reference has NaN, bitwise elsewhere); the ``nocanon`` mutant must
        time out on some rank (its 0xFFFF lines never look ready)."""
        rk, N = self.rk, self.rk.ranks
        K = L.CclKnobs
        from .harness import src_wire

        for dt in ("bf16", "f32"):
            for ar in ("oneshot", "twoshot"):
                for mut in ("none", "nocanon"):
                    if mut == "nocanon" and ar == "twoshot":
                        continue
                    sp = L.AllReduceSpec(N, 4, 2560, dt, K(form="lamport", ar=ar, mutant=mut))
                    xs = [src_wire(sp, r, 55) for r in range(N)]
                    pat = 0xFFFF if dt == "bf16" else 0xFFFFFFFF
                    v = xs[0].view(np.uint16 if dt == "bf16" else np.uint32)
                    v[::97] = pat
                    b = rk.ar_prepare(sp, 55)
                    self.reset()
                    rk.put("src", xs[rk.rank])
                    rk.launch(b, rk.ar_args(b, 3, dl_first_s=1.0, dl_s=0.3))
                    done = wait_launch_done(rk, 30)
                    got = rk.ar_result(sp)
                    want = rk.ar_ref(sp, 55, 3, xs=xs)
                    rec = rk.group.record()
                    nan_g, nan_w = np.isnan(got), np.isnan(want)
                    same = bool(np.array_equal(nan_g, nan_w)) and first_mismatch(np.where(nan_w, 0, got).astype(np.float32), np.where(nan_w, 0, want).astype(np.float32)) is None
                    codes = rk.all_gather(rec.name if rec.status else "OK")
                    if mut == "none":
                        ok, expect = done and same and rec.status == 0 and int(nan_w.sum()) > 0, "correct (NaN-aware)"
                    else:
                        ok, expect = done and "PEER_TIMEOUT" in codes, "must-fail: PEER_TIMEOUT on some rank"
                    self.log("lm_canon", ok, spec=sp.name(), expect=expect, nans=int(nan_w.sum()), same=same, codes=codes)
                    self.reset()

    @_lm_growth
    def t_lm_mutants(self):
        """CCL-LM negative controls and stress: ``noreset`` must give wrong
        results; ``gsearly`` (stripe sync releasing one arrival early) must
        trip the hooks build's release check under delay injection while the
        real stripe build with the same delays stays clean and bitwise
        correct; lamport delay-injection stress (one- and two-shot) stays
        correct; a dropped lamport use times out at that use. Stale-read
        twins (pull ``slot``, lamport ``noreset``) run right after their real
        build, same seed, no reset, at the suite's and the timed chain
        length: the non-contracting chain must reject them (the same runs
        with W3's contracting scales are logged as info: they pass)."""
        rk, N = self.rk, self.rk.ranks
        K = L.CclKnobs
        from dataclasses import replace

        # stale-read twins right after the real build (the case the contracting chain missed)
        cases = [(K(form="pull", sync="grid", gsync="stripe", unit_kib=4, ctas=48, header=False), "slot", 128, 10240),
                 (K(form="pull", unit_kib=2), "slot", 32, 2560),
                 (K(form="lamport", sync="grid", gsync="stripe", header=False), "noreset", 32, 2560),
                 (K(form="lamport", ar="twoshot"), "noreset", 16, 2560)]
        for kn, mut, r, cols in cases:
            for iters in (3, 96):
                det_by_growth = {}
                for g in (RF.LM_GROWTH, 1.0):
                    rk.chain_growth = g
                    sp = L.AllReduceSpec(N, r, cols, "bf16", kn)
                    seed = self._lm_seed(sp, iters)
                    b = rk.ar_prepare(sp, seed)
                    rk.launch(b, rk.ar_args(b, iters))
                    if not wait_launch_done(rk, 60):
                        os._exit(3)
                    real_ok = first_mismatch(rk.ar_result(sp), rk.ar_ref(sp, seed, iters)) is None and rk.group.record().status == 0
                    msp = L.AllReduceSpec(N, r, cols, "bf16", replace(kn, mutant=mut))
                    mb = rk.ar_prepare(msp, seed)  # same layout family, no reset in between
                    rk.launch(mb, rk.ar_args(mb, iters, dl_first_s=2.0, dl_s=0.5))
                    done = wait_launch_done(rk, 60)
                    if not done:
                        os._exit(3)
                    bad = first_mismatch(rk.ar_result(msp), rk.ar_ref(msp, seed, iters)) is not None or rk.group.record().status != 0
                    det_by_growth[g] = (all(rk.all_gather(real_ok)), any(rk.all_gather(bool(bad))))
                    self.reset()
                rk.chain_growth = RF.LM_GROWTH
                (real_g, det_g), (real_1, det_1) = det_by_growth[RF.LM_GROWTH], det_by_growth[1.0]
                self.log("lm_mutants", real_g and det_g, mutant=mut, form=kn.form, case="stale twin after the real build", iters=iters, spec=msp.name(),
                         real_bitwise=real_g, detected=det_g, info_contracting_chain={"real_bitwise": real_1, "detected": det_1}, expect="must-fail")
        # pull form reading the other slot: must give wrong results
        sp = L.AllReduceSpec(N, 8, 2560, "bf16", K(form="pull", mutant="slot"))
        b = rk.ar_prepare(sp, 60)
        self.reset()
        rk.launch(b, rk.ar_args(b, 4, dl_first_s=1.0, dl_s=0.3))
        done = wait_launch_done(rk, 30)
        bad = first_mismatch(rk.ar_result(sp), rk.ar_ref(sp, 60, 4)) is not None or rk.group.record().status != 0
        det = rk.all_gather(bad)
        self.log("lm_mutants", done and any(det), mutant="slot", form="pull", detected_on=[i for i, d in enumerate(det) if d], expect="must-fail")
        self.reset()
        # noreset (S = 2: use 2 reads use 0's stale lines)
        for ar in ("oneshot", "twoshot"):
            sp = L.AllReduceSpec(N, 8, 2560, "bf16", K(form="lamport", ar=ar, mutant="noreset"))
            b = rk.ar_prepare(sp, 61)
            self.reset()
            rk.launch(b, rk.ar_args(b, 4, dl_first_s=1.0, dl_s=0.3))
            done = wait_launch_done(rk, 30)
            bad = first_mismatch(rk.ar_result(sp), rk.ar_ref(sp, 61, 4)) is not None or rk.group.record().status != 0
            det = rk.all_gather(bad)
            self.log("lm_mutants", done and any(det), mutant="noreset", ar=ar, detected_on=[i for i, d in enumerate(det) if d], expect="must-fail")
            self.reset()
        # gsearly vs the real stripe build, same delays; CTA 0 (stripe 0) arrives late at every grid sync, so an
        # early release of stripe 0 is caught by the release check deterministically
        n = 10 if self.quick else 30
        for mut in ("gsearly", "none"):
            sp = L.AllReduceSpec(N, 128, 2560, "bf16", K(sync="grid", gsync="stripe", hooks=True, mutant=mut))
            b = rk.ar_prepare(sp, 62)
            self.reset()
            rnd = random.Random(4242)
            res, hits = [], 0
            for li in range(n):
                rk.set_hooks(salt=rnd.randrange(1, 1 << 30), smax=600, late_cta=0, late_units=3000)
                rk.launch(b, rk.ar_args(b, 4, dl_first_s=2.0, dl_s=1.0))
                if not wait_launch_done(rk, 60):
                    os._exit(3)
                rec = rk.group.record()
                if rec.status:
                    hits += 1
                    self.reset()
                else:
                    res.append(rk.ar_result(sp))
            rk.set_hooks()
            allhits = rk.all_gather(hits)
            want = rk.ar_ref(sp, 62, 4)
            correct = all(first_mismatch(g, want) is None for g in res)
            if mut == "gsearly":
                ok, expect = sum(allhits) > 0, "must-fail: the release check fires"
            else:
                ok, expect = sum(allhits) == 0 and correct and len(res) == n, "clean and bitwise"
            self.log("lm_mutants", ok, mutant=mut, gsync="stripe", launches=n, release_check_hits=allhits, correct=correct, expect=expect)
            self.reset()
        # lamport delay-injection stress
        m = 50 if self.quick else 600
        for ar in ("oneshot", "twoshot"):
            sp = L.AllReduceSpec(N, 16, 2560, "bf16", K(form="lamport", ar=ar, hooks=True))
            b = rk.ar_prepare(sp, 63)
            rnd = random.Random(77 + (ar == "twoshot"))
            res = []
            for li in range(m):
                rk.set_hooks(salt=rnd.randrange(1, 1 << 30), smax=400)
                rk.launch(b, rk.ar_args(b, 4))
                if not wait_launch_done(rk, 60):
                    os._exit(3)
                res.append(rk.ar_result(sp))
            rk.set_hooks()
            self.check_ar("lm_mutants", sp, 63, 4, res, stress_launches=m)
        # dropped lamport use
        sp = L.AllReduceSpec(N, 4, 2560, "bf16", K(form="lamport", hooks=True))
        b = rk.ar_prepare(sp, 64)
        self.reset()
        rk.set_hooks(drop_use=2 if rk.rank == 1 else -1)
        rk.launch(b, rk.ar_args(b, 5, dl_first_s=2.0, dl_s=0.3))
        done = wait_launch_done(rk, 30)
        rec = rk.group.record()
        rk.set_hooks()
        codes = rk.all_gather(rec.name if rec.status else "OK")
        ok = done and rec.status == 1 and rec.name in ("PEER_TIMEOUT", "PEER_ABORT") and "PEER_TIMEOUT" in codes
        if rec.name == "PEER_TIMEOUT":
            ok = ok and rec.use == 2
        self.log("lm_mutants", ok, mutant="drop_use", form="lamport", err=rec.name, use=rec.use, codes=codes)
        self.reset()

    def run(self, only=None):
        tests = [
            ("single_rank", self.t_single_rank),
            ("bitwise", self.t_bitwise),
            ("allgather", self.t_allgather),
            ("argmax", self.t_argmax),
            ("consecutive", self.t_consecutive),
            ("wrap", self.t_wrap),
            ("stress", self.t_stress),
            ("dropped_peer", self.t_dropped_peer),
            ("drop_use", self.t_drop_use),
            ("abort_peer", self.t_abort_peer),
            ("mismatch", self.t_mismatch),
            ("stale", self.t_stale),
            ("mutants", self.t_mutants),
            ("silent", self.t_silent),
            ("coherence", self.t_coherence),
            ("lm_bitwise", self.t_lm_bitwise),
            ("lm_canon", self.t_lm_canon),
            ("lm_mutants", self.t_lm_mutants),
        ]
        for name, fn in tests:
            if only and name not in only:
                continue
            t0 = time.monotonic()
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                self.log(name, False, exception=repr(e)[:400])
                try:
                    self.reset()
                except Exception:  # noqa: BLE001
                    pass
            self.rk.barrier()
            if self.rk.rank == 0:
                print(f"[test] {name} done in {time.monotonic() - t0:.1f}s", flush=True)
        return self.summary()


def RF_tag(epoch: int, k: int) -> int:
    """Tag ``k`` uses after (k >= 0) or before (k < 0) ``epoch``."""
    return (epoch - 1 + k) % TAG_MOD + 1


# ===================================================================== bench


def _median(xs):
    return float(np.median(np.asarray(xs, dtype=np.float64))) if xs else float("nan")


def _rccl_groups(rk: Rank, protos):
    """One NCCL communicator per protocol: ``NCCL_PROTO`` is set while the
    group is created and its first collective runs (RCCL reads it at
    communicator init); ``default`` leaves it unset (RCCL's own choice)."""
    import torch
    import torch.distributed as dist

    saved = os.environ.get("NCCL_PROTO")
    groups = {}
    for p in protos:
        if p == "default":
            os.environ.pop("NCCL_PROTO", None)
        else:
            os.environ["NCCL_PROTO"] = p
        g = dist.new_group(backend="nccl")
        t = torch.ones(4096, device=f"cuda:{rk.device}")
        dist.all_reduce(t, group=g)
        torch.cuda.synchronize()
        groups[p] = g
    if saved is None:
        os.environ.pop("NCCL_PROTO", None)
    else:
        os.environ["NCCL_PROTO"] = saved
    return groups


# rocKE arms (microbenchmark builds: header off, DESIGN-CCL 4.10)
_FORMS = {"line_os": dict(), "pair_os": dict(form="pair"), "line_ts": dict(ar="twoshot"), "pair_ts": dict(form="pair", ar="twoshot")}
CHAIN_ARMS = {
    **{f"rk_{f}": L.CclKnobs(header=False, **kw) for f, kw in _FORMS.items()},
    "rk_pair_os_cmem": L.CclKnobs(header=False, form="pair", carry="mem"),
    "rk_line_os_cmem": L.CclKnobs(header=False, carry="mem"),
    "rk_line_os_hd": L.CclKnobs(header=False, homing="dual"),
    "rk_line_os_pcsc1": L.CclKnobs(header=False, poll_cpol="sc1"),
    "rk_line_os_hdcm": L.CclKnobs(header=False, homing="dual", carry="mem"),
    "rk_line_ts_hd": L.CclKnobs(header=False, ar="twoshot", homing="dual"),
    "rk_line_ts_hdcm": L.CclKnobs(header=False, ar="twoshot", homing="dual", carry="mem"),
    "rk_pair_ts_cmem": L.CclKnobs(header=False, form="pair", ar="twoshot", carry="mem"),
}
GS_ARMS = {f"rk_{f}_gs": L.CclKnobs(header=False, sync="grid", **kw) for f, kw in _FORMS.items()}
GS_CTAS = 32
ONE_ARMS = [f"rk_{f}_1l" for f in _FORMS]


def bench(rk: Rank, out: str, rounds: int, iters: int, standalone_k: int, sizes, quick: bool, protos=("default", "LL", "LL128", "Simple"), extra=None):
    import torch
    import torch.distributed as dist

    N = rk.ranks
    tdt = {"bf16": torch.bfloat16, "f32": torch.float32}
    rows_all = []
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    rk_arms = {**CHAIN_ARMS, **GS_ARMS}
    groups = _rccl_groups(rk, protos)
    ext = None
    if extra:
        mod, fn = extra.split(":")
        ext = getattr(importlib.import_module(mod), fn)(rk, iters=iters)
    # dual homing: calibrate the IOD-local flag copy of every page that holds dual flags
    hd_arms = [a for a, kn in rk_arms.items() if kn.homing == "dual"]
    hd_pages = set()
    for a in hd_arms:
        for r, c, d in sizes:
            lay = L.make_layout([L.AllReduceSpec(N, r, c, d, rk_arms[a])])
            hd_pages |= {lay.flag_page(e.name) for e in lay.exchanges if e.flags >= 0}
    hd_flips = rk.calibrate_flips(hd_pages)
    flips_of = {a: hd_flips for a in hd_arms}
    if rk.rank == 0:
        print(f"[bench] code={code_digest()} node={socket.gethostname()} buses={rk.group.buses} protos={list(protos)} extra={extra}", flush=True)
        print(f"[bench] dual-homing flips (rank 0) {hd_flips:#x} pages {sorted(hd_pages)}", flush=True)
    for rows, cols, dt in sizes:
        specs = {a: L.AllReduceSpec(N, rows, cols, dt, kn) for a, kn in rk_arms.items()}
        specs["rk_auto_hdr"] = L.auto_allreduce_spec(N, rows, cols, dt)  # the default (header-on) model build
        built, numer = {}, {}
        # numerics first (bitwise vs the CPU reference, every rank)
        for a, sp in specs.items():
            b = rk.ar_prepare(sp, 42)
            rk.launch(b, rk.ar_args(b, iters, flips=flips_of.get(a, 0)))
            rk.sync()
            got = rk.ar_result(sp)
            ok = first_mismatch(got, rk.ar_ref(sp, 42, iters)) is None and rk.group.record().status == 0
            numer[a] = all(rk.all_gather(ok))
            built[a] = b
        for a in GS_ARMS:  # the grid-synced chain on at most GS_CTAS CTAs (any grid works with the xbuf carry)
            bc = rk.with_ctas(built[a], min(GS_CTAS, built[a].nctas))
            sp = specs[a]
            rk.ar_prepare(sp, 42)
            rk.launch(bc, rk.ar_args(bc, iters))
            rk.sync()
            ok = first_mismatch(rk.ar_result(sp), rk.ar_ref(sp, 42, iters)) is None and rk.group.record().status == 0
            numer[a + "_c32"] = all(rk.all_gather(ok))
            built[a + "_c32"] = bc
        # RCCL buffers, numerics per protocol, graphs
        x = torch.randn(rows, cols, device=f"cuda:{rk.device}").to(tdt[dt])
        allx = [None] * N
        dist.all_gather_object(allx, x.float().cpu().numpy(), group=rk.gloo)
        ref = np.sum(np.stack(allx).astype(np.float64), axis=0)
        tol = 2e-2 if dt == "bf16" else 1e-5
        xr = x.clone()
        graphs, eager = {}, {}
        for p, g in groups.items():
            y = x.clone()
            dist.all_reduce(y, group=g)
            torch.cuda.synchronize()
            err = float(np.max(np.abs(y.float().cpu().numpy() - ref)) / (np.max(np.abs(ref)) + 1e-30))
            numer[f"rccl_{p}"] = all(rk.all_gather(err < tol))
            eager[f"rccl_eager_{p}"] = g
            try:
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
                if p == "default":
                    g2 = torch.cuda.CUDAGraph()
                    sc = 1.0 / N
                    with torch.cuda.graph(g2):
                        for _ in range(iters):
                            dist.all_reduce(xr, group=g)
                            xr.mul_(sc)
                    graphs["rccl_graph_mul_default"] = g2
            except Exception as e:  # noqa: BLE001
                if rk.rank == 0:
                    print(f"[bench] graph capture failed ({p}): {e!r}", flush=True)
        ext_arms, ext_info = {}, {}
        if ext is not None:
            ed = ext.prepare(rows, cols, dt, x, ref, tol)
            ext_arms = ed.get("arms", {})
            for a in ext_arms:
                numer[a] = bool(ed.get("numerics_ok", {}).get(a, False))
            ext_info = ed.get("info", {})
        arms = list(CHAIN_ARMS) + ["rk_line_os_b"] + list(GS_ARMS) + [a + "_c32" for a in GS_ARMS] + ONE_ARMS + ["rk_auto_1l_hdr"]
        arms += list(eager) + list(graphs) + list(ext_arms)
        chain_like = list(CHAIN_ARMS) + ["rk_line_os_b"] + list(GS_ARMS) + [a + "_c32" for a in GS_ARMS]
        times = {a: [] for a in arms}
        dev_lat = {a: [] for a in chain_like}
        for rnd in range(rounds):
            order = arms[:]
            random.Random(rnd * 7919 + rows * 31 + cols + (dt == "f32")).shuffle(order)
            for a in order:
                rk.barrier()
                torch.cuda.synchronize()
                if a in chain_like:
                    b = built[a[:-2] if a == "rk_line_os_b" else a]
                    base = a[:-2] if a == "rk_line_os_b" else (a[:-4] if a.endswith("_c32") else a)
                    args = rk.ar_args(b, iters, flips=flips_of.get(base, 0))
                    ev0.record()
                    rk.launch(b, args)
                    ev1.record()
                    torch.cuda.synchronize()
                    times[a].append(ev0.elapsed_time(ev1) * 1e3 / iters)  # us per collective incl. launch
                    st = rk.stamps(b)
                    dev_lat[a].append(float(st[:, 2].max() - st[:, 1].max()) / (iters - 1) * 10.0)  # ns per collective
                elif a in ONE_ARMS or a == "rk_auto_1l_hdr":
                    b = built["rk_auto_hdr" if a == "rk_auto_1l_hdr" else a[:-3]]
                    args = rk.ar_args(b, 1)
                    ev0.record()
                    for _ in range(standalone_k):
                        rk.launch(b, args)
                    ev1.record()
                    torch.cuda.synchronize()
                    times[a].append(ev0.elapsed_time(ev1) * 1e3 / standalone_k)
                elif a in eager:
                    g = eager[a]
                    ev0.record()
                    for _ in range(standalone_k):
                        dist.all_reduce(xr, group=g)
                    ev1.record()
                    torch.cuda.synchronize()
                    times[a].append(ev0.elapsed_time(ev1) * 1e3 / standalone_k)
                elif a in graphs:
                    ev0.record()
                    graphs[a].replay()
                    ev1.record()
                    torch.cuda.synchronize()
                    times[a].append(ev0.elapsed_time(ev1) * 1e3 / iters)
                else:
                    kind, fn = ext_arms[a]
                    k = iters if kind == "graph" else standalone_k
                    ev0.record()
                    if kind == "graph":
                        fn()
                    else:
                        for _ in range(standalone_k):
                            fn()
                    ev1.record()
                    torch.cuda.synchronize()
                    times[a].append(ev0.elapsed_time(ev1) * 1e3 / k)
                if rk.group.record().status:
                    raise RuntimeError(f"rank {rk.rank}: CCL error in {a}: {rk.group.record()}")
        med = {a: _median(v) for a, v in times.items()}
        dmed = {a: _median(v) / 1e3 for a, v in dev_lat.items()}
        allmed = rk.all_gather({"wall_us": med, "dev_us": dmed})
        row = {
            "rows": rows, "cols": cols, "dtype": dt, "ranks": N, "iters": iters, "rounds": rounds,
            "numerics_ok": numer, "per_rank": allmed, "bus": rk.group.buses, "mem": rk.mem, "node": socket.gethostname(),
            "code": code_digest(), "protos": list(protos), "ext_info": ext_info, "gs_ctas": GS_CTAS,
            "raw_wall_us_rank0": times if rk.rank == 0 else None,
        }  # fmt: skip
        rows_all.append(row)
        if rk.rank == 0:
            # rank-max of medians (a collective finishes when every rank has it)
            def worst(kind, a):
                return max(m[kind][a] for m in allmed)

            def best(kind, names):
                names = [n for n in names if n in allmed[0][kind]]
                return min((worst(kind, n), n) for n in names) if names else (float("nan"), "-")

            rccl_all = list(eager) + list(graphs)
            r_best, r_best_n = best("wall_us", rccl_all)
            r_def, _ = best("wall_us", [a for a in rccl_all if a.endswith("_default")])
            r_graph, r_graph_n = best("wall_us", list(graphs))
            ch, ch_n = best("dev_us", list(CHAIN_ARMS))
            gs, gs_n = best("dev_us", list(GS_ARMS))
            gc, gc_n = best("dev_us", [a + "_c32" for a in GS_ARMS])
            one, one_n = best("wall_us", ONE_ARMS)
            hdr1 = worst("wall_us", "rk_auto_1l_hdr")
            ctrl = worst("dev_us", "rk_line_os_b") / worst("dev_us", "rk_line_os")
            msg = (
                f"[bench] N={N} rows={rows} cols={cols} {dt}: RCCL best={r_best_n}; chain({ch_n})/RCCL best = {ch / r_best:.3f} "
                f"[vs default proto only {ch / r_def:.3f}], gridsync chain({gs_n})/RCCL best = {gs / r_best:.3f}, "
                f"gridsync c{GS_CTAS}({gc_n})/RCCL best = {gc / r_best:.3f}, 1-launch({one_n})/RCCL graph best({r_graph_n}) = {one / r_graph:.3f}, "
                f"1-launch hdr auto/RCCL graph best = {hdr1 / r_graph:.3f}, A/A control = {ctrl:.3f}, numerics {all(numer.values())}"
            )
            if ext_arms:
                e, e_n = best("wall_us", list(ext_arms))
                msg += f"; ext best={e_n}: chain/ext = {ch / e:.3f}, gridsync/ext = {gs / e:.3f}, 1-launch/ext = {one / e:.3f}, ext/RCCL best = {e / r_best:.3f}"
            print(msg, flush=True)
            if not all(numer.values()):
                print(f"[bench] NUMERICS FAILED: {[a for a, v in numer.items() if not v]}", flush=True)
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, f"bench_n{N}{os.environ.get('CCL_TAG', '')}.json"), "w") as fh:
                json.dump(rows_all, fh, indent=1)
        del graphs
        if ext is not None:
            ext.release()
        torch.cuda.synchronize()
    return rows_all


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["test", "bench"])
    ap.add_argument("--out", default=os.environ.get("Q38_OUT", "ccl_out"))
    ap.add_argument("--only", default="")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--iters", type=int, default=96)
    ap.add_argument("--standalone-k", type=int, default=64)
    ap.add_argument("--rows", default="1,2,4,8,16,32,64,128")
    ap.add_argument("--cols", default="2560,10240")
    ap.add_argument("--dtypes", default="bf16,f32")
    ap.add_argument("--rccl-protos", default="default,LL,LL128,Simple")
    ap.add_argument("--extra-arms", default="", help="module:function returning an extra-arm provider (see bench())")
    a = ap.parse_args(argv)
    dev, gloo = _init(backend_nccl=a.cmd == "bench")
    mem = os.environ.get("CCL_MEM", "uncached")
    rk = Rank(device=dev, mem=mem, group=gloo)
    rk.gloo = gloo
    if rk.rank == 0:
        print(f"[ccl] ranks={rk.ranks} mem={mem} buses={rk.group.buses} shift={os.environ.get('CCL_DEVICE_SHIFT', '0')} code={code_digest()} "
              f"node={socket.gethostname()}", flush=True)
    ok = True
    if a.cmd == "test":
        ok = Suite(rk, a.out, a.quick).run(only=set(a.only.split(",")) if a.only else None)
    else:
        sizes = [(r, c, d) for d in a.dtypes.split(",") for c in map(int, a.cols.split(",")) for r in map(int, a.rows.split(","))]
        bench(rk, a.out, a.rounds, a.iters, a.standalone_k, sizes, a.quick, protos=tuple(a.rccl_protos.split(",")), extra=a.extra_arms or None)
    rk.close()
    import torch.distributed as dist

    dist.destroy_process_group()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
