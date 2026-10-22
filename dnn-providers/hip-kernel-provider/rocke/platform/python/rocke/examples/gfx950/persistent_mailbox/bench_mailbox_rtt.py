# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Mailbox hop-latency microbenchmarks (gfx950).

  python -m rocke.examples.gfx950.persistent_mailbox.bench_mailbox_rtt \
      {gate,pingpong,a2a,gap,tagwrap,bulk,devops,all} [--out DIR] [--quick]

* gate      day-1 device-op gate (ptr_from_i64 forms, asm rsrc_vec, ${:uid})
* pingpong  1-hop round trip between two CTAs, per knob value
            (impl x scope x pause x payload width x CTA distance)
* a2a       160-producer all-to-all round on the full grid (256 x 512),
            per impl x consume mode x pause x NUMA dual copy (calibrated)
* gap       kernel launch gap (stream and hipGraph) of a 256 x 512 kernel
* tagwrap   tag rule across the u32 wrap (step near 2**26)
* bulk      release_bulk / acquire_bulk rows, put_bf bf16 pairs,
            payload_wait and atomic_add_scoped across XCDs (numerics only),
            plus the no-release mutation (expected to show stale rows)
* devops    wave_umax_dpp, fresh_v / fresh_s, memtime / s_sleep (numerics)

The ``ir`` implementation is Python-engine-only and lives in the rocKE
library (``kernels.gfx950.qwen38_mono.mailbox_ir``); with the library root
off ``sys.path`` its cells are skipped and the console says so.

Every timed cell checks its numerics first; a cell that fails numerics is
reported with ``ok: false`` and never ranked. Raw results (device ticks and
derived ns) go to ``DIR/<cmd>.json``; the console prints ratios only.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

from rocke.helpers import mailbox as M

from . import harness as H
from . import kernels as K


def _impls() -> list[str]:
    if K.load_ir_impl():
        return ["asm", "ir"]
    print(f"NOTE: {K.IR_IMPL_MODULE} not importable: ir cells skipped")
    return ["asm"]


def _arch() -> str:
    from rocke.runtime.hip_module import get_device_arch

    return get_device_arch(0) or "gfx950"


def _dump(out: Path, name: str, obj) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(obj, indent=1, default=str))


def cmd_gate(dev, arch, args):
    recs = []
    for form in K.GATE_FORMS:
        backends = ["python"] if form == "ir" else ["python", "cpp"]
        for be in backends:
            if be == "cpp":
                import importlib.util

                if importlib.util.find_spec("rocke_engine") is None:
                    recs.append(
                        {
                            "form": form,
                            "backend": "cpp",
                            "ok": None,
                            "skip": "no rocke_engine",
                        }
                    )
                    continue
            try:
                recs.append(H.run_gate(dev, form, arch, be))
            except Exception as e:  # noqa: BLE001 -- record and continue
                recs.append(
                    {
                        "form": form,
                        "backend": be,
                        "ok": False,
                        "error": f"{type(e).__name__}: {str(e)[:400]}",
                    }
                )
    for r in recs:
        print("gate", r)
    return recs


def _knob_grid(quick: bool):
    impls = _impls()
    scopes = ["agent"] if quick else ["agent", "system"]
    pauses = ["s_nop"] if quick else ["s_nop", "s_sleep1", "none"]
    for impl, scope, pause in itertools.product(impls, scopes, pauses):
        yield M.MailboxKnobs(impl=impl, scope=scope, pause=pause)


def cmd_pingpong(dev, arch, args):
    iters = 2000 if args.quick else 20000
    recs = []
    # distance 1: neighbouring CTA ids (different XCDs under round-robin
    # dispatch); 8: the same XCD; 3: another XCD. xcc ids are recorded.
    for kn in _knob_grid(args.quick):
        for pairs in (1, 2):
            for dist in (1, 8) if args.quick else (1, 3, 8, 16):
                spec = K.PingPongSpec(knobs=kn, pairs=pairs)
                rs = H.run_pingpong(
                    dev, spec, arch, a=0, bb=dist, iters=iters, reps=args.reps
                )
                ok = all(r.ok and r.monotone for r in rs)
                rec = {
                    "name": spec.name(),
                    "impl": kn.impl,
                    "scope": kn.scope,
                    "pause": kn.pause,
                    "pairs": pairs,
                    "dist": dist,
                    "xcc_a": rs[0].xcc_a,
                    "xcc_b": rs[0].xcc_b,
                    "backend": rs[0].backend,
                    "ok": ok,
                    "iters": iters,
                    "rtt_ticks": [r.rtt_ticks for r in rs],
                    "rtt_ns_median": statistics.median(r.rtt_ns for r in rs),
                }
                recs.append(rec)
    base = [
        r
        for r in recs
        if r["impl"] == "ir"
        and r["scope"] == "agent"
        and r["pause"] == "s_nop"
        and r["pairs"] == 1
        and r["dist"] == 1
    ]
    ref = base[0]["rtt_ns_median"] if base else None
    for r in recs:
        rel = f"{r['rtt_ns_median'] / ref:.3f}x" if ref else "-"
        print(
            f"pingpong {r['name']:<46} dist={r['dist']:>2} xcc={r['xcc_a']}->"
            f"{r['xcc_b']} ok={r['ok']} rel_to_ir_sc1_nop_d1={rel}"
        )
    return recs


def cmd_a2a(dev, arch, args):
    phases = 50 if args.quick else 400
    recs = []
    for kn in (
        M.MailboxKnobs(impl=i, pause=p, numa_copies=n)
        for i in _impls()
        for p in (("s_nop",) if args.quick else ("s_nop", "s_sleep1"))
        for n in (1, 2)
    ):
        cells = [(c, "bare") for c in K.A2A_CONSUME] + [("wave0", "sync")]
        for consume, barrier in cells:
            # numerics first (check=True), then the timed build (check=False)
            chk = H.run_all_to_all(
                dev,
                K.AllToAllSpec(knobs=kn, consume=consume, check=True, barrier=barrier),
                arch,
                phases=20,
                reps=1,
            )
            ok = all(r.ok for r in chk)
            rs = (
                H.run_all_to_all(
                    dev,
                    K.AllToAllSpec(knobs=kn, consume=consume, barrier=barrier),
                    arch,
                    phases=phases,
                    reps=args.reps,
                )
                if ok
                else []
            )
            recs.append(
                {
                    "name": K.AllToAllSpec(
                        knobs=kn, consume=consume, barrier=barrier
                    ).name(),
                    "impl": kn.impl,
                    "pause": kn.pause,
                    "consume": consume,
                    "barrier": barrier,
                    "ok": ok and all(r.ok for r in rs),
                    "phases": phases,
                    "backend": chk[0].backend,
                    "round_ticks": [r.round_ticks for r in rs],
                    "round_ns_median": statistics.median(r.round_ns for r in rs)
                    if rs
                    else None,
                    "xcc_hist": chk[0].xcc_hist,
                }
            )
    for r in recs:
        print("a2a", {k: r[k] for k in ("name", "ok", "backend")})
    return recs


def cmd_gap(dev, arch, args):
    recs = []
    for mode in ("stream", "graph"):
        try:
            r = H.run_launch_gap(dev, arch, mode=mode, launches=50)
            recs.append(asdict(r) | {"gap_ns": r.gap_ns, "ok": True})
        except Exception as e:  # noqa: BLE001
            recs.append({"mode": mode, "ok": False, "error": str(e)[:300]})
    for r in recs:
        print("gap", r.get("mode"), "ok", r.get("ok"))
    return recs


def cmd_tagwrap(dev, arch, args):
    wrap = 1 << 26
    steps = [0, 1, wrap - 3, wrap - 2, wrap - 1, wrap, wrap + 1, 2 * wrap - 1, 2 * wrap]
    recs = []
    for impl in _impls():
        rs = H.run_tag_wrap(
            dev, arch, steps, [0, 47, 48], knobs=M.MailboxKnobs(impl=impl)
        )
        for r in rs:
            r["impl"] = impl
            r["tag_nonzero"] = r["tag"] != 0
        recs += rs
    bad = [r for r in recs if not (r["ok"] and r["tag_nonzero"])]
    print(f"tagwrap launches={len(recs)} failures={len(bad)}")
    for r in bad:
        print("  FAIL", r)
    return recs


def cmd_bulk(dev, arch, args):
    recs = []
    steps = 10 if args.quick else 40
    for impl in _impls():
        for fence_impl in ("asm", "ir"):
            kn = M.MailboxKnobs(impl=impl, fence_impl=fence_impl)
            for rows in (1, 8):
                r = H.run_bulk(dev, K.BulkSpec(knobs=kn, rows=rows), arch, steps=steps)
                recs.append(r)
    # mutation: no L2 write-back before the ready pair. Reported, not gated
    # (whether stale lines are observed depends on cache state).
    for impl in _impls():
        r = H.run_bulk(
            dev,
            K.BulkSpec(knobs=M.MailboxKnobs(impl=impl), fence="norelease"),
            arch,
            steps=steps,
        )
        r["mutation"] = True
        r.pop("ok")
        recs.append(r)
    for r in recs:
        print(
            "bulk",
            {
                k: r.get(k)
                for k in (
                    "name",
                    "ok",
                    "row_mismatch",
                    "bf_mismatch",
                    "counter_bad",
                    "ticket_bad",
                    "backend",
                )
            },
        )
    return recs


def cmd_devops(dev, arch, args):
    recs = [H.run_device_ops(dev, arch, seed=s) for s in range(3)]
    for r in recs:
        print("devops", r)
    return recs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "cmd",
        choices=["gate", "pingpong", "a2a", "gap", "tagwrap", "bulk", "devops", "all"],
    )
    ap.add_argument("--out", default=os.environ.get("Q38_OUT", "."))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args(argv)
    arch = _arch()
    dev = H.Dev()
    out = Path(args.out)
    cmds = (
        ["gate", "devops", "tagwrap", "bulk", "pingpong", "gap", "a2a"]
        if args.cmd == "all"
        else [args.cmd]
    )
    rc = 0
    try:
        for c in cmds:
            recs = globals()[f"cmd_{c}"](dev, arch, args)
            _dump(out, c, {"arch": arch, "records": recs})
            if any(r.get("ok") is False for r in recs):
                rc = 1
    finally:
        dev.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
