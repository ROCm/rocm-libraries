#!/usr/bin/env python3
"""AICK-3062: the RDNA tuned-table selection run (agreed rule, 2026-11-13).

On RDNA the rocKE backward is several times slower than PyTorch's own backward, so
the repo's confirm rule (an entry beats the competitor on every cell) admits no
entry. The agreed RDNA rule picks the best rocKE configuration instead:

  an entry for tuning key K is a runtime-stage finalist that, on every cohort cell
  of K, (1) is correct (arm status ok, pruning-set gate on), (2) has a clean A/A
  control (its own #aa twin is not judged different from it either way), and
  (3) is not slower than rocKE's shipped plan (paired verdict finalist vs shipped
  is ``pass``, or ``ambiguous`` with rho <= 1 + TIE_EPS); among finalists passing
  all cells the one with the lowest geomean time ratio vs shipped wins, and only
  if that geomean is < 1 (an entry equal to the shipped plan changes nothing).

Modes:
  time  <arch> <d>   time shipped + finalists (each with an A/A twin) + PyTorch's
                     default arms on the slice's cohort cells (resumable)
  judge <arch>       apply the rule over d64 / d128, write entries_<arch>.json
                     (checked by the arch module's general_entries) and a report
"""

import dataclasses
import json
import math
import os
import sys
from pathlib import Path

# work directory (outside the source tree): sweep/, best_rocke/, constants, outputs
J = Path(os.environ.get("AICK_WORK", Path(__file__).resolve().parent)).resolve()
SWEEP_D = (64, 128)


def _slice(sw, arch, d):
    return next(
        s
        for s in sw.all_primary_slices()
        if s.arch == arch and s.family == "general" and s.head_size == d
    )


def _out(arch, d):
    return J / "best_rocke" / arch / f"d{d}"


def time_mode(arch, d):
    from benchmarks.common import attention_bwd_bench as bb
    from benchmarks.common import attention_bwd_sweep as sw
    from benchmarks.common import attention_bwd_rocke_arm as ra

    sl = _slice(sw, arch, d)
    finalists = sw.ranked(J / "sweep" / arch, sl, "runtime", 2)
    if not finalists:
        raise SystemExit(f"{arch} d{d}: no runtime finalists")
    out = _out(arch, d)
    out.mkdir(parents=True, exist_ok=True)
    cfg = out / "configs.json"
    ra.write_configs(cfg, [ra.SHIPPED] + [c.config() for c in finalists])
    # THD cells key no table entry (cell_request is dense-layout only)
    cells = [c for c in sw.cohort_cells(sl) if not c.is_thd]
    argv = [
        "--ids",
        ",".join(c.id for c in cells),
        "--arch",
        arch,
        "--out-dir",
        str(out),
        "--rocke-configs",
        str(cfg),
        "--competitor-arms",
        "default",
        "--compile-cache",
        str(J / "sweep" / arch / "compile_cache"),
        "--attempts",
        "6",
        "--aa-control",
        "all",
        "--gate",
        "tests._attention_bwd_perf_gate:gate",
        "--gate-constants",
        str(J / f"gate_constants_{arch}.json"),
        "--expect-flavor",
        "llvm22",
        "--expect-comgr",
        os.environ["EXPECT_COMGR"],
    ]
    print(
        f"time {arch} d{d}: cells={len(cells)} finalists={[c.key()[:20] for c in finalists]}",
        flush=True,
    )
    return bb.main(argv)


def _verdict_vs(bb, a, b, consts):
    v = bb.paired_verdict(a["attempt_medians_ms"], b["attempt_medians_ms"], consts)
    rho = a["median_ms"] / b["median_ms"]
    if v == "ambiguous" and rho <= 1.0 + consts["TIE_EPS"]:
        v = "tie"
    return v, rho


def judge_mode(arch):
    from benchmarks.common import attention_bwd_bench as bb
    from benchmarks.common import attention_bwd_sweep as sw
    from dispatch.attention_bwd import routing
    from kernels.common.attention_bwd_plan import attn_bwd_tuning_key

    consts = bb.load_gate_constants(J / f"gate_constants_{arch}.json")
    env = None
    report = {"arch": arch, "rule": __doc__.split("Modes:")[0].strip(), "slices": {}}
    entries = []
    for d in SWEEP_D:
        out = _out(arch, d)
        summ = bb.latest_summaries(out / "cells.jsonl")
        if env is None:
            env = json.loads((out / "env.jsonl").read_text().splitlines()[0])
        num_cus = int(env["cu_count"])
        sl = _slice(sw, arch, d)
        finalists = sw.ranked(J / "sweep" / arch, sl, "runtime", 2)
        cells = sw.cohort_cells(sl)
        by_key = {}
        for c in cells:
            if c.is_thd or c.id not in summ:
                continue
            req = dataclasses.replace(sw.cell_request(c), num_cus=num_cus)
            by_key.setdefault(attn_bwd_tuning_key(req, arch), []).append(c.id)
        sl_rep = {
            "cells_timed": sum(c.id in summ for c in cells),
            "cells": len(cells),
            "keys": {},
        }
        for key, ids in sorted(by_key.items(), key=lambda kv: repr(kv[0])):
            best, rows = None, {}
            for f in finalists:
                arm = "rocke|" + f.key()
                ok, ratios, why = True, [], []
                for cid in ids:
                    arms = summ[cid]["arms"]
                    a, s, t = (
                        arms.get(arm, {}),
                        arms.get("rocke|shipped", {}),
                        arms.get(arm + "#aa", {}),
                    )
                    if (
                        a.get("status") != "ok"
                        or s.get("status") != "ok"
                        or t.get("status") != "ok"
                    ):
                        ok = False
                        why.append(f"{cid}:status")
                        continue
                    aa1, _ = _verdict_vs(bb, t, a, consts)
                    aa2, _ = _verdict_vs(bb, a, t, consts)
                    if "fail" in (aa1, aa2):
                        ok = False
                        why.append(f"{cid}:aa")
                        continue
                    v, rho = _verdict_vs(bb, a, s, consts)
                    ratios.append(rho)
                    if v not in ("pass", "tie"):
                        ok = False
                        why.append(f"{cid}:{v}")
                gm = (
                    math.exp(sum(map(math.log, ratios)) / len(ratios))
                    if ratios
                    else None
                )
                rows[f.key()[:20]] = {
                    "ok": ok,
                    "geomean_vs_shipped": gm,
                    "why": why[:5],
                }
                if (
                    ok
                    and gm is not None
                    and gm < 1.0
                    and (best is None or gm < best[1])
                ):
                    best = (f, gm)
            sl_rep["keys"][repr(key)] = {
                "cells": len(ids),
                "finalists": rows,
                "entry": best[0].key()[:20] if best else None,
            }
            if best:
                k = {n: v for n, v in best[0].tuning_knobs().items() if v is not None}
                k.update(best[0].r)
                entries.append({"key": list(key), "knobs": k})
        report["slices"][f"d{d}"] = sl_rep
    mod = routing.ARCH_TUNING[arch]
    checked = mod.general_entries({"entries": entries})  # raises on a bad entry
    (J / f"entries_{arch}.json").write_text(
        json.dumps({"entries": entries}, indent=1, default=list) + "\n"
    )
    (J / f"judge_{arch}.json").write_text(
        json.dumps(report, indent=1, default=str) + "\n"
    )
    print(
        f"judge {arch}: entries={len(checked)} (checked by {mod.__name__}.general_entries)"
    )
    for sname, s in report["slices"].items():
        print(
            f"  {sname}: cells timed {s['cells_timed']}/{s['cells']}, keys {len(s['keys'])}, "
            f"entries {sum(1 for v in s['keys'].values() if v['entry'])}"
        )


if __name__ == "__main__":
    mode, arch = sys.argv[1], sys.argv[2]
    if mode == "time":
        raise SystemExit(time_mode(arch, int(sys.argv[3])))
    judge_mode(arch)
