"""RT: compile-only census of the kernels the asm-carrying suites build (direct compiler, strict: a refusal is
reported, never a COMGR fallback). Local, no GPU.

    python rt_census_asm.py [--p4 DIR] [--p5 DIR] [--ccl]

P4 / P5: the benches' own ``build_all`` with a stand-in context whose ``fn`` compiles (direct, strict) and records
the outcome. CCL: every AllReduce / AllGather / Argmax spec of the multi-process suite's tests (``bench_ccl``'s knob
lists, ranks 2 and 4) through ``build_*_kernel``.
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
import traceback

os.environ["ROCKE_DIRECT_STRICT"] = "1"


class Recorder:
    def __init__(self):
        self.ok = 0
        self.refused = collections.Counter()
        self.examples = {}
        self.seen = set()

    def compile(self, kernel):
        from rocke.helpers.compile import compile_kernel
        from rocke.runtime.code_object import DirectUnsupported

        if kernel.name in self.seen:
            return
        self.seen.add(kernel.name)
        try:
            compile_kernel(kernel, arch="gfx950", backend="python", code_object_backend="direct")
            self.ok += 1
        except DirectUnsupported as e:
            msg = str(e).split("ROCKE_DIRECT_UNSUPPORTED:")[-1].strip()[:160]
            self.refused[msg] += 1
            self.examples.setdefault(msg, kernel.name)


class FakeCx:
    """The benches' context: ``fn`` compiles; everything else is absent (build_all only calls fn)."""

    def __init__(self, rec):
        self.rec = rec
        self.tag = 1

    def fn(self, kernel):
        self.rec.compile(kernel)
        return None

    def tags(self, n):
        b = self.tag
        self.tag += n + 1
        return b


def p4(rec, d):
    sys.path.insert(0, d)
    import p4_bench

    p4_bench.build_all(FakeCx(rec))


def p5(rec, d):
    sys.path.insert(0, d)
    import p5_bench
    from kernels.gfx950.qwen38_mono.probes import p5_coherence as P

    for _key, spec in p5_bench.all_specs(("kv", "sc", "host")).items():
        if isinstance(spec, P.KvSpec):
            rec.compile(P.build_kv_kernel(spec))
        elif isinstance(spec, P.ScSpec):
            rec.compile(P.build_scalar_kernel(spec))
        elif spec == "hostw":
            rec.compile(P.build_host_writer_kernel())
        else:
            rec.compile(P.build_host_kernel(spec))


def ccl(rec):
    from kernels.gfx950.qwen38_mono.ccl import collectives as C
    from kernels.gfx950.qwen38_mono.ccl import layout as L

    K = L.CclKnobs
    forms = [K(form="lamport"), K(form="lamport", ar="twoshot"), K(form="lamport", unit_kib=2), K(form="lamport", ar="twoshot", unit_kib=4),
             K(unit_kib=2), K(unit_kib=4, ar="twoshot"), K(form="lamport", self_local=True), K(form="lamport", self_local=True, unit_kib=4)]
    knobs = [K(publish="counted", carry="mem"), K(publish="comm_wave"), K(sync="grid", gsync="stripe"), K(form="lamport", sync="grid", gsync="stripe"),
             K(form="lamport", sync="grid"), K(form="lamport", ar="twoshot", sync="grid", gsync="stripe", unit_kib=2), K(sync="grid", diag=True),
             K(form="lamport", sync="grid", gsync="stripe", diag=True), K(form="lamport", waves=16), K(form="lamport", slots=3),
             K(form="lamport", pause="s_sleep1"), K(form="lamport", push_cpol="sc0sc1nt"), K(form="lamport", carry="mem"),
             K(form="lamport", header=False), K(form="lamport", self_local=True, sync="grid", gsync="stripe"),
             K(ar="twoshot", sync="grid", gsync="stripe", unit_kib=4), K(publish="counted", sync="grid", gsync="stripe"),
             K(ar_dtype="bf16"), K(ar_dtype="bf16", form="lamport"), K(ar_dtype="bf16", form="lamport", ar="twoshot"),
             K(ar_dtype="bf16", ar="twoshot", unit_kib=2), K(ar_dtype="bf16", form="lamport", sync="grid", gsync="stripe"), K()]
    for n in (2, 4):
        for kn in forms + knobs:
            for cols, dt in [(2560, "bf16"), (10240, "f32"), (2560, "f32")]:
                for r in (1, 5, 16, 17, 128):
                    try:
                        sp = L.AllReduceSpec(n, r, cols, dt, kn)
                    except (ValueError, TypeError):
                        continue
                    try:
                        rec.compile(C.build_allreduce_kernel(sp))
                    except Exception as e:  # a builder refusal is not ours
                        rec.refused[f"builder: {type(e).__name__}: {str(e)[:80]}"] += 1
        for r in (1, 4, 16):
            for args in ((n, r), ):
                for maker, spec in ((C.build_allgather_kernel, L.AllGatherSpec), (C.build_argmax_kernel, L.ArgmaxSpec)):
                    try:
                        rec.compile(maker(spec(*args) if spec is L.ArgmaxSpec else spec(*args, 2560)))
                    except Exception as e:
                        rec.refused[f"builder: {type(e).__name__}: {str(e)[:80]}"] += 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p4")
    ap.add_argument("--p5")
    ap.add_argument("--ccl", action="store_true")
    a = ap.parse_args(argv)
    for name, run in (("p4", (lambda r: p4(r, a.p4)) if a.p4 else None), ("p5", (lambda r: p5(r, a.p5)) if a.p5 else None),
                      ("ccl", ccl if a.ccl else None)):  # fmt: skip
        if run is None:
            continue
        rec = Recorder()
        try:
            run(rec)
        except Exception:
            traceback.print_exc()
        print(f"== {name}: {rec.ok} direct, {sum(rec.refused.values())} refused")
        for msg, n in rec.refused.most_common():
            print(f"   {n:4d}  {msg}  (e.g. {rec.examples.get(msg, '')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
