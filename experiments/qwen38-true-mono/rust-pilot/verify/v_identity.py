"""Verifier (w2-vrust): knob-off byte identity of the R2 rust tree against the mono tree.

R2 changed Python core and library files (``core/ir.py``, ``core/lower_llvm.py``, ``helpers/mailbox.py``,
``qwen38_mono/{irops,framework,stamps,toy_chain,direct_pilot}.py``, ``mx_decode_gemv.py``) after R1's identity run,
and R2's own census only covers the Rust crate. This re-runs R1's ``q38_identity`` dump on both trees with the
R1 list (14 builds) plus the R2 hybrid head as shipped (``head_hybrid``: rows_max 4, fused, waves_per_eu 1).

    python v_identity.py dump --pilot <rust direct_pilot.py> --out <json> [--mutant irops]
    python v_identity.py compare <a.json> <b.json> [--expect-differ LABEL,...]

``--mutant irops`` (must-fail control) turns the R2 ``irops`` knobs ON in the head and toy builds: the compare
against mono must then report exactly those labels as differing.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import sys
import time


def _sha(data) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _builds(mod, mutant: str):
    rows = list(mod.representative_builds())
    hybrid = {k: v for k, v in mod.HYBRID_HEAD.items() if k not in mod.HEAD}
    rows.append(("head_hybrid", lambda: mod._head(**hybrid)))
    if mutant != "irops":
        return rows
    from kernels.gfx950 import mx_decode_gemv as M
    from kernels.gfx950.qwen38_mono import toy_chain as T

    out = []
    for label, build in rows:
        if label.startswith("head_"):
            kw = {"head_logits": {}, "head_fused_argmax": {"epilogue": "fused"}, "head_nt2": {"nt": 2}, "head_hybrid": hybrid}[label]
            out.append((label, lambda kw=kw: M.build_mx_decode_gemv(M.MxDecodeGemvSpec(**mod.HEAD, irops=True, **kw))))
        elif label in ("toy_best_d4w8_l2", "toy_best_d8w0"):
            fn = {"toy_best_d4w8_l2": lambda: T.best_spec(4, 8, layers=2), "toy_best_d8w0": lambda: T.best_spec(8, 0)}[label]

            def b(fn=fn):
                s = fn()
                return T.build_toy_chain(dataclasses.replace(s, knobs=dataclasses.replace(s.knobs, irops=True)))

            out.append((label, b))
        else:
            out.append((label, build))
    return out


def dump(pilot: str, out: str, mutant: str = "") -> int:
    spec = importlib.util.spec_from_file_location("q38_direct_pilot_by_path", pilot)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import rocke
    from rocke.core.ir_print import print_ir
    from rocke.core.ir_serialize import serialize
    from rocke.helpers.compile import compile_kernel

    rows = {}
    for label, build in _builds(mod, mutant):
        t0 = time.perf_counter()
        k = build()
        rec = {"name": k.name, "ir": _sha(print_ir(k)), "serialized": _sha(serialize(k))}
        art = compile_kernel(k, arch="gfx950", backend="python")
        rec.update(llvm=_sha(art.llvm_text), hsaco=_sha(art.hsaco), hsaco_bytes=len(art.hsaco))
        for field in ("code_object_backend", "direct_fallback"):
            if hasattr(art, field):
                rec[field] = getattr(art, field)
        rows[label] = rec
        print(f"{label}: {rec['name'][:60]} {rec['hsaco'][:16]} ({time.perf_counter() - t0:.1f} s)", flush=True)
    meta = {"rocke": rocke.__file__, "mutant": mutant}
    with open(out, "w", newline="\n") as fh:
        json.dump({"meta": meta, "builds": rows}, fh, indent=1, sort_keys=True)
    return 0


def compare(a: str, b: str, expect_differ: set[str]) -> int:
    da, db = (json.load(open(p))["builds"] for p in (a, b))
    bad = 0
    differ = set()
    for label in sorted(set(da) | set(db)):
        ra, rb = da.get(label), db.get(label)
        if ra is None or rb is None:
            print(f"{label}: missing in {'a' if ra is None else 'b'}")
            bad += 1
            continue
        keys = ("name", "ir", "serialized", "llvm", "hsaco", "hsaco_bytes")
        diff = [k for k in keys if ra.get(k) != rb.get(k)]
        prov = {k: rb.get(k, ra.get(k)) for k in ("code_object_backend", "direct_fallback") if k in ra or k in rb}
        if prov and prov != {"code_object_backend": "comgr", "direct_fallback": ""}:
            diff.append(f"provenance {prov}")
        print(f"{label}: {'IDENTICAL' if not diff else 'DIFFERS ' + ','.join(diff)}")
        if diff:
            differ.add(label)
    if expect_differ:
        ok = differ == expect_differ and not bad
        print(f"MUTANT {'KILLED' if ok else 'NOT KILLED'}: differ={sorted(differ)} expected={sorted(expect_differ)}")
    else:
        ok = not differ and not bad
        print(f"IDENTITY {'OK' if ok else 'FAIL'}: {len(da)} builds, {len(differ) + bad} differ")
    return 0 if ok else 1


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        args = dict(zip(sys.argv[2::2], sys.argv[3::2]))
        sys.exit(dump(args["--pilot"], args["--out"], args.get("--mutant", "")))
    args = dict(zip(sys.argv[4::2], sys.argv[5::2]))
    exp = {x for x in args.get("--expect-differ", "").split(",") if x}
    sys.exit(compare(sys.argv[2], sys.argv[3], exp))
