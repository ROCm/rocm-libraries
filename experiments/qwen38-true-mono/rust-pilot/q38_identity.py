"""Knob-off byte identity of the Qwen3.8 builds between two rocKE trees.

    python q38_identity.py dump --pilot <direct_pilot.py> --out <json>
    python q38_identity.py compare <a.json> <b.json>

``dump`` runs with PYTHONPATH on ONE tree (platform/python + library) and
compiles every ``direct_pilot.representative_builds()`` kernel the default
way (``compile_kernel(k, arch="gfx950", backend="python")``: COMGR, no
knob), recording the kernel name and the sha256 of its printed IR, its
serialized IR, its LLVM text and its code object. ``direct_pilot.py`` is
loaded by path, so the same build list runs against a tree that does not
have the file (it uses only builders both trees have). ``compare`` exits 1
unless both dumps hold the same labels with equal records.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time


def _sha(data) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def dump(pilot: str, out: str) -> int:
    spec = importlib.util.spec_from_file_location("q38_direct_pilot_by_path", pilot)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import rocke
    from rocke.core.ir_print import print_ir
    from rocke.core.ir_serialize import serialize
    from rocke.helpers.compile import compile_kernel

    rows = {}
    for label, build in mod.representative_builds():
        t0 = time.perf_counter()
        k = build()
        rec = {"name": k.name, "ir": _sha(print_ir(k)), "serialized": _sha(serialize(k))}
        art = compile_kernel(k, arch="gfx950", backend="python")
        rec.update(llvm=_sha(art.llvm_text), hsaco=_sha(art.hsaco), hsaco_bytes=len(art.hsaco))
        for field in ("code_object_backend", "direct_fallback"):
            if hasattr(art, field):
                rec[field] = getattr(art, field)
        rows[label] = rec
        print(f"{label}: {rec['hsaco'][:16]} ({time.perf_counter() - t0:.1f} s)", flush=True)
    meta = {"rocke": rocke.__file__}
    with open(out, "w", newline="\n") as fh:
        json.dump({"meta": meta, "builds": rows}, fh, indent=1, sort_keys=True)
    return 0


def compare(a: str, b: str) -> int:
    da, db = (json.load(open(p))["builds"] for p in (a, b))
    bad = 0
    for label in sorted(set(da) | set(db)):
        ra, rb = da.get(label), db.get(label)
        if ra is None or rb is None:
            print(f"{label}: missing in {'a' if ra is None else 'b'}")
            bad += 1
            continue
        # The overlaid tree records the code-object provenance; the mono tree has none.
        keys = ("name", "ir", "serialized", "llvm", "hsaco", "hsaco_bytes")
        diff = [k for k in keys if ra.get(k) != rb.get(k)]
        prov = {k: rb.get(k, ra.get(k)) for k in ("code_object_backend", "direct_fallback") if k in ra or k in rb}
        if prov and prov != {"code_object_backend": "comgr", "direct_fallback": ""}:
            diff.append(f"provenance {prov}")
        print(f"{label}: {'IDENTICAL' if not diff else 'DIFFERS ' + ','.join(diff)}")
        bad += bool(diff)
    print(f"IDENTITY {'OK' if not bad else 'FAIL'}: {len(da)} builds, {bad} differ")
    return 1 if bad else 0


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        args = dict(zip(sys.argv[2::2], sys.argv[3::2]))
        sys.exit(dump(args["--pilot"], args["--out"]))
    sys.exit(compare(sys.argv[2], sys.argv[3]))
