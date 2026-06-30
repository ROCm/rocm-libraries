#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# run_diff.py -- differential harness spine for the dual-backend path.
#
# Drives EVERY tests/parity/<family>_emit.{c,py} pair through a differential
# layer and sha256-compares the C engine against the Python reference across all
# sampled configs of every family. Produces a full-surface parity dashboard
# (JSON + table) and a drift inventory (the families/configs that diverge).
#
#   L3  mode=ll      lower_kernel_to_llvm        (the existing .ll byte-identity)
#   L2  mode=ir      ck.dsl.ir/v1 serialization  (catches builder drift upstream
#                                                  of .ll; needs emitter mode arg)
#   L1  mode=verify  verifier diagnostics        (needs emitter mode arg)
#
# An emitter that does not yet accept a mode arg is exercised in its default
# (.ll) mode for L3 and reported as MODE_UNSUPPORTED for L2/L1 -- so this runner
# already works against the current emitters and gains L2/L1 coverage family by
# family as emitters are extended.
#
# Classification per (family, config):
#   IDENTICAL          both non-empty, sha equal
#   MISMATCH           both non-empty, sha differ              (drift)
#   BOTH_REJECTED      both empty + positive rc, not end-of-range (parity-faithful)
#   ASYMMETRIC         one empty / rc disagree                 (drift)
#   CRASH              either side died on a signal (rc < 0)   (drift)
#   END                both report "unknown config" (range end; stops the family)
#
# Build output goes to /tmp (repo tree is slow NFS). No git operations.

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROCKE = HERE.parents[2]  # rocKE root (differential -> instances -> tests -> rocKE)
CKC = ROCKE  # engine build root (cmake -S <ROCKE> produces librocke_core.a)
PYROOT = ROCKE / "Python"  # holds rocke
PARITY = ROCKE / "tests" / "instances" / "parity"
INCLUDE = ROCKE / "Cpp" / "include"
TMP = Path(tempfile.gettempdir()) / "rocke_diff"
TMP.mkdir(parents=True, exist_ok=True)

# L5 golden anchor: committed per-(mode,family,config) reference shas. This is
# the absolute regression anchor -- any future change that alters an emitted
# .ll / canonical-ir for a blessed config fails --check-golden.
GOLDEN_DIR = HERE / "golden"
GOLDEN_FILE = GOLDEN_DIR / "llvm_gfx_all.json"


def golden_key(mode, canonical):
    """Stable key under which a run's shas are stored in the golden file."""
    return "ir_canonical" if (mode == "ir" and canonical) else mode


def collect_ref_shas(results):
    """{family: {str(idx): ref_sha}} for every config that produced output."""
    out = {}
    for r in results:
        fam = {}
        for c in r.get("configs", []):
            if c.get("ref_sha"):
                fam[str(c["idx"])] = c["ref_sha"]
        if fam:
            out[r["family"]] = fam
    return out


MAX_CFG = 128  # hard cap on config enumeration per family
TIMEOUT = 120  # seconds per emitter invocation

_CANON = None


def _canon(text):
    """Canonicalize ck.dsl.ir/v1 text (stable SSA ids, loc stripped) for
    semantic diff. Lazily imports the Python reference canonicalizer; used on
    BOTH the C and Python emitted IR so a byte mismatch that is only incidental
    SSA-id renumbering shows up as CANON_EQUAL, while real structural drift
    (differing attrs / op order) shows up as STRUCT_DRIFT."""
    global _CANON
    if _CANON is None:
        if str(PYROOT) not in sys.path:
            sys.path.insert(0, str(PYROOT))
        from rocke.core.ir_serialize import canonicalize, parse

        _CANON = (canonicalize, parse)
    canonicalize, parse = _CANON
    # canonicalize takes a KernelDef -> parse the emitted text first. A parse
    # failure here means the text is not valid/round-trippable ck.dsl.ir/v1.
    return canonicalize(parse(text))


def sh(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def archive_build_id(archive: Path) -> str:
    """Build-id stamped into the engine archive (link a 2-line probe that calls
    rocke_build_id). Returns '<unavailable: ...>' if the probe cannot be built so
    this stays purely informational and never fails the run."""
    probe_c = TMP / "_build_id_probe.cpp"
    probe_bin = TMP / "_build_id_probe"
    probe_c.write_text(
        'extern "C" const char* rocke_build_id(void);\n'
        'extern "C" const char* rocke_engine_version(void);\n'
        "#include <cstdio>\n"
        'int main(){printf("%s %s\\n", rocke_build_id(), rocke_engine_version());'
        "return 0;}\n"
    )
    try:
        cc = subprocess.run(
            [
                "c++",
                "-std=c++20",
                str(probe_c),
                str(archive),
                "-lm",
                "-o",
                str(probe_bin),
            ],
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )
        if cc.returncode != 0:
            return "<unavailable: link failed>"
        run = subprocess.run(
            [str(probe_bin)], capture_output=True, text=True, timeout=30
        )
        return run.stdout.strip() or "<unavailable: empty>"
    except Exception as e:  # noqa: BLE001
        return f"<unavailable: {e}>"


def find_families():
    fams = []
    for c in sorted(PARITY.glob("*_emit.c")):
        name = c.name[: -len("_emit.c")]
        py = PARITY / f"{name}_emit.py"
        if py.exists():
            fams.append(name)
    return fams


def compile_c(name, archive):
    src = PARITY / f"{name}_emit.c"
    out = TMP / f"{name}_emit_c"
    # The engine archive is C++20; emitters are compiled as C++20 against it.
    cmd = [
        "c++",
        "-std=c++20",
        "-I",
        str(INCLUDE),
        str(src),
        str(archive),
        "-lm",
        "-o",
        str(out),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
    if p.returncode != 0:
        return None, p.stderr
    return out, ""


def run_c(binpath, idx, mode):
    args = [str(binpath), str(idx)]
    if mode != "ll":
        args.append(mode)
    try:
        p = subprocess.run(args, capture_output=True, timeout=TIMEOUT)
        return p.returncode, p.stdout, p.stderr.decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        return 124, b"", "TIMEOUT"


# Reference-side Python root: defaults to this branch's tree, but --pyroot can
# point it at another tree (e.g. the merge-target rocke-prototype) to measure
# C++(this branch) vs Python(target) drift. SHIM_DIR (optional) is
# prepended so modules the target lacks (ir_serialize/verify) import as stubs.
PY_REF_ROOT = PYROOT
SHIM_DIR = None


def run_py(name, idx, mode):
    env = dict(os.environ)
    roots = [str(PY_REF_ROOT)]
    if SHIM_DIR:
        roots.insert(0, str(SHIM_DIR))
    env["PYTHONPATH"] = os.pathsep.join(roots) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    args = [sys.executable, str(PARITY / f"{name}_emit.py"), str(idx)]
    if mode != "ll":
        args.append(mode)
    try:
        p = subprocess.run(args, capture_output=True, timeout=TIMEOUT, env=env)
        return p.returncode, p.stdout, p.stderr.decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        return 124, b"", "TIMEOUT"


def is_end(rc, out, err):
    return rc != 0 and not out and "unknown config" in err.lower()


def mode_unsupported(err):
    e = err.lower()
    return "usage" in e or "unknown mode" in e or "too many arg" in e


def classify(cr, co, ce, pr, po, pe):
    # end-of-range only when BOTH say unknown-config
    if is_end(cr, co, ce) and is_end(pr, po, pe):
        return "END", None
    if co and po:
        return ("IDENTICAL" if sh(co) == sh(po) else "MISMATCH"), (sh(co), sh(po))
    if not co and not po:
        # A negative returncode means the process died on a signal
        # (e.g. -11 SIGSEGV, -6 SIGABRT) -- exactly the UB this harness
        # exists to catch. Never let a crash masquerade as parity:
        # surface it as a distinct verdict instead of BOTH_REJECTED.
        if cr < 0 or pr < 0:
            return "CRASH", None
        # both produced nothing:
        #   rc0/rc0  -> both succeeded with no output (e.g. verify mode, no diags) = IDENTICAL
        #   >0/>0    -> both cleanly rejected the spec (parity-faithful)
        #   otherwise-> one rejected, one didn't = drift
        if cr == 0 and pr == 0:
            return "IDENTICAL", None
        if cr > 0 and pr > 0:
            return "BOTH_REJECTED", None
        return "ASYMMETRIC", None
    return "ASYMMETRIC", (sh(co) if co else None, sh(po) if po else None)


def run_family(name, archive, mode, canonical=False):
    binpath, cerr = compile_c(name, archive)
    if binpath is None:
        return {
            "family": name,
            "status": "COMPILE_FAIL",
            "error": cerr.strip().splitlines()[-3:],
            "configs": [],
        }
    configs = []
    range_drift = None
    for idx in range(MAX_CFG):
        cr, co, ce = run_c(binpath, idx, mode)
        pr, po, pe = run_py(name, idx, mode)
        # mode not supported by this emitter -> stop trying this mode
        if mode != "ll" and (mode_unsupported(ce) or mode_unsupported(pe)):
            return {
                "family": name,
                "status": "MODE_UNSUPPORTED",
                "mode": mode,
                "configs": configs,
            }
        c_end, p_end = is_end(cr, co, ce), is_end(pr, po, pe)
        if c_end and p_end:
            break  # clean shared end-of-range
        if c_end or p_end:  # one side has fewer sampled configs
            range_drift = {"idx": idx, "c_end": c_end, "p_end": p_end}
            break
        verdict, shas = classify(cr, co, ce, pr, po, pe)
        # canonical re-triage: split byte-MISMATCH into incidental id drift
        # (CANON_EQUAL) vs real structural drift (STRUCT_DRIFT)
        if canonical and mode == "ir" and verdict == "MISMATCH":
            try:
                cc = _canon(co.decode("utf-8", "replace"))
                pc = _canon(po.decode("utf-8", "replace"))
                verdict = "CANON_EQUAL" if cc == pc else "STRUCT_DRIFT"
            except Exception as e:  # noqa
                verdict = "CANON_ERROR"
        # L5 golden anchor: the reference sha to bless. For ll it is the Python
        # (== C, when GREEN) byte sha; for ir --canonical it is the canonical
        # form's sha (stable across incidental SSA-id renumbering). Only recorded
        # for configs both sides emitted; reject/empty configs carry None.
        ref_sha = None
        if po:
            if canonical and mode == "ir":
                try:
                    ref_sha = sh(_canon(po.decode("utf-8", "replace")).encode("utf-8"))
                except Exception:  # noqa
                    ref_sha = sh(po)
            else:
                ref_sha = sh(po)
        configs.append(
            {
                "idx": idx,
                "verdict": verdict,
                "shas": shas,
                "ref_sha": ref_sha,
                "c_rc": cr,
                "p_rc": pr,
            }
        )
    nbad = sum(
        1
        for c in configs
        if c["verdict"]
        in ("MISMATCH", "ASYMMETRIC", "STRUCT_DRIFT", "CANON_ERROR", "CRASH")
    )
    ncanon = sum(1 for c in configs if c["verdict"] == "CANON_EQUAL")
    if nbad:
        status = "DRIFT"
    elif ncanon:
        status = "CANON_ONLY"  # benign: differs only in SSA-id numbering
    else:
        status = "GREEN"
    if range_drift is not None and status == "GREEN":
        status = "RANGE_DRIFT"
    return {
        "family": name,
        "status": status,
        "n": len(configs),
        "bad": nbad,
        "canon": ncanon,
        "range_drift": range_drift,
        "configs": configs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="ll", choices=["ll", "ir", "verify"])
    ap.add_argument(
        "--canonical",
        action="store_true",
        help="for mode=ir, re-triage byte MISMATCH into CANON_EQUAL "
        "(incidental SSA-id drift) vs STRUCT_DRIFT (real divergence)",
    )
    ap.add_argument(
        "--archive",
        default=str(Path(tempfile.gettempdir()) / "rocke_verify" / "librocke_core.a"),
    )
    ap.add_argument(
        "--only", default="", help="comma-separated family substrings to include"
    )
    ap.add_argument(
        "--pyroot",
        default="",
        help="override the reference-side Python root (e.g. the merge-target "
        "tree) to measure C++(this branch) vs Python(other tree) drift",
    )
    ap.add_argument(
        "--shim",
        default="",
        help="dir prepended to PYTHONPATH for the reference side (stub modules "
        "the target tree lacks, e.g. ir_serialize/verify); ll-mode only",
    )
    ap.add_argument("--json", default=str(TMP / "dashboard.json"))
    ap.add_argument(
        "--record-golden",
        action="store_true",
        help="bless the current reference shas into the committed "
        f"golden file ({GOLDEN_FILE}). Run ONLY from a verified-"
        "good state (gates GREEN). Re-blessing is intentional and "
        "should accompany a reviewed, expected output change.",
    )
    ap.add_argument(
        "--check-golden",
        action="store_true",
        help="re-run and fail on any config whose reference sha "
        "differs from (or is missing in) the golden file.",
    )
    args = ap.parse_args()

    global PY_REF_ROOT, SHIM_DIR
    if args.pyroot:
        PY_REF_ROOT = Path(args.pyroot)
        print(f"[diff] reference Python root overridden -> {PY_REF_ROOT}")
    if args.shim:
        SHIM_DIR = Path(args.shim)

    archive = Path(args.archive)
    if not archive.exists():
        sys.exit(
            f"archive not found: {archive}  (build: cmake -S {CKC} -B /tmp/rocke_verify && cmake --build /tmp/rocke_verify -j)"
        )

    fams = find_families()
    if args.only:
        subs = [s for s in args.only.split(",") if s]
        fams = [f for f in fams if any(s in f for s in subs)]

    print(f"mode={args.mode}  families={len(fams)}  archive={archive}")
    print(f"engine build-id: {archive_build_id(archive)}")
    results = []
    for name in fams:
        r = run_family(name, archive, args.mode, canonical=args.canonical)
        results.append(r)
        tag = r["status"]
        extra = ""
        if tag in ("GREEN", "DRIFT", "CANON_ONLY"):
            extra = f"  configs={r['n']} bad={r['bad']} canon={r.get('canon', 0)}"
        print(f"  {tag:16s} {name}{extra}")
        if tag == "DRIFT":
            for c in r["configs"]:
                if c["verdict"] in (
                    "MISMATCH",
                    "ASYMMETRIC",
                    "STRUCT_DRIFT",
                    "CANON_ERROR",
                    "CRASH",
                ):
                    print(
                        f"        cfg[{c['idx']}] {c['verdict']} c_rc={c['c_rc']} p_rc={c['p_rc']}"
                    )

    Path(args.json).write_text(json.dumps(results, indent=2))
    # summary
    by = {}
    for r in results:
        by[r["status"]] = by.get(r["status"], 0) + 1
    print("\n=== SUMMARY (mode={}) ===".format(args.mode))
    for k in sorted(by):
        print(f"  {k:16s} {by[k]}")
    drift = [r["family"] for r in results if r["status"] == "DRIFT"]
    cfail = [r["family"] for r in results if r["status"] == "COMPILE_FAIL"]
    if drift:
        print("  DRIFT families:", ", ".join(drift))
    if cfail:
        print("  COMPILE_FAIL families:", ", ".join(cfail))
    print(f"\ndashboard: {args.json}")

    # ---- L5 golden anchor -------------------------------------------------
    if args.record_golden or args.check_golden:
        key = golden_key(args.mode, args.canonical)
        cur = collect_ref_shas(results)
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        store = {}
        if GOLDEN_FILE.exists():
            store = json.loads(GOLDEN_FILE.read_text())

    if args.record_golden:
        # The golden records the PYTHON reference sha per config, which is the
        # source of truth regardless of whether the C engine currently agrees.
        # A C-vs-Python DRIFT family is reported as a caveat (e.g. the known
        # pre-existing gfx950_attention_tiled_2d_fastkv_regp ir drift) but does
        # not invalidate the reference shas being blessed.
        if drift:
            print(
                "\nNOTE: recording golden with C-vs-Python DRIFT present in: "
                + ", ".join(drift)
                + "\n  The golden anchors the Python "
                "reference shas, which remain well-defined for these families."
            )
        store[key] = cur
        nblessed = sum(len(v) for v in cur.values())
        GOLDEN_FILE.write_text(json.dumps(store, indent=2, sort_keys=True) + "\n")
        print(
            f"\nrecorded golden[{key}]: {len(cur)} families, {nblessed} configs "
            f"-> {GOLDEN_FILE}"
        )
        return 0

    if args.check_golden:
        ref = store.get(key)
        if not ref:
            print(
                f"\nNO GOLDEN for key '{key}'. Bless first: "
                f"run_diff.py --mode {args.mode}"
                f"{' --canonical' if args.canonical else ''} --record-golden"
            )
            return 1
        mism, missing, extra = [], [], []
        for fam, cfgs in ref.items():
            cur_fam = cur.get(fam, {})
            for idx, sha in cfgs.items():
                got = cur_fam.get(idx)
                if got is None:
                    missing.append((fam, idx))
                elif got != sha:
                    mism.append((fam, idx))
        for fam, cfgs in cur.items():
            for idx in cfgs:
                if idx not in ref.get(fam, {}):
                    extra.append((fam, idx))
        print(f"\n=== GOLDEN CHECK (key={key}) ===")
        print(f"  blessed configs : {sum(len(v) for v in ref.values())}")
        print(f"  UNBLESSED CHANGE: {len(mism)}")
        print(f"  MISSING (gone)  : {len(missing)}")
        print(f"  NEW (unblessed) : {len(extra)}")
        for fam, idx in (mism + missing)[:30]:
            print(f"    {fam}[{idx}]")
        if mism or missing:
            print(
                "\nGOLDEN MISMATCH: emitted output changed vs the committed "
                "anchor. If this change is expected & reviewed, re-bless with "
                "--record-golden; otherwise you introduced a regression."
            )
            return 1
        if extra:
            print("\nNew configs not in golden (informational; re-bless to add).")
        print("GOLDEN OK")
        return 0

    # exit nonzero if any drift (compile-fail tracked separately, not a parity fail)
    return 1 if drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
