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
# family as emitters are extended. MODE_UNSUPPORTED is also the one gate-passing
# status that means "not compared" (see GATE_PASS below), which is admissible
# only because the gating L3/ll lane cannot produce it.
#
# Two levels, and the words are not interchangeable: a VERDICT classifies one
# (family, config); a STATUS rolls a family's verdicts up, and the gate rules on
# statuses. Hence DRIFT_VERDICTS below -- a set of verdicts, none of them named
# DRIFT, that force the family status DRIFT.
#
# Verdict per (family, config):
#   IDENTICAL          both non-empty, sha equal
#   MISMATCH           both non-empty, sha differ              (drift)
#   BOTH_REJECTED      both empty + positive rc, not end-of-range (parity-faithful)
#   ASYMMETRIC         one empty / rc disagree                 (drift)
#   CRASH              either side died on a signal (rc < 0)   (drift)
#   END                both report "unknown config" (range end; stops the family)
#
# Per-family statuses, and which of them the gate lets exit zero, are the
# GATE_PASS / GATE_FAIL maps below; the rule over them is gate_verdict(), unit-
# tested host-only in test_gate_verdict.py next to this file.
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
from typing import NamedTuple

HERE = Path(__file__).resolve().parent
ROCKE = HERE.parents[2]  # rocKE root (differential -> instances -> tests -> rocKE)
CKC = ROCKE  # engine build root (cmake -S <ROCKE> produces librocke_core.a)
PYROOT = ROCKE / "python"  # holds rocke
LIB_ROOT = (
    ROCKE.parent / "library"
)  # kernels.*, builders.*, dispatch.* top-level packages
PARITY = ROCKE / "tests" / "instances" / "parity"
LIB_PARITY = ROCKE.parent / "library" / "tests" / "parity"
INCLUDE = ROCKE / "cpp" / "include"
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
    for src_dir in (PARITY, LIB_PARITY):
        if not src_dir.exists():
            continue
        for c in sorted(src_dir.glob("*_emit.c")):
            name = c.name[: -len("_emit.c")]
            py = src_dir / f"{name}_emit.py"
            if py.exists():
                fams.append((name, src_dir))
    return fams


def compile_c(name, archive, src_dir=None):
    if src_dir is None:
        src_dir = PARITY
    src = src_dir / f"{name}_emit.c"
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


def run_py(name, idx, mode, src_dir=None):
    if src_dir is None:
        src_dir = PARITY
    env = dict(os.environ)
    roots = [str(PY_REF_ROOT), str(LIB_ROOT), str(PARITY)]
    if SHIM_DIR:
        roots.insert(0, str(SHIM_DIR))
    env["PYTHONPATH"] = os.pathsep.join(roots) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    args = [sys.executable, str(src_dir / f"{name}_emit.py"), str(idx)]
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


# Per-config verdicts that force the family to DRIFT: the engines disagreed, or
# one of them misbehaved outright (CRASH is either side dying, CANON_ERROR is the
# canonicalizer refusing the text). STRUCT_DRIFT/CANON_ERROR/CANON_EQUAL come
# from the canonical re-triage in run_family, not from classify().
DRIFT_VERDICTS = ("MISMATCH", "ASYMMETRIC", "STRUCT_DRIFT", "CANON_ERROR", "CRASH")


class FamilyStatus(NamedTuple):
    """A family's rolled-up status plus the counts that justify it."""

    status: str
    nbad: int
    ncanon: int


def family_status(configs, range_drift):
    """Roll a family's per-config verdicts up into one status.

    Most-specific first. RANGE_DRIFT outranks the two benign statuses because a
    family whose engines disagree about how many configs exist has not been
    fully compared, however clean the configs it did compare were. The last two
    are the vacuous passes: ALL_REJECTED sampled configs but byte-compared none
    of them, and NO_CONFIGS sampled nothing at all. Neither may read as GREEN.
    """
    nbad = sum(1 for c in configs if c["verdict"] in DRIFT_VERDICTS)
    ncanon = sum(1 for c in configs if c["verdict"] == "CANON_EQUAL")
    nrejected = sum(1 for c in configs if c["verdict"] == "BOTH_REJECTED")
    if nbad:
        status = "DRIFT"
    elif range_drift is not None:
        status = "RANGE_DRIFT"
    elif ncanon:
        status = "CANON_ONLY"  # benign: differs only in SSA-id numbering
    elif not configs:
        status = "NO_CONFIGS"
    elif nrejected == len(configs):
        status = "ALL_REJECTED"
    else:
        status = "GREEN"
    return FamilyStatus(status, nbad, ncanon)


def run_family(name, archive, mode, canonical=False, src_dir=None):
    if src_dir is None:
        src_dir = PARITY
    binpath, cerr = compile_c(name, archive, src_dir=src_dir)
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
        pr, po, pe = run_py(name, idx, mode, src_dir=src_dir)
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
    status, nbad, ncanon = family_status(configs, range_drift)
    return {
        "family": name,
        "status": status,
        "n": len(configs),
        "bad": nbad,
        "canon": ncanon,
        "range_drift": range_drift,
        "configs": configs,
    }


# ---- gate verdict ---------------------------------------------------------
# Every status run_family can return is named in exactly one of these two maps,
# plus NO_FAMILIES, which gate_verdict raises for the run as a whole.
# A status in NEITHER map fails the gate (see gate_verdict), so a status added
# later fails loudly until someone classifies it. The gate must never widen by
# omission: a one-status "fail if DRIFT" rule passes every other way the run can
# go wrong, including the ways that have not been invented yet.
#
# Only the keys are consulted. Both maps' values are prose for the reader; the
# failing ones are also printed as the gate's reason.
GATE_PASS = {
    "GREEN": "every sampled config agreed -- identical bytes, or both engines "
    "rejected the spec",
    "CANON_ONLY": "differs only in incidental SSA-id numbering",
    # ll mode -- the gating mode -- never yields this; only the non-gating
    # ir/verify modes can, where an emitter that predates the mode arg is a
    # known coverage gap rather than a parity failure.
    "MODE_UNSUPPORTED": "emitter does not accept this mode yet",
}
GATE_FAIL = {
    "DRIFT": "configs diverge between the engines",
    "RANGE_DRIFT": "the engines disagree about how many configs exist",
    "NO_CONFIGS": "the family enumerated zero configs",
    "ALL_REJECTED": "every sampled config was rejected, so no bytes were compared",
    "COMPILE_FAIL": "the C++ emitter did not build, so nothing was compared",
    "NO_FAMILIES": "the run discovered no families to compare",
}


class GateFailure(NamedTuple):
    """One family the gate refuses to pass, and why."""

    family: str
    status: str
    reason: str


def _gate_detail(r):
    """Evidence for a failing family, from whichever fields its record carries.

    A dashboard read back from ``--json`` is complete, but a caller may hand in
    a partial record. Missing evidence yields no clause: a gate that guesses
    which engine truncated prints a confident falsehood.
    """
    status = r.get("status")
    if status == "DRIFT" and r.get("n") is not None and r.get("bad") is not None:
        return f" ({r['bad']} of {r['n']} configs)"
    if status == "RANGE_DRIFT":
        rd = r.get("range_drift") or {}
        if "c_end" in rd:
            return f" (config {rd.get('idx')}: {'C++' if rd['c_end'] else 'Python'} ended first)"
    return ""


def gate_verdict(results):
    """The pass/fail rule over a dashboard.

    Returns ``(exit_code, failures)``, a list of :class:`GateFailure`. Pure over
    ``results`` -- no archive, no subprocess, no filesystem -- so the rule itself
    is testable on synthetic dashboards without a build.

    An empty dashboard fails. A run that compared no families is the whole-run
    form of NO_CONFIGS, and the gate's GREEN line would be claiming byte
    identity over nothing.
    """
    if not results:
        return 1, [GateFailure("<none>", "NO_FAMILIES", GATE_FAIL["NO_FAMILIES"])]
    failures = []
    for r in results:
        family = r.get("family", "<unnamed>")
        status = r.get("status")
        if status in GATE_FAIL:
            failures.append(
                GateFailure(family, status, GATE_FAIL[status] + _gate_detail(r))
            )
        elif status not in GATE_PASS:
            failures.append(
                GateFailure(
                    family,
                    str(status),
                    "unrecognized status -- classify it in GATE_PASS or GATE_FAIL",
                )
            )
    return (1 if failures else 0), failures


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
        "differs from (or is missing in) the golden file. This is the "
        "anchor check only -- its exit code answers 'does output match the "
        "blessed shas', not the parity question the default run gates on.",
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
        fams = [(n, d) for n, d in fams if any(s in n for s in subs)]

    print(f"mode={args.mode}  families={len(fams)}  archive={archive}")
    print(f"engine build-id: {archive_build_id(archive)}")
    results = []
    for name, src_dir in fams:
        r = run_family(
            name, archive, args.mode, canonical=args.canonical, src_dir=src_dir
        )
        results.append(r)
        tag = r["status"]
        extra = ""
        if r.get("n") is not None:
            extra = f"  configs={r['n']} bad={r['bad']} canon={r.get('canon', 0)}"
        print(f"  {tag:16s} {name}{extra}")
        if tag == "DRIFT":
            for c in r["configs"]:
                if c["verdict"] in DRIFT_VERDICTS:
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
    exit_code, failures = gate_verdict(results)
    if failures:
        print("\n=== GATE FAILURES ===")
        for family, status, reason in failures:
            print(f"  {status:16s} {family}: {reason}")
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
        if failures:
            print(
                "\nNOTE: recording golden from a run the gate would fail: "
                + ", ".join(f"{f.family} ({f.status})" for f in failures)
                + "\n  The golden anchors the Python reference shas, which "
                "remain well-defined. But a family that ended early or compared "
                "nothing blesses FEWER configs than a clean run would, and the "
                "missing ones return later as NEW (unblessed), not as MISSING."
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

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
