#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# ir_artifact_diff.py -- prove the IR-artifact lowering path is byte-identical
# to Python's own lowering, across every kernel family.
#
# This exercises the single-lowerer seam: Python authors and serializes the IR
# (the `ck.dsl.ir/v1` artifact), and the C++ engine -- with NO per-family C
# builder involved -- parses that artifact and lowers it to LLVM IR. For every
# family/config we compare:
#
#   ir_text = python <fam>_emit.py <idx> ir   (Python serialize)
#   ll_ref  = python <fam>_emit.py <idx> ll   (Python lower = reference)
#   ll_art  = ir_lower_cli <arch> < ir_text   (C parse + C lower of the artifact)
#
#   IDENTICAL  ll_art == ll_ref byte-for-byte         (seam collapses cleanly)
#   DRIFT      both non-empty, bytes differ           (real divergence)
#   SKIP       both sides reject the config            (parity-faithful)
#   ERROR      asymmetric / parse / lower failure      (artifact path gap)
#
# Family discovery and config enumeration reuse run_diff.py so the family list
# is never hardcoded. The lowering arch per family is derived from its name
# prefix (gfx1151_/gfx1201_/gfx942_/gfx950_), defaulting to gfx950; because a
# few families lower different configs for different arches, a config that does
# not match under its primary arch is retried across the full candidate arch set
# and counted IDENTICAL if any arch reproduces the reference byte-for-byte (the
# matching arch is reported). The IR artifact itself is arch-independent; only
# the lowering target varies.

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_diff  # noqa: E402  (reuse family discovery + enumeration constants)

PARITY = run_diff.PARITY
PYROOT = run_diff.PYROOT
MAX_CFG = run_diff.MAX_CFG
TIMEOUT = run_diff.TIMEOUT

# Arch derivation. A family whose name starts with one of these prefixes lowers
# for that arch; everything else defaults to gfx950. The full candidate set is
# tried as a fallback for the handful of families that lower per-config arches.
ARCH_PREFIXES = (
    ("gfx1151_", "gfx1151"),
    ("gfx1201_", "gfx1201"),
    ("gfx942_", "gfx942"),
    ("gfx950_", "gfx950"),
)
DEFAULT_ARCH = "gfx950"
CANDIDATE_ARCHES = ("gfx950", "gfx942", "gfx1201", "gfx1151", "gfx11-generic")


def primary_arch(family):
    for prefix, arch in ARCH_PREFIXES:
        if family.startswith(prefix):
            return arch
    return DEFAULT_ARCH


def arch_candidates(family):
    """Primary arch first, then the rest of the candidate set (deduped). The
    primary covers all single-arch families directly; the remainder catches the
    conv/matmul families that lower different configs for different arches."""
    prim = primary_arch(family)
    order = [prim] + [a for a in CANDIDATE_ARCHES if a != prim]
    return order


def run_py(family, idx, mode):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PYROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    args = [sys.executable, str(PARITY / f"{family}_emit.py"), str(idx)]
    if mode != "ll":
        args.append(mode)
    try:
        p = subprocess.run(args, capture_output=True, timeout=TIMEOUT, env=env)
        return p.returncode, p.stdout, p.stderr.decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        return 124, b"", "TIMEOUT"


def run_cli(cli, arch, ir_text):
    try:
        p = subprocess.run(
            [str(cli), arch],
            input=ir_text,
            capture_output=True,
            timeout=TIMEOUT,
        )
        return p.returncode, p.stdout, p.stderr.decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        return 124, b"", "TIMEOUT"


def is_end(rc, out, err):
    return rc != 0 and not out and "unknown config" in err.lower()


_HDR = b"ckdsl.ir v1"


def doc_spans(ir):
    """Byte spans of each 'ckdsl.ir v1' document in a (possibly multi-kernel)
    artifact: from the header line through the newline after the kernel's
    closing brace. Brace-aware, skipping ';' comment lines and quoted strings.
    Returns [] for a single-document artifact (the common case)."""
    spans = []
    n = len(ir)
    i = 0
    while i < n:
        if ir[i : i + len(_HDR)] == _HDR and (i == 0 or ir[i - 1 : i] == b"\n"):
            start = i
            depth = 0
            started = False
            in_str = False
            line_comment = False
            at_line_start = True
            j = i
            while j < n:
                c = ir[j : j + 1]
                if at_line_start:
                    line_comment = c == b";"
                    at_line_start = False
                if c == b"\n":
                    at_line_start = True
                    line_comment = False
                    j += 1
                    if started and depth <= 0:
                        break
                    continue
                if line_comment:
                    j += 1
                    continue
                if in_str:
                    if c == b"\\":
                        j += 2
                        continue
                    if c == b'"':
                        in_str = False
                    j += 1
                    continue
                if c == b'"':
                    in_str = True
                elif c == b"{":
                    depth += 1
                    started = True
                elif c == b"}":
                    depth -= 1
                j += 1
            spans.append((start, j))
            i = j
        else:
            i += 1
    return spans if len(spans) > 1 else []


def _lower_stitched(cli, arch, ir_text, spans):
    """Lower each document in a multi-kernel artifact and stitch the results
    back into the original framing (inter-document comment lines preserved
    byte-for-byte), mirroring how the multi-kernel emitter concatenates the
    per-kernel .ll. Returns (assembled_bytes_or_None, err)."""
    parts = []
    last = 0
    for s, e in spans:
        parts.append(ir_text[last:s])  # framing before this document
        rc, out, err = run_cli(cli, arch, ir_text[s:e])
        if not out:
            return None, err
        parts.append(out)
        last = e
    parts.append(ir_text[last:])  # trailing framing after the last document
    return b"".join(parts), ""


def lower_artifact(cli, family, ir_text, ll_ref):
    """Parse+lower the IR artifact with the C CLI. Tries the family's primary
    arch first, then the remaining candidates, returning (rc, out, err, arch)
    for the arch that byte-matches the reference -- or the primary-arch result
    if none match (so the caller can report a real DRIFT/ERROR). Multi-kernel
    artifacts (several 'ckdsl.ir v1' documents) are lowered document-by-document
    and stitched back into their framing."""
    spans = doc_spans(ir_text)
    primary = None
    for arch in arch_candidates(family):
        if spans:
            out, err = _lower_stitched(cli, arch, ir_text, spans)
            rc = 0 if out else 1
        else:
            rc, out, err = run_cli(cli, arch, ir_text)
        if primary is None:
            primary = (rc, out or b"", err, arch)
        if out and out == ll_ref:
            return rc, out, err, arch
    return primary


def run_family(cli, family):
    rows = []
    for idx in range(MAX_CFG):
        pr_ir, ir_text, ir_err = run_py(family, idx, "ir")
        pr_ll, ll_ref, ll_err = run_py(family, idx, "ll")

        # End-of-range: Python rejects with "unknown config" on both modes.
        if is_end(pr_ir, ir_text, ir_err) and is_end(pr_ll, ll_ref, ll_err):
            break

        # Both-reject (a config the spec legitimately declines): no IR, nonzero
        # rc, but NOT end-of-range. Parity-faithful -> SKIP.
        if not ir_text and not ll_ref and pr_ir != 0 and pr_ll != 0:
            rows.append({"idx": idx, "verdict": "SKIP", "arch": "", "note": ""})
            continue

        # A few emitters ignore the mode arg and always print serialized IR
        # (their "ll" output is itself ck.dsl.ir/v1, not lowerable .ll). Those
        # are IR-only emitters; the artifact lowering path does not apply, so we
        # report the family as N/A rather than a spurious DRIFT.
        if ll_ref.lstrip().startswith(b"ckdsl.ir"):
            rows.append({"idx": idx, "verdict": "NA", "arch": "", "note": ""})
            continue

        # Python produced a reference .ll but no serializable IR (or vice
        # versa): the artifact path cannot be exercised for this config.
        if not ir_text or not ll_ref:
            rows.append(
                {
                    "idx": idx,
                    "verdict": "ERROR",
                    "arch": "",
                    "note": "py ir/ll asymmetric (ir=%d ll=%d bytes)"
                    % (len(ir_text), len(ll_ref)),
                }
            )
            continue

        cr, c_out, c_err, arch = lower_artifact(cli, family, ir_text, ll_ref)
        if not c_out:
            note = (c_err.strip().splitlines() or ["(no stderr)"])[-1]
            rows.append({"idx": idx, "verdict": "ERROR", "arch": arch, "note": note})
            continue
        if c_out == ll_ref:
            rows.append({"idx": idx, "verdict": "IDENTICAL", "arch": arch, "note": ""})
        else:
            note = first_diff(ll_ref, c_out)
            rows.append({"idx": idx, "verdict": "DRIFT", "arch": arch, "note": note})
    return rows


def first_diff(ref, art):
    """Short human description of where the two .ll texts first diverge."""
    ref_lines = ref.decode("utf-8", "replace").splitlines()
    art_lines = art.decode("utf-8", "replace").splitlines()
    n = min(len(ref_lines), len(art_lines))
    for i in range(n):
        if ref_lines[i] != art_lines[i]:
            return "line %d: ref %r vs art %r" % (
                i + 1,
                ref_lines[i][:60],
                art_lines[i][:60],
            )
    if len(ref_lines) != len(art_lines):
        return "len differs: ref %d lines, art %d lines (%d vs %d bytes)" % (
            len(ref_lines),
            len(art_lines),
            len(ref),
            len(art),
        )
    return "byte-only diff (%d vs %d bytes)" % (len(ref), len(art))


def family_status(rows):
    n_id = sum(1 for r in rows if r["verdict"] == "IDENTICAL")
    n_dr = sum(1 for r in rows if r["verdict"] == "DRIFT")
    n_er = sum(1 for r in rows if r["verdict"] == "ERROR")
    n_sk = sum(1 for r in rows if r["verdict"] == "SKIP")
    n_na = sum(1 for r in rows if r["verdict"] == "NA")
    if n_dr or n_er:
        status = "DRIFT"
    elif n_id:
        status = "IDENTICAL"
    elif n_na and not n_sk:
        status = "N/A"
    else:
        status = "SKIP_ONLY"
    return status, n_id, n_dr, n_er, n_sk, n_na


def _run_pass(cli, fams, arch_label):
    """One Python-IR -> C-parse -> C-lower vs Python-lower pass over `fams`.
    Returns 1 on any real DRIFT/ERROR, else 0."""
    print(
        f"IR-artifact lowering parity  families={len(fams)}  cli={cli}  arch={arch_label}"
    )
    print("-" * 78)

    results = []
    tot_id = tot_dr = tot_er = tot_sk = tot_na = 0
    for family in fams:
        rows = run_family(cli, family)
        status, n_id, n_dr, n_er, n_sk, n_na = family_status(rows)
        results.append((family, status, rows))
        tot_id += n_id
        tot_dr += n_dr
        tot_er += n_er
        tot_sk += n_sk
        tot_na += n_na
        extra = f" na={n_na}" if n_na else ""
        print(
            f"  {status:11s} {family:42s} "
            f"identical={n_id} drift={n_dr} error={n_er} skip={n_sk}{extra}"
        )
        for r in rows:
            if r["verdict"] in ("DRIFT", "ERROR"):
                print(
                    f"        cfg[{r['idx']}] {r['verdict']} arch={r['arch']} "
                    f"{r['note']}"
                )

    fam_identical = [f for f, s, _ in results if s == "IDENTICAL"]
    fam_drift = [f for f, s, _ in results if s == "DRIFT"]
    fam_skip = [f for f, s, _ in results if s == "SKIP_ONLY"]
    fam_na = [f for f, s, _ in results if s == "N/A"]

    print("\n" + "=" * 78)
    print(
        f"SUMMARY [arch={arch_label}] -- Python-IR -> C-parse -> C-lower vs Python-lower"
    )
    print("=" * 78)
    print(f"  families IDENTICAL : {len(fam_identical)}")
    print(f"  families DRIFT     : {len(fam_drift)}")
    print(f"  families SKIP-only : {len(fam_skip)}")
    print(f"  families N/A       : {len(fam_na)} (IR-only emitters)")
    print(f"  total configs identical : {tot_id}")
    print(f"  total configs drift     : {tot_dr}")
    print(f"  total configs error     : {tot_er}")
    print(f"  total configs skip      : {tot_sk}")
    print(f"  total configs n/a       : {tot_na}")
    if fam_drift:
        print("\n  DRIFT/ERROR families:")
        for f, s, rows in results:
            if s == "DRIFT":
                bad = [r for r in rows if r["verdict"] in ("DRIFT", "ERROR")]
                print(f"    {f}: {len(bad)} bad config(s)")

    # Nonzero exit only on real drift/error; a SKIP-only family is fine.
    return 1 if fam_drift else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cli",
        default=str(Path(tempfile.gettempdir()) / "rocke_irart" / "ir_lower_cli"),
        help="path to the compiled ir_lower_cli tool",
    )
    ap.add_argument(
        "--only", default="", help="comma-separated family substrings to include"
    )
    args = ap.parse_args()

    cli = Path(args.cli)
    if not cli.exists():
        sys.exit(
            f"ir_lower_cli not found: {cli}\n"
            "build it: cmake -S <rocKE> -B /tmp/rocke_irart -DCMAKE_BUILD_TYPE="
            "Release && cmake --build /tmp/rocke_irart --target rocke_core -j && "
            "c++ -std=c++20 -I <rocKE>/Cpp/include <rocKE>/tests/core/"
            "ir_lower_cli.cpp /tmp/rocke_irart/librocke_core.a -lm "
            "-o /tmp/rocke_irart/ir_lower_cli"
        )

    fams = run_diff.find_families()
    if args.only:
        subs = [s for s in args.only.split(",") if s]
        fams = [f for f in fams if any(s in f for s in subs)]

    # Each family is built+lowered at its own (spec, arch); byte-identity is a
    # per-(spec, arch) property and arch-incompatible configs reject on both
    # sides (counted SKIP). Arch coverage comes from the configs the emitters
    # enumerate (the gfx-prefixed families + any gfx942/gfx1151 configs), not
    # from a global arch override.
    return _run_pass(cli, fams, "per-family")


if __name__ == "__main__":
    raise SystemExit(main())
