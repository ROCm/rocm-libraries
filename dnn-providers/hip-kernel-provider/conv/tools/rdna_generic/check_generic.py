#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Compile conv kernel configs for an RDNA chip and its LLVM generic target, and check that the
generic build keeps the chip build's matrix-core code.

Input is a list of HIPRTC compiles, one JSON object a line: {"name": "kernels/ConvIgemm.cpp",
"options": ["-std=c++17", "--offload-arch=gfx1100", "-DCONV_IGEMM_T=_Float16", ...]}, as
rtc_hook.c's HKP_RTC_LOG writes them while the provider prepares plans (any GPU test or
census run on an RDNA device). The chip is the record's --offload-arch, the generic target
the LLVM generic whose members contain it (gfx11-generic, gfx12-generic). Records on any
other arch are skipped.

Each unique (source, defines) compiles offline with clang for both targets, from the kernel
sources under --kernels-root (the descriptors/conv tree), and fails when:
  - either compile fails;
  - the generic code object's matrix-core instructions (v_wmma*, v_swmmac*, v_mfma*) differ
    from the chip's, as a multiset of mnemonics: a WMMA config that silently compiled to its
    trap fallback has none;
  - the generic code object has more s_trap than the chip one.

--baseline-root OLD_TREE instead compiles each config for its chip from OLD_TREE and from
--kernels-root and fails unless the disassembly is identical (a guard refactor must not change
an exact-chip build).

Exit status 0 only when every config passes. Uses -j processes (default 2).
"""

import argparse
import collections
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile

GENERIC_OF = {}
for chip in (
    "gfx1100",
    "gfx1101",
    "gfx1102",
    "gfx1103",
    "gfx1150",
    "gfx1151",
    "gfx1152",
    "gfx1153",
):
    GENERIC_OF[chip] = "gfx11-generic"
for chip in ("gfx1200", "gfx1201"):
    GENERIC_OF[chip] = "gfx12-generic"

MATRIX_RE = re.compile(r"\b(v_wmma\w*|v_swmmac\w*|v_mfma\w*|v_smfmac\w*)\b")
TRAP_RE = re.compile(r"\bs_trap\b")
ARCH_PREFIX = "--offload-arch="


def config_key(record):
    """The source and every option but the target and the language standard."""
    opts = [
        o
        for o in record["options"]
        if not o.startswith(ARCH_PREFIX) and not o.startswith("-std=")
    ]
    return record["name"], tuple(sorted(opts))


def chip_of(record):
    for o in record["options"]:
        if o.startswith(ARCH_PREFIX):
            return o[len(ARCH_PREFIX) :].split(":")[0]
    return None


def find_source(root, name):
    base = os.path.basename(name)
    for d in sorted(os.listdir(root)):
        path = os.path.join(root, d, "kernels", base)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"{name}: no kernels/{base} under {root}")


def include_flags(root):
    return [
        f"-I{os.path.join(root, d, 'kernels')}"
        for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d, "kernels"))
    ]


def compile_one(args, root, name, opts, arch, out):
    cmd = [
        args.clang,
        "-x",
        "hip",
        "--offload-device-only",
        "--no-gpu-bundle-output",
        f"--offload-arch={arch}",
        "-O3",
        "-std=c++17",
        "-w",
        f"-I{os.path.join(args.rocm, 'include')}",
        "-include",
        "hip/hip_runtime.h",
        *include_flags(root),
        *opts,
        find_source(root, name),
        "-o",
        out,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return proc.stderr.strip().splitlines()[-20:]
    return None


def disassemble(args, path):
    proc = subprocess.run(
        [args.objdump, "-d", "--no-show-raw-insn", "--no-leading-addr", path],
        capture_output=True,
        text=True,
        check=True,
    )
    # Drop the file header line, which names the path.
    return "\n".join(
        line for line in proc.stdout.splitlines() if not line.startswith(path)
    )


def check(args, item):
    (name, opts), chip = item
    tag = hashlib.sha1(repr((name, opts, chip)).encode()).hexdigest()[:12]
    label = f"{name} [{chip}] {' '.join(o for o in opts if 'ATOM' in o or '_T=' in o)}"
    with tempfile.TemporaryDirectory(prefix="rdg-") as tmp:
        if args.baseline_root:
            outs = {}
            for side, root in (("old", args.baseline_root), ("new", args.kernels_root)):
                outs[side] = os.path.join(tmp, f"{side}.co")
                err = compile_one(args, root, name, list(opts), chip, outs[side])
                if err:
                    return (
                        tag,
                        label,
                        False,
                        f"{side} tree failed to compile: " + " | ".join(err),
                    )
            same = disassemble(args, outs["old"]) == disassemble(args, outs["new"])
            return (
                tag,
                label,
                same,
                "identical" if same else "ISA differs between trees",
            )

        generic = GENERIC_OF[chip]
        texts = {}
        for arch in (chip, generic):
            out = os.path.join(tmp, f"{arch}.co")
            err = compile_one(args, args.kernels_root, name, list(opts), arch, out)
            if err:
                return (
                    tag,
                    label,
                    False,
                    f"{arch} failed to compile: " + " | ".join(err),
                )
            texts[arch] = disassemble(args, out)
        mm = {a: collections.Counter(MATRIX_RE.findall(t)) for a, t in texts.items()}
        traps = {a: len(TRAP_RE.findall(t)) for a, t in texts.items()}
        ok = mm[chip] == mm[generic] and traps[generic] <= traps[chip]
        same = texts[chip].split("\n", 1)[-1] == texts[generic].split("\n", 1)[-1]
        detail = (
            f"matrix chip={sum(mm[chip].values())} generic={sum(mm[generic].values())}"
            f" traps chip={traps[chip]} generic={traps[generic]}"
            f" isa={'identical' if same else 'differs'}"
        )
        return tag, label, ok, detail


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--configs", required=True, nargs="+", help="HKP_RTC_LOG JSONL file(s)"
    )
    p.add_argument("--kernels-root", required=True, help="descriptors/conv directory")
    p.add_argument(
        "--baseline-root", help="old descriptors/conv tree: exact-chip ISA identity"
    )
    p.add_argument("--rocm", default=os.environ.get("ROCM_PATH", "/opt/rocm"))
    p.add_argument("--clang")
    p.add_argument("--objdump")
    p.add_argument("-j", "--jobs", type=int, default=2)
    p.add_argument(
        "--sample",
        type=int,
        default=0,
        help="at most N configs per (source, chip, matrix atom); 0 = all",
    )
    p.add_argument("--report", help="write one CSV line per config here")
    args = p.parse_args()
    args.clang = args.clang or os.path.join(args.rocm, "llvm", "bin", "clang")
    args.objdump = args.objdump or os.path.join(
        args.rocm, "llvm", "bin", "llvm-objdump"
    )

    items = {}
    for path in args.configs:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                chip = chip_of(rec)
                if chip not in GENERIC_OF or rec.get("result", 0) != 0:
                    continue
                items.setdefault((config_key(rec), chip), None)
    work = sorted(items)
    if args.sample > 0:
        kept, seen = [], collections.Counter()
        for key, chip in work:
            atom = next((o for o in key[1] if "ATOM" in o), "")
            group = (key[0], chip, atom)
            if seen[group] < args.sample:
                seen[group] += 1
                kept.append((key, chip))
        work = kept
    if not work:
        print("no RDNA configs in the input", file=sys.stderr)
        return 2

    failures = 0
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for tag, label, ok, detail in pool.map(check, [args] * len(work), work):
            rows.append((tag, label, ok, detail))
            if not ok:
                failures += 1
                print(f"FAIL {tag} {label}: {detail}", flush=True)
    if args.report:
        with open(args.report, "w") as f:
            f.write("tag,config,ok,detail\n")
            for tag, label, ok, detail in rows:
                f.write(f'{tag},"{label}",{int(ok)},"{detail}"\n')
    print(
        f"{len(rows) - failures}/{len(rows)} configs pass"
        + (
            " (exact-chip ISA identity)"
            if args.baseline_root
            else " (generic keeps matrix core)"
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
