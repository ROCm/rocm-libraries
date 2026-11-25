#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Compile logged conv kernel configs offline and fail on the long-branch return hazard.

A device function the inliner leaves out of its entry point is called with its return address
in s[30:31]. Past the short-branch range (2^15 instructions, 128 KiB) its branches become long
branches (s_getpc_b64 / s_add / s_setpc_b64 through an SGPR pair), and the compiler can build
them in s[30:31] without saving it: the function's return then jumps back into its own body and
the kernel never ends. Kernels have no return address, so a long branch in a kernel is safe.

Input is HIPRTC compiles, one JSON object a line: {"name": "kernels/ConvGroup.cpp", "options":
["-std=c++17", "--offload-arch=gfx950", "-DCONV_GROUP_T=_Float16", ...]}, as
../rdna_generic/rtc_hook.c's HKP_RTC_LOG writes them (with HKP_RTC_DRY=1 under the census test
TestConvCompileCensus, nothing is compiled while logging). Each distinct (source, options)
compiles with the ROCm clang from the kernel sources under --kernels-root (descriptors/conv), and
a config fails when:
  - it does not compile (unless --allow-compile-errors: configs past a device's LDS, which the
    host gates decline, do not compile);
  - a device function other than a kernel is --max-function-bytes (128 KiB) or larger;
  - a device function other than a kernel builds a long branch in s[30:31].
Kernels of --max-function-bytes or more are listed, not failed.

Exit status 0 only when every config passes. --report writes one CSV line per config.

The records of every shipped config the census reaches, on every catalog architecture, from a
build with HIPKERNELPROVIDER_ENABLE_TESTS (no GPU needed):

    cc -shared -fPIC -O2 -o librtc_hook.so conv/tools/rdna_generic/rtc_hook.c -ldl -lpthread
    HKP_RTC_DRY=1 HKP_RTC_LOG=rtc.jsonl HIPDNN_DISABLE_CACHE=1 \\
    HKP_CONV_CENSUS_TREE=src/engines/kernel_ingestor_engine/descriptors/conv \\
    HKP_CONV_CENSUS_MAP=src/integration_tests/kernel_ingestor_engine/hip_kernel_provider_conv_coverage_map.json \\
    HKP_CONV_CENSUS_OUT=census.jsonl LD_PRELOAD=./librtc_hook.so \\
        hip_kernel_provider_tests --gtest_filter='TestConvCompileCensus.*'
    check_code_size.py --configs rtc.jsonl --kernels-root src/engines/kernel_ingestor_engine/descriptors/conv \\
        --rocm <rocm> --allow-compile-errors -j <cores>

census.jsonl lists, per architecture, every catalog config with the graphs that admitted it, so
the configs the census did not reach are known.
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

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rdna_generic")
)
from check_generic import find_source, include_flags  # noqa: E402

ARCH_PREFIX = "--offload-arch="
SHORT_BRANCH_BYTES = 1 << 17
SYMBOL_RE = re.compile(r"^([0-9a-f]+) ([0-9a-f]+) ([A-Za-z]) (\S+)$")
FUNC_RE = re.compile(r"^[0-9a-f]+ <(\S+)>:$")
LONG_BRANCH_RE = re.compile(r"\bs_getpc_b64\s+s\[30:31\]")


def arch_of(options):
    for o in options:
        if o.startswith(ARCH_PREFIX):
            return o[len(ARCH_PREFIX) :].split(":")[0]
    return None


def compile_one(args, name, opts, out):
    cmd = [
        args.clang,
        "-x",
        "hip",
        "--offload-device-only",
        "--no-gpu-bundle-output",
        "-O3",
        "-w",
        f"-I{os.path.join(args.rocm, 'include')}",
        "-include",
        "hip/hip_runtime.h",
        *include_flags(args.kernels_root),
        *opts,
        find_source(args.kernels_root, name),
        "-o",
        out,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return " | ".join(proc.stderr.strip().splitlines()[-4:])
    return None


def functions(args, path):
    """{symbol: (size, is_kernel)} of the code object's functions."""
    proc = subprocess.run(
        [args.nm, "-S", "--defined-only", path],
        capture_output=True,
        text=True,
        check=True,
    )
    found, names = {}, set()
    for line in proc.stdout.splitlines():
        m = SYMBOL_RE.match(line.strip())
        if not m:
            continue
        names.add(m.group(4))
        if m.group(3) in "Tt":
            found[m.group(4)] = int(m.group(2), 16)
    return {s: (size, f"{s}.kd" in names) for s, size in found.items()}


def long_branch_functions(args, path):
    """Functions with a long branch built in s[30:31]."""
    proc = subprocess.run(
        [args.objdump, "-d", "--no-show-raw-insn", path],
        capture_output=True,
        text=True,
        check=True,
    )
    hits, current = set(), None
    for line in proc.stdout.splitlines():
        m = FUNC_RE.match(line)
        if m:
            current = m.group(1)
        elif current is not None and LONG_BRANCH_RE.search(line):
            hits.add(current)
    return hits


def check(args, item):
    name, opts = item
    arch = arch_of(opts)
    tag = hashlib.sha1(repr(item).encode()).hexdigest()[:12]
    label = f"{name} [{arch}] " + " ".join(o for o in opts if o.startswith("-D"))
    with tempfile.TemporaryDirectory(prefix="hkp-code-gate-") as tmp:
        out = os.path.join(tmp, "k.co")
        err = compile_one(args, name, list(opts), out)
        if err:
            return (
                tag,
                arch,
                label,
                "compile_error" if args.allow_compile_errors else "fail",
                err,
                0,
                0,
            )
        funcs = functions(args, out)
        outlined = {s: n for s, (n, kernel) in funcs.items() if not kernel and n > 0}
        biggest_kernel = max((n for n, kernel in funcs.values() if kernel), default=0)
        biggest_outlined = max(outlined.values(), default=0)
        problems = [
            f"{s} is {n} bytes, not inlined"
            for s, n in outlined.items()
            if n >= args.max_function_bytes
        ]
        if outlined:
            for s in sorted(long_branch_functions(args, out) & set(outlined)):
                problems.append(f"{s} builds a long branch in s[30:31]")
        if problems:
            return (
                tag,
                arch,
                label,
                "fail",
                "; ".join(problems),
                biggest_kernel,
                biggest_outlined,
            )
        note = (
            f"kernel {biggest_kernel} bytes"
            if biggest_kernel >= args.max_function_bytes
            else ""
        )
        if outlined:
            note = (
                (note + "; " if note else "")
                + "outlined: "
                + ",".join(f"{s}={n}" for s, n in sorted(outlined.items()))
            )
        return tag, arch, label, "ok", note, biggest_kernel, biggest_outlined


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--configs", required=True, nargs="+", help="HKP_RTC_LOG JSONL file(s)"
    )
    p.add_argument("--kernels-root", required=True, help="descriptors/conv directory")
    p.add_argument("--rocm", default=os.environ.get("ROCM_PATH", "/opt/rocm"))
    p.add_argument("--clang")
    p.add_argument("--objdump")
    p.add_argument("--nm")
    p.add_argument(
        "--arch", action="append", help="only records for this arch (repeatable)"
    )
    p.add_argument(
        "--source",
        action="append",
        help="only records of this kernels/<file> (repeatable)",
    )
    p.add_argument("--max-function-bytes", type=int, default=SHORT_BRANCH_BYTES)
    p.add_argument("--allow-compile-errors", action="store_true")
    p.add_argument("-j", "--jobs", type=int, default=2)
    p.add_argument("--report", help="write one CSV line per config here")
    args = p.parse_args()
    llvm = os.path.join(args.rocm, "lib", "llvm", "bin")
    if not os.path.isdir(llvm):
        llvm = os.path.join(args.rocm, "llvm", "bin")
    args.clang = args.clang or os.path.join(llvm, "clang")
    args.objdump = args.objdump or os.path.join(llvm, "llvm-objdump")
    args.nm = args.nm or os.path.join(llvm, "llvm-nm")

    items = {}
    for path in args.configs:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                arch = arch_of(rec["options"])
                if args.arch and arch not in args.arch:
                    continue
                if args.source and os.path.basename(rec["name"]) not in args.source:
                    continue
                items.setdefault((rec["name"], tuple(rec["options"])), None)
    work = sorted(items)
    if not work:
        print("no configs in the input", file=sys.stderr)
        return 2

    rows, counts = [], collections.Counter()
    worst = collections.defaultdict(lambda: (0, ""))
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for row in pool.map(check, [args] * len(work), work, chunksize=4):
            rows.append(row)
            tag, arch, label, status, detail, kernel_bytes, outlined_bytes = row
            counts[status] += 1
            source = label.split(" ", 1)[0]
            if kernel_bytes > worst[(source, arch)][0]:
                worst[(source, arch)] = (kernel_bytes, tag)
            if status == "fail":
                print(f"FAIL {tag} {label}: {detail}", flush=True)
    if args.report:
        with open(args.report, "w") as f:
            f.write("tag,arch,status,kernel_bytes,outlined_bytes,detail,config\n")
            for tag, arch, label, status, detail, kb, ob in rows:
                f.write(f'{tag},{arch},{status},{kb},{ob},"{detail}","{label}"\n')
    print("largest kernel per source and arch:")
    for (source, arch), (size, tag) in sorted(worst.items()):
        print(f"  {source} {arch}: {size} bytes ({tag})")
    print(
        f"{counts['ok']}/{len(rows)} configs pass, {counts['fail']} fail"
        + (
            f", {counts['compile_error']} do not compile"
            if counts["compile_error"]
            else ""
        )
    )
    return 1 if counts["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
