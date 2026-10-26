#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# ABI presubmit guard for libhiptensor.so.
#
# Fails if the shared library exports any defined symbol that is not on the allowlist. This
# codifies the hidden-visibility baseline (library/src/CMakeLists.txt) + version script
# (library/src/hiptensor.map): only the public C API and the two shared-state singletons
# (hiptensor::HiptensorOptions, hiptensor::Logger, reached via hiptensor::LazySingleton) may
# leave the .so. Everything else — internal C++ classes, Composable Kernel template
# instantiations — must stay local.
#
# Keep ALLOWLIST in lockstep with the `global:` section of library/src/hiptensor.map.
#
# Usage:
#   python3 scripts/check_exported_symbols.py <path/to/libhiptensor.so> [--nm nm]
#
# Exit status: 0 = clean, 1 = disallowed exports found, 2 = usage/tooling error.

import argparse
import os
import re
import shutil
import subprocess
import sys

# Each entry matches a mangled symbol name as emitted by `nm -D`.
ALLOWLIST = [
    # extern "C" public API. Coarse prefix: admits any `hiptensor`-prefixed export, including
    # the test-only hiptensor*Reference helpers. Internal C++ (`_ZN9hiptensor...`) is still caught.
    re.compile(r"^hiptensor"),
    # hiptensor::HiptensorOptions (shared singleton) + its LazySingleton plumbing. The
    # LazySingleton instance()/static/guard forms (_ZN / _ZZN / _ZGVZN) only emit out-of-line
    # in low-inlining builds (e.g. Debug); matched here to stay in lockstep with hiptensor.map.
    re.compile(r"^_ZN9hiptensor16HiptensorOptions"),
    re.compile(r"^_Z(N|ZN|GVZN)9hiptensor13LazySingletonINS_16HiptensorOptionsEE"),
    # hiptensor::Logger (shared singleton) + its LazySingleton plumbing
    re.compile(r"^_ZN9hiptensor6Logger"),
    re.compile(r"^_Z(N|ZN|GVZN)9hiptensor13LazySingletonINS_6LoggerEE"),
    # Standard ELF/CRT entries. The version script already demotes these to local, so normally
    # none are exported; kept as an explicit safety net for a .so built without the script.
    re.compile(r"^_init$"),
    re.compile(r"^_fini$"),
    re.compile(r"^_edata$"),
    re.compile(r"^_end$"),
    re.compile(r"^__bss_start$"),
]

# nm type codes for an externally exported defined symbol (global/weak; locals excluded).
# T = text (code), D = data, B = bss, R = read-only data, W = weak, V = weak object.
EXPORTED_TYPES = set("TDBRWV")


def is_allowed(name: str) -> bool:
    return any(pat.search(name) for pat in ALLOWLIST)


def main() -> int:
    parser = argparse.ArgumentParser(description="ABI export allowlist guard for libhiptensor.so")
    parser.add_argument("lib", help="Path to libhiptensor.so (or versioned .so)")
    parser.add_argument(
        "--nm",
        action="append",
        default=[],
        help="nm binary to use. May be given multiple times; the first one found on PATH "
        "(or as an absolute path) wins. Defaults to trying llvm-nm then nm.",
    )
    args = parser.parse_args()

    # Resolve nm at runtime so no build-time absolute path is baked into the test command.
    # An absolute path is used as-is if it exists; a bare name is looked up on PATH.
    candidates = args.nm or ["llvm-nm", "nm"]
    nm = None
    for cand in candidates:
        if os.path.isabs(cand):
            if os.path.exists(cand):
                nm = cand
                break
        else:
            found = shutil.which(cand)
            if found:
                nm = found
                break
    if nm is None:
        print(
            f"error: no nm tool found (tried: {', '.join(candidates)})",
            file=sys.stderr,
        )
        return 2

    try:
        out = subprocess.run(
            [nm, "-D", "--defined-only", args.lib],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        print(f"error: nm failed on {args.lib}: {exc.stderr.strip()}", file=sys.stderr)
        return 2

    total = 0
    offenders = []
    for line in out.splitlines():
        parts = line.split()
        # Expected form: "<addr> <type> <name>"; undefined/no-value lines have fewer fields.
        if len(parts) < 3:
            continue
        sym_type, name = parts[-2], parts[-1]
        if sym_type not in EXPORTED_TYPES:
            continue
        total += 1
        if not is_allowed(name):
            offenders.append((sym_type, name))

    print(f"checked {total} exported defined symbols in {args.lib}")
    if offenders:
        print(f"FAIL: {len(offenders)} symbol(s) exported outside the allowlist:", file=sys.stderr)
        for sym_type, name in offenders[:50]:
            print(f"  {sym_type} {name}", file=sys.stderr)
        if len(offenders) > 50:
            print(f"  ... and {len(offenders) - 50} more", file=sys.stderr)
        print(
            "\nThese symbols must be hidden. Tag public API with HIPTENSOR_EXPORT and keep "
            "everything else at hidden visibility; update library/src/hiptensor.map and this "
            "script's ALLOWLIST together if the public surface legitimately changed.",
            file=sys.stderr,
        )
        return 1

    print("PASS: all exported symbols are on the allowlist")
    return 0


if __name__ == "__main__":
    sys.exit(main())
