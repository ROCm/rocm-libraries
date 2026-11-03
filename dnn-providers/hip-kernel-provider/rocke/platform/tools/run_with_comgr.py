# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Apply a build-tree compiler default without replacing a caller's override."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--default-comgr", default="")
    parser.add_argument("--needs-llvm-flavor", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a test command is required")
    env = dict(os.environ)
    if not env.get("ROCKE_COMGR_LIB") and args.default_comgr:
        env["ROCKE_COMGR_LIB"] = args.default_comgr
    # Only explicitly marked native AUTO-lowering tests need owner evidence.
    # Host-only binaries and Python tests keep their own compiler-free/session
    # contracts; command filenames are not a resource-policy classifier.
    if args.needs_llvm_flavor and not env.get("ROCKE_LLVM_FLAVOR"):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
        from rocke.runtime.comgr import _require_compiler_info
        from rocke.core.llvm_profile import _flavor_for_llvm

        if env.get("ROCKE_COMGR_LIB"):
            os.environ["ROCKE_COMGR_LIB"] = env["ROCKE_COMGR_LIB"]
        info = _require_compiler_info()
        env["ROCKE_LLVM_FLAVOR"] = _flavor_for_llvm(info.llvm_version[0])
    return subprocess.run(command, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
