# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Usage: python -I -B run_emit_wrap.py <rocke_root> <emit_py> <idx>
import sys, os, runpy, pathlib

root = pathlib.Path(sys.argv[1])
emit = pathlib.Path(sys.argv[2])
idx = sys.argv[3]
os.environ.pop("ROCKE_DEBUG_LOC", None)
for p in (
    str(emit.parent),
    str(root / "platform" / "tests" / "instances" / "parity"),
    str(root / "library"),
    str(root / "platform" / "python"),
):
    sys.path.insert(0, p)
sys.argv = [str(emit), idx]
runpy.run_path(str(emit), run_name="__main__")
