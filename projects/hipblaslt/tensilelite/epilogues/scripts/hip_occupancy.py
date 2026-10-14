#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Ask the HIP runtime how many workgroups of a kernel fit on one CU.

Loads a code object with hipModuleLoad and calls
hipModuleOccupancyMaxActiveBlocksPerMultiprocessor, which is the runtime's own
answer to "how many blocks per CU".  Used as ground truth for the analytic
occupancy model in rocprof_analyze.py.

Usage: hip_occupancy.py --co FILE --kernel NAME --block-size N [--dyn-smem BYTES]
Prints one JSON object on stdout.  Run it with ROCR_VISIBLE_DEVICES pinned to the
profiled GPU so that device 0 is that GPU.
"""

import argparse
import ctypes
import json
import os
import sys


class HipError(RuntimeError):
    pass


def _check(hip: ctypes.CDLL, status: int, what: str) -> None:
    if status == 0:
        return
    hip.hipGetErrorString.restype = ctypes.c_char_p
    msg = hip.hipGetErrorString(status)
    raise HipError(f"{what} failed: {status} ({msg.decode() if msg else '?'})")


def _load_hip() -> ctypes.CDLL:
    rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
    return ctypes.CDLL(os.path.join(rocm, "lib", "libamdhip64.so"))


def max_active_blocks_per_cu(co_path: str, kernel_name: str, block_size: int, dyn_smem: int = 0) -> dict:
    hip = _load_hip()
    _check(hip, hip.hipInit(0), "hipInit")
    _check(hip, hip.hipSetDevice(0), "hipSetDevice")

    module = ctypes.c_void_p()
    _check(hip, hip.hipModuleLoad(ctypes.byref(module), co_path.encode()), f"hipModuleLoad({co_path})")
    func = ctypes.c_void_p()
    _check(
        hip,
        hip.hipModuleGetFunction(ctypes.byref(func), module, kernel_name.encode()),
        "hipModuleGetFunction",
    )
    blocks = ctypes.c_int(-1)
    _check(
        hip,
        hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
            ctypes.byref(blocks), func, ctypes.c_int(block_size), ctypes.c_size_t(dyn_smem)
        ),
        "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor",
    )
    return {"max_active_blocks_per_cu": blocks.value, "block_size": block_size, "dyn_smem": dyn_smem}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--co", required=True)
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--block-size", type=int, required=True)
    ap.add_argument("--dyn-smem", type=int, default=0)
    args = ap.parse_args()
    try:
        result = max_active_blocks_per_cu(args.co, args.kernel, args.block_size, args.dyn_smem)
    except (HipError, OSError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
