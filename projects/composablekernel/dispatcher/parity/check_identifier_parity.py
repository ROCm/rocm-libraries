#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Identifier parity checker -- parity deliverable (b).

For every dispatcher config produced from a Tile Engine config JSON, this
compares two independent producers of the registry-lookup identifier:

  * Python oracle  -- identifier.encode_identifier() (drives codegen-side naming)
  * C++   oracle   -- KernelKey::encode_identifier() via cpp_identifier_oracle
                      (the real runtime path in kernel_key.hpp)

If they agree byte-for-byte for every config, the registry key computed offline
during codegen will match the one computed at runtime, so dispatch lookups
cannot silently miss. This needs only g++ + python3 -- NO GPU, NO hipcc, NO
cmake -- because kernel_key.hpp is pure host C++.

Usage:
    python check_identifier_parity.py configs/single_fp16_rcr.json
    python check_identifier_parity.py configs/single_fp16_rcr.json --verbose
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from identifier import encode_identifier
from te_to_dispatcher import TranslationError, translate_file

_HERE = Path(__file__).resolve().parent
_ORACLE_SRC = _HERE / "cpp_identifier_oracle.cpp"
_ORACLE_BIN = _HERE / "cpp_identifier_oracle"
# kernel_key.hpp lives at <dispatcher>/include/ck_tile/dispatcher/kernel_key.hpp
_INCLUDE_DIR = _HERE.parent / "include"
_KERNEL_KEY_HPP = _INCLUDE_DIR / "ck_tile" / "dispatcher" / "kernel_key.hpp"


def _bool_field(value: Any) -> str:
    """Serialize a Python bool to the flat format the C++ oracle parses ('1'/'0')."""
    return "1" if value else "0"


def _serialize(cfg: Dict[str, Any]) -> str:
    """Flatten a dispatcher config dict to `key=value` lines for the C++ oracle.

    Mirrors every field cpp_identifier_oracle.cpp reads via require(). Keeping the
    two field lists in sync is enforced by the oracle aborting on a missing key.
    """
    sig = cfg["signature"]
    alg = cfg["algorithm"]
    fields = {
        "dtype_a": sig["dtype_a"],
        "dtype_b": sig["dtype_b"],
        "dtype_c": sig["dtype_c"],
        "dtype_acc": sig["dtype_acc"],
        "layout_a": sig["layout_a"],
        "layout_b": sig["layout_b"],
        "layout_c": sig["layout_c"],
        "transpose_a": _bool_field(sig["transpose_a"]),
        "transpose_b": _bool_field(sig["transpose_b"]),
        "grouped": _bool_field(sig["grouped"]),
        "split_k": sig["split_k"],
        "elementwise_op": sig["elementwise_op"],
        "num_d_tensors": sig["num_d_tensors"],
        "structured_sparsity": _bool_field(sig["structured_sparsity"]),
        "tile_m": alg["tile_m"],
        "tile_n": alg["tile_n"],
        "tile_k": alg["tile_k"],
        "warp_m": alg["warp_m"],
        "warp_n": alg["warp_n"],
        "warp_k": alg["warp_k"],
        "warp_tile_m": alg["warp_tile_m"],
        "warp_tile_n": alg["warp_tile_n"],
        "warp_tile_k": alg["warp_tile_k"],
        "pipeline": alg["pipeline"],
        "scheduler": alg["scheduler"],
        "epilogue": alg["epilogue"],
        "block_size": alg["block_size"],
        "double_buffer": _bool_field(alg["double_buffer"]),
        "persistent": _bool_field(alg["persistent"]),
        "preshuffle": _bool_field(alg["preshuffle"]),
        "transpose_c": _bool_field(alg["transpose_c"]),
        "num_wave_groups": alg["num_wave_groups"],
        "pad_m": _bool_field(alg["pad_m"]),
        "pad_n": _bool_field(alg["pad_n"]),
        "pad_k": _bool_field(alg["pad_k"]),
        "gfx_arch": cfg["gfx_arch"],
    }
    return "\n".join(f"{k}={v}" for k, v in fields.items()) + "\n"


def _ensure_oracle() -> Path:
    """Compile the C++ oracle if needed; return its path. Host compiler only."""
    if _ORACLE_BIN.exists():
        bin_mtime = _ORACLE_BIN.stat().st_mtime
        # Recompile if either the oracle source or the inline header it includes
        # is newer than the binary. Checking only cpp_identifier_oracle.cpp misses
        # edits to kernel_key.hpp (where encode_identifier() is defined inline),
        # which would leave a stale binary silently returning wrong identifiers.
        src_mtime = _ORACLE_SRC.stat().st_mtime
        hdr_mtime = _KERNEL_KEY_HPP.stat().st_mtime if _KERNEL_KEY_HPP.exists() else 0.0
        if bin_mtime >= max(src_mtime, hdr_mtime):
            return _ORACLE_BIN

    cxx = shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")
    if cxx is None:
        raise RuntimeError("no host C++ compiler (g++/c++/clang++) found on PATH")

    cmd = [
        cxx,
        "-std=c++17",
        f"-I{_INCLUDE_DIR}",
        str(_ORACLE_SRC),
        "-o",
        str(_ORACLE_BIN),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"failed to compile C++ oracle:\n{proc.stderr}")
    return _ORACLE_BIN


def _cpp_identifiers(oracle: Path, configs: List[Dict[str, Any]]) -> List[str]:
    """Run all configs through one oracle process (batched with '---' separators)."""
    payload = "---\n".join(_serialize(cfg) for cfg in configs)
    proc = subprocess.run(
        [str(oracle)],
        input=payload,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"C++ oracle failed (rc={proc.returncode}):\n{proc.stderr}")
    out = proc.stdout.splitlines()
    if len(out) != len(configs):
        raise RuntimeError(
            f"C++ oracle returned {len(out)} identifiers for {len(configs)} configs"
        )
    return out


def check(config_path: str | Path, verbose: bool = False) -> int:
    configs: List[Dict[str, Any]] = translate_file(config_path)
    if not configs:
        print(f"error: no valid dispatcher configs from {config_path}", file=sys.stderr)
        return 1

    oracle = _ensure_oracle()
    cpp_ids = _cpp_identifiers(oracle, configs)

    mismatches = 0
    for i, (cfg, cpp_id) in enumerate(zip(configs, cpp_ids)):
        py_id = encode_identifier(cfg)
        ok = py_id == cpp_id
        if not ok:
            mismatches += 1
        if verbose or not ok:
            mark = "OK  " if ok else "FAIL"
            print(f"[{mark}] #{i}")
            print(f"       py : {py_id}")
            print(f"       cpp: {cpp_id}")

    total = len(configs)
    passed = total - mismatches
    print(f"\nidentifier parity: {passed}/{total} configs match "
          f"(python encode_identifier vs C++ KernelKey::encode_identifier)")
    return 0 if mismatches == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=Path, help="Tile Engine config JSON")
    ap.add_argument("--verbose", action="store_true", help="Print every config, not just failures")
    args = ap.parse_args()

    try:
        return check(args.config, verbose=args.verbose)
    except (TranslationError, RuntimeError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
