################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
################################################################################
"""Compare ``STINKY_TOTAL_INST_BYTES`` in an emitted ``.s`` to the ``.text`` section size of the ``.o``.

Same contract as the legacy shell helper:

- Exit **0**: marker absent (skip) or sizes match.
- Exit **1**: mismatch.
- Exit **2**: usage / missing file / could not read ``.text``.

Uses **llvm-readelf** from the **ROCm** install (``ROCM_PATH/bin``) when set, else **LLVM_BIN**,
then ``PATH``. On Windows the probe tries ``llvm-readelf.exe`` first under those directories.

CLI (no rocisa; safe ``python -m`` entry):

    python -m Tensile.verify_stinky_comment_vs_elf_text path/to/kernel.s path/to/kernel.o
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

_STINKY_RE = re.compile(r"STINKY_TOTAL_INST_BYTES:\s*([0-9]+)")

# ROCm installs LLVM's llvm-readelf under ``ROCM_PATH/bin`` (same toolchain as the SDK), not a
# separate llvm-dev package. Try ``.exe`` first so Windows finds the real file name.
_READELF_CANDIDATES = (
    "llvm-readelf.exe",
    "llvm-readelf",
    "readelf.exe",
    "readelf",
)


def _find_readelf() -> Optional[List[str]]:
    """Return argv prefix ``[exe]`` for readelf/llvm-readelf, or ``None`` if not found.

    Order: ``ROCM_PATH/bin`` (ROCm SDK LLVM), then ``LLVM_BIN``, then ``PATH`` via ``shutil.which``.
    Tries ``llvm-readelf`` before ``readelf``; ``.exe`` suffix first for directory probes (Windows).
    """
    def pick_from_dir(d: Path) -> Optional[List[str]]:
        for cand in _READELF_CANDIDATES:
            exe = d / cand
            if exe.is_file():
                return [str(exe)]
        return None

    rocm = (os.environ.get("ROCM_PATH") or "").strip()
    if rocm:
        hit = pick_from_dir(Path(rocm) / "bin")
        if hit:
            return hit

    llvm_bin = (os.environ.get("LLVM_BIN") or "").strip()
    if llvm_bin:
        hit = pick_from_dir(Path(llvm_bin))
        if hit:
            return hit

    for name in _READELF_CANDIDATES:
        p = shutil.which(name)
        if p:
            return [p]

    return None


def _readelf_section_headers(o_path: Path) -> Tuple[int, str, str]:
    """Run readelf/llvm-readelf ``-W -S`` on ``o_path``. Returns (returncode, stdout, stderr)."""
    exe = _find_readelf()
    if not exe:
        hint = (
            "no readelf/llvm-readelf found (set ROCM_PATH, LLVM_BIN, or put llvm-readelf on PATH). "
            "Windows: ensure ROCm or LLVM bin contains llvm-readelf.exe."
        )
        return 127, "", hint

    def run(args: List[str]) -> Tuple[int, str, str]:
        r = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        return r.returncode, r.stdout or "", r.stderr or ""

    rc, out, err = run(exe + ["-W", "-S", str(o_path)])
    if rc != 0:
        rc2, out2, err2 = run(exe + ["-S", str(o_path)])
        if rc2 == 0:
            return rc2, out2, err2
    return rc, out, err


def _parse_dot_text_size(readelf_stdout: str) -> Optional[int]:
    """Parse ``.text`` section size (bytes) from ``readelf -W -S`` style output."""
    for line in readelf_stdout.splitlines():
        parts = line.split()
        if ".text" not in parts:
            continue
        try:
            i = parts.index(".text")
        except ValueError:
            continue
        # Standard: ... .text PROGBITS address offset size ...
        if i + 4 >= len(parts):
            continue
        # GNU readelf / llvm-readelf: Size is **hexadecimal** (often without ``0x``), same as the
        # shell script: ``text=$((0x${hex#0x}))``. Parsing as decimal breaks when e.g. ``9660`` means
        # 0x9660 (38496), not 9660.
        sz_tok = parts[i + 4]
        if sz_tok.startswith(("0x", "0X")):
            sz_tok = sz_tok[2:]
        if not sz_tok:
            continue
        try:
            return int(sz_tok, 16)
        except ValueError:
            continue
    return None


def extract_stinky_total_inst_bytes(s_text: str) -> Optional[int]:
    """Return the first ``STINKY_TOTAL_INST_BYTES`` integer from asm text, or ``None``."""
    m = _STINKY_RE.search(s_text)
    if not m:
        return None
    return int(m.group(1), 10)


def verify_stinky_paths(s_path: Path, o_path: Path) -> Tuple[int, str, str]:
    """Run verification. Returns ``(exit_code, stdout, stderr)`` for logging.

    Exit codes: 0 ok/skip, 1 mismatch, 2 error.
    """
    out_buf: List[str] = []
    err_buf: List[str] = []

    if not s_path.is_file() or not o_path.is_file():
        err_buf.append("verify_stinky_comment_vs_elf_text: missing file")
        return 2, "\n".join(out_buf), "\n".join(err_buf)

    try:
        s_text = s_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        err_buf.append(f"verify_stinky_comment_vs_elf_text: read {s_path}: {e}")
        return 2, "\n".join(out_buf), "\n".join(err_buf)

    cost = extract_stinky_total_inst_bytes(s_text)
    if cost is None:
        out_buf.append(
            f"verify_stinky_comment_vs_elf_text: no STINKY_TOTAL_INST_BYTES in {s_path} (Stinky path off or old emitter); skip"
        )
        return 0, "\n".join(out_buf), "\n".join(err_buf)

    rc, rel_out, rel_err = _readelf_section_headers(o_path)
    if rc != 0:
        err_buf.append(rel_err.strip() or f"readelf failed (code {rc}) for {o_path}")
        if rel_out.strip():
            err_buf.append(rel_out.strip())
        return 2, "\n".join(out_buf), "\n".join(err_buf)

    text_sz = _parse_dot_text_size(rel_out)
    if text_sz is None:
        err_buf.append(f"verify_stinky_comment_vs_elf_text: could not find .text in {o_path}")
        return 2, "\n".join(out_buf), "\n".join(err_buf)

    if cost == text_sz:
        out_buf.append(f"OK STINKY_TOTAL_INST_BYTES={cost} == ELF .text={text_sz} ({o_path})")
        return 0, "\n".join(out_buf), "\n".join(err_buf)

    err_buf.append(
        f"MISMATCH STINKY_TOTAL_INST_BYTES={cost} (from {s_path}) vs ELF .text={text_sz} ({o_path}) diff={text_sz - cost}"
    )
    return 1, "\n".join(out_buf), "\n".join(err_buf)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI: ``python -m Tensile.verify_stinky_comment_vs_elf_text <.s> <.o>``."""
    ap = argparse.ArgumentParser(
        description="Compare STINKY_TOTAL_INST_BYTES in .s to ELF .text size in .o",
    )
    ap.add_argument("s_path", type=Path, help="Path to kernel .s")
    ap.add_argument("o_path", type=Path, help="Path to kernel .o")
    args = ap.parse_args(argv)

    code, out, err = verify_stinky_paths(args.s_path, args.o_path)
    if out:
        sys.stdout.write(out)
        if not out.endswith("\n"):
            sys.stdout.write("\n")
    if err:
        sys.stderr.write(err)
        if not err.endswith("\n"):
            sys.stderr.write("\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
