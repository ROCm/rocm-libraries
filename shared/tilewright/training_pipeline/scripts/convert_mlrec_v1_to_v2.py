#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Convert MLREC_v1 model files to MLREC_v2.

The payload (splits, cells, weights) is copied byte for byte; the v2 header
adds the arch constants the model was trained with (lib/hardware.py, looked up
from the arch stored in the v1 header unless --arch is given). Files that are
already MLREC_v2 are left alone.

    convert_mlrec_v1_to_v2.py --in-place weights/*.tilewright.bin
    convert_mlrec_v1_to_v2.py old.bin -o new.bin
    convert_mlrec_v1_to_v2.py --out-dir converted/ a.bin b.bin
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from lib import mlrec  # noqa: E402
from lib.hardware import arch_constants  # noqa: E402


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def convert_file(src: Path, dst: Path, arch: str | None) -> str:
    data = src.read_bytes()
    if data[:8] == mlrec.MAGIC_V2:
        mlrec.read_model(data)
        if dst != src:
            _atomic_write(dst, data)
        return f"{src}: already MLREC_v2"
    header = mlrec.read_model(data)
    use_arch = arch or header["arch"]
    out = mlrec.convert_v1_to_v2(data, arch_constants(use_arch))
    _atomic_write(dst, out)
    return (
        f"{src} -> {dst}: arch={header['arch']} constants={use_arch} "
        f"weights={header['weight_dtype_name']} cells={len(header['cells'])} "
        f"splits={len(header['splits'])} bytes {len(data)} -> {len(out)}"
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", nargs="+", type=Path, help="MLREC_v1 files")
    dest = ap.add_mutually_exclusive_group(required=True)
    dest.add_argument("-o", "--output", type=Path, help="output file (one input)")
    dest.add_argument("--out-dir", type=Path, help="write <out-dir>/<input name>")
    dest.add_argument("--in-place", action="store_true", help="replace the inputs")
    ap.add_argument(
        "--arch",
        default=None,
        help="arch whose constants to embed (default: the arch in the v1 header)",
    )
    args = ap.parse_args(argv)
    if args.output is not None and len(args.inputs) != 1:
        ap.error("-o/--output takes exactly one input")
    rc = 0
    for src in args.inputs:
        if args.output is not None:
            dst = args.output
        elif args.out_dir is not None:
            dst = args.out_dir / src.name
        else:
            dst = src
        try:
            print(convert_file(src, dst, args.arch))
        except (OSError, ValueError) as e:
            print(f"{src}: {e}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
