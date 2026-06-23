#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Compress/decompress TensileLite Logic YAML files using zstandard (zstd).

Usage:
    python3 compress_logic_yamls.py compress [--level LEVEL] [--device DEVICE] [--jobs N] [--dry-run] LOGIC_DIR
    python3 compress_logic_yamls.py decompress [--device DEVICE] [--jobs N] [--dry-run] LOGIC_DIR
    python3 compress_logic_yamls.py stats LOGIC_DIR

The Logic YAML directory (e.g. asm_full/) contains per-device subdirectories,
each with category subdirectories (Equality, GridBased, FreeSize, StreamK, etc.)
holding .yaml files.

Compression uses zstandard (zstd) which typically achieves 19:1 ratios on YAML text.

.gitattributes note:
    If the repo uses Git LFS for *.yaml files, you should also add a rule for
    compressed files:
        *.yaml.zst filter=lfs diff=lfs merge=lfs -text
    This tool does NOT modify .gitattributes automatically.
"""

import argparse
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

LARGE_FILE_THRESHOLD = 100 * 1024 * 1024  # 100 MB

# Try importing zstandard; fall back to CLI
_USE_PYTHON_ZSTD = False
try:
    import zstandard

    _USE_PYTHON_ZSTD = True
except ImportError:
    _zstd_bin = shutil.which("zstd")
    if not _zstd_bin:
        print(
            "ERROR: Neither the 'zstandard' Python package nor the 'zstd' CLI is available.\n"
            "Install one of:\n"
            "  pip install zstandard\n"
            "  apt install zstd  /  brew install zstd",
            file=sys.stderr,
        )
        sys.exit(1)


def _human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


def _compress_file_python(src: Path, dst: Path, level: int) -> int:
    """Compress using the zstandard Python library. Returns compressed size."""
    cctx = zstandard.ZstdCompressor(level=level)
    src_size = src.stat().st_size
    if src_size > LARGE_FILE_THRESHOLD:
        with open(src, "rb") as fin, open(dst, "wb") as fout:
            cctx.copy_stream(fin, fout)
    else:
        with open(src, "rb") as fin:
            data = fin.read()
        compressed = cctx.compress(data)
        with open(dst, "wb") as fout:
            fout.write(compressed)
    return dst.stat().st_size


def _decompress_file_python(src: Path, dst: Path) -> int:
    """Decompress using the zstandard Python library. Returns decompressed size."""
    dctx = zstandard.ZstdDecompressor()
    src_size = src.stat().st_size
    if src_size > LARGE_FILE_THRESHOLD:
        with open(src, "rb") as fin, open(dst, "wb") as fout:
            dctx.copy_stream(fin, fout)
    else:
        with open(src, "rb") as fin:
            compressed = fin.read()
        data = dctx.decompress(compressed, max_output_size=2 * 1024 * 1024 * 1024)
        with open(dst, "wb") as fout:
            fout.write(data)
    return dst.stat().st_size


def _compress_file_cli(src: Path, dst: Path, level: int) -> int:
    """Compress using the zstd CLI. Returns compressed size."""
    subprocess.run(
        ["zstd", f"-{level}", "-f", "-q", str(src), "-o", str(dst)],
        check=True,
        capture_output=True,
    )
    return dst.stat().st_size


def _decompress_file_cli(src: Path, dst: Path) -> int:
    """Decompress using the zstd CLI. Returns decompressed size."""
    subprocess.run(
        ["zstd", "-d", "-f", "-q", str(src), "-o", str(dst)],
        check=True,
        capture_output=True,
    )
    return dst.stat().st_size


def compress_one(args: tuple) -> dict:
    """Compress a single file. Designed for use with ProcessPoolExecutor."""
    src_path_str, level, dry_run = args
    src = Path(src_path_str)
    dst = src.with_suffix(src.suffix + ".zst")
    result = {
        "src": str(src),
        "dst": str(dst),
        "original_size": src.stat().st_size,
        "compressed_size": 0,
        "error": None,
    }
    if dry_run:
        return result
    try:
        if _USE_PYTHON_ZSTD:
            result["compressed_size"] = _compress_file_python(src, dst, level)
        else:
            result["compressed_size"] = _compress_file_cli(src, dst, level)
        if dst.stat().st_size == 0:
            result["error"] = "Compressed file is empty"
            dst.unlink(missing_ok=True)
        else:
            src.unlink()
    except Exception as e:
        result["error"] = str(e)
        dst.unlink(missing_ok=True)
    return result


def decompress_one(args: tuple) -> dict:
    """Decompress a single file. Designed for use with ProcessPoolExecutor."""
    src_path_str, dry_run = args
    src = Path(src_path_str)
    dst = Path(str(src)[: -len(".zst")])
    result = {
        "src": str(src),
        "dst": str(dst),
        "compressed_size": src.stat().st_size,
        "decompressed_size": 0,
        "error": None,
    }
    if dry_run:
        return result
    try:
        if _USE_PYTHON_ZSTD:
            result["decompressed_size"] = _decompress_file_python(src, dst)
        else:
            result["decompressed_size"] = _decompress_file_cli(src, dst)
        if dst.stat().st_size == 0:
            result["error"] = "Decompressed file is empty"
            dst.unlink(missing_ok=True)
        else:
            src.unlink()
    except Exception as e:
        result["error"] = str(e)
        dst.unlink(missing_ok=True)
    return result


def find_yaml_files(logic_dir: Path, device: str | None = None) -> list[Path]:
    """Find all .yaml files under logic_dir, optionally filtered by device."""
    files = []
    if device:
        device_dir = logic_dir / device
        if not device_dir.is_dir():
            for d in sorted(logic_dir.iterdir()):
                if d.is_dir() and device.lower() in d.name.lower():
                    device_dir = d
                    break
            else:
                print(f"WARNING: Device directory '{device}' not found in {logic_dir}", file=sys.stderr)
                return files
        for f in sorted(device_dir.rglob("*.yaml")):
            if f.is_file():
                files.append(f)
    else:
        for f in sorted(logic_dir.rglob("*.yaml")):
            if f.is_file():
                files.append(f)
    return files


def find_zst_files(logic_dir: Path, device: str | None = None) -> list[Path]:
    """Find all .yaml.zst files under logic_dir, optionally filtered by device."""
    files = []
    if device:
        device_dir = logic_dir / device
        if not device_dir.is_dir():
            for d in sorted(logic_dir.iterdir()):
                if d.is_dir() and device.lower() in d.name.lower():
                    device_dir = d
                    break
            else:
                print(f"WARNING: Device directory '{device}' not found in {logic_dir}", file=sys.stderr)
                return files
        for f in sorted(device_dir.rglob("*.yaml.zst")):
            if f.is_file():
                files.append(f)
    else:
        for f in sorted(logic_dir.rglob("*.yaml.zst")):
            if f.is_file():
                files.append(f)
    return files


def _detect_dir_level(logic_dir: Path) -> str:
    """Detect whether logic_dir is at the asm_full level or a device level.

    Returns 'asm_full' if logic_dir contains device subdirs (which contain
    category subdirs), or 'device' if it directly contains category subdirs
    (like Equality/, GridBased/).
    """
    known_categories = {"Equality", "GridBased", "FreeSize", "StreamK", "Origami"}
    for child in logic_dir.iterdir():
        if child.is_dir() and child.name in known_categories:
            return "device"
    return "asm_full"


def _classify_file(filepath: Path, logic_dir: Path, dir_level: str) -> tuple[str, str]:
    """Extract (device, category) from a file path relative to logic_dir.

    dir_level='asm_full': relative path is device/category/file.yaml
    dir_level='device':   relative path is category/file.yaml (device = dir name)
    """
    try:
        rel = filepath.relative_to(logic_dir)
        parts = rel.parts
        if dir_level == "device":
            device = logic_dir.name
            category = parts[0] if len(parts) > 1 else "unknown"
        else:
            device = parts[0] if len(parts) > 0 else "unknown"
            category = parts[1] if len(parts) > 1 else "unknown"
        return device, category
    except ValueError:
        return "unknown", "unknown"


def cmd_compress(args):
    logic_dir = Path(args.logic_dir).resolve()
    if not logic_dir.is_dir():
        print(f"ERROR: {logic_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    files = find_yaml_files(logic_dir, args.device)
    if not files:
        print("No .yaml files found to compress.")
        return

    print(f"Found {len(files)} .yaml file(s) to compress (level={args.level})")
    if args.dry_run:
        print("DRY RUN -- no files will be modified\n")

    backend = "zstandard (Python)" if _USE_PYTHON_ZSTD else "zstd (CLI)"
    print(f"Backend: {backend}")
    print(f"Workers: {args.jobs}\n")

    work = [(str(f), args.level, args.dry_run) for f in files]
    results = []

    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(compress_one, w): w for w in work}
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            name = Path(r["src"]).name
            if r["error"]:
                print(f"  ERROR {name}: {r['error']}")
            elif args.dry_run:
                print(f"  {name}: {_human_size(r['original_size'])}")
            else:
                ratio = r["original_size"] / r["compressed_size"] if r["compressed_size"] else 0
                print(
                    f"  {name}: {_human_size(r['original_size'])} -> "
                    f"{_human_size(r['compressed_size'])} ({ratio:.1f}x)"
                )

    total_original = sum(r["original_size"] for r in results)
    total_compressed = sum(r["compressed_size"] for r in results)
    errors = sum(1 for r in results if r["error"])

    print(f"\n{'='*60}")
    print(f"Files processed: {len(results)}")
    print(f"Errors: {errors}")
    print(f"Total original:   {_human_size(total_original)}")
    if not args.dry_run:
        ratio = total_original / total_compressed if total_compressed else 0
        print(f"Total compressed: {_human_size(total_compressed)}")
        print(f"Compression ratio: {ratio:.1f}x")
        saved = total_original - total_compressed
        print(f"Space saved: {_human_size(saved)} ({100*saved/total_original:.1f}%)" if total_original else "")


def cmd_decompress(args):
    logic_dir = Path(args.logic_dir).resolve()
    if not logic_dir.is_dir():
        print(f"ERROR: {logic_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    files = find_zst_files(logic_dir, args.device)
    if not files:
        print("No .yaml.zst files found to decompress.")
        return

    print(f"Found {len(files)} .yaml.zst file(s) to decompress")
    if args.dry_run:
        print("DRY RUN -- no files will be modified\n")

    backend = "zstandard (Python)" if _USE_PYTHON_ZSTD else "zstd (CLI)"
    print(f"Backend: {backend}")
    print(f"Workers: {args.jobs}\n")

    work = [(str(f), args.dry_run) for f in files]
    results = []

    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(decompress_one, w): w for w in work}
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            name = Path(r["src"]).name
            if r["error"]:
                print(f"  ERROR {name}: {r['error']}")
            elif args.dry_run:
                print(f"  {name}: {_human_size(r['compressed_size'])}")
            else:
                ratio = r["decompressed_size"] / r["compressed_size"] if r["compressed_size"] else 0
                print(
                    f"  {name}: {_human_size(r['compressed_size'])} -> "
                    f"{_human_size(r['decompressed_size'])} ({ratio:.1f}x expansion)"
                )

    total_compressed = sum(r["compressed_size"] for r in results)
    total_decompressed = sum(r["decompressed_size"] for r in results)
    errors = sum(1 for r in results if r["error"])

    print(f"\n{'='*60}")
    print(f"Files processed: {len(results)}")
    print(f"Errors: {errors}")
    print(f"Total compressed:   {_human_size(total_compressed)}")
    if not args.dry_run:
        print(f"Total decompressed: {_human_size(total_decompressed)}")


def cmd_stats(args):
    logic_dir = Path(args.logic_dir).resolve()
    if not logic_dir.is_dir():
        print(f"ERROR: {logic_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    yaml_files = list(logic_dir.rglob("*.yaml"))
    # Exclude files that are actually .yaml.zst matched by *.yaml glob
    yaml_files = [f for f in yaml_files if not f.name.endswith(".yaml.zst") and f.is_file()]
    zst_files = [f for f in logic_dir.rglob("*.yaml.zst") if f.is_file()]

    yaml_size = sum(f.stat().st_size for f in yaml_files)
    zst_size = sum(f.stat().st_size for f in zst_files)

    print(f"Logic directory: {logic_dir}")
    print(f"{'='*60}")
    print(f"Uncompressed .yaml files:  {len(yaml_files):>6}  ({_human_size(yaml_size)})")
    print(f"Compressed .yaml.zst files:{len(zst_files):>6}  ({_human_size(zst_size)})")
    print(f"Total files:               {len(yaml_files)+len(zst_files):>6}  ({_human_size(yaml_size + zst_size)})")

    if yaml_files and zst_files:
        print(f"\nMixed state: some files compressed, some not.")
    elif zst_files and not yaml_files:
        print(f"\nAll files are compressed.")
    elif yaml_files and not zst_files:
        print(f"\nAll files are uncompressed.")
    else:
        print(f"\nNo YAML or ZST files found.")
        return

    dir_level = _detect_dir_level(logic_dir)

    # Per-device breakdown
    device_stats = defaultdict(lambda: {"yaml_count": 0, "yaml_size": 0, "zst_count": 0, "zst_size": 0})
    for f in yaml_files:
        device, _ = _classify_file(f, logic_dir, dir_level)
        device_stats[device]["yaml_count"] += 1
        device_stats[device]["yaml_size"] += f.stat().st_size
    for f in zst_files:
        device, _ = _classify_file(f, logic_dir, dir_level)
        device_stats[device]["zst_count"] += 1
        device_stats[device]["zst_size"] += f.stat().st_size

    print(f"\n{'='*60}")
    print("Per-device breakdown:")
    print(f"  {'Device':<20} {'YAML':>6} {'Size':>10}  {'ZST':>6} {'Size':>10}")
    print(f"  {'-'*18:<20} {'-'*6:>6} {'-'*10:>10}  {'-'*6:>6} {'-'*10:>10}")
    for device in sorted(device_stats):
        s = device_stats[device]
        print(
            f"  {device:<20} {s['yaml_count']:>6} {_human_size(s['yaml_size']):>10}  "
            f"{s['zst_count']:>6} {_human_size(s['zst_size']):>10}"
        )

    # Per-category breakdown
    cat_stats = defaultdict(lambda: {"yaml_count": 0, "yaml_size": 0, "zst_count": 0, "zst_size": 0})
    for f in yaml_files:
        _, category = _classify_file(f, logic_dir, dir_level)
        cat_stats[category]["yaml_count"] += 1
        cat_stats[category]["yaml_size"] += f.stat().st_size
    for f in zst_files:
        _, category = _classify_file(f, logic_dir, dir_level)
        cat_stats[category]["zst_count"] += 1
        cat_stats[category]["zst_size"] += f.stat().st_size

    print(f"\n{'='*60}")
    print("Per-category breakdown:")
    print(f"  {'Category':<20} {'YAML':>6} {'Size':>10}  {'ZST':>6} {'Size':>10}")
    print(f"  {'-'*18:<20} {'-'*6:>6} {'-'*10:>10}  {'-'*6:>6} {'-'*10:>10}")
    for cat in sorted(cat_stats):
        s = cat_stats[cat]
        print(
            f"  {cat:<20} {s['yaml_count']:>6} {_human_size(s['yaml_size']):>10}  "
            f"{s['zst_count']:>6} {_human_size(s['zst_size']):>10}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # compress
    p_compress = sub.add_parser("compress", help="Compress .yaml files to .yaml.zst")
    p_compress.add_argument("logic_dir", help="Path to Logic YAML directory (e.g. asm_full/)")
    p_compress.add_argument("--level", type=int, default=3, help="Compression level 1-22 (default: 3)")
    p_compress.add_argument("--device", help="Only process files for this device (e.g. navi31)")
    p_compress.add_argument("--jobs", "-j", type=int, default=os.cpu_count(), help="Parallel workers (default: num CPUs)")
    p_compress.add_argument("--dry-run", action="store_true", help="Report sizes without compressing")

    # decompress
    p_decompress = sub.add_parser("decompress", help="Decompress .yaml.zst files back to .yaml")
    p_decompress.add_argument("logic_dir", help="Path to Logic YAML directory (e.g. asm_full/)")
    p_decompress.add_argument("--device", help="Only process files for this device")
    p_decompress.add_argument("--jobs", "-j", type=int, default=os.cpu_count(), help="Parallel workers (default: num CPUs)")
    p_decompress.add_argument("--dry-run", action="store_true", help="Report sizes without decompressing")

    # stats
    p_stats = sub.add_parser("stats", help="Report compression state of Logic YAML directory")
    p_stats.add_argument("logic_dir", help="Path to Logic YAML directory (e.g. asm_full/)")

    args = parser.parse_args()

    if args.command == "compress":
        if args.level < 1 or args.level > 22:
            parser.error("--level must be between 1 and 22")
        cmd_compress(args)
    elif args.command == "decompress":
        cmd_decompress(args)
    elif args.command == "stats":
        cmd_stats(args)


if __name__ == "__main__":
    main()
