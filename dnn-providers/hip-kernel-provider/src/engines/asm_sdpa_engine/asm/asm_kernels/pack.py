# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Pack ASM SDPA .co kernel binaries into a .kpack archive.

Usage:
    python pack.py --kernel-dir <path> --arch <arch> --out-dir <path>

Requires rocm_kpack on PYTHONPATH (consumed from source, not pip install).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pack ASM SDPA .co kernels into a .kpack archive"
    )
    parser.add_argument(
        "--kernel-dir",
        type=Path,
        required=True,
        help="Root directory containing <arch>/ subdirectories with .co files",
    )
    parser.add_argument(
        "--arch",
        type=str,
        required=True,
        help="GPU architecture to pack (e.g. gfx942)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for .kpack file",
    )
    args = parser.parse_args()

    kernel_dir: Path = args.kernel_dir
    arch: str = args.arch
    out_dir: Path = args.out_dir

    # Validate inputs
    arch_dir = kernel_dir / arch
    if not arch_dir.is_dir():
        print(f"Error: architecture directory not found: {arch_dir}", file=sys.stderr)
        return 1

    # Glob all .co files under <kernel-dir>/<arch>/
    co_files = sorted(arch_dir.rglob("*.co"))
    if not co_files:
        print(f"Error: no .co files found under {arch_dir}", file=sys.stderr)
        return 1

    # Import rocm_kpack (expected on PYTHONPATH)
    try:
        from rocm_kpack.compression import ZstdCompressor
        from rocm_kpack.kpack import PackedKernelArchive
    except ImportError as exc:
        print(
            f"Error: failed to import rocm_kpack: {exc}\n"
            "Ensure PYTHONPATH includes the rocm_kpack Python source directory.",
            file=sys.stderr,
        )
        return 1

    # Build TOC key -> (co_path, original_sha256) mapping
    entries: list[tuple[str, Path, bytes]] = []
    for co_path in co_files:
        # TOC key = relative path from <arch>/ (strip the arch prefix)
        # e.g. fmha_v3_fwd/MI300/fwd_hd128_bf16_rtne.co
        toc_key = co_path.relative_to(arch_dir).as_posix()
        hsaco_data = co_path.read_bytes()
        entries.append((toc_key, co_path, hsaco_data))

    print(f"Packing {len(entries)} .co files for {arch}...")

    # Create archive
    archive = PackedKernelArchive(
        group_name="hip_kernel_provider_sdpa",
        gfx_arch_family=arch,
        gfx_arches=[arch],
        compressor=ZstdCompressor(compression_level=3),
    )

    # Prepare and add each kernel
    original_hashes: dict[str, str] = {}
    for toc_key, _, hsaco_data in entries:
        original_hashes[toc_key] = _sha256(hsaco_data)
        prepared = archive.prepare_kernel(
            relative_path=toc_key,
            gfx_arch=arch,
            hsaco_data=hsaco_data,
        )
        archive.add_kernel(prepared)

    # Finalize and write
    archive.finalize_archive()

    out_dir.mkdir(parents=True, exist_ok=True)
    kpack_path = out_dir / f"hip_kernel_provider_sdpa_{arch}.kpack"
    archive.write(kpack_path)

    print(f"Written: {kpack_path} ({kpack_path.stat().st_size} bytes)")

    # Round-trip verification
    written = PackedKernelArchive.read(kpack_path)
    failures = 0
    for toc_key, _, _ in entries:
        packed = written.get_kernel(toc_key, arch)
        if packed is None:
            print(
                f"  VERIFY FAIL: key '{toc_key}' not found in archive",
                file=sys.stderr,
            )
            failures += 1
            continue
        packed_hash = _sha256(packed)
        if packed_hash != original_hashes[toc_key]:
            print(
                f"  VERIFY FAIL: key '{toc_key}' SHA256 mismatch: "
                f"expected {original_hashes[toc_key]}, got {packed_hash}",
                file=sys.stderr,
            )
            failures += 1

    if failures:
        print(
            f"Error: {failures}/{len(entries)} round-trip verification failures",
            file=sys.stderr,
        )
        return 1

    print(f"Round-trip verification passed for all {len(entries)} kernels.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
