#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pack per-architecture .hsaco files for FMHA BWD OGradDotO variants into a kpack archive.

Archive format:

    [0x00]  "KPAK"           4 bytes   Magic
    [0x04]  version          4 bytes   uint32 LE (currently 1)
    [0x08]  toc_offset       8 bytes   uint64 LE
    [0x10]  blob_0           variable  Raw .hsaco bytes
            blob_1           variable  ...
    [toc_offset]  TOC        variable  MessagePack table of contents
"""

import struct
import sys
from pathlib import Path

try:
    import msgpack
except ImportError:
    print(
        "Error: msgpack package required. Install with: pip install msgpack",
        file=sys.stderr,
    )
    sys.exit(1)


KPACK_MAGIC = b"KPAK"
KPACK_VERSION = 1
HEADER_SIZE = 16  # 4 (magic) + 4 (version) + 8 (toc_offset)

# Variant metadata mirrors the constexpr ALL_FMHA_BWD_VARIANTS table in
# rocm_fmha_bwd_registry.hpp.
VARIANTS = [
    {
        "name": "fmha_bwd_ograd_dot_o_fp16_d128_batch",
        "dtype": "fp16",
        "hdim_v": 128,
        "mode": "batch",
        "pad_seqlen_q": True,
        "pad_hdim_v": True,
        "block_per_cu": 2,
        "block_size": 64,
    },
    {
        "name": "fmha_bwd_ograd_dot_o_bf16_d128_batch",
        "dtype": "bf16",
        "hdim_v": 128,
        "mode": "batch",
        "pad_seqlen_q": True,
        "pad_hdim_v": True,
        "block_per_cu": 2,
        "block_size": 64,
    },
    {
        "name": "fmha_bwd_ograd_dot_o_fp16_d64_batch",
        "dtype": "fp16",
        "hdim_v": 64,
        "mode": "batch",
        "pad_seqlen_q": True,
        "pad_hdim_v": True,
        "block_per_cu": 2,
        "block_size": 64,
    },
    {
        "name": "fmha_bwd_ograd_dot_o_fp16_d128_group",
        "dtype": "fp16",
        "hdim_v": 128,
        "mode": "group",
        "pad_seqlen_q": True,
        "pad_hdim_v": True,
        "block_per_cu": 2,
        "block_size": 64,
    },
    {
        "name": "fmha_bwd_ograd_dot_o_fp16_d128_batch_npad",
        "dtype": "fp16",
        "hdim_v": 128,
        "mode": "batch",
        "pad_seqlen_q": False,
        "pad_hdim_v": False,
        "block_per_cu": 2,
        "block_size": 64,
    },
]
ARCHITECTURES = ["gfx90a", "gfx942", "gfx950"]


def main() -> None:
    # Accept an optional build directory argument (used by CMake).
    # Defaults to ./build relative to this script.
    build_dir = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "build"
    )
    output_path = build_dir / "kernels.kpack"

    # Read .hsaco blobs for each variant x architecture
    blobs: list[bytes] = []
    found_arches: set[str] = set()
    # Map: variant_name -> {arch -> ordinal} for building the TOC
    variant_map: dict[str, dict[str, int]] = {}

    for variant in VARIANTS:
        name = variant["name"]
        variant_entries: dict[str, int] = {}
        for arch in ARCHITECTURES:
            hsaco_path = build_dir / f"{name}_{arch}.hsaco"
            if not hsaco_path.exists():
                print(f"  Skipping {name}/{arch}: {hsaco_path.name} not found")
                continue
            blob = hsaco_path.read_bytes()
            ordinal = len(blobs)
            blobs.append(blob)
            found_arches.add(arch)
            variant_entries[arch] = ordinal
            print(f"  Read {hsaco_path.name} ({len(blob)} bytes)")

        if variant_entries:
            variant_map[name] = variant_entries

    if not blobs:
        print("Error: no .hsaco files found in build/", file=sys.stderr)
        sys.exit(1)

    # Stable arch ordering for the TOC
    sorted_arches = sorted(found_arches)

    # Write the archive
    with output_path.open("wb") as out:
        # Header (toc_offset patched after writing blobs)
        out.write(KPACK_MAGIC)
        out.write(struct.pack("<I", KPACK_VERSION))
        out.write(struct.pack("<Q", 0))  # toc_offset placeholder

        # Concatenate blobs, recording offsets
        blob_infos: list[dict] = []
        for blob in blobs:
            offset = out.tell()
            out.write(blob)
            blob_infos.append({"offset": offset, "size": len(blob)})

        # Build and write the MessagePack TOC
        toc_offset = out.tell()

        # Variant metadata: tuning surface parameters for each variant
        variant_metadata = {}
        for v in VARIANTS:
            if v["name"] in variant_map:
                variant_metadata[v["name"]] = {
                    "dtype": v["dtype"],
                    "hdim_v": v["hdim_v"],
                    "mode": v["mode"],
                    "pad_seqlen_q": v["pad_seqlen_q"],
                    "pad_hdim_v": v["pad_hdim_v"],
                    "block_per_cu": v["block_per_cu"],
                    "block_size": v["block_size"],
                }

        toc = {
            "compression_scheme": "none",
            "gfx_arches": sorted_arches,
            "blobs": blob_infos,
            "variant_metadata": variant_metadata,
            "toc": {
                variant_name: {
                    arch: {
                        "ordinal": ordinal,
                        "original_size": len(blobs[ordinal]),
                        "type": "hsaco",
                    }
                    for arch, ordinal in arch_entries.items()
                }
                for variant_name, arch_entries in variant_map.items()
            },
        }
        out.write(msgpack.packb(toc, use_bin_type=True))

        # Patch header with actual toc_offset
        out.seek(8)
        out.write(struct.pack("<Q", toc_offset))

    total_size = sum(len(b) for b in blobs)
    print(f"\nCreated {output_path}")
    print(f"  Architectures: {', '.join(sorted_arches)}")
    print(f"  Variants: {', '.join(variant_map.keys())}")
    print(f"  Total kernel data: {total_size} bytes")


if __name__ == "__main__":
    main()
