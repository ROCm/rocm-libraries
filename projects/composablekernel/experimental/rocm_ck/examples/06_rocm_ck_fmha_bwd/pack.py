#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pack per-architecture .hsaco files for all FMHA BWD kernel families into a kpack archive.

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
    # --- OGradDotO variants ---
    {
        "name": "fmha_bwd_ograd_dot_o_fp16_d128_batch",
        "family": "ograd_dot_o",
        "dtype": "fp16",
        "hdim_v": 128,
        "mode": "batch",
        "block_size": 64,
    },
    {
        "name": "fmha_bwd_ograd_dot_o_bf16_d128_batch",
        "family": "ograd_dot_o",
        "dtype": "bf16",
        "hdim_v": 128,
        "mode": "batch",
        "block_size": 64,
    },
    {
        "name": "fmha_bwd_ograd_dot_o_fp16_d64_batch",
        "family": "ograd_dot_o",
        "dtype": "fp16",
        "hdim_v": 64,
        "mode": "batch",
        "block_size": 64,
    },
    {
        "name": "fmha_bwd_ograd_dot_o_fp16_d128_group",
        "family": "ograd_dot_o",
        "dtype": "fp16",
        "hdim_v": 128,
        "mode": "group",
        "block_size": 64,
    },
    {
        "name": "fmha_bwd_ograd_dot_o_fp16_d128_batch_npad",
        "family": "ograd_dot_o",
        "dtype": "fp16",
        "hdim_v": 128,
        "mode": "batch",
        "block_size": 64,
    },
    # --- DqDkDv variants ---
    {
        "name": "fmha_bwd_dqdkdv_fp16_d128_batch",
        "family": "dqdkdv",
        "dtype": "fp16",
        "hdim_q": 128,
        "hdim_v": 128,
        "mode": "batch",
        "block_size": 256,
    },
    {
        "name": "fmha_bwd_dqdkdv_bf16_d128_batch",
        "family": "dqdkdv",
        "dtype": "bf16",
        "hdim_q": 128,
        "hdim_v": 128,
        "mode": "batch",
        "block_size": 256,
    },
    {
        "name": "fmha_bwd_dqdkdv_fp16_d128_batch_cmask",
        "family": "dqdkdv",
        "dtype": "fp16",
        "hdim_q": 128,
        "hdim_v": 128,
        "mode": "batch",
        "has_mask": True,
        "block_size": 256,
    },
    {
        "name": "fmha_bwd_dqdkdv_fp16_d128_batch_det",
        "family": "dqdkdv",
        "dtype": "fp16",
        "hdim_q": 128,
        "hdim_v": 128,
        "mode": "batch",
        "is_deterministic": True,
        "block_size": 256,
    },
    {
        "name": "fmha_bwd_dqdkdv_fp16_d128_group",
        "family": "dqdkdv",
        "dtype": "fp16",
        "hdim_q": 128,
        "hdim_v": 128,
        "mode": "group",
        "block_size": 256,
    },
    {
        "name": "fmha_bwd_dqdkdv_fp16_d128_batch_ebias",
        "family": "dqdkdv",
        "dtype": "fp16",
        "hdim_q": 128,
        "hdim_v": 128,
        "mode": "batch",
        "bias_type": "elementwise",
        "block_size": 256,
    },
    {
        "name": "fmha_bwd_dqdkdv_fp16_d128_batch_alibi",
        "family": "dqdkdv",
        "dtype": "fp16",
        "hdim_q": 128,
        "hdim_v": 128,
        "mode": "batch",
        "bias_type": "alibi",
        "block_size": 256,
    },
    {
        "name": "fmha_bwd_dqdkdv_fp16_d128_batch_ebias_dbias",
        "family": "dqdkdv",
        "dtype": "fp16",
        "hdim_q": 128,
        "hdim_v": 128,
        "mode": "batch",
        "bias_type": "elementwise",
        "has_bias_grad": True,
        "block_size": 256,
    },
    {
        "name": "fmha_bwd_dqdkdv_fp16_d128_batch_dropout",
        "family": "dqdkdv",
        "dtype": "fp16",
        "hdim_q": 128,
        "hdim_v": 128,
        "mode": "batch",
        "has_dropout": True,
        "block_size": 256,
    },
    {
        "name": "fmha_bwd_dqdkdv_fp16_d128_batch_cmask_det",
        "family": "dqdkdv",
        "dtype": "fp16",
        "hdim_q": 128,
        "hdim_v": 128,
        "mode": "batch",
        "has_mask": True,
        "is_deterministic": True,
        "block_size": 256,
    },
    # --- ConvertDQ variants ---
    {
        "name": "fmha_bwd_convert_dq_fp16_d128_batch_det",
        "family": "convert_dq",
        "dtype": "fp16",
        "hdim_q": 128,
        "mode": "batch",
        "block_size": 256,
    },
    {
        "name": "fmha_bwd_convert_dq_fp16_d128_group_det",
        "family": "convert_dq",
        "dtype": "fp16",
        "hdim_q": 128,
        "mode": "group",
        "block_size": 256,
    },
    {
        "name": "fmha_bwd_convert_dq_bf16_d128_batch_det",
        "family": "convert_dq",
        "dtype": "bf16",
        "hdim_q": 128,
        "mode": "batch",
        "block_size": 256,
    },
    {
        "name": "fmha_bwd_convert_dq_bf16_d128_group_det",
        "family": "convert_dq",
        "dtype": "bf16",
        "hdim_q": 128,
        "mode": "group",
        "block_size": 256,
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

        # Variant metadata: all variant fields except 'name'
        variant_metadata = {}
        for v in VARIANTS:
            if v["name"] in variant_map:
                meta = {k: val for k, val in v.items() if k != "name"}
                variant_metadata[v["name"]] = meta

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
