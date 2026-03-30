#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pack per-architecture .hsaco files for the GEMM kernel into a kpack archive.

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

VARIANTS = [
    {
        "name": "gemm_fp32",
        "a_dtype": "fp32",
        "b_dtype": "fp32",
        "c_dtype": "fp32",
        "acc_dtype": "fp32",
        "block_m": 128,
        "block_n": 128,
        "block_k": 32,
        "waves_m": 2,
        "waves_n": 2,
        "waves_k": 1,
        "mfma_m": 16,
        "mfma_n": 16,
        "mfma_k": 16,
        "workgroup_size": 256,
    },
    {
        "name": "gemm_fp16",
        "a_dtype": "fp16",
        "b_dtype": "fp16",
        "c_dtype": "fp16",
        "acc_dtype": "fp32",
        "block_m": 128,
        "block_n": 128,
        "block_k": 32,
        "waves_m": 2,
        "waves_n": 2,
        "waves_k": 1,
        "mfma_m": 16,
        "mfma_n": 16,
        "mfma_k": 16,
        "workgroup_size": 256,
    },
    {
        "name": "gemm_bf16",
        "a_dtype": "bf16",
        "b_dtype": "bf16",
        "c_dtype": "bf16",
        "acc_dtype": "fp32",
        "block_m": 128,
        "block_n": 128,
        "block_k": 32,
        "waves_m": 2,
        "waves_n": 2,
        "waves_k": 1,
        "mfma_m": 16,
        "mfma_n": 16,
        "mfma_k": 16,
        "workgroup_size": 256,
    },
    {
        "name": "gemm_fp16_w32",
        "a_dtype": "fp16",
        "b_dtype": "fp16",
        "c_dtype": "fp16",
        "acc_dtype": "fp32",
        "block_m": 128,
        "block_n": 128,
        "block_k": 32,
        "waves_m": 2,
        "waves_n": 2,
        "waves_k": 1,
        "mfma_m": 32,
        "mfma_n": 32,
        "mfma_k": 16,
        "workgroup_size": 256,
    },
    {
        "name": "gemm_fp16_add",
        "a_dtype": "fp16",
        "b_dtype": "fp16",
        "c_dtype": "fp16",
        "acc_dtype": "fp32",
        "combine": "add",
        "d0_dtype": "fp16",
        "block_m": 128,
        "block_n": 128,
        "block_k": 32,
        "waves_m": 2,
        "waves_n": 2,
        "waves_k": 1,
        "mfma_m": 16,
        "mfma_n": 16,
        "mfma_k": 16,
        "workgroup_size": 256,
    },
    {
        "name": "gemm_fp16_add_relu",
        "a_dtype": "fp16",
        "b_dtype": "fp16",
        "c_dtype": "fp16",
        "acc_dtype": "fp32",
        "combine": "add",
        "activation": "relu",
        "d0_dtype": "fp16",
        "block_m": 128,
        "block_n": 128,
        "block_k": 32,
        "waves_m": 2,
        "waves_n": 2,
        "waves_k": 1,
        "mfma_m": 16,
        "mfma_n": 16,
        "mfma_k": 16,
        "workgroup_size": 256,
    },
    # Layout variants: A×B layout combinations beyond the R×C default
    {
        "name": "gemm_fp16_rr",
        "a_dtype": "fp16",
        "b_dtype": "fp16",
        "c_dtype": "fp16",
        "acc_dtype": "fp32",
        "a_layout": "row",
        "b_layout": "row",
        "c_layout": "row",
        "block_m": 128,
        "block_n": 128,
        "block_k": 32,
        "waves_m": 2,
        "waves_n": 2,
        "waves_k": 1,
        "mfma_m": 16,
        "mfma_n": 16,
        "mfma_k": 16,
        "workgroup_size": 256,
    },
    {
        "name": "gemm_fp16_cr",
        "a_dtype": "fp16",
        "b_dtype": "fp16",
        "c_dtype": "fp16",
        "acc_dtype": "fp32",
        "a_layout": "col",
        "b_layout": "row",
        "c_layout": "row",
        "block_m": 128,
        "block_n": 128,
        "block_k": 32,
        "waves_m": 2,
        "waves_n": 2,
        "waves_k": 1,
        "mfma_m": 16,
        "mfma_n": 16,
        "mfma_k": 16,
        "workgroup_size": 256,
    },
    {
        "name": "gemm_fp16_cc",
        "a_dtype": "fp16",
        "b_dtype": "fp16",
        "c_dtype": "fp16",
        "acc_dtype": "fp32",
        "a_layout": "col",
        "b_layout": "col",
        "c_layout": "row",
        "block_m": 128,
        "block_n": 128,
        "block_k": 32,
        "waves_m": 2,
        "waves_n": 2,
        "waves_k": 1,
        "mfma_m": 16,
        "mfma_n": 16,
        "mfma_k": 16,
        "workgroup_size": 256,
    },
    # Split-K: partition K dimension across blockIdx.z
    {
        "name": "gemm_fp16_splitk",
        "a_dtype": "fp16",
        "b_dtype": "fp16",
        "c_dtype": "fp16",
        "acc_dtype": "fp32",
        "k_batch": 4,
        "block_m": 128,
        "block_n": 128,
        "block_k": 32,
        "waves_m": 2,
        "waves_n": 2,
        "waves_k": 1,
        "mfma_m": 16,
        "mfma_n": 16,
        "mfma_k": 16,
        "workgroup_size": 256,
    },
    # Pipeline V3: compute-optimized pipeline
    {
        "name": "gemm_fp16_v3",
        "a_dtype": "fp16",
        "b_dtype": "fp16",
        "c_dtype": "fp16",
        "acc_dtype": "fp32",
        "pipeline": "V3",
        "block_m": 128,
        "block_n": 128,
        "block_k": 32,
        "waves_m": 2,
        "waves_n": 2,
        "waves_k": 1,
        "mfma_m": 16,
        "mfma_n": 16,
        "mfma_k": 16,
        "workgroup_size": 256,
    },
    # Multi-D: two D tensors (Add+Add: result = A*B + D0 + D1)
    {
        "name": "gemm_fp16_add_add",
        "a_dtype": "fp16",
        "b_dtype": "fp16",
        "c_dtype": "fp16",
        "acc_dtype": "fp32",
        "combine": "add",
        "d0_dtype": "fp16",
        "d1_dtype": "fp16",
        "block_m": 128,
        "block_n": 128,
        "block_k": 32,
        "waves_m": 2,
        "waves_n": 2,
        "waves_k": 1,
        "mfma_m": 16,
        "mfma_n": 16,
        "mfma_k": 16,
        "workgroup_size": 256,
    },
    # Batched GEMM: batch dimension via blockIdx.y
    {
        "name": "gemm_fp16_batched",
        "a_dtype": "fp16",
        "b_dtype": "fp16",
        "c_dtype": "fp16",
        "acc_dtype": "fp32",
        "block_m": 128,
        "block_n": 128,
        "block_k": 32,
        "waves_m": 2,
        "waves_n": 2,
        "waves_k": 1,
        "mfma_m": 16,
        "mfma_n": 16,
        "mfma_k": 16,
        "workgroup_size": 256,
    },
    # Architecture-adaptive: gfx90a config (128x128x32, 16x16x16 MFMA tile)
    {
        "name": "gemm_fp16_gfx90a",
        "a_dtype": "fp16",
        "b_dtype": "fp16",
        "c_dtype": "fp16",
        "acc_dtype": "fp32",
        "block_m": 128,
        "block_n": 128,
        "block_k": 32,
        "waves_m": 2,
        "waves_n": 2,
        "waves_k": 1,
        "mfma_m": 16,
        "mfma_n": 16,
        "mfma_k": 16,
        "workgroup_size": 256,
    },
    # Architecture-adaptive: gfx942 config (256x256x32, 32x32x16 MFMA tile)
    {
        "name": "gemm_fp16_gfx942",
        "a_dtype": "fp16",
        "b_dtype": "fp16",
        "c_dtype": "fp16",
        "acc_dtype": "fp32",
        "block_m": 256,
        "block_n": 256,
        "block_k": 32,
        "waves_m": 2,
        "waves_n": 2,
        "waves_k": 1,
        "mfma_m": 32,
        "mfma_n": 32,
        "mfma_k": 16,
        "workgroup_size": 256,
    },
    # Preshuffle: weight preshuffle pipeline
    {
        "name": "gemm_fp16_preshuffle",
        "a_dtype": "fp16",
        "b_dtype": "fp16",
        "c_dtype": "fp16",
        "acc_dtype": "fp32",
        "pipeline": "Preshuffle",
        "block_m": 128,
        "block_n": 128,
        "block_k": 32,
        "waves_m": 2,
        "waves_n": 2,
        "waves_k": 1,
        "mfma_m": 16,
        "mfma_n": 16,
        "mfma_k": 16,
        "workgroup_size": 256,
    },
]
ARCHITECTURES = ["gfx90a", "gfx942", "gfx950"]


def main() -> None:
    # Accept an optional build directory argument (used by CMake).
    build_dir = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "build"
    )
    output_path = build_dir / "gemm.kpack"

    # Read .hsaco blobs for each variant x architecture
    blobs: list[bytes] = []
    found_arches: set[str] = set()
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

        variant_metadata = {}
        for v in VARIANTS:
            if v["name"] in variant_map:
                meta = {
                    "a_dtype": v["a_dtype"],
                    "b_dtype": v["b_dtype"],
                    "c_dtype": v["c_dtype"],
                    "acc_dtype": v["acc_dtype"],
                    "block_m": v["block_m"],
                    "block_n": v["block_n"],
                    "block_k": v["block_k"],
                    "waves_m": v["waves_m"],
                    "waves_n": v["waves_n"],
                    "waves_k": v["waves_k"],
                    "mfma_m": v["mfma_m"],
                    "mfma_n": v["mfma_n"],
                    "mfma_k": v["mfma_k"],
                    "workgroup_size": v["workgroup_size"],
                }
                # Include epilogue metadata when present
                if "combine" in v:
                    meta["combine"] = v["combine"]
                if "activation" in v:
                    meta["activation"] = v["activation"]
                if "d0_dtype" in v:
                    meta["d0_dtype"] = v["d0_dtype"]
                if "d1_dtype" in v:
                    meta["d1_dtype"] = v["d1_dtype"]
                if "a_layout" in v:
                    meta["a_layout"] = v["a_layout"]
                if "b_layout" in v:
                    meta["b_layout"] = v["b_layout"]
                if "c_layout" in v:
                    meta["c_layout"] = v["c_layout"]
                if "k_batch" in v:
                    meta["k_batch"] = v["k_batch"]
                if "pipeline" in v:
                    meta["pipeline"] = v["pipeline"]
                if "scheduling" in v:
                    meta["scheduling"] = v["scheduling"]
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
