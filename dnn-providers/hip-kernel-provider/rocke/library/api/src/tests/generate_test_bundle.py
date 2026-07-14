#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Generate a minimal rocKE-client integration-test probe bundle:
#   rocke_client_<arch>.kpack  - zstd-compressed kpack archive
#   rocke_client_<arch>.json   - bundle manifest
#
# The bundle contains a single trivial kernel compiled from probe.hip.
# It is consumed by loadForDevice() to prove the AOT load path end-to-end
# without executing any real SDPA workloads.
#
# toc_key: rocke/test/probe/rocke_test_probe
# symbol:  rocke_test_probe
#
# Usage (called by CMake add_custom_command):
#   python generate_test_bundle.py \
#       --hsaco probe_gfx942.hsaco \
#       --arch gfx942 \
#       --outdir /path/to/build/bin/hip_kernel_provider/tests/aot_test_bundles/valid/gfx942 \
#       --kpack-python-dir /path/to/rocm-systems/shared/kpack/python

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Pack a probe HSACO into a rocKE-client test bundle"
)
parser.add_argument(
    "--hsaco",
    required=True,
    type=Path,
    help="Path to the unbundled bare HSACO code object",
)
parser.add_argument(
    "--arch", required=True, help="GFX architecture string (e.g. gfx942)"
)
parser.add_argument(
    "--outdir",
    required=True,
    type=Path,
    help="Output directory for the kpack + manifest",
)
parser.add_argument(
    "--kpack-python-dir",
    required=True,
    type=Path,
    help="Path to rocm-systems/shared/kpack/python",
)
args = parser.parse_args()

sys.path.insert(0, str(args.kpack_python_dir))
try:
    from rocm_kpack.kpack import PackedKernelArchive
    from rocm_kpack.compression import ZstdCompressor
except ImportError as exc:
    print(
        f"ERROR: cannot import rocm_kpack from {args.kpack_python_dir}: {exc}",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants — must match what AotCatalog::loadForDevice() parses
# ---------------------------------------------------------------------------
TOC_KEY = "rocke/test/probe/rocke_test_probe"
SYMBOL = "rocke_test_probe"

# ---------------------------------------------------------------------------
# Pack
# ---------------------------------------------------------------------------
hsaco = args.hsaco.read_bytes()
if hsaco[:4] != b"\x7fELF":
    print(f"ERROR: {args.hsaco} does not start with ELF magic", file=sys.stderr)
    sys.exit(1)

archive = PackedKernelArchive(
    group_name="rocke_client",
    gfx_arch_family=args.arch,
    gfx_arches=[args.arch],
    compressor=ZstdCompressor(compression_level=3),
)
prepared = archive.prepare_kernel(
    relative_path=TOC_KEY,
    gfx_arch=args.arch,
    hsaco_data=hsaco,
    metadata={"cache_key": f"test_probe_{args.arch}", "symbol": SYMBOL},
)
archive.add_kernel(prepared)
archive.finalize_archive()

args.outdir.mkdir(parents=True, exist_ok=True)
kpack_name = f"rocke_client_{args.arch}.kpack"
manifest_name = f"rocke_client_{args.arch}.json"
kpack_path = args.outdir / kpack_name
manifest_path = args.outdir / manifest_name

archive.write(kpack_path)

# ---------------------------------------------------------------------------
# Bundle manifest — schema matches what AotCatalog::loadForDevice() parses:
#   entries[0]["toc_key"], entries[0]["symbol"]
# ---------------------------------------------------------------------------
manifest = {
    "schema": "rocke.aot.bundle/v1",
    "arch": args.arch,
    "kpack": kpack_name,
    "entries": [
        {
            "toc_key": TOC_KEY,
            "symbol": SYMBOL,
            "cache_key": f"test_probe_{args.arch}",
        }
    ],
}
manifest_path.write_text(json.dumps(manifest, indent=2))

print(
    f"Generated {kpack_path} ({kpack_path.stat().st_size} bytes) "
    f"and {manifest_path}"
)
