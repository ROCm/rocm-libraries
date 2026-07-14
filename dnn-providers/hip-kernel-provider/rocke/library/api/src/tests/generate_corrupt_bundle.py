#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Generate a CORRUPT rocKE-client test bundle for the negative integration test.
# Produces:
#   rocke_client_<arch>.kpack  -- invalid bytes (kpack_open will fail)
#   rocke_client_<arch>.json   -- valid manifest (parser succeeds; kpack open fails)
#
# The corrupt kpack lets TestRockeClientAotLoad::CorruptBundleFailsLoudly confirm
# that the AOT load path is actually exercised and emits the LOAD FAILED marker
# rather than being silently skipped.
#
# Usage (called by CMake add_custom_command):
#   python generate_corrupt_bundle.py --arch gfx942 --outdir /path/to/corrupt/gfx942

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description="Generate a corrupt rocKE test bundle")
parser.add_argument(
    "--arch", required=True, help="GFX architecture string (e.g. gfx942)"
)
parser.add_argument("--outdir", required=True, type=Path, help="Output directory")
args = parser.parse_args()

args.outdir.mkdir(parents=True, exist_ok=True)

# Corrupt kpack: intentionally invalid bytes so kpack_open fails.
kpack_path = args.outdir / f"rocke_client_{args.arch}.kpack"
kpack_path.write_bytes(b"CORRUPT_KPACK_FOR_TESTING_NOT_A_VALID_ARCHIVE")

# Valid manifest so loadForDevice reaches kpack_open before failing.
# toc_key and symbol match what generate_test_bundle.py produces.
manifest = {
    "schema": "rocke.aot.bundle/v1",
    "arch": args.arch,
    "kpack": f"rocke_client_{args.arch}.kpack",
    "entries": [
        {
            "toc_key": "rocke/test/skeleton/rocke_test_probe",
            "symbol": "rocke_test_probe",
        }
    ],
}
(args.outdir / f"rocke_client_{args.arch}.json").write_text(
    json.dumps(manifest, indent=2)
)

print(f"Generated corrupt bundle in {args.outdir}")
