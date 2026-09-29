# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Derive packer-side output variants of a synthetic bundle, simulating hkp_pack changes.

  noprov : drop every UKD's `provenance` (the KDP header's specialization_contract stays)
  min    : serialize without indentation (separators ',' ':')

Usage: python variant_kdp.py <synthetic root> <out root> [noprov] [min]
"""
import json
import shutil
import sys
from pathlib import Path

KDP = Path("gfx950/rocKE/gfx950_attention_dense/gfx950_attention_dense.kdp.json")


def main(src, out, opts):
    src, out = Path(src), Path(out)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src, out, ignore=shutil.ignore_patterns("*.kdp.json"))
    kdp = json.loads((src / KDP).read_text())
    if "noprov" in opts:
        for ukd in kdp["kernelDescriptors"]:
            ukd.pop("provenance", None)
    if "min" in opts:
        text = json.dumps(kdp, separators=(",", ":")) + "\n"
    else:
        text = json.dumps(kdp, indent=2) + "\n"
    (out / KDP).write_text(text)
    print(json.dumps({"out": str(out), "opts": opts, "kdp_bytes": len(text)}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
