#!/usr/bin/env python3
"""Regenerate exactly the flyDSL attention HSACOs referenced by the descriptor pack.

Reads flydsl_attention.kdp.json, and for every unique `metadata.hsaco` it references,
reconstructs the build_flash_attn_real.py invocation from the baked config
(num_query_heads, num_kv_heads, head_size, causal, dtype) plus any knob suffix encoded
in the filename (_w4 -> --waves 4, _nostag -> --no-stagger). This makes the whole
registered family reproducible from source on a fresh gfx950 checkout:

    /path/to/flydsl-venv/bin/python \
        flydsl_poc_scratch/flydsl_build/build_all_flydsl_attention.py

HSACOs land next to the pack (FLYDSL_ATTENTION_HSACO_DIR = flydsl_poc_scratch/).
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.path.dirname(HERE)  # flydsl_poc_scratch
KDP = os.path.join(
    SCRATCH, "flydsl_descriptors_attention", "flydsl_attention", "flydsl_attention.kdp.json"
)
BUILDER = os.path.join(HERE, "build_flash_attn_real.py")
DTYPE_ARG = {"BF16": "bf16", "FP16": "f16"}


def main():
    d = json.load(open(KDP))
    seen = {}
    for k in d["kernelDescriptors"]:
        m = k["metadata"]
        seen.setdefault(m["hsaco"], m)
    print(f"{len(seen)} unique HSACOs to (re)build from {os.path.basename(KDP)}\n")
    rc = 0
    for hsaco, m in sorted(seen.items()):
        nqh = int(m["num_query_heads"])
        nkv = int(m["num_kv_heads"])
        hs = int(m["head_size"])
        causal = "1" if str(m.get("causal")).lower() in ("1", "true") else "0"
        dt = DTYPE_ARG[m["dtype"]]
        args = [sys.executable, BUILDER, str(nqh), str(hs), causal, dt, "--kv", str(nkv)]
        if "_w4" in hsaco:
            args += ["--waves", "4"]
        elif "_w1" in hsaco:
            args += ["--waves", "1"]
        elif "_w3" in hsaco:
            args += ["--waves", "3"]
        if "_nostag" in hsaco:
            args += ["--no-stagger"]
        print(f"--> {hsaco}\n    {' '.join(args[1:])}")
        r = subprocess.run(args, cwd=SCRATCH)
        if r.returncode != 0:
            print(f"    !! build FAILED for {hsaco}")
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
