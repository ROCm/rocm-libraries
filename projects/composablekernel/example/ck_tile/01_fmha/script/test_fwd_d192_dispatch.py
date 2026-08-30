#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

COMMON_ARGS = (
    "-prec=bf16",
    "-b=1",
    "-h=8",
    "-h_k=8",
    "-d=192",
    "-d_v=128",
    "-s=128",
    "-s_k=128",
    "-bias=n",
    "-p_drop=0",
    "-lse=0",
    "-iperm=0",
    "-operm=0",
    "-vlayout=r",
    "-mask=0",
    "-v=1",
    "-kname=1",
    "-warmup=0",
    "-repeat=1",
    "-timer=cpu",
    "-init=uf",
    "-seed=20260825",
)

CASES = {
    "dispatch_s1_fallback": ("-s=1", "-s_k=1"),
    "dispatch_s31_fallback": ("-s=31", "-s_k=31"),
    "dispatch_s32_fallback": ("-s=32", "-s_k=32"),
    "dispatch_s63_fallback": ("-s=63", "-s_k=63"),
    "dispatch_s64_fallback": ("-s=64", "-s_k=64"),
    "dispatch_exact_s128": (),
    "dispatch_s129": ("-s=129", "-s_k=129"),
    "dispatch_s257": ("-s=257", "-s_k=257"),
    "dispatch_s512": ("-s=512", "-s_k=512"),
    "dispatch_s1024": ("-s=1024", "-s_k=1024"),
    "dispatch_exact_s2048": ("-s=2048", "-s_k=2048"),
    "batch_q129_k257": ("-s=129", "-s_k=257"),
    "batch_q257_k129": ("-s=257", "-s_k=129"),
    "batch_empty_k": ("-s=128", "-s_k=0"),
    "dispatch_s127_fallback": ("-s=127", "-s_k=127"),
    "dispatch_hdim160_v128_fallback": ("-d=160",),
    "batch_causal_tl": ("-mask=t",),
    "batch_causal_br": ("-mask=b",),
    "batch_swa_tile_boundary": ("-mask=b:128,32",),
    "batch_lse": ("-lse=1",),
    "gqa_h8_hkv2": ("-h_k=2",),
    "mqa_h8_hkv1": ("-h_k=1",),
    "group_uneven_113_257_1024": (
        "-mode=1",
        "-b=3",
        "-s=113,257,1024",
        "-s_k=113,257,1024",
    ),
    "group_padded_113_257_1024": (
        "-mode=1",
        "-b=3",
        "-s=113,257,1024",
        "-s_k=113,257,1024",
        "-s_qpad=128,384,1024",
        "-s_kpad=128,384,1024",
    ),
    "fallback_logits": ("-logits_soft_cap=1",),
    "fallback_bias": ("-bias=e",),
    "fallback_dropout": ("-p_drop=0.2",),
    "fallback_sink": ("-init_sink=1",),
    "rejected_qscale": ("-qscale=pt",),
    "d128_regression_s128": ("-d=128", "-d_v=128"),
    "d128_group_regression": (
        "-mode=1",
        "-b=3",
        "-d=128",
        "-d_v=128",
        "-s=113,257,1024",
        "-s_k=113,257,1024",
    ),
}

EXPECTED_TOKEN = {
    "candidate": "_qr_tdm_d192_v128_",
    "generic": "_qr_vr_",
    "d128": "_qr_tdm_",
    "rejected": None,
}
_D192_PIPELINE_TOKEN = EXPECTED_TOKEN["candidate"]


def _replace_args(overrides):
    values = {arg.split("=", 1)[0]: arg for arg in COMMON_ARGS}
    for arg in overrides:
        values[arg.split("=", 1)[0]] = arg
    return list(values.values())


def _set_opt_in(environment, opt_in):
    if opt_in == "unset":
        environment.pop("CK_TILE_FMHA_GFX125_D192_TDM", None)
    else:
        environment["CK_TILE_FMHA_GFX125_D192_TDM"] = opt_in


def _selected_kernel(output):
    names = re.findall(r"fmha_fwd_d[0-9]+_[A-Za-z0-9_]+", output)
    return names[-1] if names else None


def main():
    parser = argparse.ArgumentParser(
        description="Run one D192 dispatch/correctness case in a fresh process."
    )
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--case", required=True, choices=sorted(CASES))
    parser.add_argument("--expect", required=True, choices=sorted(EXPECTED_TOKEN))
    parser.add_argument("--opt-in", choices=("unset", "0", "1"), default="unset")
    parser.add_argument(
        "--asic-revision", choices=("A0", "B0", "unknown"), default="unknown"
    )
    parser.add_argument("--asic-revision-source", default="")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.expect == "candidate" and args.opt_in != "1":
        parser.error("candidate expectation requires --opt-in 1")

    binary = args.binary.resolve()
    command = [str(binary), *_replace_args(CASES[args.case])]
    environment = os.environ.copy()
    _set_opt_in(environment, args.opt_in)

    print(f"case={args.case}")
    print(f"expect={args.expect}")
    print(f"opt_in={args.opt_in}")
    print(f"asic_revision={args.asic_revision}")
    print(f"asic_revision_source={args.asic_revision_source or 'unrecorded'}")
    print(f"command={shlex.join(command)}")
    if args.dry_run:
        return 0

    if not binary.is_file():
        print(f"error: binary not found: {binary}", file=sys.stderr)
        return 2

    try:
        completed = subprocess.run(
            command,
            env=environment,
            capture_output=True,
            text=True,
            timeout=args.timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        print(f"error: case timed out after {args.timeout}s", file=sys.stderr)
        if error.stdout:
            print(error.stdout, end="")
        if error.stderr:
            print(error.stderr, end="", file=sys.stderr)
        return 8
    output = completed.stdout + completed.stderr
    print(output, end="")

    if args.expect == "rejected":
        if completed.returncode == 0 or "not supported yet" not in output:
            print("error: expected an explicit no-instance rejection", file=sys.stderr)
            return 9
        if _D192_PIPELINE_TOKEN in output:
            print("error: rejected case selected the D192 candidate", file=sys.stderr)
            return 10
        print("selected_kernel=none (expected rejection)")
        return 0

    if completed.returncode != 0:
        print(f"error: binary exited with {completed.returncode}", file=sys.stderr)
        return 3

    kernel = _selected_kernel(output)
    if kernel is None:
        print("error: generated kernel name was not reported", file=sys.stderr)
        return 4

    expected_token = EXPECTED_TOKEN[args.expect]
    if expected_token not in kernel:
        print(
            f"error: expected {expected_token!r}, selected {kernel!r}",
            file=sys.stderr,
        )
        return 5
    if args.expect != "candidate" and _D192_PIPELINE_TOKEN in kernel:
        print(f"error: unexpected D192 candidate: {kernel}", file=sys.stderr)
        return 6
    if "valid:y" not in output:
        print("error: numerical validation did not report valid:y", file=sys.stderr)
        return 7

    print(f"selected_kernel={kernel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
