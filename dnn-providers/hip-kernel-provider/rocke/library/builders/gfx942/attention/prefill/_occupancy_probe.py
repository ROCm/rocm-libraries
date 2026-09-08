#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Per-phase spill / occupancy probe for the gfx942 dense kernel.

Kept across P0-P4: re-run after every lever to confirm 0 spill and to see which
resource (VGPR at D64, LDS at D128) is the occupancy limiter. Compiles the gfx942 dense
kernel FOR gfx942 and parses AMDHSA notes (VGPR/AGPR/SGPR/spill/LDS) via the
shipped probe_occupancy helpers. Static (comgr only, no GPU launch).

The shipped probe_occupancy() loop hardcodes the default compile arch, so we call
its parse/estimate helpers directly with an explicit arch='gfx942' compile.

IMPORTING THIS MODULE MUST STAY FREE OF SIDE EFFECTS. It lives under builders/,
which pytest can be pointed at; the sys.path surgery, the kernel compiles and the
sys.exit() all happen inside main() for that reason. Importing it and then never
calling main() must do nothing.
"""
import os
import sys


def main() -> int:
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _RK = os.path.abspath(os.path.join(_HERE, "../../../../.."))
    sys.path.insert(0, _RK + "/platform/python")
    sys.path.insert(0, _RK + "/library")
    sys.path.insert(
        0, _RK + "/platform/dsl_docs/optimization/utilities/tools/dsl_probes"
    )

    from kernels.gfx942.attention_dense import (
        AttentionDenseSpec,
        build_attention_dense,
        _tuned_waves_per_eu,
    )
    from rocke.helpers.compile import compile_kernel
    from probe_occupancy import (
        parse_hsaco_notes,
        estimate_occupancy,
        ARCH_GFX942,
    )

    print(
        f"{'label':<24} {'vgpr':>5} {'agpr':>5} {'sgpr':>5} {'spill':>6} "
        f"{'lds B':>7} {'waves/CU':>9} {'wg/CU':>6} {'limit':>8}"
    )
    print("-" * 84)
    worst_spill = 0
    for label, s in SHAPES:
        # waves_per_eu is NOT left at the dataclass default: it is a per-config
        # tuned value that the dispatch spec factory fills from the kernel's own
        # policy, and it changes register allocation. Probing at the default would
        # report the spill/occupancy of a config nobody ships -- which is exactly
        # what hid bf16 D64 (the one config with an override) from this gate.
        spec = AttentionDenseSpec(
            batch=1,
            block_n=64,
            waves_per_eu=_tuned_waves_per_eu(s["head_size"], s["dtype"]),
            **s,
        )
        kd = build_attention_dense(spec, arch="gfx942")
        art = compile_kernel(kd, arch="gfx942", capture_ir_text=False)
        notes = parse_hsaco_notes(art.hsaco)
        occ = estimate_occupancy(
            notes=notes,
            waves_per_wg=8,
            arch=ARCH_GFX942,
            waves_per_eu_hint=int(spec.waves_per_eu),
        )
        vspill = notes.get("vgpr_spill_count", 0)
        sspill = notes.get("sgpr_spill_count", 0)
        worst_spill = max(worst_spill, vspill, sspill)
        print(
            f"{label:<24} {notes.get('vgpr_count',0):>5} {notes.get('agpr_count',0):>5} "
            f"{notes.get('sgpr_count',0):>5} {vspill+sspill:>6} "
            f"{notes.get('lds_size',0):>7} "
            f"{occ['waves_per_cu']:>9} {occ['wgs_per_cu']:>6} {occ['limited_by']:>8}"
        )
    print("-" * 84)
    print(
        "RESULT:",
        (
            "0 SPILL (resource gate PASS)"
            if worst_spill == 0
            else f"SPILL={worst_spill} (FAIL)"
        ),
    )
    return 0 if worst_spill == 0 else 1


SHAPES = [
    (
        "d128_fp16_16x4_s2048",
        dict(
            seqlen_q=2048,
            seqlen_kv=2048,
            num_query_heads=16,
            num_kv_heads=4,
            head_size=128,
            dtype="fp16",
            causal=True,
        ),
    ),
    (
        "d128_bf16_16x4_s2048",
        dict(
            seqlen_q=2048,
            seqlen_kv=2048,
            num_query_heads=16,
            num_kv_heads=4,
            head_size=128,
            dtype="bf16",
            causal=True,
        ),
    ),
    (
        "d64_fp16_16x4_s512",
        dict(
            seqlen_q=512,
            seqlen_kv=512,
            num_query_heads=16,
            num_kv_heads=4,
            head_size=64,
            dtype="fp16",
            causal=True,
        ),
    ),
    # bf16 D64 is the ONE shipped config with a waves_per_eu override (4, not 2), so
    # it is the row this probe most needs: the override is what forces the allocator
    # down far enough for a 2nd workgroup, and it is the row most likely to spill if
    # a lever raises register pressure. It was missing until now.
    (
        "d64_bf16_16x4_s512",
        dict(
            seqlen_q=512,
            seqlen_kv=512,
            num_query_heads=16,
            num_kv_heads=4,
            head_size=64,
            dtype="bf16",
            causal=True,
        ),
    ),
]

if __name__ == "__main__":
    sys.exit(main())
