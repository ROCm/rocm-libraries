#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gett_f64_pack_emit.py -- Python reference emitter for the f64
# GETT kernels of hipTensor's kernel pack, resolved through the gett_f64 spec
# policy (gemm_problem, arch_spec, TUNED_CONFIGS, runtime_template,
# _with_k_block). gett_f64_pack_emit.c resolves the same specs through the C
# policy API (gett_f64_policy.cpp), so run_diff.py proves the policy and the
# builder byte-identical together.
#
# Config order: per arch (gfx950, gfx942, gfx90a), real then complex, every
# tuned shape class (sorted) then "default", D layout m / n / x, scale /
# bilinear, without / with k_block -- the pack's descriptor order. A config
# whose k_block variant does not exist (or whose spec is invalid) is rejected
# by both engines.
import sys
from dataclasses import replace

from rocke.instances.common.gett_f64 import (
    TUNED_CONFIGS,
    _with_k_block,
    arch_spec,
    build_gett_f64,
    gemm_problem,
    is_valid_spec,
    runtime_template,
)

from _emit_common import run_emit

ARCHES = ("gfx950", "gfx942", "gfx90a")


def entries():
    out = []
    for arch in ARCHES:
        for cplx in (False, True):
            kind = "complex" if cplx else "real"
            classes = sorted(
                c for (a, k, c) in TUNED_CONFIGS if a == arch and k == kind
            )
            for cls in classes + ["default"]:
                for dclass in ("m", "n", "x"):
                    for bilinear in (False, True):
                        for kblock in (False, True):
                            out.append((arch, cplx, cls, dclass, bilinear, kblock))
    return out


def pack_spec(arch, cplx, cls, dclass, bilinear, kblock):
    """The spec of one pack descriptor, or None when it does not apply."""
    M, N, K, L = 64, 64, 64, 2
    d_strides = {"m": (1, M), "n": (N, 1), "x": (2, 2 * M)}[dclass]
    prob = gemm_problem(
        M,
        N,
        K,
        a_strides=(K, 1),
        b_strides=(K, 1),
        d_strides=d_strides,
        batch=L,
        batch_strides=(M * K, N * K, 4 * M * N),
        complex_=cplx,
    )
    kind = "complex" if cplx else "real"
    cfg = {} if cls == "default" else dict(TUNED_CONFIGS[(arch, kind, cls)])
    cfg.pop("split_k", None)
    spec = arch_spec(prob, arch, bilinear=bilinear, runtime_sizes=True, **cfg)
    spec = replace(spec, split_k=1, problem=runtime_template(prob))
    if kblock:
        kb = _with_k_block(spec, arch)
        if kb.k_block == spec.k_block:
            return None
        spec = kb
    if not is_valid_spec(spec, arch=arch)[0]:
        return None
    return spec


def _spec(idx):
    es = entries()
    if idx < 0 or idx >= len(es):
        sys.stderr.write(f"unknown config index {idx}\n")
        sys.exit(2)
    arch = es[idx][0]
    spec = pack_spec(*es[idx])
    if spec is None:
        sys.stderr.write("config does not apply\n")
        sys.exit(1)
    return spec, arch


def main() -> int:
    return run_emit(
        _spec,
        build_gett_f64,
        usage="usage: gett_f64_pack_emit.py <config_index> [ll|ir|verify]\n",
    )


if __name__ == "__main__":
    sys.exit(main())
