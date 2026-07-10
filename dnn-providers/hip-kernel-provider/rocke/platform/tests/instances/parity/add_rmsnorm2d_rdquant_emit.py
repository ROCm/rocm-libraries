#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/add_rmsnorm2d_rdquant_emit.py -- Python reference emitter for the
# add_rmsnorm2d_rdquant parity harness. Selects one of the sampled configs by
# argv[1] (the config index), builds the AddRmsnorm2DRdquantSpec, builds the
# kernel via build_add_rmsnorm2d_rdquant(arch='gfx950') and prints
# _native_lower(arch='gfx950') to stdout so it can be byte-compared with
# the C emitter add_rmsnorm2d_rdquant_emit.c.
import sys

from rocke.instances.common.add_rmsnorm2d_rdquant import (
    AddRmsnorm2DRdquantSpec,
    build_add_rmsnorm2d_rdquant,
    is_valid_spec,
)

try:
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as _native_lower
except ImportError:  # pragma: no cover - older reference tree
    from rocke import lower_kernel_to_llvm as _native_lower
from rocke.core.ir_serialize import serialize
from rocke.core.verify import verify


def _spec(idx: int) -> AddRmsnorm2DRdquantSpec:
    if idx == 0:
        return AddRmsnorm2DRdquantSpec(
            n_per_block=4096,
            dtype="f16",
            out_dtype="i8",
            block_size=256,
            vec=4,
            save_residual=True,
            save_yscale=True,
            wave_size=64,
        )
    if idx == 1:
        return AddRmsnorm2DRdquantSpec(
            n_per_block=8192,
            dtype="f16",
            out_dtype="fp8e4m3",
            block_size=256,
            vec=8,
            save_residual=True,
            save_yscale=True,
            wave_size=64,
        )
    if idx == 2:
        return AddRmsnorm2DRdquantSpec(
            n_per_block=2048,
            dtype="bf16",
            out_dtype="bf8e5m2",
            block_size=128,
            vec=4,
            save_residual=False,
            save_yscale=False,
            wave_size=64,
        )
    if idx == 3:
        return AddRmsnorm2DRdquantSpec(
            n_per_block=16384,
            dtype="f16",
            out_dtype="i8",
            block_size=256,
            vec=4,
            save_residual=True,
            save_yscale=True,
            wave_size=64,
        )
    if idx == 4:
        return AddRmsnorm2DRdquantSpec(
            n_per_block=1024,
            dtype="bf16",
            out_dtype="i8",
            block_size=64,
            vec=4,
            save_residual=True,
            save_yscale=True,
            wave_size=64,
        )
    if idx == 5:
        return AddRmsnorm2DRdquantSpec(
            n_per_block=6144,
            dtype="f16",
            out_dtype="fp8e4m3",
            block_size=256,
            vec=2,
            save_residual=True,
            save_yscale=False,
            wave_size=64,
        )
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write(
            "usage: add_rmsnorm2d_rdquant_emit.py <config_index 0..5> [ll|ir|verify]\n"
        )
        return 2
    idx = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "ll"
    if mode not in ("ll", "ir", "verify"):
        sys.stderr.write(f"unknown mode {mode}\n")
        return 2
    spec = _spec(idx)
    ok, reason = is_valid_spec(spec, "gfx950")
    if not ok:
        sys.stderr.write(f"invalid spec: {reason}\n")
        return 1
    kernel = build_add_rmsnorm2d_rdquant(spec, arch="gfx950")
    if mode == "ll":
        text = _native_lower(kernel, arch="gfx950")
        sys.stdout.write(text)
    elif mode == "ir":
        sys.stdout.write(serialize(kernel))
    else:  # verify
        sys.stdout.write("".join(str(d) + "\n" for d in verify(kernel)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
