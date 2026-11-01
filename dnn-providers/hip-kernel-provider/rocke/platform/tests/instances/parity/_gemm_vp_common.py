# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/_gemm_vp_common.py -- shared driver of the gemm_vp_* parity
# families: the register-staged global prefetch body of the universal GEMM
# (TraitSpec.vgpr_prefetch, the K-contiguous RCR form, and its main-loop levers)
# on gfx942, built by the Python engine (gemm_vp_*_emit.py) and through the C API
# (gemm_vp_*_emit.c) so run_diff.py byte-compares the two engines.
#
# The configurations are a frozen table (_gemm_vp_cases.CASES, one list per
# family): a dispatcher change does not move them. Each row is
#   (name, arch, dtype, (tile_m, tile_n, tile_k, warp_m, warp_n, warp_k,
#    warp_tile_m, warp_tile_n, warp_tile_k), block_size, {TraitSpec field:
#    value} (non-default fields only), {Prologue field: value} or None,
#    batched)
# The C table of a family is generated from it:
#   python gemm_vp_<family>_emit.py --c-cases
# prints the case rows of gemm_vp_<family>_emit.c.
import sys

from _gemm_vp_cases import CASES


def spec_for(family: str, idx: int):
    """(spec, arch) of config ``idx`` of ``family`` (exit 1: a spec the
    validity gate / the prologue refuses; exit 2: past the end)."""
    from rocke.instances.common.gemm_prologue import Prologue, attach_prologue
    from rocke.instances.common.gemm_universal import (
        DataSpec,
        TileSpec,
        TraitSpec,
        UniversalGemmSpec,
        is_valid_spec,
    )

    rows = CASES[family]
    if idx < 0 or idx >= len(rows):
        sys.stderr.write(f"unknown config index {idx}\n")
        sys.exit(2)
    name, arch, dt, tile, bs, trait, pro, batched = rows[idx]
    spec = UniversalGemmSpec(
        name=name,
        tile=TileSpec(*tile),
        trait=TraitSpec(**trait),
        data=DataSpec(dtype_a=dt, dtype_b=dt, dtype_c=dt),
        block_size=bs,
        batched=batched,
    )
    if pro is not None:
        try:
            attach_prologue(spec, Prologue(**pro))
        except ValueError as e:
            sys.stderr.write(f"prologue: {e}\n")
            sys.exit(1)
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        sys.stderr.write(f"invalid spec: {why}\n")
        sys.exit(1)
    return spec, arch


def build(spec, arch):
    from rocke.instances.common.gemm_universal import build_universal_gemm

    return build_universal_gemm(spec, arch)


def _c_value(v) -> str:
    if v is None:
        return "none"
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def c_cases(family: str) -> str:
    """The rows of the static case table of gemm_vp_<family>_emit.c."""
    lines = []
    for name, arch, dt, tile, bs, trait, pro, batched in CASES[family]:
        tr = ",".join(f"{k}={_c_value(v)}" for k, v in trait.items())
        pr = ",".join(f"{k}={_c_value(v)}" for k, v in (pro or {}).items())
        t = ", ".join(str(x) for x in tile)
        lines.append(
            f'    {{"{name}", "{arch}", "{dt}", {{{t}}}, {bs}, "{tr}", '
            f'{"NULL" if pro is None else chr(34) + pr + chr(34)}, {int(batched)}}},'
        )
    return "\n".join(lines) + "\n"


def main(family: str) -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--c-cases":
        sys.stdout.write(c_cases(family))
        return 0
    from _emit_common import run_emit

    return run_emit(
        lambda idx: spec_for(family, idx),
        build,
        usage=f"usage: {family}_emit.py <config_index> [ll|ir|verify]\n",
    )
