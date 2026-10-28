#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gemm_prologue_unary_emit.py -- Python reference emitter for the
# operand operators of the fused GEMM operand prologue
# (instances/common/gemm_prologue.py: Prologue.unary_a / unary_b, unary_impl,
# fast_exp) on the named universal GEMM bodies of dispatch/gemm/cf32.py.
# gemm_prologue_unary_emit.c builds the same specs through the C API so
# run_diff.py byte-compares the two engines.
#
# CONFIGS: every operator on A of an f32-source x6 body (split at the MFMA
# use), on B of the row-interleaved x6 body, and on both operands of a 16-bit
# cvt body (split at the load); the accurate exp / sigmoid (fast_exp off); the
# packed-bit forms of abs / neg / relu (unary_impl "bits") on f16 and bf16
# fragments, at the load and at the use; an operator with the complex-B
# prologue and with the x9 recipe.
#
# The C cases are generated from this table: `python
# gemm_prologue_unary_emit.py --c-cases` prints the body of make_spec() in
# gemm_prologue_unary_emit.c.
import sys

ARCH = "gfx950"

_H16 = dict(src="same", recipe="cvt", out_f32=True)
_X6 = dict(src="f32", recipe="x6", out_f32=True, split_at="use", psub=True)
_CF = dict(_X6, cplx_b=True)

UNARY = (
    "relu",
    "abs",
    "neg",
    "exp",
    "sq",
    "sigmoid",
    "sqrt",
    "rcp",
    "tanh",
    "log",
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "asin",
    "acos",
    "atan",
    "asinh",
    "acosh",
    "atanh",
    "ceil",
    "floor",
)


def _configs():
    out = []
    for u in UNARY:
        out.append(
            dict(name=f"x6a_{u}", body="C3", dtype="bf16", pro=dict(_X6, unary_a=u))
        )
        out.append(
            dict(name=f"x6b_{u}", body="RIW", dtype="bf16", pro=dict(_X6, unary_b=u))
        )
        out.append(
            dict(
                name=f"cvt_{u}",
                body="C3",
                dtype="fp16",
                pro=dict(_H16, unary_a=u, unary_b=u),
            )
        )
    for u in ("exp", "sigmoid"):
        out.append(
            dict(
                name=f"x6a_{u}_acc",
                body="C3",
                dtype="bf16",
                pro=dict(_X6, unary_a=u, fast_exp=False),
            )
        )
        out.append(
            dict(
                name=f"cvt_{u}_acc",
                body="C3",
                dtype="bf16",
                pro=dict(_H16, unary_b=u, fast_exp=False),
            )
        )
    for u in ("abs", "neg", "relu"):
        for dt in ("fp16", "bf16"):
            out.append(
                dict(
                    name=f"bits_{u}_{dt}",
                    body="C3",
                    dtype=dt,
                    pro=dict(_H16, unary_a=u, unary_impl="bits"),
                )
            )
            out.append(
                dict(
                    name=f"bits_use_{u}_{dt}",
                    body="RIW",
                    dtype=dt,
                    pro=dict(_H16, unary_b=u, unary_impl="bits", split_at="use"),
                )
            )
    out += [
        dict(
            name="cf_neg_a",
            body="RIW",
            dtype="bf16",
            pro=dict(_CF, unary_a="neg"),
        ),
        dict(
            name="cf_sqrt_b_q3sk",
            body="Q3_SK",
            dtype="bf16",
            pro=dict(_CF, unary_b="sqrt"),
        ),
        dict(
            name="x9_tanh_b",
            body="C3",
            dtype="bf16",
            pro=dict(_X6, recipe="x9", unary_b="tanh"),
        ),
    ]
    return out


CONFIGS = _configs()


def _spec(idx: int):
    from rocke.dispatch.gemm.cf32 import cf32_body
    from rocke.instances.common.gemm_prologue import Prologue, attach_prologue
    from rocke.instances.common.gemm_universal import (
        DataSpec,
        UniversalGemmSpec,
        is_valid_spec,
    )

    if idx < 0 or idx >= len(CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    c = CONFIGS[idx]
    tile, trait, bs = cf32_body(c["body"], ARCH)
    dt = c["dtype"]
    spec = UniversalGemmSpec(
        name=c["name"],
        tile=tile,
        trait=trait,
        data=DataSpec(dtype_a=dt, dtype_b=dt, dtype_c=dt),
        block_size=bs,
    )
    ok, why = is_valid_spec(spec, arch=ARCH)
    if not ok:
        sys.stderr.write(f"invalid spec: {why}\n")
        sys.exit(1)
    try:
        attach_prologue(spec, Prologue(**c["pro"]))
    except ValueError as e:
        sys.stderr.write(f"prologue: {e}\n")
        sys.exit(1)
    return spec, ARCH


def _build(spec, arch):
    from rocke.instances.common.gemm_universal import build_universal_gemm

    return build_universal_gemm(spec, arch)


def _c_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    return f'"{v}"'


def c_cases() -> str:
    """The case list of make_spec() in gemm_prologue_unary_emit.c."""
    lines = []
    for i, c in enumerate(CONFIGS):
        lines.append(f"    case {i}:")
        lines.append(
            f'        base(spec, "{c["name"]}", "{c["body"]}", "{c["dtype"]}");'
        )
        for k, v in c["pro"].items():
            lines.append(f"        pro->{k} = {_c_value(v)};")
        lines.append("        return 1;")
    return "\n".join(lines) + "\n"


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--c-cases":
        sys.stdout.write(c_cases())
        return 0
    from _emit_common import run_emit

    return run_emit(
        _spec,
        _build,
        usage="usage: gemm_prologue_unary_emit.py <config_index> [ll|ir|verify]\n",
    )


if __name__ == "__main__":
    sys.exit(main())
