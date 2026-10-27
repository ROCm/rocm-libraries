#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gemm_cf32_emit.py -- Python reference emitter for the named
# universal GEMM bodies of the emulated-f32 routes (dispatch/gemm/cf32.py
# cf32_body / _adjust_k) with their fused operand prologues
# (instances/common/gemm_prologue.py), as hipTensor's kernel pack builds them.
# gemm_cf32_emit.c builds the same specs through the C API
# (rocke_gemm_cf32_body, rocke_gemm_cf32_adjust_k, rocke_gemm_prologue_t) so
# run_diff.py byte-compares the two engines.
#
# CONFIGS: first the pack's GEMM descriptors in its order (routes h16 / x6 /
# cf32, every body, batched, the 64-bit-address body), then the levers and
# prologue forms around them: every recipe, split_at load / use, order, cap,
# pin_v, agpr_out, conjugation in both complex-row modes, the row interleave
# without its vector epilogue / with rilv_pad 32, the levers without a
# prologue, dtl_soff0 on a swizzled body, the non-ring DTL body and the VGPR
# body. A config that is not valid (is_valid_spec) is rejected by both engines.
#
# The C cases are generated from this table: `python gemm_cf32_emit.py
# --c-cases` prints the body of make_spec() in gemm_cf32_emit.c.
import sys

ARCH = "gfx950"

_H16 = dict(src="same", recipe="cvt", out_f32=True)
_X6 = dict(src="f32", recipe="x6", out_f32=True, split_at="use", psub=True)
_CF = dict(_X6, cplx_b=True)
# the non-ring bodies: no ring levers
_FLAT = dict(
    dtl_stages=0,
    dtl_frag_prefetch=False,
    mfma_interleave=False,
    ilv_merge=False,
    ilv_barrier_after=0,
    lds_swizzle_r=-1,
    dtl_hoist=False,
    interwave_setprio=False,
)


def _pack_configs():
    """make_rocke_pack.py kernel_descriptors(): the GEMM family."""
    bodies = {
        "h16": ("RIW", "RIW_SK", "C3", "Q3_SK"),
        "x6": ("RIW", "RIW_SK", "C3"),
        "cf32": ("RIW", "RIW_SK", "Q3_SK"),
    }
    out = []
    for route, bs in bodies.items():
        dts = ("fp16", "bf16") if route == "h16" else ("bf16",)
        pro = {"h16": _H16, "x6": _X6, "cf32": _CF}[route]
        for dt in dts:
            for body in bs:
                for batched in (False, True):
                    out.append(
                        dict(
                            name=f"pk_{route}_{body.lower()}",
                            body=body,
                            dtype=dt,
                            batched=batched,
                            pro=pro,
                        )
                    )
            out.append(
                dict(
                    name=f"pk_{route}_riw_w64",
                    body="RIW",
                    dtype=dt,
                    wide=True,
                    pro=pro,
                )
            )
    return out


def _extra_configs():
    out = []
    for body in ("RIW", "C3"):
        for recipe in ("x3", "x4", "x9"):
            out.append(
                dict(
                    name=f"ex_{recipe}_{body.lower()}",
                    body=body,
                    dtype="bf16",
                    pro=dict(_X6, recipe=recipe),
                )
            )
        out.append(
            dict(
                name=f"ex_load_{body.lower()}",
                body=body,
                dtype="bf16",
                pro=dict(src="f32", recipe="x6", out_f32=True),
            )
        )
    out += [
        dict(
            name="ex_bfcap",
            body="RIW",
            dtype="bf16",
            pro=dict(_X6, order="big_first", cap=True),
        ),
        dict(
            name="ex_cvtf32",
            body="C3",
            dtype="bf16",
            pro=dict(src="f32", recipe="cvt", out_f32=True),
        ),
        dict(name="ex_pinv", body="C3", dtype="bf16", pro=dict(_X6, pin_v=True)),
        dict(name="ex_agprout", body="RIW", dtype="bf16", pro=dict(_X6, agpr_out=True)),
        dict(
            name="ex_agprout_sk",
            body="RIW_SK",
            dtype="bf16",
            pro=dict(_X6, agpr_out=True),
        ),
        dict(
            name="ex_noof32",
            body="RIW",
            dtype="fp16",
            pro=dict(src="same", recipe="cvt"),
        ),
        dict(name="ex_x6_nof32", body="C3", dtype="bf16", pro=dict(_X6, out_f32=False)),
    ]
    for body in ("RIW", "Q3_SK", "RIW_SK"):
        for ca, cb in ((True, False), (False, True), (True, True)):
            out.append(
                dict(
                    name=f"ex_cj{int(ca)}{int(cb)}_{body.lower()}",
                    body=body,
                    dtype="bf16",
                    pro=dict(_CF, conj_a=ca, conj_b=cb),
                )
            )
    out += [
        # the levers without a prologue (16-bit output through the vector epilogue)
        dict(name="lv_riw", body="RIW", dtype="bf16"),
        dict(name="lv_riw_fp16_bat", body="RIW", dtype="fp16", batched=True),
        dict(name="lv_riw_sk", body="RIW_SK", dtype="fp16"),
        dict(name="lv_riw_w64", body="RIW", dtype="bf16", wide=True),
        dict(
            name="lv_riw_nt0", body="RIW", dtype="bf16", trait=dict(rve_store_nt=False)
        ),
        # the row interleave without its vector epilogue (permuted DTL rows)
        dict(
            name="lv_rilv_scatter",
            body="RIW",
            dtype="bf16",
            trait=dict(rilv_vec_epilogue=False, rve_store_nt=False),
        ),
        dict(
            name="lv_rilv_scatter_x6",
            body="RIW",
            dtype="bf16",
            trait=dict(rilv_vec_epilogue=False, rve_store_nt=False),
            pro=_X6,
        ),
        dict(
            name="lv_rilv_pad32",
            body="RIW",
            dtype="bf16",
            trait=dict(rilv_pad=32),
            pro=_X6,
        ),
        dict(
            name="lv_rilv_pad64_sk",
            body="RIW_SK",
            dtype="fp16",
            trait=dict(rilv_pad=64),
        ),
        # dtl_soff0 on the swizzled interleaved ring (no 64-bit base)
        dict(name="lv_soff0_c3", body="C3", dtype="bf16", trait=dict(dtl_soff0=True)),
        dict(
            name="lv_soff0_c3_x6",
            body="C3",
            dtype="bf16",
            trait=dict(dtl_soff0=True),
            pro=_X6,
        ),
        dict(
            name="lv_soff0_riw",
            body="RIW",
            dtype="bf16",
            trait=dict(dtl_soff0=True),
            pro=_CF,
        ),
        # the 64-bit-address body with tight resources instead of dtl_soff0
        dict(
            name="lv_w64_tight",
            body="RIW",
            dtype="bf16",
            wide=True,
            trait=dict(dtl_soff0=False, dtl_tight_rsrc=True),
            pro=_X6,
        ),
        # the non-ring DTL body (dtl_prefetch, no stages) and the VGPR body
        dict(name="lv_flat_dtl_x6", body="C3", dtype="bf16", trait=_FLAT, pro=_X6),
        dict(name="lv_flat_dtl_cf", body="C3", dtype="bf16", trait=_FLAT, pro=_CF),
        dict(
            name="lv_vgpr_x6",
            body="C3",
            dtype="bf16",
            trait=dict(_FLAT, direct_to_lds=False, dtl_prefetch=False, dtl_cache_b=2),
            pro=_X6,
        ),
        dict(
            name="lv_vgpr_h16",
            body="C3",
            dtype="fp16",
            trait=dict(_FLAT, direct_to_lds=False, dtl_prefetch=False, dtl_cache_b=2),
            pro=_H16,
        ),
    ]
    return out


CONFIGS = _pack_configs() + _extra_configs()


def _spec(idx: int):
    from dataclasses import replace

    from rocke.dispatch.gemm.cf32 import _adjust_k, cf32_body
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
        batched=bool(c.get("batched", False)),
    )
    if c.get("wide"):
        spec = _adjust_k(spec, tile.tile_m, tile.tile_n, True)
    if c.get("trait"):
        spec = replace(spec, trait=replace(spec.trait, **c["trait"]))
    ok, why = is_valid_spec(spec, arch=ARCH)
    if not ok:
        sys.stderr.write(f"invalid spec: {why}\n")
        sys.exit(1)
    if c.get("pro") is not None:
        try:
            attach_prologue(spec, Prologue(**c["pro"]))
        except ValueError as e:  # pragma: no cover - not a refused form today
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
    """The case list of make_spec() in gemm_cf32_emit.c."""
    lines = []
    for i, c in enumerate(CONFIGS):
        lines.append(f"    case {i}:")
        lines.append(
            f'        base(spec, "{c["name"]}", "{c["body"]}", "{c["dtype"]}", '
            f'{_c_value(bool(c.get("batched")))}, {_c_value(bool(c.get("wide")))});'
        )
        for k, v in (c.get("trait") or {}).items():
            lines.append(f"        spec->trait.{k} = {_c_value(v)};")
        if c.get("pro") is not None:
            lines.append("        *has_pro = true;")
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
        usage="usage: gemm_cf32_emit.py <config_index> [ll|ir|verify]\n",
    )


if __name__ == "__main__":
    sys.exit(main())
