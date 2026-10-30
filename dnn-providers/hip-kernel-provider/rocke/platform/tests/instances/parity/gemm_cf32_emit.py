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
# Then the gfx942 configs (key "arch"): the pack's gfx942 GEMM descriptors (the
# register-staged bodies of every route, batched, the 64-bit-address body) and
# the recipes, conjugation and lever forms around them.
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
    """The GEMM family of hipTensor's rocKE kernel pack (its pack writers' descriptor tables)."""
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


# the gfx942 pack's bodies per route (make_rocke_pack.py GEMM_BODIES["gfx942"]: x3 / x4 / x9 on
# x6's, cf32x3 / cf32x4 / cf32x9 on cf32's) and its 64-bit-address bodies (GEMM_WIDE_BODY)
_G942 = {
    "h16": (
        "G64x64x128L3",
        "G128x128x128L3",
        "G64x32x128U",
        "G192x128x64L3",
        "G256x256x64L3",
        "G32x32x128U",
        "G256x224x64L3",
    ),
    "x6": (
        "G64x64x128L3",
        "G32x32x128L3",
        "G128x128x128L3",
        "G192x128x64L1S3",
        "G256x224x64L1",
        "G128x64x64",
        "G64x32x128U",
    ),
    "cf32": (
        "G64x64x128L3",
        "G32x32x128L3",
        "G128x128x128L3",
        "G192x128x64US3",
        "G256x224x64L1",
        "G128x64x64U",
        "G64x32x128U",
    ),
}
_G942_WIDE = {"h16": "G256x224x64L3", "x6": "G256x224x64L1", "cf32": "G256x224x64L1"}
# the gfx942 split of x6 / x9 / cf32 (rocke.dispatch.gemm.cf32.CF32_SPLIT_MODE)
_X6T = dict(_X6, split_mode="trunc")
_CFT = dict(_CF, split_mode="trunc")


def _g942_configs():
    """The gfx942 pack's GEMM descriptors of the h16 / x6 / cf32 routes, then
    the other recipes, conjugation, the bodies without a prologue and the
    batched 64-bit-address forms."""
    out = []
    for route in ("h16", "x6", "cf32"):
        dts = ("fp16", "bf16") if route == "h16" else ("bf16",)
        pro = {"h16": _H16, "x6": _X6T, "cf32": _CFT}[route]
        for dt in dts:
            for body in _G942[route]:
                for batched in (False, True):
                    out.append(
                        dict(
                            name=f"pk942_{route}_{body.lower()}",
                            body=body,
                            dtype=dt,
                            batched=batched,
                            pro=pro,
                        )
                    )
            out.append(
                dict(
                    name=f"pk942_{route}_{_G942_WIDE[route].lower()}_w64",
                    body=_G942_WIDE[route],
                    dtype=dt,
                    wide=True,
                    pro=pro,
                )
            )
    # the other recipes: x9 truncated, x3 / x4 rounded (as the pack builds them)
    for body in ("G128x128x128L3", "G64x32x128U"):
        for recipe, pro in (
            ("x9", dict(_X6T, recipe="x9")),
            ("x3", dict(_X6, recipe="x3")),
            ("x4", dict(_X6, recipe="x4")),
        ):
            out.append(
                dict(
                    name=f"ex942_{recipe}_{body.lower()}",
                    body=body,
                    dtype="bf16",
                    pro=pro,
                )
            )
        out.append(
            dict(
                name=f"ex942_cf32x3_{body.lower()}",
                body=body,
                dtype="bf16",
                pro=dict(_CF, recipe="x3"),
            )
        )
    for body in ("G256x224x64L1", "G32x32x128L3"):
        for ca, cb in ((True, False), (False, True), (True, True)):
            out.append(
                dict(
                    name=f"ex942_cj{int(ca)}{int(cb)}_{body.lower()}",
                    body=body,
                    dtype="bf16",
                    pro=dict(_CFT, conj_a=ca, conj_b=cb),
                )
            )
    out += [
        # the bodies without a prologue (16-bit output)
        dict(name="lv942_g192x128x64l1s3", body="G192x128x64L1S3", dtype="bf16"),
        dict(
            name="lv942_g128x64x64u_bat", body="G128x64x64U", dtype="fp16", batched=True
        ),
        dict(
            name="lv942_g256x256x64l3_w64",
            body="G256x256x64L3",
            dtype="bf16",
            wide=True,
        ),
        # the 64-bit-address form of a batched launch (64-bit batch origins)
        dict(
            name="lv942_g256x224x64l1_w64_bat_x6",
            body="G256x224x64L1",
            dtype="bf16",
            batched=True,
            wide=True,
            pro=_X6T,
        ),
        dict(
            name="lv942_g64x64x128l3_w64_bat_cf",
            body="G64x64x128L3",
            dtype="bf16",
            batched=True,
            wide=True,
            pro=_CFT,
        ),
        # the rounded split on a truncated route's body (rocKE's dispatch default elsewhere)
        dict(name="lv942_g128x64x64_rne", body="G128x64x64", dtype="bf16", pro=_X6),
    ]
    return [dict(c, arch="gfx942") for c in out]


CONFIGS = _pack_configs() + _extra_configs() + _g942_configs()


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
    arch = c.get("arch", ARCH)
    tile, trait, bs = cf32_body(c["body"], arch)
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
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        sys.stderr.write(f"invalid spec: {why}\n")
        sys.exit(1)
    if c.get("pro") is not None:
        try:
            attach_prologue(spec, Prologue(**c["pro"]))
        except ValueError as e:  # pragma: no cover - not a refused form today
            sys.stderr.write(f"prologue: {e}\n")
            sys.exit(1)
    return spec, arch


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
        if c.get("arch", ARCH) != ARCH:
            lines.append(f'        g_arch = "{c["arch"]}";')
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
