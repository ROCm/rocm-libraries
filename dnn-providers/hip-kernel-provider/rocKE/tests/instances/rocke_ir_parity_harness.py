#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from collections import Counter
from pathlib import Path


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe(name: str) -> str:
    return name.replace("/", "__").replace(":", "_") + ".ll"


def current_flavor() -> str:
    """The llvm flavor this host would autodetect (llvm20 for ROCm < 7.2,
    llvm22 otherwise). The golden stores both; the gate compares only this."""
    from rocke.core.lower_llvm import _resolve_llvm_flavor

    return _resolve_llvm_flavor()


def lower_case(case, flavor):
    # Pin the NATIVE python lowerer (not the backend-dispatched
    # lower_kernel_to_llvm, whose default 'cpp' path silently falls back to
    # python) and an EXPLICIT llvm flavor. The recorded sha is then a function
    # of the committed IR alone -- independent of whether rocke_engine is built or
    # which ROCm/comgr vintage the host happens to detect. (llvm20 vs llvm22
    # emit different datalayout/type encodings, so an unpinned flavor would make
    # the same code hash differently across hosts.)
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    kernel = case["build"]()
    llvm = _lower_kernel_to_llvm_python(kernel, arch=case["arch"], llvm_flavor=flavor)
    return {
        "status": "ok",
        "family": case["family"],
        "arch": case["arch"],
        "kernel_name": getattr(kernel, "name", "<unknown>"),
        "sha256": sha(llvm),
        "bytes": len(llvm.encode("utf-8")),
    }, llvm


def gemm_spec(
    name,
    arch,
    tile_m,
    tile_n,
    tile_k,
    warp_m,
    warp_n,
    wtm,
    wtn,
    wtk,
    pipeline,
    epilogue,
    wave_size,
):
    from rocke.instances.common.gemm_universal import (
        DataSpec,
        TileSpec,
        TraitSpec,
        UniversalGemmSpec,
    )

    return UniversalGemmSpec(
        name=name,
        tile=TileSpec(
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            warp_m=warp_m,
            warp_n=warp_n,
            warp_k=1,
            warp_tile_m=wtm,
            warp_tile_n=wtn,
            warp_tile_k=wtk,
        ),
        trait=TraitSpec(
            pipeline=pipeline,
            scheduler="intrawave",
            epilogue=epilogue,
            pad_m=True,
            pad_n=True,
            pad_k=True,
        ),
        data=DataSpec(dtype_a="fp16", dtype_b="fp16", dtype_c="fp16"),
        wave_size=wave_size,
    )


def build_gemm(name, arch, *args):
    def _build():
        from rocke.instances.common.gemm_universal import build_universal_gemm

        return build_universal_gemm(gemm_spec(name, arch, *args), arch=arch)

    return _build


def build_conv(
    name,
    arch,
    problem_args,
    *,
    wave_size,
    wtm,
    wtn,
    wtk,
    tile_m,
    tile_n,
    tile_k,
    pipeline="mem",
    epilogue="default",
    groups=1,
):
    def _build():
        from rocke.instances.common.conv_implicit_gemm import (
            ConvProblem,
            ImplicitGemmConvSpec,
            build_implicit_gemm_conv,
        )

        p = ConvProblem(*problem_args)
        spec = ImplicitGemmConvSpec(
            problem=p,
            name=name,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            warp_m=2,
            warp_n=1 if wave_size == 32 else 2,
            warp_tile_m=wtm,
            warp_tile_n=wtn,
            warp_tile_k=wtk,
            wave_size=wave_size,
            pipeline=pipeline,
            epilogue=epilogue,
            groups=groups,
        )
        return build_implicit_gemm_conv(spec, arch=arch)

    return _build


def attn_problem(**kw):
    from rocke.instances.common.attention_unified import UnifiedAttentionProblem

    return UnifiedAttentionProblem(**kw)


def build_attn_2d(name, pkw):
    def _build():
        from rocke.instances.common.attention_unified import (
            UnifiedAttention2DSpec,
            build_unified_attention_2d,
        )

        return build_unified_attention_2d(
            UnifiedAttention2DSpec(attn_problem(**pkw), name=name)
        )

    return _build


def build_attn_3d(name, pkw, segs):
    def _build():
        from rocke.instances.common.attention_unified import (
            UnifiedAttention3DSpec,
            build_unified_attention_3d,
        )

        return build_unified_attention_3d(
            UnifiedAttention3DSpec(attn_problem(**pkw), name=name, num_segments=segs)
        )

    return _build


def build_attn_reduce(name, pkw, segs):
    def _build():
        from rocke.instances.common.attention_unified import (
            UnifiedAttentionReduceSpec,
            build_unified_attention_reduce,
        )

        return build_unified_attention_reduce(
            UnifiedAttentionReduceSpec(
                attn_problem(**pkw), num_segments=segs, name=name
            )
        )

    return _build


def build_moe_sort(phase, tokens, topk, experts, block, arch):
    def _build():
        from rocke.instances.common.moe_sorting import (
            MoeSortingSpec,
            build_moe_sort_histogram,
            build_moe_sort_scan,
            build_moe_sort_scatter,
        )

        spec = MoeSortingSpec(
            tokens=tokens,
            topk=topk,
            experts=experts,
            block_size=block,
            name=f"irhash_moe_sort_{phase}",
        )
        return {
            "histogram": build_moe_sort_histogram,
            "scan": build_moe_sort_scan,
            "scatter": build_moe_sort_scatter,
        }[phase](spec, arch=arch)

    return _build


def build_fused_moe(phase, tokens, experts, topk, hidden, intermediate, dtype="f16"):
    def _build():
        from rocke.instances.common.fused_moe import (
            FusedMoeSpec,
            build_moe_gather,
            build_moe_silu_mul,
            build_moe_topk_weighted_reduce,
        )

        spec = FusedMoeSpec(
            tokens=tokens,
            experts=experts,
            topk=topk,
            hidden=hidden,
            intermediate=intermediate,
            dtype=dtype,
            block_size=128,
            vec=4,
            name=f"irhash_fused_moe_{phase}",
        )
        return {
            "gather": build_moe_gather,
            "silu": build_moe_silu_mul,
            "reduce": build_moe_topk_weighted_reduce,
        }[phase](spec)

    return _build


def build_deep(kind, arch, **kw):
    def _build():
        if kind == "common":
            from rocke.instances.common.deep_fused_conv_pool import (
                build_deep_fused_conv_pool,
                make_deep_fused_conv_pool_spec,
            )
        elif kind == "gfx950":
            from rocke.instances.gfx950.deep_fused_conv_pool import (
                build_deep_fused_conv_pool,
                make_deep_fused_conv_pool_spec,
            )
        elif kind == "gfx1201":
            from rocke.instances.gfx1201.deep_fused_conv_pool import (
                build_deep_fused_conv_pool,
                make_deep_fused_conv_pool_spec,
            )
        else:
            from rocke.instances.gfx1151.deep_fused_conv_pool import (
                build_deep_fused_conv_pool,
                make_deep_fused_conv_pool_spec,
            )
        return build_deep_fused_conv_pool(
            make_deep_fused_conv_pool_spec(**kw), arch=arch
        )

    return _build


def cases():
    out = []

    def add(family, case_id, arch, build):
        out.append({"family": family, "case_id": case_id, "arch": arch, "build": build})

    # GEMM: tile/atom/pipeline variants across CDNA and RDNA arches.
    add(
        "gemm",
        "gemm/gfx942/t64x64x16",
        "gfx942",
        build_gemm(
            "irhash_gemm_942_a",
            "gfx942",
            64,
            64,
            16,
            2,
            2,
            16,
            16,
            16,
            "mem",
            "default",
            64,
        ),
    )
    add(
        "gemm",
        "gemm/gfx942/t128x128x16",
        "gfx942",
        build_gemm(
            "irhash_gemm_942_b",
            "gfx942",
            128,
            128,
            16,
            2,
            2,
            32,
            32,
            8,
            "compv4",
            "cshuffle",
            64,
        ),
    )
    add(
        "gemm",
        "gemm/gfx950/t128x128x32",
        "gfx950",
        build_gemm(
            "irhash_gemm_950_a",
            "gfx950",
            128,
            128,
            32,
            2,
            2,
            32,
            32,
            16,
            "compv4",
            "cshuffle",
            64,
        ),
    )
    add(
        "gemm",
        "gemm/gfx950/t64x128x32",
        "gfx950",
        build_gemm(
            "irhash_gemm_950_b",
            "gfx950",
            64,
            128,
            32,
            2,
            2,
            16,
            16,
            32,
            "mem",
            "default",
            64,
        ),
    )
    add(
        "gemm",
        "gemm/gfx1151/t32x32x16",
        "gfx1151",
        build_gemm(
            "irhash_gemm_1151_a",
            "gfx1151",
            32,
            32,
            16,
            2,
            2,
            16,
            16,
            16,
            "mem",
            "default",
            32,
        ),
    )
    add(
        "gemm",
        "gemm/gfx1151/t64x32x16",
        "gfx1151",
        build_gemm(
            "irhash_gemm_1151_b",
            "gfx1151",
            64,
            32,
            16,
            2,
            1,
            16,
            16,
            16,
            "mem",
            "default",
            32,
        ),
    )
    add(
        "gemm",
        "gemm/gfx1201/t32x32x16",
        "gfx1201",
        build_gemm(
            "irhash_gemm_1201_a",
            "gfx1201",
            32,
            32,
            16,
            2,
            2,
            16,
            16,
            16,
            "mem",
            "default",
            32,
        ),
    )
    add(
        "gemm",
        "gemm/gfx1201/t64x32x16",
        "gfx1201",
        build_gemm(
            "irhash_gemm_1201_b",
            "gfx1201",
            64,
            32,
            16,
            2,
            1,
            16,
            16,
            16,
            "mem",
            "default",
            32,
        ),
    )
    # gfx90a is wave64/MFMA like gfx942; mirror its 16x16x16 atom variants.
    add(
        "gemm",
        "gemm/gfx90a/t64x64x16",
        "gfx90a",
        build_gemm(
            "irhash_gemm_90a_a",
            "gfx90a",
            64,
            64,
            16,
            2,
            2,
            16,
            16,
            16,
            "mem",
            "default",
            64,
        ),
    )
    add(
        "gemm",
        "gemm/gfx90a/t128x128x16",
        "gfx90a",
        build_gemm(
            "irhash_gemm_90a_b",
            "gfx90a",
            128,
            128,
            16,
            2,
            2,
            32,
            32,
            8,
            "compv4",
            "cshuffle",
            64,
        ),
    )
    # gfx1250 is wave32/WMMA like gfx1201, but its fp16 atom is K=32 (16x16x32),
    # so warp_tile_k and tile_k are 32 rather than 16. NOTE: these lower through
    # the Python engine only -- the C++ engine has no gfx1250 ISA backend yet
    # (rocke_ll_backend_for rejects gfx1250; see Cpp/core/lower_llvm/mma.cpp), so
    # they are not part of the C-vs-Python byte-identity gate until that lands.
    add(
        "gemm",
        "gemm/gfx1250/t32x32x32",
        "gfx1250",
        build_gemm(
            "irhash_gemm_1250_a",
            "gfx1250",
            32,
            32,
            32,
            2,
            2,
            16,
            16,
            32,
            "mem",
            "default",
            32,
        ),
    )
    add(
        "gemm",
        "gemm/gfx1250/t64x32x32",
        "gfx1250",
        build_gemm(
            "irhash_gemm_1250_b",
            "gfx1250",
            64,
            32,
            32,
            2,
            1,
            16,
            16,
            32,
            "mem",
            "default",
            32,
        ),
    )

    # Conv: problem-shape and arch variants.
    conv1 = (1, 8, 8, 16, 32, 3, 3, 1, 1, 1, 1, 1, 1)
    conv2 = (2, 16, 16, 32, 32, 1, 1, 1, 1, 0, 0, 1, 1)
    add(
        "conv",
        "conv/gfx942/n1h8c16k32r3",
        "gfx942",
        build_conv(
            "irhash_conv_942_a",
            "gfx942",
            conv1,
            wave_size=64,
            wtm=16,
            wtn=16,
            wtk=16,
            tile_m=64,
            tile_n=32,
            tile_k=16,
        ),
    )
    add(
        "conv",
        "conv/gfx950/n1h8c16k32r3",
        "gfx950",
        build_conv(
            "irhash_conv_950_a",
            "gfx950",
            conv1,
            wave_size=64,
            wtm=32,
            wtn=32,
            wtk=16,
            tile_m=64,
            tile_n=64,
            tile_k=32,
            pipeline="compv4",
            epilogue="cshuffle",
        ),
    )
    add(
        "conv",
        "conv/gfx950/n2h16c32k32r1",
        "gfx950",
        build_conv(
            "irhash_conv_950_b",
            "gfx950",
            conv2,
            wave_size=64,
            wtm=16,
            wtn=16,
            wtk=16,
            tile_m=64,
            tile_n=32,
            tile_k=16,
        ),
    )
    add(
        "conv",
        "conv/gfx1151/n1h8c16k32r3",
        "gfx1151",
        build_conv(
            "irhash_conv_1151_a",
            "gfx1151",
            conv1,
            wave_size=32,
            wtm=16,
            wtn=16,
            wtk=16,
            tile_m=32,
            tile_n=32,
            tile_k=16,
        ),
    )
    add(
        "conv",
        "conv/gfx1151/n2h16c32k32r1",
        "gfx1151",
        build_conv(
            "irhash_conv_1151_b",
            "gfx1151",
            conv2,
            wave_size=32,
            wtm=16,
            wtn=16,
            wtk=16,
            tile_m=32,
            tile_n=32,
            tile_k=16,
        ),
    )
    add(
        "conv",
        "conv/gfx1201/n1h8c16k32r3",
        "gfx1201",
        build_conv(
            "irhash_conv_1201_a",
            "gfx1201",
            conv1,
            wave_size=32,
            wtm=16,
            wtn=16,
            wtk=16,
            tile_m=32,
            tile_n=32,
            tile_k=16,
        ),
    )
    # gfx90a conv mirrors the gfx942 MFMA path (wave64, 16x16x16 atom). gfx1250
    # has no conv case: WMMA conv requires the 16x16x16 atom, but gfx1250's fp16
    # warp_tile is 16x16x32, so the two constraints are mutually exclusive.
    add(
        "conv",
        "conv/gfx90a/n1h8c16k32r3",
        "gfx90a",
        build_conv(
            "irhash_conv_90a_a",
            "gfx90a",
            conv1,
            wave_size=64,
            wtm=16,
            wtn=16,
            wtk=16,
            tile_m=64,
            tile_n=32,
            tile_k=16,
        ),
    )

    # Unified attention: scalar 2D/3D/reduce variants lowered for multiple arches.
    p_decode = dict(
        total_q=4,
        num_seqs=4,
        num_query_heads=4,
        num_kv_heads=2,
        head_size=64,
        block_size=16,
        max_seqlen_q=1,
        max_seqlen_k=64,
        dtype="fp16",
    )
    p_prefill = dict(
        total_q=64,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=2,
        head_size=128,
        block_size=16,
        max_seqlen_q=32,
        max_seqlen_k=128,
        dtype="bf16",
        sliding_window=32,
        softcap=10.0,
        use_sinks=True,
    )
    add(
        "unified_attention",
        "ua/gfx942/2d_decode_fp16",
        "gfx942",
        build_attn_2d("irhash_ua_2d_decode", p_decode),
    )
    add(
        "unified_attention",
        "ua/gfx950/2d_prefill_bf16_sw",
        "gfx950",
        build_attn_2d("irhash_ua_2d_prefill", p_prefill),
    )
    add(
        "unified_attention",
        "ua/gfx1151/2d_decode_fp16",
        "gfx1151",
        build_attn_2d("irhash_ua_2d_decode", p_decode),
    )
    add(
        "unified_attention",
        "ua/gfx950/3d_prefill",
        "gfx950",
        build_attn_3d("irhash_ua_3d_prefill", p_prefill, 8),
    )
    add(
        "unified_attention",
        "ua/gfx942/reduce_prefill",
        "gfx942",
        build_attn_reduce("irhash_ua_reduce_prefill", p_prefill, 8),
    )
    add(
        "unified_attention",
        "ua/gfx1201/reduce_decode",
        "gfx1201",
        build_attn_reduce("irhash_ua_reduce_decode", p_decode, 4),
    )

    # MoE: sorting phases and fused-MoE streaming phases.
    for arch in ("gfx942", "gfx950", "gfx1151", "gfx1201"):
        add(
            "moe",
            f"moe_sort/{arch}/hist_t32_k2_e8",
            arch,
            build_moe_sort("histogram", 32, 2, 8, 128, arch),
        )
    add(
        "moe",
        "moe_sort/gfx950/scan_t64_k2_e16",
        "gfx950",
        build_moe_sort("scan", 64, 2, 16, 128, "gfx950"),
    )
    add(
        "moe",
        "moe_sort/gfx950/scatter_t64_k2_e16",
        "gfx950",
        build_moe_sort("scatter", 64, 2, 16, 128, "gfx950"),
    )
    add(
        "moe",
        "fused_moe/gfx950/gather_h128",
        "gfx950",
        build_fused_moe("gather", 32, 8, 2, 128, 256),
    )
    add(
        "moe",
        "fused_moe/gfx950/silu_i256",
        "gfx950",
        build_fused_moe("silu", 32, 8, 2, 128, 256),
    )
    add(
        "moe",
        "fused_moe/gfx950/reduce_h128",
        "gfx950",
        build_fused_moe("reduce", 32, 8, 2, 128, 256),
    )

    # Deep fused conv: arch shims and shape/toggle samples.
    add(
        "deep_fused_conv",
        "deep/gfx950/h16w16_k32",
        "gfx950",
        build_deep(
            "gfx950",
            "gfx950",
            h=16,
            w=16,
            c=16,
            k0=32,
            k1=32,
            pool_tile_h=4,
            pool_tile_w=4,
        ),
    )
    add(
        "deep_fused_conv",
        "deep/gfx950/h24w24_k32",
        "gfx950",
        build_deep(
            "gfx950",
            "gfx950",
            h=24,
            w=24,
            c=16,
            k0=32,
            k1=32,
            pool_tile_h=4,
            pool_tile_w=4,
        ),
    )
    add(
        "deep_fused_conv",
        "deep/common_gfx950_wt16/h16w16_k32",
        "gfx950",
        build_deep(
            "common",
            "gfx950",
            name="irhash_deep_gfx950_wt16",
            wave_size=64,
            warp_tile_m=16,
            warp_tile_n=16,
            h=16,
            w=16,
            c=16,
            k0=32,
            k1=32,
            pool_tile_h=4,
            pool_tile_w=4,
        ),
    )
    add(
        "deep_fused_conv",
        "deep/common_gfx950_wt16/h24w24_k32",
        "gfx950",
        build_deep(
            "common",
            "gfx950",
            name="irhash_deep_gfx950_wt16b",
            wave_size=64,
            warp_tile_m=16,
            warp_tile_n=16,
            h=24,
            w=24,
            c=16,
            k0=32,
            k1=32,
            pool_tile_h=4,
            pool_tile_w=4,
        ),
    )
    add(
        "deep_fused_conv",
        "deep/gfx1201/h16w16_k32",
        "gfx1201",
        build_deep(
            "gfx1201",
            "gfx1201",
            h=16,
            w=16,
            c=16,
            k0=32,
            k1=32,
            pool_tile_h=4,
            pool_tile_w=4,
        ),
    )
    add(
        "deep_fused_conv",
        "deep/gfx1201/h24w24_k32",
        "gfx1201",
        build_deep(
            "gfx1201",
            "gfx1201",
            h=24,
            w=24,
            c=16,
            k0=32,
            k1=32,
            pool_tile_h=4,
            pool_tile_w=4,
        ),
    )
    add(
        "deep_fused_conv",
        "deep/gfx1151/native_h16w16",
        "gfx1151",
        build_deep(
            "gfx1151",
            "gfx1151",
            h=16,
            w=16,
            c=16,
            k0=32,
            k1=32,
            pool_tile_h=4,
            pool_tile_w=4,
            native_int=True,
        ),
    )
    add(
        "deep_fused_conv",
        "deep/gfx1151/native_h24w24",
        "gfx1151",
        build_deep(
            "gfx1151",
            "gfx1151",
            h=24,
            w=24,
            c=16,
            k0=32,
            k1=32,
            pool_tile_h=4,
            pool_tile_w=4,
            native_int=True,
        ),
    )
    add(
        "deep_fused_conv",
        "deep/gfx11_generic/native_h16w16",
        "gfx11-generic",
        build_deep(
            "gfx1151",
            "gfx11-generic",
            h=16,
            w=16,
            c=16,
            k0=32,
            k1=32,
            pool_tile_h=4,
            pool_tile_w=4,
            native_int=True,
        ),
    )
    return out


# The golden stores one sub-document per llvm flavor under this key; the gate
# (check_golden) compares only the flavor the running host autodetects, so the
# same committed golden is valid on both ROCm < 7.2 (llvm20) and >= 7.2 (llvm22).
GOLDEN_FLAVORS = ("llvm20", "llvm22")
GOLDEN_SCHEMA = "ck.dsl.ir_golden_sha256/v2"


def run(ir_dir: Path | None = None, *, flavor: str):
    results = {}
    failures = {}
    if ir_dir:
        ir_dir.mkdir(parents=True, exist_ok=True)
    for case in cases():
        cid = case["case_id"]
        try:
            rec, llvm = lower_case(case, flavor)
            results[cid] = rec
            if ir_dir:
                (ir_dir / safe(cid)).write_text(llvm)
        except Exception as e:
            failures[cid] = {
                "type": type(e).__name__,
                "message": str(e),
                "trace": traceback.format_exc(limit=5),
            }
    return {
        "summary": {
            "ok_cases": len(results),
            "expected_failures": len(failures),
            "families": dict(
                sorted(Counter(v["family"] for v in results.values()).items())
            ),
            "arches": dict(
                sorted(Counter(v["arch"] for v in results.values()).items())
            ),
        },
        "cases": results,
        "expected_failures": {
            k: {"type": v["type"], "message": v["message"]} for k, v in failures.items()
        },
    }


def build_golden() -> dict:
    """Run every case under each golden flavor and return the flavor-keyed doc."""
    return {
        "schema": GOLDEN_SCHEMA,
        "flavors": {fl: run(flavor=fl) for fl in GOLDEN_FLAVORS},
    }


def check_golden(golden_path: Path, flavor: str | None = None) -> list[str]:
    """Compare a fresh run against the golden sub-doc for ``flavor`` (defaults to
    the host's autodetected flavor). Returns a list of drift strings; empty == OK.
    """
    flavor = flavor or current_flavor()
    doc = json.loads(golden_path.read_text())
    base = doc.get("flavors", {}).get(flavor)
    if base is None:
        have = sorted(doc.get("flavors", {}))
        return [f"golden has no entry for flavor {flavor!r} (have {have})"]
    return compare(base, run(flavor=flavor))


def compare(base, cur):
    errors = []
    for section in ("cases", "expected_failures"):
        bkeys = set(base.get(section, {}))
        ckeys = set(cur.get(section, {}))
        for missing in sorted(bkeys - ckeys):
            errors.append(f"{section}: missing current {missing}")
        for new in sorted(ckeys - bkeys):
            errors.append(f"{section}: new current {new}")
    for cid, brec in sorted(base.get("cases", {}).items()):
        crec = cur.get("cases", {}).get(cid)
        if not crec:
            continue
        if brec.get("sha256") != crec.get("sha256"):
            errors.append(f"{cid}: {brec.get('sha256')} -> {crec.get('sha256')}")
    for cid, brec in sorted(base.get("expected_failures", {}).items()):
        crec = cur.get("expected_failures", {}).get(cid)
        if not crec:
            continue
        if brec.get("type") != crec.get("type") or brec.get("message") != crec.get(
            "message"
        ):
            errors.append(f"{cid}: failure changed {brec} -> {crec}")
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--write",
        type=Path,
        help="(re-)bless the flavor-keyed golden: run every case under each of "
        f"{GOLDEN_FLAVORS} and write the result. Run only from a verified-good "
        "tree; re-blessing should accompany a reviewed, expected output change.",
    )
    ap.add_argument(
        "--check",
        type=Path,
        help="compare a fresh run against the golden sub-doc for THIS host's "
        "autodetected llvm flavor; exit 1 on any drift.",
    )
    ap.add_argument(
        "--flavor",
        choices=GOLDEN_FLAVORS,
        help="override the autodetected llvm flavor (for --check / --ir-dir).",
    )
    ap.add_argument("--ir-dir", type=Path)
    ns = ap.parse_args()

    if ns.write:
        doc = build_golden()
        ns.write.parent.mkdir(parents=True, exist_ok=True)
        ns.write.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
        for fl, sub in doc["flavors"].items():
            print(
                f"wrote {fl}: {len(sub['cases'])} ok, "
                f"{len(sub['expected_failures'])} failures"
            )
        print(f"-> {ns.write}")
        return

    if ns.check:
        flavor = ns.flavor or current_flavor()
        errors = check_golden(ns.check, flavor)
        if errors:
            for e in errors:
                print(f"IR DRIFT [{flavor}]: {e}")
            raise SystemExit(1)
        print(f"IR parity OK [{flavor}] vs {ns.check}")
        return

    flavor = ns.flavor or current_flavor()
    doc = run(ns.ir_dir, flavor=flavor)
    print(json.dumps(doc, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
