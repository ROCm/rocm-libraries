#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import contextlib
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
    llvm22 for 7.2-7.12, llvm23 for 7.13+). The golden stores all of them; the
    gate compares only this one."""
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
    cshuffle_no_alias=False,
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
            cshuffle_no_alias=cshuffle_no_alias,
        ),
        data=DataSpec(dtype_a="fp16", dtype_b="fp16", dtype_c="fp16"),
        wave_size=wave_size,
    )


def build_gemm(name, arch, *args, **kwargs):
    def _build():
        from rocke.instances.common.gemm_universal import build_universal_gemm

        return build_universal_gemm(gemm_spec(name, arch, *args, **kwargs), arch=arch)

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
    vector_size_c=None,
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
            vector_size_c=vector_size_c,
        )
        return build_implicit_gemm_conv(spec, arch=arch)

    return _build


def build_conv_wgrad(
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
    split_k=1,
    dtype_d="fp16",
):
    def _build():
        from rocke.instances.common.conv_implicit_gemm_wgrad import (
            WgradConvSpec,
            build_implicit_gemm_conv_wgrad,
        )
        from rocke.instances.common._conv_implicit_gemm_common import (
            ConvDataSpec,
            ConvProblem,
        )

        p = ConvProblem(*problem_args)
        spec = WgradConvSpec(
            problem=p,
            name=name,
            data=ConvDataSpec(dtype_a="fp16", dtype_b="fp16", dtype_d=dtype_d),
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
            split_k=split_k,
        )
        return build_implicit_gemm_conv_wgrad(spec, arch=arch)

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


def build_attention_2d(
    name,
    arch,
    *,
    head_size,
    block_size,
    num_query_heads,
    num_kv_heads,
    dtype,
    sliding_window=0,
    has_softcap=False,
    use_alibi=False,
    **kw,
):
    def _build():
        from kernels.common.attention_unified import _tiled_2d_impl

        SpecCls, _build_fn, _ = _tiled_2d_impl(arch)
        spec = SpecCls(
            head_size=head_size,
            block_size=block_size,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            dtype=dtype,
            use_sinks=False,
            sliding_window=sliding_window,
            has_softcap=has_softcap,
            use_alibi=use_alibi,
            **kw,
        )
        return _build_fn(spec, arch=arch)

    return _build


def build_attention_3d(
    name,
    arch,
    *,
    head_size,
    block_size,
    num_query_heads,
    num_kv_heads,
    dtype,
    num_segments,
    sliding_window=0,
    has_softcap=False,
):
    def _build():
        from kernels import (
            UnifiedAttention3DTiledSpec,
            build_unified_attention_3d_tiled,
        )

        spec = UnifiedAttention3DTiledSpec(
            head_size=head_size,
            block_size=block_size,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            dtype=dtype,
            use_sinks=False,
            sliding_window=sliding_window,
            has_softcap=has_softcap,
            num_segments=num_segments,
        )
        return build_unified_attention_3d_tiled(spec)

    return _build


def build_attention_reduce(
    name, arch, *, head_size, num_query_heads, num_kv_heads, dtype, num_segments
):
    def _build():
        from kernels import (
            UnifiedAttentionReduceTiledSpec,
            build_unified_attention_reduce_tiled,
        )

        spec = UnifiedAttentionReduceTiledSpec(
            head_size=head_size,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            dtype=dtype,
            num_segments=num_segments,
        )
        return build_unified_attention_reduce_tiled(spec)

    return _build


def build_attention_dense(arch, **over):
    """Dense flash-attn prefill spec (library ``kernels/gfx950/attention_dense``).

    ``over`` patches the shared base spec; a small Sq keeps the IR compact while
    still exercising the full pipeline (both the default one-CTA-per-q-block grid
    and the persistent grid-stride grid).
    """

    def _build():
        from kernels.gfx950.attention_dense import (
            AttentionDenseSpec,
            build_attention_dense as _build_dense,
        )

        spec = dict(
            batch=1,
            seqlen_q=512,
            seqlen_kv=512,
            num_query_heads=128,
            num_kv_heads=8,
            head_size=128,
            causal=True,
            dtype="bf16",
        )
        spec.update(over)
        return _build_dense(AttentionDenseSpec(**spec))

    return _build


def _d256_problem():
    """Validated D256 cohort point (GQA 16/2, hd256, bs16, sq4096 bf16)."""
    from kernels.common.attention_unified import UnifiedAttentionProblem

    return UnifiedAttentionProblem(
        total_q=4096,
        num_seqs=1,
        num_query_heads=16,
        num_kv_heads=2,
        head_size=256,
        block_size=16,
        max_seqlen_q=4096,
        max_seqlen_k=4096,
        dtype="bf16",
        num_sms=120,
    )


@contextlib.contextmanager
def _pinned_attention_arch(arch):
    """Pin the *memoized runtime device arch* the D256 fast routes gate on.

    Those routes consult ``_resolve_attention_arch`` -- NOT any per-case argument
    -- and the spec builder holds a *bound* import of it, so both bindings are
    patched wholesale (per its docstring: "Tests that monkeypatch this function
    replace it wholesale"). Without this a non-gfx950 host silently selects the
    FALLBACK spec and the recorded sha becomes host-dependent.
    """
    import builders.common.attention_spec_builder as asb
    import kernels.common.attention_unified as au

    pin = lambda: arch  # noqa: E731
    o_au, o_asb = au._resolve_attention_arch, asb._resolve_attention_arch
    au._resolve_attention_arch = pin
    asb._resolve_attention_arch = pin
    try:
        yield
    finally:
        au._resolve_attention_arch = o_au
        asb._resolve_attention_arch = o_asb


def build_attention_d256_gfx950(arch):
    """D256 bf16 prefill *fast* spec: 32x32-transposed + ``use_kq_lds_pad``
    (slab-granularity K_lds pad) + ``use_softmax_mfma_interleave``.

    The slab pad's async-DMA write and its padded read remap must agree
    byte-for-byte or numerics silently corrupt, and an addressing drift would pass
    every other CPU test and fail only on GPU (#9233 review). Raises if the fast
    spec / pad+interleave is not selected, so this can never pin the fallback.
    """

    def _build():
        from kernels import build_unified_attention_2d_tiled
        from kernels.common.attention_unified import (
            _d256_gfx950_fast,
            _tiled_spec_from_problem,
        )

        problem = _d256_problem()
        with _pinned_attention_arch(arch):
            if not _d256_gfx950_fast(problem):
                raise RuntimeError(
                    f"D256 fast spec not selected under pinned arch {arch!r}; "
                    "would pin the fallback (no pad/interleave)"
                )
            spec = _tiled_spec_from_problem(problem)
            if not (spec.use_kq_lds_pad and spec.use_softmax_mfma_interleave):
                raise RuntimeError(
                    "expected pad+interleave in the D256 fast spec; got "
                    f"pad={spec.use_kq_lds_pad} "
                    f"interleave={spec.use_softmax_mfma_interleave}"
                )
            return build_unified_attention_2d_tiled(spec, arch=arch)

    return _build


def build_attention_d256_gfx942(arch):
    """D256 gfx942 4-warp GQA fast path.

    Mirrors ``build_attention_d256_gfx950`` but pins the *gfx942* cohort: raises
    if ``_d256_gfx942_fast`` is not selected (so this can never pin the fallback),
    then lowers the dedicated ``build_gfx942_4warp_gqa`` builder -- the
    byte-sensitive V wide-load -> ``V_lds`` transpose kernel worth pinning.
    """

    def _build():
        from kernels.common.attention_unified import (
            _d256_gfx942_fast,
            _tiled_spec_from_problem,
        )
        from kernels.gfx942.attention_tiled_2d import build_gfx942_4warp_gqa

        problem = _d256_problem()
        with _pinned_attention_arch(arch):
            if not _d256_gfx942_fast(problem):
                raise RuntimeError(
                    f"D256 gfx942 fast route not selected under pinned arch "
                    f"{arch!r}; would pin the fallback (slow scalar path)"
                )
            spec = _tiled_spec_from_problem(problem)
            return build_gfx942_4warp_gqa(spec, arch=arch)

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
        "gemm/gfx950/t128x128x32/cshuffle_no_alias",
        "gfx950",
        build_gemm(
            "irhash_gemm_950_a_noalc",
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
            cshuffle_no_alias=True,
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
    # (rocke_ll_backend_for rejects gfx1250; see cpp/core/lower_llvm/mma.cpp), so
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
            epilogue="cshuffle",
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
            epilogue="cshuffle",
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
            epilogue="cshuffle",
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
            epilogue="cshuffle",
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
            epilogue="cshuffle",
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
            epilogue="cshuffle",
        ),
    )
    # pipeline=basic: single-buffer global-read/compute overlap on gfx950.
    add(
        "conv",
        "conv/gfx950/n1h8c16k32r3/basic_default",
        "gfx950",
        build_conv(
            "irhash_conv_950_basic_a",
            "gfx950",
            conv1,
            wave_size=64,
            wtm=32,
            wtn=32,
            wtk=16,
            tile_m=64,
            tile_n=64,
            tile_k=32,
            pipeline="basic",
            epilogue="default",
            vector_size_c=1,
        ),
    )
    add(
        "conv",
        "conv/gfx950/n1h8c16k32r3/basic_cshuffle",
        "gfx950",
        build_conv(
            "irhash_conv_950_basic_b",
            "gfx950",
            conv1,
            wave_size=64,
            wtm=32,
            wtn=32,
            wtk=16,
            tile_m=64,
            tile_n=64,
            tile_k=32,
            pipeline="basic",
            epilogue="cshuffle",
        ),
    )

    # Conv wgrad: implicit-GEMM backward-weight direction (NHWK x NHWC -> KYXC).
    # wgrad1: small 3x3 conv (N8 H8 W8 C16 K32 Y3 X3, stride/pad=defaults).
    # wgrad2: 1x1 conv (N2 H16 W16 C32 K32, no filter spatial).
    wgrad1 = (1, 8, 8, 16, 32, 3, 3, 1, 1, 1, 1, 1, 1)
    wgrad2 = (2, 16, 16, 32, 32, 1, 1, 1, 1, 0, 0, 1, 1)
    add(
        "conv_wgrad",
        "conv_wgrad/gfx942/n1h8c16k32r3",
        "gfx942",
        build_conv_wgrad(
            "irhash_wgrad_942_a",
            "gfx942",
            wgrad1,
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
        "conv_wgrad",
        "conv_wgrad/gfx950/n1h8c16k32r3",
        "gfx950",
        build_conv_wgrad(
            "irhash_wgrad_950_a",
            "gfx950",
            wgrad1,
            wave_size=64,
            wtm=32,
            wtn=32,
            wtk=16,
            tile_m=64,
            tile_n=64,
            tile_k=32,
            pipeline="mem",
            epilogue="default",
        ),
    )
    add(
        "conv_wgrad",
        "conv_wgrad/gfx950/n2h16c32k32r1",
        "gfx950",
        build_conv_wgrad(
            "irhash_wgrad_950_b",
            "gfx950",
            wgrad2,
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
        "conv_wgrad",
        "conv_wgrad/gfx950/n1h8c16k32r3_spk4",
        "gfx950",
        build_conv_wgrad(
            "irhash_wgrad_950_spk4",
            "gfx950",
            wgrad1,
            wave_size=64,
            wtm=16,
            wtn=16,
            wtk=16,
            tile_m=64,
            tile_n=32,
            tile_k=16,
            split_k=4,
        ),
    )
    add(
        "conv_wgrad",
        "conv_wgrad/gfx1151/n1h8c16k32r3",
        "gfx1151",
        build_conv_wgrad(
            "irhash_wgrad_1151_a",
            "gfx1151",
            wgrad1,
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
        "conv_wgrad",
        "conv_wgrad/gfx1201/n1h8c16k32r3",
        "gfx1201",
        build_conv_wgrad(
            "irhash_wgrad_1201_a",
            "gfx1201",
            wgrad1,
            wave_size=32,
            wtm=16,
            wtn=16,
            wtk=16,
            tile_m=32,
            tile_n=32,
            tile_k=16,
        ),
    )
    # gfx90a wgrad mirrors the gfx942 MFMA path.
    add(
        "conv_wgrad",
        "conv_wgrad/gfx90a/n1h8c16k32r3",
        "gfx90a",
        build_conv_wgrad(
            "irhash_wgrad_90a_a",
            "gfx90a",
            wgrad1,
            wave_size=64,
            wtm=16,
            wtn=16,
            wtk=16,
            tile_m=64,
            tile_n=32,
            tile_k=16,
        ),
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
            epilogue="cshuffle",
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
            epilogue="cshuffle",
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

    # Attention: 2D tiled (gfx950 wide-K), 3D tiled + reduce (gfx950),
    # and 2D tiled narrow (gfx942 MFMA-16x16x16 path).
    add(
        "attention",
        "attention/gfx950/2d_fp16_d128_b64",
        "gfx950",
        build_attention_2d(
            "irhash_attn_950_2d_fp16",
            "gfx950",
            head_size=128,
            block_size=64,
            num_query_heads=4,
            num_kv_heads=2,
            dtype="fp16",
        ),
    )
    add(
        "attention",
        "attention/gfx950/2d_bf16_d128_b64",
        "gfx950",
        build_attention_2d(
            "irhash_attn_950_2d_bf16",
            "gfx950",
            head_size=128,
            block_size=64,
            num_query_heads=4,
            num_kv_heads=2,
            dtype="bf16",
        ),
    )
    add(
        "attention",
        "attention/gfx950/2d_fp16_d128_b64_softcap",
        "gfx950",
        build_attention_2d(
            "irhash_attn_950_2d_fp16_sc",
            "gfx950",
            head_size=128,
            block_size=64,
            num_query_heads=4,
            num_kv_heads=2,
            dtype="fp16",
            has_softcap=True,
        ),
    )
    add(
        "attention",
        "attention/gfx950/2d_fp16_d128_b64_alibi",
        "gfx950",
        build_attention_2d(
            "irhash_attn_950_2d_fp16_alibi",
            "gfx950",
            head_size=128,
            block_size=64,
            num_query_heads=4,
            num_kv_heads=2,
            dtype="fp16",
            use_alibi=True,
        ),
    )
    add(
        "attention",
        "attention/gfx950/3d_fp16_d128_b64",
        "gfx950",
        build_attention_3d(
            "irhash_attn_950_3d_fp16",
            "gfx950",
            head_size=128,
            block_size=64,
            num_query_heads=4,
            num_kv_heads=2,
            dtype="fp16",
            num_segments=4,
        ),
    )
    add(
        "attention",
        "attention/gfx950/3d_bf16_d128_b64",
        "gfx950",
        build_attention_3d(
            "irhash_attn_950_3d_bf16",
            "gfx950",
            head_size=128,
            block_size=64,
            num_query_heads=4,
            num_kv_heads=2,
            dtype="bf16",
            num_segments=4,
        ),
    )
    add(
        "attention",
        "attention/gfx950/reduce_fp16_d128",
        "gfx950",
        build_attention_reduce(
            "irhash_attn_950_reduce_fp16",
            "gfx950",
            head_size=128,
            num_query_heads=4,
            num_kv_heads=2,
            dtype="fp16",
            num_segments=4,
        ),
    )
    add(
        "attention",
        "attention/gfx950/reduce_bf16_d128",
        "gfx950",
        build_attention_reduce(
            "irhash_attn_950_reduce_bf16",
            "gfx950",
            head_size=128,
            num_query_heads=4,
            num_kv_heads=2,
            dtype="bf16",
            num_segments=4,
        ),
    )
    add(
        "attention",
        "attention/gfx942/2d_fp16_d128_b64",
        "gfx942",
        build_attention_2d(
            "irhash_attn_942_2d_fp16",
            "gfx942",
            head_size=128,
            block_size=64,
            num_query_heads=4,
            num_kv_heads=2,
            dtype="fp16",
        ),
    )
    add(
        "attention",
        "attention/gfx942/2d_bf16_d128_b64",
        "gfx942",
        build_attention_2d(
            "irhash_attn_942_2d_bf16",
            "gfx942",
            head_size=128,
            block_size=64,
            num_query_heads=4,
            num_kv_heads=2,
            dtype="bf16",
        ),
    )
    add(
        "attention",
        "attention/gfx942/3d_fp16_d128_b64",
        "gfx942",
        build_attention_3d(
            "irhash_attn_942_3d_fp16",
            "gfx942",
            head_size=128,
            block_size=64,
            num_query_heads=4,
            num_kv_heads=2,
            dtype="fp16",
            num_segments=4,
        ),
    )
    add(
        "attention",
        "attention/gfx942/3d_bf16_d128_b64",
        "gfx942",
        build_attention_3d(
            "irhash_attn_942_3d_bf16",
            "gfx942",
            head_size=128,
            block_size=64,
            num_query_heads=4,
            num_kv_heads=2,
            dtype="bf16",
            num_segments=4,
        ),
    )
    add(
        "attention",
        "attention/gfx942/reduce_fp16_d128",
        "gfx942",
        build_attention_reduce(
            "irhash_attn_942_reduce_fp16",
            "gfx942",
            head_size=128,
            num_query_heads=4,
            num_kv_heads=2,
            dtype="fp16",
            num_segments=4,
        ),
    )
    add(
        "attention",
        "attention/gfx942/reduce_bf16_d128",
        "gfx942",
        build_attention_reduce(
            "irhash_attn_942_reduce_bf16",
            "gfx942",
            head_size=128,
            num_query_heads=4,
            num_kv_heads=2,
            dtype="bf16",
            num_segments=4,
        ),
    )

    # Dense flash-attn prefill (gfx950-only builder). Registered as a table rather
    # than 18 add() blocks because every case is the same base spec with one or two
    # overrides; the variant name is the only thing that differs case to case.
    for _variant, _over in (
        # --- default (one CTA per q-block/head) grid ---
        ("default_causal_sq512", {}),
        ("swa_w128_sq512", {"sliding_window": 128}),
        ("varlen_sq512", {"varlen": True}),
        ("lazy_off_sq512", {"lazy_rescale": False}),
        ("fp16_h64_sq512", {"dtype": "fp16", "head_size": 64}),
        ("bn128_sq512", {"block_n": 128}),
        ("noncausal_sq512", {"causal": False}),
        # --- persistent (grid-stride) grid + decode variants ---
        ("persistent_causal_sq512", {"persistent": True, "num_persistent": 256}),
        (
            "persist_qbmaj_sq512",
            {"persistent": True, "num_persistent": 256, "persist_decode": "qb_major"},
        ),
        (
            "persist_hkvmaj_sq512",
            {"persistent": True, "num_persistent": 256, "persist_decode": "hkv_major"},
        ),
        (
            "persist_intl_sq512",
            {"persistent": True, "num_persistent": 256, "interleave": True},
        ),
        (
            "persist_lazy_off_sq512",
            {"persistent": True, "num_persistent": 256, "lazy_rescale": False},
        ),
        (
            "persist_swa_w128_sq512",
            {"persistent": True, "num_persistent": 256, "sliding_window": 128},
        ),
        # D=64 packed-row DMA loader (2 rows/instr, unpadded LDS) on the persistent
        # builder -- locks the head_size=64 fix (fp16_h64 above only exercises the
        # default builder).
        (
            "persist_h64_sq512",
            {"persistent": True, "num_persistent": 256, "head_size": 64},
        ),
        # ragged (non-256 seqlen) in-kernel path: on-chip boundary padding. causal
        # (no key mask), non-causal (ktok<seqlen_kv key mask), D=64, and the
        # persistent variant -- lock all four ragged codegen shapes.
        ("ragged_causal_sq500", {"seqlen_q": 500, "seqlen_kv": 500, "ragged": True}),
        (
            "ragged_full_sq500",
            {"seqlen_q": 500, "seqlen_kv": 500, "ragged": True, "causal": False},
        ),
        (
            "ragged_h64_sq500",
            {"seqlen_q": 500, "seqlen_kv": 500, "ragged": True, "head_size": 64},
        ),
        (
            "persist_ragged_sq500",
            {
                "seqlen_q": 500,
                "seqlen_kv": 500,
                "ragged": True,
                "persistent": True,
                "num_persistent": 256,
            },
        ),
    ):
        add(
            "attention_dense",
            f"attention_dense/gfx950/{_variant}",
            "gfx950",
            build_attention_dense("gfx950", **_over),
        )

    # D256 bf16 prefill fast specs: the byte-sensitive slab-pad (gfx950) and
    # 4-warp GQA V-transpose (gfx942) routes.
    add(
        "attention_d256",
        "attention_d256/gfx950/pad_interleave",
        "gfx950",
        build_attention_d256_gfx950("gfx950"),
    )
    add(
        "attention_d256",
        "attention_d256/gfx942/4warp_gqa",
        "gfx942",
        build_attention_d256_gfx942("gfx942"),
    )
    return out


# The golden stores one sub-document per llvm flavor under this key, and the
# gate (check_golden) verifies all of them from any host: the flavor is an
# argument to lowering, so nothing about the running ROCm vintage limits which
# sub-documents can be checked. The same committed golden is therefore valid,
# and verified, on ROCm < 7.2 (llvm20), 7.2-7.12 (llvm22), and 7.13+ (llvm23).
GOLDEN_FLAVORS = ("llvm20", "llvm22", "llvm23")
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
    """Compare a fresh run against the golden sub-doc(s). Empty list == OK.

    With no ``flavor``, every flavor in :data:`GOLDEN_FLAVORS` is checked, not
    just the one this host autodetects. Lowering takes the flavor as an
    argument, so the extra runs cost a few hundred milliseconds -- whereas
    checking only the host's flavor leaves the other sub-documents unverified
    by any machine that does not happen to run that ROCm vintage. The llvm23
    sub-document, for instance, is only reachable on ROCm >= 7.13, so it would
    otherwise sit in the golden untested.

    Drift strings are prefixed with the flavor when more than one is checked.
    """
    doc = json.loads(golden_path.read_text())
    have = doc.get("flavors", {})
    wanted = [flavor] if flavor else list(GOLDEN_FLAVORS)
    errors: list[str] = []
    for fl in wanted:
        base = have.get(fl)
        if base is None:
            errors.append(
                f"golden has no entry for flavor {fl!r} (have {sorted(have)})"
            )
            continue
        prefix = "" if len(wanted) == 1 else f"[{fl}] "
        errors.extend(prefix + e for e in compare(base, run(flavor=fl)))
    return errors


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
        help="compare a fresh run against EVERY golden sub-doc (narrow it with "
        "--flavor); exit 1 on any drift.",
    )
    ap.add_argument(
        "--flavor",
        choices=GOLDEN_FLAVORS,
        help="check (or dump) just this llvm flavor instead of all of them; "
        "for --ir-dir it overrides the autodetected flavor.",
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
        checked = ns.flavor or ", ".join(GOLDEN_FLAVORS)
        errors = check_golden(ns.check, ns.flavor)
        if errors:
            for e in errors:
                print(f"IR DRIFT: {e}")
            raise SystemExit(1)
        print(f"IR parity OK [{checked}] vs {ns.check}")
        return

    flavor = ns.flavor or current_flavor()
    doc = run(ns.ir_dir, flavor=flavor)
    print(json.dumps(doc, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
