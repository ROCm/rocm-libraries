################################################################################
#
# MIT License
#
# Copyright 2024-2025 AMD ROCm(TM) Software
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

from rrperf.problems import GEMMRun, CodeGenRun, TensileRun, TypeParameters
from rrperf.utils import rocm_gfx

import pathlib
from itertools import product
from typing import List

repo_dir = pathlib.Path(__file__).resolve().parent.parent.parent.parent

fp4fp4_fp32 = TypeParameters(
    type_A="fp4",
    type_B="fp4",
    type_C="float",
    type_D="float",
)

fp6fp6_fp32 = TypeParameters(
    type_A="fp6",
    type_B="fp6",
    type_C="float",
    type_D="float",
)

bf6bf6_fp32 = TypeParameters(
    type_A="bf6",
    type_B="bf6",
    type_C="float",
    type_D="float",
)

fp8fp8_fp32 = TypeParameters(
    type_A="fp8",
    type_B="fp8",
    type_C="float",
    type_D="float",
)

bf8bf8_fp32 = TypeParameters(
    type_A="bf8",
    type_B="bf8",
    type_C="float",
    type_D="float",
)

fp8bf8_fp32 = TypeParameters(
    type_A="fp8",
    type_B="bf8",
    type_C="float",
    type_D="float",
)

bf8fp8_fp32 = TypeParameters(
    type_A="bf8",
    type_B="fp8",
    type_C="float",
    type_D="float",
)

fp16 = TypeParameters(
    type_A="half",
    type_B="half",
    type_C="half",
    type_D="half",
)

bf16_fp32 = TypeParameters(
    type_A="bf16",
    type_B="bf16",
    type_C="float",
    type_D="float",
)

bf16_bf16 = TypeParameters(
    type_A="bf16",
    type_B="bf16",
    type_C="bf16",
    type_D="bf16",
)

fp32 = TypeParameters(
    type_A="float",
    type_B="float",
    type_C="float",
    type_D="float",
)

SGEMM_3072x4096x4096 = dict(
    M=3072, N=4096, K=4096, mac_m=64, mac_n=64, mac_k=64, types=fp32
)

HGEMM_7680x8448x8192 = dict(
    M=7680, N=8448, K=8192, mac_m=64, mac_n=64, mac_k=64, types=fp16
)

HGEMM_7680x8448x8448 = dict(
    M=7680,
    N=8448,
    K=8448,  # matches the guidepost unit test
    mac_m=64,
    mac_n=64,
    mac_k=64,
    workgroup_size_x=128,
    workgroup_size_y=2,
    types=TypeParameters(fp16, trans_A="N", trans_B="T"),
)


def update_parameters(*args, **kwargs):
    rv = {}
    for d in args:
        rv.update(d)
    rv.update(kwargs)
    return rv


def mkGEMM(*args, **kwargs):
    return GEMMRun(**update_parameters(*args, **kwargs))


def unit():
    default = dict(
        M=1024,
        N=1024,
        K=128,
        mac_m=64,
        mac_n=64,
        mac_k=64,
        numWarmUp=1,
        numOuter=1,
        numInner=1,
    )
    yield mkGEMM(default, types=fp32)
    yield mkGEMM(default, types=fp16)
    yield from tail_loop_reproducer()


def unit_gfx120X():
    default = dict(
        M=1024,
        N=1024,
        K=128,
        mac_m=64,
        mac_n=64,
        mac_k=64,
        wave_m=16,
        wave_n=16,
        wave_k=16,
        numWarmUp=1,
        numOuter=1,
        numInner=1,
    )
    yield mkGEMM(default, types=fp16)
    yield mkGEMM(default, types=bf16_fp32)


def sgemm():
    yield mkGEMM(SGEMM_3072x4096x4096)
    yield mkGEMM(SGEMM_3072x4096x4096, mac_m=128, mac_n=64, mac_k=16)


def hgemm_tensile_guidepost():
    yield mkGEMM(HGEMM_7680x8448x8448)


def hgemm():
    yield mkGEMM(HGEMM_7680x8448x8192)
    yield mkGEMM(
        HGEMM_7680x8448x8192,
        mac_m=128,
        mac_n=256,
        mac_k=16,
        workgroup_size_x=128,
        workgroup_size_y=2,
    )
    yield mkGEMM(
        HGEMM_7680x8448x8192,
        mac_m=128,
        mac_n=256,
        mac_k=16,
        workgroup_size_x=128,
        workgroup_size_y=2,
        prefetchInFlight=2,
        prefetchLDSFactor=2,
        types=TypeParameters(
            HGEMM_7680x8448x8192["types"],
            trans_A="N",
            trans_B="T",
        ),
    )
    yield mkGEMM(
        HGEMM_7680x8448x8192,
        mac_m=128,
        mac_n=256,
        mac_k=16,
        workgroup_size_x=64,
        workgroup_size_y=4,
    )

    yield mkGEMM(
        HGEMM_7680x8448x8192,
        types=TypeParameters(HGEMM_7680x8448x8192["types"], trans_A="T", trans_B="N"),
    )
    yield mkGEMM(
        HGEMM_7680x8448x8192,
        types=TypeParameters(HGEMM_7680x8448x8192["types"], trans_A="T", trans_B="T"),
    )
    yield mkGEMM(
        HGEMM_7680x8448x8192,
        types=TypeParameters(HGEMM_7680x8448x8192["types"], trans_A="N", trans_B="T"),
    )

    yield mkGEMM(HGEMM_7680x8448x8448)
    yield mkGEMM(
        HGEMM_7680x8448x8448,
        mac_m=128,
        mac_n=256,
        mac_k=16,
        workgroup_size_x=256,
        workgroup_size_y=1,
    )
    yield mkGEMM(
        HGEMM_7680x8448x8448,
        mac_m=128,
        mac_n=256,
        mac_k=16,
        workgroup_size_x=128,
        workgroup_size_y=2,
    )
    yield mkGEMM(
        HGEMM_7680x8448x8448,
        mac_m=128,
        mac_n=256,
        mac_k=16,
        workgroup_size_x=64,
        workgroup_size_y=4,
    )
    yield mkGEMM(
        HGEMM_7680x8448x8448,
        mac_m=128,
        mac_n=256,
        mac_k=16,
        workgroup_size_x=128,
        workgroup_size_y=2,
        prefetchInFlight=2,
        prefetchLDSFactor=2,
    )

    for sched in ["Priority", "Cooperative", "Sequential"]:
        yield mkGEMM(
            M=3840,
            N=4224,
            K=4224,
            mac_m=128,
            mac_n=128,
            mac_k=32,
            workgroup_size_x=128,
            workgroup_size_y=2,
            visualize=False,
            scheduler=sched,
            types=TypeParameters(
                fp16,
                trans_A="N",
                trans_B="T",
            ),
        )

        yield mkGEMM(
            M=1024,
            N=50304,
            K=8192,
            mac_m=128,
            mac_n=128,
            mac_k=32,
            workgroup_size_x=128,
            workgroup_size_y=2,
            visualize=False,
            scheduler=sched,
            types=TypeParameters(
                fp16,
                trans_A="N",
                trans_B="T",
            ),
        )

        yield mkGEMM(
            M=3840,
            N=4224,
            K=4224,
            mac_m=128,
            mac_n=128,
            mac_k=32,
            workgroup_size_x=128,
            workgroup_size_y=2,
            betaInFma=False,
            visualize=False,
            scheduler=sched,
            types=TypeParameters(
                fp16,
                trans_A="N",
                trans_B="T",
            ),
        )

        yield mkGEMM(
            M=1024,
            N=50304,
            K=8192,
            mac_m=128,
            mac_n=128,
            mac_k=32,
            workgroup_size_x=128,
            workgroup_size_y=2,
            betaInFma=False,
            visualize=False,
            scheduler=sched,
            types=TypeParameters(
                fp16,
                trans_A="N",
                trans_B="T",
            ),
        )

        yield mkGEMM(
            M=8448,
            N=8448,
            K=128,
            mac_m=128,
            mac_n=256,
            mac_k=16,
            workgroup_size_x=128,
            workgroup_size_y=2,
            visualize=False,
            scheduler=sched,
            types=TypeParameters(
                fp16,
                trans_A="N",
                trans_B="T",
            ),
        )

        yield mkGEMM(
            M=7680,
            N=8448,
            K=8192,
            mac_m=128,
            mac_n=256,
            mac_k=16,
            workgroup_size_x=256,
            workgroup_size_y=1,
            visualize=False,
            scheduler=sched,
            types=TypeParameters(
                fp16,
                trans_A="N",
                trans_B="T",
            ),
        )

    # TODO: Enable once visualizer is working
    # yield from visualizer()


def hgemm_gfx120X():
    params = dict(
        wave_m=16,
        wave_n=16,
        wave_k=16,
        workgroup_size_x=64,
        prefetch=False,
    )

    type_specifiers = [
        ("half", fp16),
        ("bf16", bf16_bf16),
        ("float", fp16),
        ("float", fp8fp8_fp32),
        ("float", fp8bf8_fp32),
        ("float", bf8bf8_fp32),
        ("float", bf8fp8_fp32),
    ]

    for sched in ["Priority", "Cooperative", "Sequential"]:
        for a, b in product("NT", repeat=2):
            for acc, abcd in type_specifiers:
                yield mkGEMM(
                    HGEMM_7680x8448x8192,
                    types=TypeParameters(abcd, type_acc=acc, trans_A=a, trans_B=b),
                    scheduler=sched,
                    **params,
                )


def visualizer():
    yield mkGEMM(
        M=512,
        N=768,
        K=512,
        mac_m=128,
        mac_n=256,
        mac_k=16,
        alpha=2.0,
        beta=0.5,
        workgroup_size_x=256,
        workgroup_size_y=1,
        storeLDS_D=False,
        visualize=True,
        prefetch=False,
        types=TypeParameters(
            fp16,
            trans_A="N",
            trans_B="T",
        ),
    )


def tail_loop_reproducer():
    yield mkGEMM(
        M=64,
        N=128,
        K=8,
        wave_m=32,
        wave_n=32,
        wave_k=2,
        wave_b=1,
        mac_m=64,
        mac_n=64,
        mac_k=8,
        types=TypeParameters(
            trans_A="T",
            trans_B="N",
        ),
    )


def guidepost_1():
    yield mkGEMM(
        HGEMM_7680x8448x8448,
        mac_m=128,
        mac_n=256,
        mac_k=16,
        scheduler="Priority",
        prefetch=True,
        prefetchInFlight=2,
        prefetchLDSFactor=2,
    )


def guidepost_2():
    yield mkGEMM(
        HGEMM_7680x8448x8448,
        K=8192,
        mac_m=128,
        mac_n=256,
        mac_k=16,
        scheduler="Priority",
        prefetch=True,
        prefetchInFlight=2,
        prefetchLDSFactor=2,
    )


def hgemm_no_store_LDS():
    for K in [8448, 8192]:
        yield mkGEMM(
            HGEMM_7680x8448x8448,
            K=K,
            mac_m=128,
            mac_n=256,
            mac_k=16,
            storeLDS_D=False,
            scheduler="Priority",
            prefetch=True,
            prefetchInFlight=2,
            prefetchLDSFactor=0,
        )


def tensile_guidepost():
    yield TensileRun(
        config=str(
            repo_dir
            / "test"
            / "unit"
            / "GemmGuidePost"
            / "HGemmGuidePost_Optimized.yaml"
        ),
    )


def tensile_sgemm_guidepost():
    yield TensileRun(
        config=str(
            repo_dir
            / "test"
            / "unit"
            / "GemmGuidePost"
            / "GemmGuidePost_Optimized.yaml"
        ),
    )


def streamk_sweep():
    for twoTile in {True, False}:
        for base in [HGEMM_7680x8448x8448]:
            # Currently these run out of LDS everywhere except gfx950.
            # + [SGEMM_3072x4096x4096]
            for mac_m in [64, 128]:
                for mac_n in [64, 128, 256]:
                    for mac_k in [16, 32, 64]:
                        if twoTile and mac_m * mac_n * mac_k >= (64 * 256 * 64):
                            # currently these run out of VGPRs.
                            pass
                        else:
                            yield mkGEMM(
                                base,
                                mac_m=mac_m,
                                mac_n=mac_n,
                                mac_k=mac_k,
                                workgroup_size_x=128,
                                workgroup_size_y=2,
                                visualize=False,
                                prefetch=False,  # TODO: Fix k loop unrolling with stream k
                                # prefetchInFlight=2,
                                # prefetchLDSFactor=2,
                                streamK=True,
                                streamKTwoTile=twoTile,
                                types=TypeParameters(
                                    base["types"],
                                    trans_A="N",
                                    trans_B="T",
                                ),
                            )


def streamk():
    for twoTile in {True, False}:
        # SGEMM
        yield mkGEMM(
            SGEMM_3072x4096x4096,
            workgroup_size_x=128,
            workgroup_size_y=2,
            visualize=False,
            prefetch=False,  # TODO: Fix k loop unrolling with stream k
            # prefetchInFlight=2,
            # prefetchLDSFactor=2,
            streamK=True,
            streamKTwoTile=twoTile,
            types=TypeParameters(
                SGEMM_3072x4096x4096["types"],
                trans_A="N",
                trans_B="T",
            ),
        )
        # HGEMM
        yield mkGEMM(
            HGEMM_7680x8448x8448,
            mac_m=128,
            mac_n=256,
            mac_k=16,
            workgroup_size_x=128,
            workgroup_size_y=2,
            prefetch=False,  # TODO: Fix k loop unrolling with stream k
            # prefetchInFlight=2,
            # prefetchLDSFactor=2,
            streamK=True,
            streamKTwoTile=twoTile,
            types=TypeParameters(
                HGEMM_7680x8448x8448["types"],
                trans_A="N",
                trans_B="T",
            ),
        )
        yield mkGEMM(
            HGEMM_7680x8448x8192,
            mac_m=128,
            mac_n=256,
            mac_k=16,
            prefetch=False,  # TODO: Fix k loop unrolling with stream k
            streamK=True,
            streamKTwoTile=twoTile,
            types=TypeParameters(
                HGEMM_7680x8448x8192["types"],
                trans_A="N",
                trans_B="T",
            ),
        )


def scalar_is_zero():
    # TODO: Make streamK and ConstantPropagation transformation can be both applied
    sgemm = update_parameters(
        SGEMM_3072x4096x4096,
        beta=0.0,
        streamK=False,
    )
    yield mkGEMM(sgemm)
    yield mkGEMM(sgemm, mac_m=128, mac_n=64, mac_k=16)

    hgemm = update_parameters(
        HGEMM_7680x8448x8192,
        beta=0.0,
        streamK=False,
    )
    yield mkGEMM(hgemm)
    yield mkGEMM(
        hgemm,
        mac_m=128,
        mac_n=256,
        mac_k=16,
        workgroup_size_x=128,
        workgroup_size_y=2,
    )
    yield mkGEMM(
        hgemm,
        mac_m=128,
        mac_n=256,
        mac_k=16,
        workgroup_size_x=128,
        workgroup_size_y=2,
        prefetchInFlight=2,
        prefetchLDSFactor=2,
        types=TypeParameters(
            hgemm["types"],
            trans_A="N",
            trans_B="T",
        ),
    )
    yield mkGEMM(
        hgemm,
        mac_m=128,
        mac_n=256,
        mac_k=16,
        beta=0.0,
        workgroup_size_x=64,
        workgroup_size_y=4,
    )


def tensile_benchmarks():
    yield from tensile_guidepost()
    yield from tensile_sgemm_guidepost()


def codegen():
    yield CodeGenRun(instCount=40000, instructions="comments")
    yield CodeGenRun(instCount=40000, instructions="simple_mi")
    yield CodeGenRun(instCount=40000, instructions="complex_mi_with_coop")


def f16gemm_16x16x32():
    params = dict(
        M=128,
        N=128,
        K=256,
        mac_m=64,
        mac_n=64,
        mac_k=32,
        wave_m=16,
        wave_n=16,
        wave_k=32,
        workgroup_size_x=256,
        workgroup_size_y=1,
    )
    for a, b in product("NT", repeat=2):
        yield GEMMRun(
            **params,
            types=TypeParameters(fp16, trans_A=a, trans_B=b),
        )


def f16gemm_32x32x16():
    params = dict(
        M=256,
        N=256,
        K=128,
        mac_m=128,
        mac_n=128,
        mac_k=16,
        wave_m=32,
        wave_n=32,
        wave_k=16,
        workgroup_size_x=256,
        workgroup_size_y=1,
    )
    for a, b in product("NT", repeat=2):
        yield GEMMRun(
            **params,
            types=TypeParameters(fp16, trans_A=a, trans_B=b),
        )


def f16gemm():
    yield from f16gemm_16x16x32()
    yield from f16gemm_32x32x16()


def f8gemm():
    yield GEMMRun(
        M=1024,
        N=1024,
        K=1024,
        mac_m=64,
        mac_n=16,
        mac_k=64,
        workgroup_size_x=256,
        workgroup_size_y=1,
        types=fp8fp8_fp32,
    )


def mx_gemm_16x16x128_f8f6f4():
    params = dict(
        M=4096,
        N=4096,
        K=8192,
        mac_m=64,
        mac_n=64,
        mac_k=128,
        wave_m=16,
        wave_n=16,
        wave_k=128,
        workgroup_size_x=256,
        workgroup_size_y=1,
    )
    for typeParam in [fp8fp8_fp32, bf8bf8_fp32, fp6fp6_fp32, bf6bf6_fp32, fp4fp4_fp32]:
        for a, b in product("NT", repeat=2):
            yield GEMMRun(
                **params,
                types=TypeParameters(typeParam, trans_A=a, trans_B=b),
            )


def mx_gemm_32x32x64_f8f6f4():
    params = dict(
        M=4096,
        N=4096,
        K=8192,
        mac_m=128,
        mac_n=128,
        mac_k=64,
        wave_m=32,
        wave_n=32,
        wave_k=64,
        workgroup_size_x=256,
        workgroup_size_y=1,
    )
    for typeParam in [fp8fp8_fp32, bf8bf8_fp32, fp6fp6_fp32, bf6bf6_fp32, fp4fp4_fp32]:
        for a, b in product("NT", repeat=2):
            yield GEMMRun(
                **params,
                types=TypeParameters(typeParam, trans_A=a, trans_B=b),
            )


def mx_gemms_f8f6f4():
    yield from mx_gemm_32x32x64_f8f6f4()
    yield from mx_gemm_16x16x128_f8f6f4()


def gemm_mixed_16x16x128_f8f6f4():
    params = dict(
        M=256,
        N=256,
        K=512,
        mac_m=64,
        mac_n=64,
        mac_k=128,
        wave_m=16,
        wave_n=16,
        wave_k=128,
        workgroup_size_x=256,
        workgroup_size_y=1,
    )
    TA = {"fp8", "bf8", "fp6", "bf6", "fp4"}
    TB = {"fp8", "bf8", "fp6", "bf6", "fp4"}
    for A in TA:
        for B in TB:
            yield GEMMRun(
                **params,
                types=TypeParameters(
                    type_A=A,
                    type_B=B,
                    type_C="float",
                    type_D="float",
                    trans_A="T",
                    trans_B="N",
                ),
            )


def gemm_mixed_32x32x64_f8f6f4():
    params = dict(
        M=256,
        N=256,
        K=512,
        mac_m=128,
        mac_n=128,
        mac_k=64,
        wave_m=32,
        wave_n=32,
        wave_k=64,
        workgroup_size_x=256,
        workgroup_size_y=1,
    )
    TA = {"fp8", "bf8", "fp6", "bf6", "fp4"}
    TB = {"fp8", "bf8", "fp6", "bf6", "fp4"}
    for A in TA:
        for B in TB:
            yield GEMMRun(
                **params,
                types=TypeParameters(
                    type_A=A,
                    type_B=B,
                    type_C="float",
                    type_D="float",
                    trans_A="T",
                    trans_B="N",
                ),
            )


def mixedgemm_f8f6f4():
    yield from gemm_mixed_16x16x128_f8f6f4()
    yield from gemm_mixed_32x32x64_f8f6f4()


def _f8f6f4_gemm_macrotiles(
    wave_m, wave_n, wave_k, gemmTypes, mac_m_factor, mac_n_factor, mac_k_factor
):
    params = dict(
        M=256,
        N=256,
        K=512,
        mac_m=wave_m * mac_m_factor,
        mac_n=wave_m * mac_n_factor,
        mac_k=wave_k * mac_k_factor,
        wave_m=wave_m,
        wave_n=wave_n,
        wave_k=wave_k,
        workgroup_size_x=64,
        workgroup_size_y=1,
    )
    yield GEMMRun(**params, types=TypeParameters(gemmTypes, trans_A="T", trans_B="N"))


def gemm_f8f6f4_different_macrotiles():
    for type in [fp8fp8_fp32, fp6fp6_fp32, fp4fp4_fp32, bf8bf8_fp32, bf6bf6_fp32]:
        for m_factor in [2, 4, 8]:
            for n_factor in [2, 4, 8]:
                for k_factor in [2, 4]:
                    yield from _f8f6f4_gemm_macrotiles(
                        16, 16, 128, type, m_factor, n_factor, k_factor
                    )
                    yield from _f8f6f4_gemm_macrotiles(
                        32, 32, 64, type, m_factor, n_factor, k_factor
                    )


def _f8f6f4_gemm_prefetch(wave_m, wave_n, wave_k, gemmTypes, prefetchFactor):
    params = dict(
        M=256,
        N=256,
        K=512,
        mac_m=wave_m * 4,
        mac_n=wave_m * 4,
        mac_k=wave_k * 2,
        wave_m=wave_m,
        wave_n=wave_n,
        wave_k=wave_k,
        workgroup_size_x=256,
        workgroup_size_y=1,
        prefetchInFlight=prefetchFactor,
        prefetchLDSFactor=prefetchFactor,
    )
    yield GEMMRun(
        **params,
        types=TypeParameters(gemmTypes, trans_A="T", trans_B="N"),
    )


def gemm_f8f6f4_test_prefetch():
    # Any factor > 2 fails to generate due to lack of registers/too large imm offset.
    # TODO: test with more aggressive prefetching once we can.
    for factor in [2]:
        for type in [fp8fp8_fp32, fp6fp6_fp32, fp4fp4_fp32, bf8bf8_fp32, bf6bf6_fp32]:
            yield from _f8f6f4_gemm_prefetch(16, 16, 128, type, factor)
            yield from _f8f6f4_gemm_prefetch(32, 32, 64, type, factor)


def bf16gemm_16x16x8():
    yield GEMMRun(
        M=1024,
        N=1024,
        K=1024,
        mac_m=64,
        mac_n=16,
        mac_k=64,
        wave_m=16,
        wave_n=16,
        wave_k=8,
        workgroup_size_x=128,
        workgroup_size_y=1,
        types=bf16_fp32,
    )


def bf16gemm_32x32x4():
    yield GEMMRun(
        M=1024,
        N=1024,
        K=1024,
        mac_m=64,
        mac_n=64,
        mac_k=64,
        wave_m=32,
        wave_n=32,
        wave_k=4,
        workgroup_size_x=128,
        workgroup_size_y=1,
        types=bf16_fp32,
    )


def bf16bf16gemm_16x16x8():
    yield GEMMRun(
        M=1024,
        N=1024,
        K=1024,
        mac_m=64,
        mac_n=16,
        mac_k=64,
        wave_m=16,
        wave_n=16,
        wave_k=8,
        workgroup_size_x=128,
        workgroup_size_y=1,
        types=bf16_bf16,
    )


def bf16bf16gemm_32x32x4():
    yield GEMMRun(
        M=1024,
        N=1024,
        K=1024,
        mac_m=64,
        mac_n=64,
        mac_k=64,
        wave_m=32,
        wave_n=32,
        wave_k=4,
        workgroup_size_x=128,
        workgroup_size_y=1,
        types=bf16_bf16,
    )


def fp4_target():
    yield GEMMRun(
        M=4096,
        N=4096,
        K=32768,
        mac_m=256,
        mac_n=256,
        mac_k=128,
        wave_m=32,
        wave_n=32,
        wave_k=64,
        wave_b=1,
        workgroup_size_x=128,
        workgroup_size_y=2,
        unroll_x=0,
        unroll_y=0,
        loadLDS_A=True,
        loadLDS_B=True,
        loadLDSScale_A=True,
        loadLDSScale_B=True,
        storeLDS_D=True,
        prefetch=True,
        prefetchInFlight=2,
        prefetchLDSFactor=2,
        betaInFma=True,
        scheduler="Priority",
        matchMemoryAccess=True,
        types=TypeParameters(
            trans_A="T",
            trans_B="N",
            type_A="fp4",
            type_B="fp4",
            type_C="half",
            type_D="half",
            type_acc="float",
            scale_A="Separate",
            scaleType_A="E8M0",
            scale_B="Separate",
            scaleType_B="E8M0",
            scaleBlockSize=32,
        ),
        numOuter=1,
        numWarmUp=1000,
        numInner=1000,
    )


def fp4_target_d2lds_mi32x32x64_pf2x1():
    yield GEMMRun(
        M=4096,
        N=4096,
        K=32768,
        mac_m=256,
        mac_n=256,
        mac_k=128,
        wave_m=32,
        wave_n=32,
        wave_k=64,
        wave_b=1,
        workgroup_size_x=128,
        workgroup_size_y=2,
        unroll_x=0,
        unroll_y=0,
        direct2LDS_A=True,
        direct2LDS_B=True,
        loadLDSScale_A=True,
        loadLDSScale_B=True,
        storeLDS_D=True,
        prefetch=True,
        prefetchInFlight=2,
        prefetchLDSFactor=1,
        betaInFma=True,
        scheduler="Priority",
        matchMemoryAccess=True,
        types=TypeParameters(
            trans_A="T",
            trans_B="N",
            type_A="fp4",
            type_B="fp4",
            type_C="half",
            type_D="half",
            type_acc="float",
            scale_A="Separate",
            scaleType_A="E8M0",
            scale_B="Separate",
            scaleType_B="E8M0",
            scaleBlockSize=32,
        ),
        numOuter=1,
        numWarmUp=1000,
        numInner=1000,
    )


def add_wgm(mapping, suite):
    for run in suite:
        run.workgroupMappingDim = mapping[0]
        run.workgroupMappingValue = mapping[1]
        yield run


def addSkipPermlane(suite: List[GEMMRun], value=True):
    for run in suite:
        run.types.scaleSkipPermlane = value
        yield run


def fp4_target_d2lds_mi32x32x64_pf2x1_wgm():
    yield from add_wgm((0, 2), fp4_target_d2lds_mi32x32x64_pf2x1())


def fp4_target_d2lds_mi32x32x64_pf4x1_sweep_wgms():
    for wgm_dim in [0, 1]:
        for wgm_value in range(1, 50):
            yield from add_wgm(
                (wgm_dim, wgm_value), fp4_target_d2lds_mi32x32x64_pf2x1()
            )


def fp4_target_d2lds_mi32x32x64_pf4x1():
    yield GEMMRun(
        M=4096,
        N=4096,
        K=32768,
        beta=0.0,
        mac_m=256,
        mac_n=256,
        mac_k=128,
        wave_m=32,
        wave_n=32,
        wave_k=64,
        wave_b=1,
        workgroup_size_x=128,
        workgroup_size_y=2,
        unroll_x=0,
        unroll_y=0,
        direct2LDS_A=True,
        direct2LDS_B=True,
        loadLDSScale_A=False,
        loadLDSScale_B=False,
        storeLDS_D=False,
        prefetch=True,
        prefetchInFlight=4,
        prefetchLDSFactor=1,
        prefetchScale=True,
        swizzleScale=True,
        prefetchMixMemOps=True,
        betaInFma=True,
        scheduler="Priority",
        matchMemoryAccess=True,
        types=TypeParameters(
            trans_A="T",
            trans_B="N",
            type_A="fp4",
            type_B="fp4",
            type_C="half",
            type_D="half",
            type_acc="float",
            scale_A="Separate",
            scaleType_A="E8M0",
            scale_B="Separate",
            scaleType_B="E8M0",
            scaleBlockSize=32,
        ),
        numOuter=1,
        numWarmUp=1000,
        numInner=1000,
    )


def fp4_target_d2lds_mi32x32x64_pf4x1_wgm():
    yield from add_wgm((0, 2), fp4_target_d2lds_mi32x32x64_pf4x1())


def fp4_target_d2lds_mi32x32x64_pf4x1_both():
    yield from fp4_target_d2lds_mi32x32x64_pf4x1()
    yield from fp4_target_d2lds_mi32x32x64_pf4x1_wgm()


def fp4_target_d2lds_mi16x16x128_pf4x1():
    yield GEMMRun(
        M=4096,
        N=4096,
        K=32768,
        beta=0.0,
        mac_m=256,
        mac_n=256,
        mac_k=128,
        wave_m=16,
        wave_n=16,
        wave_k=128,
        wave_b=1,
        workgroup_size_x=128,
        workgroup_size_y=2,
        unroll_x=0,
        unroll_y=0,
        direct2LDS_A=True,
        direct2LDS_B=True,
        loadLDSScale_A=False,
        loadLDSScale_B=False,
        storeLDS_D=False,
        prefetch=True,
        prefetchInFlight=4,
        prefetchLDSFactor=1,
        prefetchScale=True,
        swizzleScale=True,
        prefetchMixMemOps=True,
        betaInFma=True,
        scheduler="Priority",
        matchMemoryAccess=True,
        types=TypeParameters(
            trans_A="T",
            trans_B="N",
            type_A="fp4",
            type_B="fp4",
            type_C="half",
            type_D="half",
            type_acc="float",
            scale_A="Separate",
            scaleType_A="E8M0",
            scale_B="Separate",
            scaleType_B="E8M0",
            scaleBlockSize=32,
        ),
        numOuter=1,
        numWarmUp=1000,
        numInner=1000,
    )


def fp4_target_d2lds_mi16x16x128_pf4x1_wgm():
    yield from add_wgm((0, 2), fp4_target_d2lds_mi16x16x128_pf4x1())


def fp4_target_d2lds_mi16x16x128_pf4x1_both():
    yield from fp4_target_d2lds_mi16x16x128_pf4x1()
    yield from fp4_target_d2lds_mi16x16x128_pf4x1_wgm()


def does_this_fail():
    yield GEMMRun(
        M=4096,
        N=4096,
        K=32768,
        beta=0.0,
        mac_m=64,
        mac_n=64,
        mac_k=128,
        wave_m=16,
        wave_n=16,
        wave_k=128,
        wave_b=1,
        workgroup_size_x=128,
        workgroup_size_y=2,
        unroll_x=0,
        unroll_y=0,
        direct2LDS_A=True,
        direct2LDS_B=True,
        loadLDSScale_A=False,
        loadLDSScale_B=False,
        storeLDS_D=False,
        prefetch=True,
        prefetchInFlight=4,
        prefetchLDSFactor=1,
        prefetchScale=False,
        swizzleScale=False,
        prefetchMixMemOps=True,
        betaInFma=True,
        scheduler="Priority",
        matchMemoryAccess=True,
        types=TypeParameters(
            trans_A="T",
            trans_B="N",
            type_A="fp4",
            type_B="fp4",
            type_C="half",
            type_D="half",
            type_acc="float",
        ),
        numOuter=1,
        numWarmUp=1000,
        numInner=1000,
    )


def fp4_no_scale_target_d2lds_mi16x16x128_pf4x1():
    yield GEMMRun(
        M=4096,
        N=4096,
        K=32768,
        beta=0.0,
        mac_m=256,
        mac_n=256,
        mac_k=128,
        wave_m=16,
        wave_n=16,
        wave_k=128,
        wave_b=1,
        workgroup_size_x=128,
        workgroup_size_y=2,
        unroll_x=0,
        unroll_y=0,
        direct2LDS_A=True,
        direct2LDS_B=True,
        loadLDSScale_A=False,
        loadLDSScale_B=False,
        storeLDS_D=False,
        prefetch=True,
        prefetchInFlight=4,
        prefetchLDSFactor=1,
        prefetchScale=False,
        swizzleScale=False,
        prefetchMixMemOps=True,
        betaInFma=True,
        scheduler="Priority",
        matchMemoryAccess=True,
        types=TypeParameters(
            trans_A="T",
            trans_B="N",
            type_A="fp4",
            type_B="fp4",
            type_C="half",
            type_D="half",
            type_acc="float",
        ),
        numOuter=1,
        numWarmUp=1000,
        numInner=1000,
    )


def fp4_no_scale_target_d2lds_mi16x16x128_pf4x1_wgm():
    yield from add_wgm((0, 2), fp4_target_d2lds_mi16x16x128_pf4x1())


def fp4_kernels_no_wgm():
    yield from fp4_target()
    yield from fp4_target_d2lds_mi32x32x64_pf2x1()
    yield from fp4_target_d2lds_mi32x32x64_pf4x1()
    yield from fp4_target_d2lds_mi16x16x128_pf4x1()
    # yield from fp4_no_scale_target_d2lds_mi16x16x128_pf4x1()


def fp4_kernels_wgm():
    yield from fp4_target_d2lds_mi32x32x64_pf2x1_wgm()
    yield from fp4_target_d2lds_mi32x32x64_pf4x1_wgm()
    yield from fp4_target_d2lds_mi16x16x128_pf4x1_wgm()
    yield from fp4_no_scale_target_d2lds_mi16x16x128_pf4x1_wgm()


def fp4_16x16x128_scale_options():
    yield from fp4_target_d2lds_mi16x16x128_pf4x1_wgm()
    yield from addSkipPermlane(fp4_target_d2lds_mi16x16x128_pf4x1_wgm())


def fp4_32x32x64_scale_options():
    yield from fp4_target_d2lds_mi16x16x128_pf4x1_wgm()
    yield from addSkipPermlane(fp4_target_d2lds_mi32x32x64_pf4x1_wgm())


def fp4_kernels():
    yield from fp4_kernels_no_wgm()
    yield from fp4_kernels_wgm()
    yield from fp4_16x16x128_scale_options()
    yield from fp4_32x32x64_scale_options()


def fp4_target_sweep_wgms():
    for wgm_dim in [0, 1]:
        for wgm_value in range(1, 50):
            yield from add_wgm(
                (wgm_dim, wgm_value), fp4_no_scale_target_d2lds_mi16x16x128_pf4x1()
            )


def generate_gfx950():
    yield from fp4_kernels()


def all():
    if rocm_gfx().startswith("gfx95"):
        # TODO: Add here more GFX950 tests
        yield from fp4_kernels()
        yield from mx_gemms_f8f6f4()
    yield from sgemm()
    yield from hgemm()
    yield from hgemm_no_store_LDS()
    yield from streamk()
    yield from streamk_sweep()
    yield from scalar_is_zero()
    yield from codegen()


def all_gfx120X():
    yield from hgemm_gfx120X()
    yield from codegen()


def hgemm_guideposts():
    yield from guidepost_1()
    yield from guidepost_2()


def priority_problems():
    return {
        "1. HGEMM Guidepost": {
            "M": 7680,
            "N": 8448,
            "trans_A": "N",
            "trans_B": "T",
        },
        "2. Halfs": {"type_A": "half"},
        "3. Floats": {"type_A": "float"},
    }
