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

import itertools
from pathlib import Path

from rrperf.problems import TypeParameters
from rrperf.rrsuites import codegen, mkGEMM

repo_dir = Path(__file__).resolve().parent.parent.parent.parent

TYPE_CONFIGS_F8 = {
    "bf8bf8_fp16": dict(
        type_A="bf8",
        type_B="bf8",
        type_C="half",
        type_D="half",
        type_acc="half",
        wave_k=64,
    ),
    # TODO: support these types (unit_gfx1250 suite)
    # "bf8fp8_fp16": dict(
    #     type_A="bf8",
    #     type_B="fp8",
    #     type_C="half",
    #     type_D="half",
    #     type_acc="half",
    #     wave_k=64,
    # ),
    # "fp8bf8_fp16": dict(
    #     type_A="fp8",
    #     type_B="bf8",
    #     type_C="half",
    #     type_D="half",
    #     type_acc="half",
    #     wave_k=64,
    # ),
    "fp8fp8_fp16": dict(
        type_A="fp8",
        type_B="fp8",
        type_C="half",
        type_D="half",
        type_acc="half",
        wave_k=64,
    ),
    "bf8bf8_fp32": dict(
        type_A="bf8",
        type_B="bf8",
        type_C="float",
        type_D="float",
        type_acc="float",
        wave_k=64,
    ),
    "bf8fp8_fp32": dict(
        type_A="bf8",
        type_B="fp8",
        type_C="float",
        type_D="float",
        type_acc="float",
        wave_k=64,
    ),
    "fp8bf8_fp32": dict(
        type_A="fp8",
        type_B="bf8",
        type_C="float",
        type_D="float",
        type_acc="float",
        wave_k=64,
    ),
    "fp8fp8_fp32": dict(
        type_A="fp8",
        type_B="fp8",
        type_C="float",
        type_D="float",
        type_acc="float",
        wave_k=64,
    ),
}
TYPE_CONFIGS_F16 = {
    "bf16_bf16": dict(
        type_A="bf16",
        type_B="bf16",
        type_C="bf16",
        type_D="bf16",
        type_acc="bf16",
        wave_k=32,
    ),
    "fp16_fp16": dict(
        type_A="half",
        type_B="half",
        type_C="half",
        type_D="half",
        type_acc="half",
        wave_k=32,
    ),
    "fp16_fp32": dict(
        type_A="half",
        type_B="half",
        type_C="float",
        type_D="float",
        type_acc="float",
        wave_k=32,
    ),
    "bf16_fp32": dict(
        type_A="bf16",
        type_B="bf16",
        type_C="float",
        type_D="float",
        type_acc="float",
        wave_k=32,
    ),
}
TYPE_CONFIGS_F32 = {
    "fp32_fp32": dict(
        type_A="float",
        type_B="float",
        type_C="float",
        type_D="float",
        type_acc="float",
        wave_k=4,
    ),
}

TYPE_CONFIGS_MX = dict()
for a in ["fp8", "bf8", "fp6", "bf6", "fp4"]:
    for b in ["fp8", "bf8", "fp6", "bf6", "fp4"]:
        TYPE_CONFIGS_MX[f"{a}_{b}_fp32"] = dict(
            type_A=a,
            type_B=b,
            type_C="float",
            type_D="float",
            type_acc="float",
            wave_k=128,
        )

BASE_PROBLEM_CONFIGS = {
    "GEMM_512x512x512": dict(
        M=512,
        N=512,
        K=512,
    ),
}

BASE_SOLUTION_CONFIGS = {
    "1250_default": dict(
        mac_m=64,
        mac_n=64,
        mac_k=128,
        wave_m=16,
        wave_n=16,
        workgroup_size_x=64,
        workgroup_size_y=2,
    )
}


def merge_dicts(dicts: list[dict]) -> dict:
    merged = {}
    for d in dicts:
        merged.update(d)
    return merged


# Takes a dict of parameters, extracts those that belong to TypeParameters
# Returns a the input dict with a new types field containing a TypeParameters object
def create_type_parameters(params: dict) -> dict:
    type_fields = {f.name for f in TypeParameters.__dataclass_fields__.values()}
    type_params_dict = {
        k: params.pop(k) for k in list(params.keys()) if k in type_fields
    }
    params["types"] = TypeParameters(**type_params_dict)
    return params


# Run all combinations of the configurations in a grid list, with optional filter
# If parameters overlap, later dicts overwrite earlier ones
def iterate_grid(config_grid: list[list[dict]], skip_filter: callable = None):
    for config_combo in itertools.product(*config_grid):
        params = merge_dicts(config_combo)
        params = create_type_parameters(params)
        if skip_filter is not None and skip_filter(params):
            continue
        yield mkGEMM(params)


def unit_gfx1250():
    base_configs = [
        merge_dicts(
            [
                BASE_PROBLEM_CONFIGS["GEMM_512x512x512"],
                BASE_SOLUTION_CONFIGS["1250_default"],
                dict(numWarmUp=1, numOuter=1, numInner=1),
            ]
        )
    ]
    type_configs_half_float = list(TYPE_CONFIGS_F32.values()) + list(
        TYPE_CONFIGS_F16.values()
    )
    mac_configs_half_float = [
        dict(mac_m=64, mac_n=64, mac_k=64),
    ]
    type_configs_remaining = list(TYPE_CONFIGS_F8.values()) + list(
        TYPE_CONFIGS_MX.values()
    )
    yield from iterate_grid(
        [base_configs, type_configs_half_float, mac_configs_half_float]
    )
    yield from iterate_grid([base_configs, type_configs_remaining])


def sgemm_gfx1250():
    base_configs = [
        merge_dicts(
            [
                BASE_PROBLEM_CONFIGS["GEMM_512x512x512"],
                BASE_SOLUTION_CONFIGS["1250_default"],
                dict(mac_m=64, mac_n=64, mac_k=64),
            ]
        )
    ]
    scalar_configs = [
        dict(beta=0.0),
        dict(beta=0.5),
    ]
    type_configs = list(TYPE_CONFIGS_F32.values())
    mac_configs = [
        dict(mac_m=64, mac_n=64, mac_k=64),
        dict(mac_m=128, mac_n=64, mac_k=16),
    ]
    yield from iterate_grid([base_configs, type_configs, scalar_configs])
    yield from iterate_grid([base_configs, type_configs, mac_configs])


def hgemm_gfx1250():
    base_configs = [
        merge_dicts(
            [
                BASE_PROBLEM_CONFIGS["GEMM_512x512x512"],
                BASE_SOLUTION_CONFIGS["1250_default"],
                dict(mac_m=64, mac_n=64, mac_k=64),
            ]
        )
    ]
    scalar_configs = [
        dict(beta=0.0),
        dict(beta=0.5),
    ]
    type_configs = list(TYPE_CONFIGS_F16.values())
    mac_configs = [
        dict(mac_m=64, mac_n=64, mac_k=64),
        # TODO: Support this mac config
        # dict(mac_m=128, mac_n=256, mac_k=32),
    ]
    trans_configs = [
        dict(trans_A="N", trans_B="N"),
        dict(trans_A="T", trans_B="N"),
        dict(trans_A="T", trans_B="T"),
        dict(trans_A="N", trans_B="T"),
    ]
    workgroup_configs = [
        dict(workgroup_size_x=128, workgroup_size_y=1),
        dict(workgroup_size_x=64, workgroup_size_y=2),
    ]
    prefetch_configs = [
        dict(prefetch=True, prefetchInFlight=2, prefetchLDSFactor=2),
        dict(prefetch=True, prefetchInFlight=2, prefetchLDSFactor=0),
    ]
    scheduler_configs = [
        dict(scheduler="Priority"),
        dict(scheduler="Cooperative"),
        dict(scheduler="Sequential"),
    ]
    yield from iterate_grid([base_configs, type_configs, scalar_configs])
    yield from iterate_grid([base_configs, type_configs, mac_configs])
    yield from iterate_grid([base_configs, type_configs, trans_configs])
    yield from iterate_grid([base_configs, type_configs, workgroup_configs])
    yield from iterate_grid([base_configs, type_configs, prefetch_configs])
    yield from iterate_grid([base_configs, type_configs, scheduler_configs])


def f8gemm_gfx1250():
    base_configs = [
        merge_dicts(
            [
                BASE_PROBLEM_CONFIGS["GEMM_512x512x512"],
                BASE_SOLUTION_CONFIGS["1250_default"],
            ]
        )
    ]
    type_configs = list(TYPE_CONFIGS_F8.values())
    workgroupMapping_configs = [
        dict(workgroupMappingDim=1, workgroupMappingValue=2),
        dict(workgroupMappingDim=-1, workgroupMappingValue=-1),
    ]
    trans_configs = [
        dict(trans_A="N", trans_B="N"),
        dict(trans_A="T", trans_B="N"),
        dict(trans_A="T", trans_B="T"),
        dict(trans_A="N", trans_B="T"),
    ]
    prefetch_configs = [
        dict(
            prefetch=True,
            prefetchInFlight=4,
            prefetchLDSFactor=1,
            prefetchMixMemOps=True,
        ),
        dict(prefetch=False),
    ]
    # direct2lds_configs = [
    #     dict(direct2LDS_A=True, direct2LDS_B=True),
    #     dict(direct2LDS_A=False, direct2LDS_B=False),
    # ]

    yield from iterate_grid([base_configs, type_configs, workgroupMapping_configs])
    yield from iterate_grid([base_configs, type_configs, trans_configs])
    yield from iterate_grid([base_configs, type_configs, prefetch_configs])
    # yield from iterate_grid([base_configs, type_configs, direct2lds_configs])


def f8f6f4_gemm_gfx1250():
    base_configs = [
        merge_dicts(
            [
                BASE_PROBLEM_CONFIGS["GEMM_512x512x512"],
                BASE_SOLUTION_CONFIGS["1250_default"],
            ]
        )
    ]
    type_configs = list(TYPE_CONFIGS_MX.values())
    workgroupMapping_configs = [
        dict(workgroupMappingDim=0, workgroupMappingValue=2),
        dict(workgroupMappingDim=-1, workgroupMappingValue=-1),
    ]
    trans_configs = [
        dict(trans_A="N", trans_B="N"),
        dict(trans_A="T", trans_B="N"),
        dict(trans_A="T", trans_B="T"),
        dict(trans_A="N", trans_B="T"),
    ]
    # prefetch_configs = [
    #     dict(
    #         prefetch=True,
    #         prefetchInFlight=4,
    #         prefetchLDSFactor=1,
    #         prefetchMixMemOps=True,
    #     ),
    #     dict(prefetch=False),
    # ]
    # direct2lds_configs = [
    #     dict(direct2LDS_A=True, direct2LDS_B=True),
    #     dict(direct2LDS_A=False, direct2LDS_B=False),
    # ]
    scale_type_configs = [
        dict(
            scale_A="None",
            scaleType_A="None",
            scale_B="None",
            scaleType_B="None",
            scaleBlockSize=-1,
        ),
        dict(
            scale_A="Separate",
            scaleType_A="E8M0",
            scale_B="Separate",
            scaleType_B="E8M0",
            scaleBlockSize=32,
        ),
    ]
    lds_scale_configs = [
        # TODO: Support this config
        # dict(loadLDSScale_A=True, loadLDSScale_B=True),
        dict(loadLDSScale_A=False, loadLDSScale_B=False),
    ]
    scale_swizzle_prefetch_configs = [
        dict(prefetchScale=True, swizzleScale=True, storeLDS_D=False),
        dict(prefetchScale=False, swizzleScale=False, storeLDS_D=True),
    ]
    skip_permlane_configs = [
        # TODO: Support this config
        # dict(scaleSkipPermlane=True),
        dict(scaleSkipPermlane=False),
    ]
    scale_ON_config = [
        dict(
            scale_A="Separate",
            scaleType_A="E8M0",
            scale_B="Separate",
            scaleType_B="E8M0",
            scaleBlockSize=32,
        ),
    ]
    trans_TN_config = [
        dict(trans_A="T", trans_B="N"),
    ]

    yield from iterate_grid([base_configs, type_configs, workgroupMapping_configs])
    yield from iterate_grid([base_configs, type_configs, trans_configs])
    # TODO: support these configs
    # yield from iterate_grid([base_configs, type_configs, prefetch_configs])
    # yield from iterate_grid([base_configs, type_configs, direct2lds_configs])
    yield from iterate_grid([base_configs, type_configs, scale_type_configs])
    yield from iterate_grid(
        [
            base_configs,
            type_configs,
            trans_TN_config,
            scale_ON_config,
            lds_scale_configs,
            scale_swizzle_prefetch_configs,
        ],
    )
    yield from iterate_grid(
        [
            base_configs,
            type_configs,
            trans_TN_config,
            scale_ON_config,
            skip_permlane_configs,
        ]
    )


def stream_k_gfx1250(sweep=False):
    base_configs = [
        merge_dicts(
            [
                BASE_PROBLEM_CONFIGS["GEMM_512x512x512"],
                BASE_SOLUTION_CONFIGS["1250_default"],
                dict(mac_m=64, mac_n=64, mac_k=64),
            ]
        )
    ]
    type_configs = list(TYPE_CONFIGS_F16.values()) + list(TYPE_CONFIGS_F32.values())

    if sweep:
        mac_configs = [
            dict(mac_m=mac_m, mac_n=mac_n, mac_k=mac_k)
            for mac_m in [64, 128]
            for mac_n in [64, 128, 256]
            for mac_k in [32, 64]
        ]
    else:
        mac_configs = [
            dict(mac_m=128, mac_n=256, mac_k=32),
        ]
    stream_k_configs = [
        dict(streamK=True, streamKTwoTile=False),
        dict(streamK=True, streamKTwoTile=True),
    ]
    trans_configs = [
        dict(trans_A="N", trans_B="T"),
    ]

    def skip_filter(params):
        return params["streamKTwoTile"] and (
            params["mac_m"] * params["mac_n"] * params["mac_k"] >= (64 * 256 * 64)
        )

    yield from iterate_grid(
        [base_configs, type_configs, trans_configs, mac_configs, stream_k_configs],
        skip_filter=skip_filter,
    )


def all_gfx1250():
    yield from sgemm_gfx1250()
    yield from hgemm_gfx1250()
    yield from f8gemm_gfx1250()
    yield from f8f6f4_gemm_gfx1250()
    # TODO: support stream k
    # yield from stream_k_gfx1250(sweep=True)
    yield from codegen()
