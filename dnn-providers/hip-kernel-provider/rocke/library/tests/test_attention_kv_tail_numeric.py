# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""On-GPU guardrail: the tiled 2D and 3D unified-attention kernels (gfx942, gfx950) must not read
K/V outside a sequence. See ``_paged_kv_tail`` for how each case surrounds the sequences with
hostile cache data. A 2D case runs the default spec for its path, or that spec with the overrides
it names; a 3D case runs an explicit segment spec with the knobs it names.
"""

from __future__ import annotations

import math

import pytest
from _gpu_gate import gpu_arch, requires_gpu
from _paged_kv_tail import TOL, own_blocks, poison, result, spawn_selected
from test_gfx942_gqa_head_fold_numeric import _load_windowed_harness

ARCH = gpu_arch()
# Claimed cache size that switches the default spec to 64-bit K/V addressing (the tensors stay
# small: the kernel only addresses the sequences' own blocks).
I64_NUM_KV_BLOCKS = 1 << 17


def _run(case):
    """Run one case on the path it names; return max_abs against the reference."""
    import torch

    from dispatch.attention.common import AttentionTuningSpec
    from dispatch.attention.tuning_specs import (
        ExplicitAttention3DConfig,
        make_explicit_attention_3d_specs,
    )
    from kernels.common import attention_unified as au

    H = _load_windowed_harness()
    s = H.Scenario(
        name=case["id"],
        seq_lens=[tuple(x) for x in case["seq_lens"]],
        num_query_heads=case["hq"],
        num_kv_heads=case["hk"],
        head_size=case["d"],
        block_size=case["bs"],
        dtype=torch.bfloat16,
        # each sequence its own blocks, plus the spare one
        num_blocks=sum(-(-kv // case["bs"]) for _, kv in case["seq_lens"]) + 1,
    )
    data = H.make_inputs(s, seed=0)
    kv_lens = [x[1] for x in s.seq_lens]
    data["block_tables"] = own_blocks(kv_lens, s.block_size, s.num_blocks)
    fnuz = ARCH == "gfx942"
    if case["fp8"]:
        # The reference reads the same quantized values (scales are 1).
        fp8 = torch.float8_e4m3fnuz if fnuz else torch.float8_e4m3fn
        for name in ("key_cache", "value_cache"):
            data[name] = data[name].to(fp8)
        ref = H.run_reference(
            s,
            dict(
                data,
                key_cache=data["key_cache"].to(torch.bfloat16),
                value_cache=data["value_cache"].to(torch.bfloat16),
            ),
        )
    else:
        ref = H.run_reference(s, data)
    table = poison(
        data["key_cache"],
        data["value_cache"],
        data["block_tables"],
        kv_lens,
        s.block_size,
        case["poison"],
    )
    q = data["query"]
    num_kv_blocks = I64_NUM_KV_BLOCKS if case["i64"] else data["key_cache"].shape[0]
    problem = au.UnifiedAttentionProblem(
        total_q=q.shape[0],
        num_seqs=len(s.seq_lens),
        num_query_heads=s.num_query_heads,
        num_kv_heads=s.num_kv_heads,
        head_size=s.head_size,
        block_size=s.block_size,
        max_seqlen_q=data["max_query_len"],
        max_seqlen_k=data["max_kv_len"],
        dtype="bf16",
        use_fp8=case["fp8"],
        fp8_fnuz=case["fp8"] and fnuz,
        num_cus=torch.cuda.get_device_properties(0).multi_processor_count,
        num_kv_blocks=num_kv_blocks,
    )
    reduce = None
    if case["path"] == "3d":
        config = ExplicitAttention3DConfig(
            num_segments=0,
            tile_policy=case["tile"],
            knobs=tuple(case["overrides"].items()),
        )
        spec, reduce = make_explicit_attention_3d_specs(problem, config, arch=ARCH)
    else:
        spec = au._tiled_spec_from_problem(problem, ARCH, overrides=case["overrides"])
    assert spec.use_i64_kv_addr == case["i64"]
    # A dropped knob would run the default loader instead of the one the case names.
    for knob, value in case["overrides"].items():
        assert getattr(spec, knob) == value, f"{knob}={value} did not take effect"
    tuning_spec = AttentionTuningSpec(
        path=case["path"],
        arch=ARCH,
        builder_kind="tiled_3d" if case["path"] == "3d" else "tiled",
        compile_backend="llvm",
        candidate_name="kv_tail",
        tuning_id=f"kv_tail_{case['id']}",
        kernel_spec=spec,
        reduce_spec=reduce,
        num_kv_blocks=num_kv_blocks,
    )
    out = torch.empty_like(q)
    au.run_unified_attention_torch(
        problem=problem,
        q=q,
        k=data["key_cache"],
        v=data["value_cache"],
        out=out,
        cu_seqlens_q=data["cu_q"],
        seqused_k=data["kv_lens"],
        softmax_scale=1 / math.sqrt(s.head_size),
        block_table=table,
        softcap=0.0,
        tuning_spec=tuning_spec,
    )
    torch.cuda.synchronize()  # an out-of-bounds read surfaces here as a HIP error
    return H.compare(ref, out)["max_abs"]


def _shape(
    name,
    seq_lens,
    bs,
    hq=32,
    hk=8,
    d=128,
    fp8=False,
    i64=False,
    overrides=None,
    path="2d",
    tile="1x",
    arches=("gfx942", "gfx950"),
):
    return {
        "name": name,
        "seq_lens": seq_lens,
        "bs": bs,
        "hq": hq,
        "hk": hk,
        "d": d,
        "fp8": fp8,
        "i64": i64,
        "overrides": overrides or {},
        "path": path,
        "tile": tile,
        "arches": arches,
    }


# Lengths that are not a multiple of any block size or tile width; ragged batches put shorter
# sequences next to longer ones (their block-table rows end early).
_DECODE_LENS = [(1, 1001), (1, 777), (1, 130), (1, 1500)]
_SHAPES = [
    _shape("prefill_2d_bs16", [(1000, 1000)], 16),
    _shape("prefill_2d_bs64", [(1000, 1000)], 64),
    _shape("prefill_2d_ragged_bs16", [(300, 300), (1000, 1000)], 16),
    _shape("chunked_2d_small_q_bs16", [(200, 1000)], 16),
    _shape("decode_2d_bs16", _DECODE_LENS, 16),
    _shape("decode_2d_bs64", _DECODE_LENS, 64),
    _shape("decode_2d_d256_bs16", _DECODE_LENS, 16, hq=16, hk=2, d=256),
    _shape("prefill_2d_i64_bs16", [(1000, 1000)], 16, i64=True),
    _shape("prefill_2d_d64_bs32", [(1000, 1000)], 32, hq=64, hk=8, d=64),
    _shape(
        "prefill_2d_fast_d64_bs32",
        [(1000, 1000)],
        32,
        hq=64,
        hk=8,
        d=64,
        overrides={
            "use_fast_paged_kv_desc": True,
            "tile_size": 64,
            "num_warps": 4,
            # gfx942's sliced-K ring has no fast-descriptor path
            **({"use_k_sliced_ring": False} if ARCH == "gfx942" else {}),
        },
    ),
    _shape("prefill_2d_fp8_bs16", [(1000, 1000)], 16, fp8=True, arches=("gfx950",)),
    # gfx950 page copies whose copy spans pages (32 tokens of 16-token pages: each wave's
    # part has its own page) and whose copy ends mid-row (head size 192).
    *(
        _shape(
            f"{kind}_2d_{name}{'_i64' if i64 else ''}",
            lens,
            16,
            hq=hq,
            hk=hk,
            d=d,
            i64=i64,
            overrides={"num_warps": 4, "tile_size": 64},
            arches=("gfx950",),
        )
        for name, hq, hk, d in (
            ("d64_bs16_4w", 32, 8, 64),
            ("d192_bs16_4w", 16, 16, 192),
        )
        for kind, lens in (
            ("prefill", [(300, 300), (1000, 1000)]),
            ("decode", _DECODE_LENS),
        )
        for i64 in (False, True)
    ),
    # Tiled 3D (split-KV segments): each V loader variant.
    _shape("decode_3d_bs16", _DECODE_LENS, 16, path="3d"),
    _shape("decode_3d_b16_bs16", _DECODE_LENS * 4, 16, path="3d"),
    _shape(
        "decode_3d_bs16_page_ids",
        _DECODE_LENS,
        16,
        path="3d",
        overrides={"use_wave_page_id_load": True},
    ),
    _shape(
        "decode_3d_bs16_swizzle",
        _DECODE_LENS,
        16,
        path="3d",
        overrides={"use_kv_lds_swizzle": True},
    ),
    _shape(
        "decode_3d_d64_bs32_page_ids",
        _DECODE_LENS,
        32,
        hq=64,
        hk=8,
        d=64,
        path="3d",
        overrides={"use_wave_page_id_load": True},
    ),
    _shape("decode_3d_bs64", _DECODE_LENS, 64, path="3d"),
    _shape(
        "decode_3d_bs64_half_tile",
        _DECODE_LENS,
        64,
        path="3d",
        tile="half",
        arches=("gfx942",),
    ),
    _shape(
        "decode_3d_bs16_two_page_tile",
        _DECODE_LENS,
        16,
        path="3d",
        tile="2x",
        arches=("gfx950",),
    ),
    _shape("decode_3d_d64_bs32", _DECODE_LENS, 32, hq=64, hk=8, d=64, path="3d"),
    _shape("decode_3d_d256_bs16", _DECODE_LENS, 16, hq=16, hk=2, d=256, path="3d"),
    _shape(
        "decode_3d_d192_bs16",
        _DECODE_LENS,
        16,
        hq=16,
        hk=16,
        d=192,
        path="3d",
        arches=("gfx950",),
    ),
    _shape(
        "decode_3d_i64_bs16",
        _DECODE_LENS,
        16,
        i64=True,
        path="3d",
        arches=("gfx950",),
    ),
    _shape(
        "decode_3d_bs16_wide",
        _DECODE_LENS,
        16,
        path="3d",
        overrides={"use_wide_kv_load": True},
        arches=("gfx942",),
    ),
    _shape("decode_3d_fp8_bs16", _DECODE_LENS, 16, fp8=True, path="3d"),
]
CASES = [
    dict(shape, id=f"{shape['name']}_{mode}", poison=mode)
    for shape in _SHAPES
    if ARCH in shape["arches"]
    for mode in ("nan", "far")
]


@pytest.fixture(scope="module")
def results(request):
    """Start each selected case in its own fresh process up front; the tests collect the results."""
    if _load_windowed_harness() is None:
        yield None
        return
    yield from spawn_selected(request, _run)


@requires_gpu("gfx942", "gfx950")
@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_kv_tail_reads_in_bounds(case, results):
    if results is None:
        pytest.skip("windowed parity harness not present in this checkout")
    assert result(results, case["id"]) <= TOL


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
