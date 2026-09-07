# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Correction-biased top-k fused with compact active-expert packing.

One workgroup handles a small decode routing batch. For every token it applies
sigmoid scoring, selects top-k experts using correction-biased scores, retains
the unbiased scores as routing weights, normalizes them, counts active experts,
prefix-scans padded expert-block counts, and scatters the selected token metadata
into the compact block layout consumed by grouped MoE kernels.

The first implementation intentionally supports the one-group routing case.
That is still expressed through ``num_expert_groups`` / ``topk_groups`` so a
future grouped selector can extend the same ABI without silently changing the
current selection rule.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...core.arch import ArchTarget
from ...core.ir import F32, I32, IRBuilder, KernelDef, PtrType
from ...helpers.reduction import block_lds_reduce
from ...helpers.scan import block_exclusive_scan_i32
from ...helpers.spec import SignatureBuilder, kernel_name_join

_NEG_INF = -3.4028234663852886e38
_INDEX_SENTINEL = 16_777_216.0
_LOG2_E = 1.4426950408889634


@dataclass(frozen=True)
class MoeTopkActivePackSpec:
    """One small-batch top-k and compact-packing configuration."""

    tokens: int
    experts: int
    topk: int
    tile_m: int = 16
    block_size: int = 1024
    num_expert_groups: int = 1
    topk_groups: int = 1
    renormalize: bool = True
    local_experts: int | None = None
    name: str = "rocke_moe_topk_active_pack"

    @property
    def total_pairs(self) -> int:
        return self.tokens * self.topk

    @property
    def max_blocks(self) -> int:
        return self.total_pairs

    @property
    def max_padded_pairs(self) -> int:
        return self.max_blocks * self.tile_m

    @property
    def max_blocks_per_expert(self) -> int:
        return (self.tokens + self.tile_m - 1) // self.tile_m

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            f"T{self.tokens}",
            f"E{self.experts}",
            f"K{self.topk}",
            f"G{self.num_expert_groups}x{self.topk_groups}",
            f"tm{self.tile_m}",
            f"b{self.block_size}",
            flags={
                "rn": self.renormalize,
                "local": self.local_experts is not None,
            },
        )


def is_valid_spec(
    spec: MoeTopkActivePackSpec, arch: str = "gfx950"
) -> tuple[bool, str]:
    """Return whether the fused routing/packing kernel can own the request."""

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as error:
        return False, str(error)
    if spec.tokens <= 0 or spec.experts <= 0 or spec.topk <= 0:
        return False, (
            "tokens, experts, and topk must be positive "
            f"(got {spec.tokens}, {spec.experts}, {spec.topk})"
        )
    if spec.topk > spec.experts:
        return False, f"topk ({spec.topk}) must be <= experts ({spec.experts})"
    if spec.local_experts is not None and not 1 <= spec.local_experts <= spec.experts:
        return False, (
            "local_experts must be in [1, experts] "
            f"(got {spec.local_experts}/{spec.experts})"
        )
    if spec.topk > 32:
        return False, f"topk ({spec.topk}) must be <= 32"
    if spec.tile_m <= 0:
        return False, f"tile_m must be positive (got {spec.tile_m})"
    if spec.block_size not in (64, 128, 256, 512, 1024):
        return False, f"unsupported block_size {spec.block_size}"
    if spec.block_size > target.max_threads_per_block:
        return False, (
            f"block_size {spec.block_size} > max_threads_per_block "
            f"{target.max_threads_per_block} on {arch}"
        )
    if spec.experts > spec.block_size:
        return False, (
            f"experts ({spec.experts}) > block_size ({spec.block_size}); "
            "the single-workgroup scan requires one lane per expert"
        )
    if spec.total_pairs > spec.block_size:
        return False, (
            f"tokens*topk ({spec.total_pairs}) > block_size ({spec.block_size}); "
            "the decode path supports one routed pair per lane"
        )
    if spec.num_expert_groups != 1 or spec.topk_groups != 1:
        return False, (
            "v1 supports the one-group routing case only "
            f"(got {spec.num_expert_groups}/{spec.topk_groups})"
        )
    bytes_lds = spec.block_size * 4 + 3 * spec.experts * 4 + spec.total_pairs * (4 + 4)
    if not target.fits_lds(bytes_lds):
        return False, (
            f"LDS budget {bytes_lds} > {target.lds_capacity_bytes} cap on {arch}"
        )
    return True, "ok"


def _load_lds_i32(b: IRBuilder, ptr, index):
    return b.vec_extract(b.smem_load_vN(ptr, index, dtype=I32, n=1), 0)


def _load_lds_f32(b: IRBuilder, ptr, index):
    return b.vec_extract(b.smem_load_vN(ptr, index, dtype=F32, n=1), 0)


def build_moe_topk_active_pack(
    spec: MoeTopkActivePackSpec, arch: str = "gfx950"
) -> KernelDef:
    """Build the one-workgroup routing and compact-packing kernel.

    Output capacities are compile-time and intentionally conservative:

    * ``Sorted*``: ``spec.max_padded_pairs`` elements;
    * ``BlockExpertIds``: ``spec.max_blocks`` elements;
    * ``Counts`` / ``BlockOffsets``: ``spec.experts`` elements;
    * ``NumBlocks``: one i32.

    Unused output slots are initialized to ``-1`` token/top-k/expert ids and
    zero weights. A downstream grouped kernel can therefore launch the static
    ``max_blocks`` grid and skip sentinel blocks without a host readback.

    ABI::

        Logits, CorrectionBias,
        SortedTokenIds, SortedTopkIds, SortedWeights,
        BlockExpertIds, Counts, BlockOffsets, NumBlocks,
        tokens, experts, routed_scale
    """

    ok, why = is_valid_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid moe topk-active-pack spec for {arch}: {why}")

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size

    logits = b.param(
        "Logits", PtrType(F32, "global"), noalias=True, readonly=True, align=16
    )
    correction_bias = b.param(
        "CorrectionBias",
        PtrType(F32, "global"),
        noalias=True,
        readonly=True,
        align=16,
    )
    sorted_token_ids = b.param(
        "SortedTokenIds", PtrType(I32, "global"), writeonly=True, align=4
    )
    sorted_topk_ids = b.param(
        "SortedTopkIds", PtrType(I32, "global"), writeonly=True, align=4
    )
    sorted_weights = b.param(
        "SortedWeights", PtrType(F32, "global"), writeonly=True, align=4
    )
    block_expert_ids = b.param(
        "BlockExpertIds", PtrType(I32, "global"), writeonly=True, align=4
    )
    counts = b.param("Counts", PtrType(I32, "global"), writeonly=True, align=4)
    block_offsets = b.param(
        "BlockOffsets", PtrType(I32, "global"), writeonly=True, align=4
    )
    num_blocks = b.param("NumBlocks", PtrType(I32, "global"), writeonly=True, align=4)
    _tokens = b.param("tokens", I32)
    _experts = b.param("experts", I32)
    routed_scale = b.param("routed_scale", F32)
    expert_start = (
        b.param("expert_start", I32) if spec.local_experts is not None else None
    )

    tid = b.thread_id_x()
    c_zero = b.const_i32(0)
    c_one = b.const_i32(1)
    c_experts = b.const_i32(spec.experts)
    c_pairs = b.const_i32(spec.total_pairs)
    c_tile_m = b.const_i32(spec.tile_m)
    c_neg_inf = b.const_f32(_NEG_INF)
    c_index_sentinel = b.const_f32(_INDEX_SENTINEL)
    c_one_f32 = b.const_f32(1.0)
    c_neg_log2e = b.const_f32(-_LOG2_E)

    lds_reduce = b.smem_alloc(F32, [spec.block_size], name_hint="topk_reduce")
    lds_counts = b.smem_alloc(I32, [spec.experts], name_hint="expert_counts")
    lds_block_offsets = b.smem_alloc(
        I32, [spec.experts], name_hint="expert_block_offsets"
    )
    lds_counters = b.smem_alloc(I32, [spec.experts], name_hint="expert_counters")
    lds_ids = b.smem_alloc(I32, [spec.total_pairs], name_hint="topk_ids")
    lds_weights = b.smem_alloc(F32, [spec.total_pairs], name_hint="topk_weights")

    in_experts = b.cmp_lt(tid, c_experts)
    with b.scf_if(in_experts):
        b.smem_store_vN(lds_counts, [tid], c_zero, 1)
        b.smem_store_vN(lds_counters, [tid], c_zero, 1)

    in_blocks = b.cmp_lt(tid, c_pairs)
    with b.scf_if(in_blocks):
        b.global_store(block_expert_ids, tid, b.const_i32(-1), align=4)

    init_passes = (spec.max_padded_pairs + spec.block_size - 1) // spec.block_size
    for init_pass in range(init_passes):
        output_index = b.add(tid, b.const_i32(init_pass * spec.block_size))
        in_output = b.cmp_lt(output_index, b.const_i32(spec.max_padded_pairs))
        with b.scf_if(in_output):
            b.global_store(sorted_token_ids, output_index, b.const_i32(-1), align=4)
            b.global_store(sorted_topk_ids, output_index, b.const_i32(-1), align=4)
            b.global_store(sorted_weights, output_index, b.const_f32(0.0), align=4)
    with b.scf_if(b.cmp_eq(tid, c_zero)):
        b.global_store(num_blocks, c_zero, c_zero, align=4)
    b.sync()

    if spec.block_size >= spec.tokens * 64:
        # Decode fast path: one wave owns one token. Every lane keeps the
        # sigmoid+bias scores for its strided experts in registers, then 16
        # wave-level argmax reductions select top-k concurrently for all
        # tokens. This removes the token-serial block reductions that dominate
        # T8 routing at E=896.
        lane = b.mod(tid, b.const_i32(64))
        wave = b.div(tid, b.const_i32(64))
        wave_valid = b.cmp_lt(wave, b.const_i32(spec.tokens))
        safe_token = b.select(wave_valid, wave, c_zero)
        candidate_ids: list = []
        candidate_scores: list = []
        candidate_weights: list = []
        candidates_per_lane = (spec.experts + 63) // 64
        for candidate in range(candidates_per_lane):
            expert = b.add(lane, b.const_i32(candidate * 64))
            expert_valid = b.land(wave_valid, b.cmp_lt(expert, c_experts))
            safe_expert = b.select(expert_valid, expert, c_zero)
            logit_offset = b.add(b.mul(safe_token, c_experts), safe_expert)
            logit = b.global_load_f32(logits, logit_offset)
            bias = b.global_load_f32(correction_bias, safe_expert)
            score = b.rcp(b.fadd(c_one_f32, b.exp2(b.fmul(c_neg_log2e, logit))))
            candidate_ids.append(expert)
            candidate_weights.append(score)
            candidate_scores.append(
                b.select(expert_valid, b.fadd(score, bias), c_neg_inf)
            )

        for topk_slot in range(spec.topk):
            best_score = c_neg_inf
            best_id = b.const_i32(spec.experts)
            best_weight = b.const_f32(0.0)
            for candidate in range(candidates_per_lane):
                score = candidate_scores[candidate]
                expert = candidate_ids[candidate]
                better = b.lor(
                    b.fcmp("ogt", score, best_score),
                    b.land(
                        b.fcmp("oeq", score, best_score),
                        b.cmp_lt(expert, best_id),
                    ),
                )
                best_score = b.select(better, score, best_score)
                best_id = b.select(better, expert, best_id)
                best_weight = b.select(
                    better, candidate_weights[candidate], best_weight
                )

            for xor_mask in (32, 16, 8, 4, 2, 1):
                other_score = b.warp_shuffle_xor(best_score, xor_mask)
                other_id = b.warp_shuffle_xor(best_id, xor_mask)
                other_weight = b.warp_shuffle_xor(best_weight, xor_mask)
                other_better = b.lor(
                    b.fcmp("ogt", other_score, best_score),
                    b.land(
                        b.fcmp("oeq", other_score, best_score),
                        b.cmp_lt(other_id, best_id),
                    ),
                )
                best_score = b.select(other_better, other_score, best_score)
                best_id = b.select(other_better, other_id, best_id)
                best_weight = b.select(other_better, other_weight, best_weight)

            pair = b.add(
                b.mul(wave, b.const_i32(spec.topk)),
                b.const_i32(topk_slot),
            )
            is_writer = b.land(wave_valid, b.cmp_eq(lane, c_zero))
            with b.scf_if(is_writer):
                b.smem_store_vN(lds_ids, [pair], best_id, 1)
                b.smem_store_vN(lds_weights, [pair], best_weight, 1)
                b.lds_atomic_add(lds_counts, [best_id], c_one)
            candidate_scores = [
                b.select(b.cmp_eq(expert, best_id), c_neg_inf, score)
                for expert, score in zip(candidate_ids, candidate_scores)
            ]
    else:
        lane_as_f32 = b.sitofp_f32(tid)
        token_loop = b.scf_for(c_zero, b.const_i32(spec.tokens), c_one, iv_name="token")
        with token_loop as token:
            expert_valid = b.cmp_lt(tid, c_experts)
            safe_expert = b.select(expert_valid, tid, c_zero)
            logit_offset = b.add(b.mul(token, c_experts), safe_expert)
            logit = b.global_load_f32(logits, logit_offset)
            bias = b.global_load_f32(correction_bias, safe_expert)
            score = b.rcp(b.fadd(c_one_f32, b.exp2(b.fmul(c_neg_log2e, logit))))
            selected_score = b.select(expert_valid, b.fadd(score, bias), c_neg_inf)

            topk_loop = b.scf_for_iter(
                c_zero,
                b.const_i32(spec.topk),
                c_one,
                [("selected_score", selected_score)],
                iv_name="topk_slot",
            )
            with topk_loop as (topk_slot, (current_score,)):
                winning_score = block_lds_reduce(
                    b,
                    current_score,
                    lds_reduce,
                    tid,
                    block_size=spec.block_size,
                    combine="max",
                )
                is_max = b.land(
                    expert_valid, b.fcmp("oeq", current_score, winning_score)
                )
                candidate_index = b.select(is_max, lane_as_f32, c_index_sentinel)
                winning_index = block_lds_reduce(
                    b,
                    candidate_index,
                    lds_reduce,
                    tid,
                    block_size=spec.block_size,
                    combine="min",
                )
                is_winner = b.land(is_max, b.fcmp("oeq", lane_as_f32, winning_index))
                pair = b.add(b.mul(token, b.const_i32(spec.topk)), topk_slot)
                with b.scf_if(is_winner):
                    b.smem_store_vN(lds_ids, [pair], tid, 1)
                    b.smem_store_vN(lds_weights, [pair], score, 1)
                    b.lds_atomic_add(lds_counts, [tid], c_one)
                b.scf_yield(b.select(is_winner, c_neg_inf, current_score))

    b.sync()

    in_pairs = b.cmp_lt(tid, c_pairs)
    with b.scf_if(in_pairs):
        token = b.div(tid, b.const_i32(spec.topk))
        token_base = b.mul(token, b.const_i32(spec.topk))
        weight = _load_lds_f32(b, lds_weights, tid)
        if spec.renormalize:
            weight_sum = b.const_f32(0.0)
            for topk_slot in range(spec.topk):
                weight_sum = b.fadd(
                    weight_sum,
                    _load_lds_f32(
                        b,
                        lds_weights,
                        b.add(token_base, b.const_i32(topk_slot)),
                    ),
                )
            weight = b.fmul(weight, b.rcp(weight_sum))
        weight = b.fmul(weight, routed_scale)
        b.smem_store_vN(lds_weights, [tid], weight, 1)
    b.sync()

    with b.scf_if(in_experts):
        count = _load_lds_i32(b, lds_counts, tid)
        blocks = b.div(b.add(count, b.const_i32(spec.tile_m - 1)), c_tile_m)
        b.global_store(counts, tid, count, align=4)
        b.smem_store_vN(lds_block_offsets, [tid], blocks, 1)
    b.sync()

    block_exclusive_scan_i32(
        b,
        lds_block_offsets,
        tid=tid,
        block_size=spec.block_size,
        length=spec.experts,
    )

    with b.scf_if(in_experts):
        offset = _load_lds_i32(b, lds_block_offsets, tid)
        count = _load_lds_i32(b, lds_counts, tid)
        blocks = b.div(b.add(count, b.const_i32(spec.tile_m - 1)), c_tile_m)
        b.global_store(block_offsets, tid, offset, align=4)
        for block in range(spec.max_blocks_per_expert):
            with b.scf_if(b.cmp_lt(b.const_i32(block), blocks)):
                block_expert = tid
                if expert_start is not None:
                    local_expert = b.sub(tid, expert_start)
                    in_local_range = b.land(
                        b.cmp_ge(tid, expert_start),
                        b.cmp_lt(
                            local_expert,
                            b.const_i32(spec.local_experts),
                        ),
                    )
                    block_expert = b.select(
                        in_local_range,
                        local_expert,
                        b.const_i32(-1),
                    )
                b.global_store(
                    block_expert_ids,
                    b.add(offset, b.const_i32(block)),
                    block_expert,
                    align=4,
                )

    with b.scf_if(b.cmp_eq(tid, b.const_i32(spec.experts - 1))):
        last_offset = _load_lds_i32(b, lds_block_offsets, tid)
        last_count = _load_lds_i32(b, lds_counts, tid)
        last_blocks = b.div(b.add(last_count, b.const_i32(spec.tile_m - 1)), c_tile_m)
        b.global_store(num_blocks, c_zero, b.add(last_offset, last_blocks), align=4)
    b.sync()

    with b.scf_if(in_pairs):
        expert = _load_lds_i32(b, lds_ids, tid)
        weight = _load_lds_f32(b, lds_weights, tid)
        local_offset = b.lds_atomic_add(lds_counters, [expert], c_one)
        expert_block = _load_lds_i32(b, lds_block_offsets, expert)
        output_index = b.add(b.mul(expert_block, c_tile_m), local_offset)
        token = b.div(tid, b.const_i32(spec.topk))
        topk_slot = b.mod(tid, b.const_i32(spec.topk))
        b.global_store(sorted_token_ids, output_index, token, align=4)
        b.global_store(sorted_topk_ids, output_index, topk_slot, align=4)
        b.global_store(sorted_weights, output_index, weight, align=4)

    b.ret()
    return b.kernel


def moe_topk_active_pack_grid(
    spec: MoeTopkActivePackSpec,
) -> tuple[int, int, int]:
    """The complete decode routing batch is owned by one workgroup."""

    _ = spec
    return (1, 1, 1)


def moe_topk_active_pack_signature(
    spec: MoeTopkActivePackSpec,
) -> list[dict[str, str]]:
    _ = spec
    signature = (
        SignatureBuilder()
        .ptr("Logits", "f32")
        .ptr("CorrectionBias", "f32")
        .ptr("SortedTokenIds", "i32")
        .ptr("SortedTopkIds", "i32")
        .ptr("SortedWeights", "f32")
        .ptr("BlockExpertIds", "i32")
        .ptr("Counts", "i32")
        .ptr("BlockOffsets", "i32")
        .ptr("NumBlocks", "i32")
        .scalar("tokens", "i32")
        .scalar("experts", "i32")
        .scalar("routed_scale", "f32")
    )
    if spec.local_experts is not None:
        signature = signature.scalar("expert_start", "i32")
    return signature.build()


__all__ = [
    "MoeTopkActivePackSpec",
    "build_moe_topk_active_pack",
    "is_valid_spec",
    "moe_topk_active_pack_grid",
    "moe_topk_active_pack_signature",
]
