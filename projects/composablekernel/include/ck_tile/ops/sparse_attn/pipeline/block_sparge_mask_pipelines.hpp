// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core.hpp"

#include <cassert>

namespace ck_tile {

// Generic block reduction: intra-warp shuffle, then cross-warp via scratch[0..numWarps).
// `identity` fills lanes whose warp is out of range. Result broadcast through scratch[0].
template <index_t BlockSize, typename T, typename Op>
CK_TILE_DEVICE T
block_reduce_smem(T val, T* scratch, index_t tid, Op op, T identity)
{
    for(index_t off = get_warp_size() / 2; off > 0; off /= 2)
        val = op(val, warp_shuffle_down(val, static_cast<unsigned>(off)));
    if(tid % get_warp_size() == 0)
        scratch[tid / get_warp_size()] = val;
    __builtin_amdgcn_s_barrier();

    if(tid < static_cast<index_t>(get_warp_size()))
    {
        const index_t num_warps = (BlockSize + get_warp_size() - 1) / get_warp_size();
        T v = (tid < num_warps) ? scratch[tid] : identity;
        for(index_t off = get_warp_size() / 2; off > 0; off /= 2)
            v = op(v, warp_shuffle_down(v, static_cast<unsigned>(off)));
        if(tid == 0)
            scratch[0] = v;
    }
    __builtin_amdgcn_s_barrier();
    return scratch[0];
}

template <index_t BlockSize>
CK_TILE_DEVICE float
block_reduce_max_f32(float val, float* scratch, index_t tid)
{
    return block_reduce_smem<BlockSize>(
        val, scratch, tid,
        [](float a, float b) { return a > b ? a : b; },
        -INFINITY);
}
template <index_t BlockSize>
CK_TILE_DEVICE float
block_reduce_sum_f32(float val, float* scratch, index_t tid)
{
    return block_reduce_smem<BlockSize>(
        val, scratch, tid,
        [](float a, float b) { return a + b; },
        0.0f);
}
template <index_t BlockSize>
CK_TILE_DEVICE int32_t
block_reduce_min_i32(int32_t val, int32_t* scratch, index_t tid)
{
    return block_reduce_smem<BlockSize>(
        val, scratch, tid,
        [](int32_t a, int32_t b) { return a < b ? a : b; },
        INT32_MAX);
}

// In-place descending bitonic sort with int32 payload. N must be power of two.
// Inner stages with j < warp_size stay within a wavefront, so wave_barrier
// suffices in place of the CTA-wide s_barrier.
template <index_t N>
CK_TILE_DEVICE void
bitonic_sort_desc_smem(float* keys, int32_t* vals, index_t tid)
{
    static_assert((N & (N - 1)) == 0 && N > 0, "N must be power of two");

    const index_t wsz = static_cast<index_t>(get_warp_size());

    for(index_t k = 2; k <= N; k <<= 1)
    {
        for(index_t j = k >> 1; j > 0; j >>= 1)
        {
            const index_t i   = tid;
            const index_t ixj = i ^ j;
            if(i < N && ixj < N && ixj > i)
            {
                const bool flip = ((i & k) == 0)
                                      ? (keys[i] < keys[ixj])
                                      : (keys[i] > keys[ixj]);
                if(flip)
                {
                    float kt   = keys[i]; keys[i] = keys[ixj]; keys[ixj] = kt;
                    int32_t vt = vals[i]; vals[i] = vals[ixj]; vals[ixj] = vt;
                }
            }
            if(j >= wsz)
                __builtin_amdgcn_s_barrier();
            else
                __builtin_amdgcn_wave_barrier();
        }
    }
}

// Hillis-Steele inclusive prefix sum on first N elements; aux is scratch of size >= N.
template <index_t N>
CK_TILE_DEVICE void
block_scan_inclusive_sum_smem(float* buf, float* aux, index_t tid)
{
    if(tid < N) aux[tid] = buf[tid];
    __builtin_amdgcn_s_barrier();

    for(index_t d = 1; d < N; d <<= 1)
    {
        float v   = (tid < N) ? aux[tid] : 0.0f;
        float add = (tid >= d && tid < N) ? aux[tid - d] : 0.0f;
        __builtin_amdgcn_s_barrier();
        if(tid < N) aux[tid] = v + add;
        __builtin_amdgcn_s_barrier();
    }
    if(tid < N) buf[tid] = aux[tid];
    __builtin_amdgcn_s_barrier();
}

// Per-block mean + optional cosine similarity. Reads fp16/bf16, converts on the fly.
template <typename InputType_>
struct BlockSpargePreprocessPipeline
{
    using InputType = remove_cvref_t<InputType_>;

    static constexpr index_t kBlockSize   = 256;
    static constexpr float   kNormEpsilon = 1e-8f; // sqrt argument floor

    struct Params
    {
        index_t seqlen;
        index_t hdim;
        index_t block_size;
        index_t block_id;
        index_t stride_seq;     // tokens stride in elements; hdim for BHSD, H*hdim for BSHD
        float simthreshold;
        const float* km_ptr;    // [hdim] K-mean; nullptr disables (Q always nullptr)
    };

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize(index_t hdim)
    {
        return static_cast<index_t>((hdim + kBlockSize) * sizeof(float));
    }

    CK_TILE_DEVICE void operator()(
        const InputType* slice,
        float* mean_out,
        float* sim_out,
        const Params& params,
        void* smem) const
    {
        const index_t tid  = static_cast<index_t>(threadIdx.x);
        const index_t hdim = params.hdim;

        float* s_mean    = reinterpret_cast<float*>(smem);
        float* s_scratch = s_mean + hdim;

        const index_t s_start = params.block_id * params.block_size;
        const index_t s_end   = min(s_start + params.block_size, params.seqlen);
        const index_t count   = s_end - s_start;

        if(count <= 0)
        {
            for(index_t d = tid; d < hdim; d += kBlockSize)
                mean_out[d] = 0.0f;
            if(sim_out != nullptr && tid == 0)
                *sim_out = 1.0f;
            return;
        }

        // km_ptr is WG-uniform — null-check hoisted out of all inner loops.
        const bool has_km = (params.km_ptr != nullptr);
        // Pass 1: block mean (subtract km if smoothing)
        for(index_t d = tid; d < hdim; d += kBlockSize)
        {
            const float km = has_km ? params.km_ptr[d] : 0.0f;
            float sum = 0.0f;
            for(index_t s = s_start; s < s_end; ++s)
                sum += type_convert<float>(
                    slice[static_cast<long_index_t>(s) * params.stride_seq + d]) - km;
            float m     = sum / static_cast<float>(count);
            s_mean[d]   = m;
            mean_out[d] = m;
        }
        __builtin_amdgcn_s_barrier();

        // Pass 2: cosine similarity (optional; same km treatment)
        if(sim_out != nullptr && params.simthreshold > 0.0f)
        {
            float local_norm_sq = 0.0f;
            for(index_t d = tid; d < hdim; d += kBlockSize)
                local_norm_sq += s_mean[d] * s_mean[d];
            float mean_norm = __builtin_sqrtf(
                block_reduce_sum_f32<kBlockSize>(local_norm_sq, s_scratch, tid) + kNormEpsilon);

            float local_sim_sum = 0.0f;
            for(index_t s = s_start + tid; s < s_end; s += kBlockSize)
            {
                float dot      = 0.0f;
                float tok_norm = 0.0f;
                for(index_t d = 0; d < hdim; ++d)
                {
                    const float km = has_km ? params.km_ptr[d] : 0.0f;
                    float v = type_convert<float>(
                        slice[static_cast<long_index_t>(s) * params.stride_seq + d]) - km;
                    dot += v * s_mean[d];
                    tok_norm += v * v;
                }
                tok_norm = __builtin_sqrtf(tok_norm + kNormEpsilon);
                local_sim_sum += dot / (tok_norm * mean_norm);
            }
            float avg_sim =
                block_reduce_sum_f32<kBlockSize>(local_sim_sum, s_scratch, tid) /
                static_cast<float>(count);
            if(tid == 0)
                *sim_out = avg_sim;
        }
        else if(sim_out != nullptr && tid == 0)
        {
            *sim_out = 1.0f;
        }
    }

};

// Per Q-block: scores + softmax + sort-based selection -> delta-encoded LUT.
struct BlockSpargeMaskPredictionPipeline
{
    static constexpr index_t kBlockSize      = 256;
    static constexpr index_t kMaxKBlocksPow2 = 256; // seqlen_k up to 32k @ BLKK=128
    // Cross-warp scratch: kBlockSize / min_warp_size = 256/32 = 8; round up for safety.
    static constexpr index_t kReduceScratchSlots = (kBlockSize + 31) / 32;
    // OOB sentinel: finite (not -INF) so softmax max-subtract avoids inf-inf=NaN.
    static constexpr float   kScoreOOB           = -1.0e30f;
    static constexpr float   kScoreSelected      = -2.0f;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize(index_t hdim, index_t num_k_blocks)
    {
        // [q_mean | scores | sort_keys | sort_vals | aux | scratch | n_target]
        return static_cast<index_t>(
            (hdim + num_k_blocks + 3 * kMaxKBlocksPow2) * sizeof(float)
            + (kReduceScratchSlots + 1) * sizeof(int32_t));
    }

    struct MaskRunArgs
    {
        const float* k_means;
        const float* q_means;
        const float* k_sim;     // [B, Hk, num_k_blocks] or nullptr
        const float* q_sim;     // [B, Hq, num_q_blocks] or nullptr
        int32_t*     lut_row;
        int32_t*     vbn_ptr;
        index_t b, head, kv_head, q_block;
        index_t nhead_q, nhead_k, num_q_blocks, num_k_blocks, hdim;
        float head_cdfthreshd, head_topk, head_simthreshold;
        index_t causal_type;
        bool    attention_sink;
        index_t seqlen_q, seqlen_k, block_size;
        index_t window_left, window_right;
    };

    CK_TILE_DEVICE void run_with_indices(const MaskRunArgs& args, void* smem) const
    {
        const float* __restrict__ k_means = args.k_means;
        const float* __restrict__ q_means = args.q_means;
        const float* __restrict__ k_sim   = args.k_sim;
        const float* __restrict__ q_sim   = args.q_sim;
        int32_t*     __restrict__ lut_row = args.lut_row;
        int32_t*     __restrict__ vbn_ptr = args.vbn_ptr;
        const index_t b              = args.b;
        const index_t head           = args.head;
        const index_t kv_head        = args.kv_head;
        const index_t q_block        = args.q_block;
        const index_t nhead_q        = args.nhead_q;
        const index_t nhead_k        = args.nhead_k;
        const index_t num_q_blocks   = args.num_q_blocks;
        const index_t num_k_blocks   = args.num_k_blocks;
        const index_t hdim           = args.hdim;
        const float head_cdfthreshd   = args.head_cdfthreshd;
        const float head_topk          = args.head_topk;
        const float head_simthreshold = args.head_simthreshold;
        const index_t causal_type    = args.causal_type;
        const bool    attention_sink = args.attention_sink;
        const index_t seqlen_q       = args.seqlen_q;
        const index_t seqlen_k       = args.seqlen_k;
        const index_t block_size     = args.block_size;
        const index_t window_left    = args.window_left;
        const index_t window_right   = args.window_right;

        const index_t tid = static_cast<index_t>(threadIdx.x);

        float* q_mean_smem = reinterpret_cast<float*>(smem);
        float* scores_smem = q_mean_smem + hdim;

        {
            const float* q_src =
                q_means +
                (static_cast<long_index_t>(b) * nhead_q + head) * num_q_blocks * hdim +
                static_cast<long_index_t>(q_block) * hdim;
            for(index_t d = tid; d < hdim; d += kBlockSize)
                q_mean_smem[d] = q_src[d];
            __builtin_amdgcn_s_barrier();
        }

        // Causal bounds
        const index_t q_start      = q_block * block_size;
        const index_t causal_delta = (causal_type == 2) ? (seqlen_k - seqlen_q) : index_t{0};
        const index_t right_ext    = (causal_type && window_right >= 0) ? window_right : index_t{0};
        const index_t causal_max_k = causal_type
            ? min(num_k_blocks - 1,
                  (q_start + block_size - 1 + right_ext + causal_delta) / block_size)
            : (num_k_blocks - 1);
        const index_t causal_min_k = (causal_type && window_left >= 0)
            ? max(index_t{0}, (q_start - window_left + causal_delta) / block_size)
            : index_t{0};

        const float* k_mean_base =
            k_means +
            (static_cast<long_index_t>(b) * nhead_k + kv_head) * num_k_blocks * hdim;
        const float scale = 1.0f / __builtin_sqrtf(static_cast<float>(hdim));

        for(index_t k = tid; k < num_k_blocks; k += kBlockSize)
        {
            float score = kScoreOOB;
            if(!causal_type || (k >= causal_min_k && k <= causal_max_k))
            {
                const float* k_mean = k_mean_base + static_cast<long_index_t>(k) * hdim;
                float dot = 0.0f;
                for(index_t d = 0; d < hdim; ++d)
                    dot += q_mean_smem[d] * k_mean[d];
                score = dot * scale;
            }
            scores_smem[k] = score;
        }
        __builtin_amdgcn_s_barrier();

        // smem layout
        float*   sort_keys_smem = scores_smem + num_k_blocks;
        int32_t* sort_vals_smem = reinterpret_cast<int32_t*>(sort_keys_smem + kMaxKBlocksPow2);
        float*   aux_smem       = reinterpret_cast<float*>(sort_vals_smem + kMaxKBlocksPow2);
        int32_t* scratch_i32    = reinterpret_cast<int32_t*>(aux_smem + kMaxKBlocksPow2);
        int32_t* n_target_smem  = scratch_i32 + kReduceScratchSlots;
        float*   scratch_f32    = reinterpret_cast<float*>(scratch_i32);

        // Parallel softmax; normalize only for CDF mode (TopK is scaling-invariant).
        float local_max = -INFINITY;
        for(index_t k = tid; k < num_k_blocks; k += kBlockSize)
            local_max = (scores_smem[k] > local_max) ? scores_smem[k] : local_max;
        const float max_score = block_reduce_max_f32<kBlockSize>(local_max, scratch_f32, tid);

        float local_sum = 0.0f;
        for(index_t k = tid; k < num_k_blocks; k += kBlockSize)
        {
            float p = __expf(scores_smem[k] - max_score);
            scores_smem[k] = p;
            local_sum += p;
        }
        const float sum_exp = block_reduce_sum_f32<kBlockSize>(local_sum, scratch_f32, tid);

        const bool topk_mode = (head_topk > 0.0f);
        if(!topk_mode)
        {
            const float rcp = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
            for(index_t k = tid; k < num_k_blocks; k += kBlockSize)
                scores_smem[k] *= rcp;
            __builtin_amdgcn_s_barrier();
        }

        // Dispatch sort+select to smallest pow-of-2 >= num_k_blocks
        int32_t n_target = 0;
        auto do_select = [&](auto N_const) {
            constexpr index_t N_pow2 = decltype(N_const)::value;

            if(tid < N_pow2)
            {
                const bool valid = tid < num_k_blocks;
                float p          = valid ? scores_smem[tid] : -1.0f;
                if(!(p == p)) p = -1.0f;
                sort_keys_smem[tid] = p;
                sort_vals_smem[tid] = valid ? static_cast<int32_t>(tid) : int32_t{-1};
            }
            __builtin_amdgcn_s_barrier();

            bitonic_sort_desc_smem<N_pow2>(sort_keys_smem, sort_vals_smem, tid);
            // Cumsum scan only needed for the CDF threshold path.
            if(!topk_mode)
                block_scan_inclusive_sum_smem<N_pow2>(sort_keys_smem, aux_smem, tid);

            if(topk_mode)
            {
                n_target = max(int32_t{1},
                               static_cast<int32_t>(head_topk * static_cast<float>(num_k_blocks)));
            }
            else
            {
                int32_t cand = num_k_blocks + 1;
                if(tid < static_cast<index_t>(num_k_blocks) &&
                   sort_keys_smem[tid] >= head_cdfthreshd)
                    cand = static_cast<int32_t>(tid) + 1;
                int32_t reduced = block_reduce_min_i32<kBlockSize>(cand, scratch_i32, tid);
                if(tid == 0)
                {
                    if(reduced > num_k_blocks) reduced = num_k_blocks;
                    if(reduced < 1)            reduced = 1;
                    n_target_smem[0] = reduced;
                }
                __builtin_amdgcn_s_barrier();
                n_target = n_target_smem[0];
            }

            if(tid < N_pow2 && static_cast<int32_t>(tid) < n_target)
            {
                int32_t orig = sort_vals_smem[tid];
                if(orig >= 0)
                {
                    const bool causal_ok = !causal_type ||
                        (orig >= causal_min_k && orig <= causal_max_k);
                    if(causal_ok)
                        scores_smem[orig] = kScoreSelected;
                }
            }
            __builtin_amdgcn_s_barrier();
        };

        if(num_k_blocks <= 32)        do_select(integral_constant<index_t, 32>{});
        else if(num_k_blocks <= 64)   do_select(integral_constant<index_t, 64>{});
        else if(num_k_blocks <= 128)  do_select(integral_constant<index_t, 128>{});
        else                          do_select(integral_constant<index_t, kMaxKBlocksPow2>{});

        // K-sim union
        if(k_sim != nullptr && head_simthreshold > 0.0f)
        {
            const float* k_sim_row =
                k_sim +
                static_cast<long_index_t>(b) * nhead_k * num_k_blocks +
                static_cast<long_index_t>(kv_head) * num_k_blocks;
            if(tid < num_k_blocks)
            {
                const bool causal_ok = !causal_type ||
                    (tid >= causal_min_k && tid <= causal_max_k);
                if(k_sim_row[tid] < head_simthreshold && causal_ok &&
                   scores_smem[tid] != kScoreSelected)
                    scores_smem[tid] = kScoreSelected;
            }
            __builtin_amdgcn_s_barrier();
        }

        // (Empty-selection fallback runs after the LUT scan below — reuses its n_total.)

        // Q-sim union
        if(q_sim != nullptr && head_simthreshold > 0.0f)
        {
            const float q_block_sim =
                q_sim[static_cast<long_index_t>(b) * nhead_q * num_q_blocks +
                      static_cast<long_index_t>(head) * num_q_blocks +
                      q_block];
            if(q_block_sim < head_simthreshold)
            {
                const index_t lo = causal_type ? causal_min_k : index_t{0};
                const index_t hi = causal_type ? causal_max_k : (num_k_blocks - 1);
                if(tid >= lo && tid <= hi && tid < num_k_blocks &&
                   scores_smem[tid] != kScoreSelected)
                    scores_smem[tid] = kScoreSelected;
            }
            __builtin_amdgcn_s_barrier();
        }

        // Attention sink
        if(tid == 0 && attention_sink && num_k_blocks > 0 &&
           scores_smem[0] != kScoreSelected)
            scores_smem[0] = kScoreSelected;
        __builtin_amdgcn_s_barrier();

        // LUT build: flag -> scan -> compact -> delta
        if(tid < kMaxKBlocksPow2)
            sort_keys_smem[tid] = (tid < num_k_blocks &&
                                   scores_smem[tid] == kScoreSelected) ? 1.0f : 0.0f;
        __builtin_amdgcn_s_barrier();

        block_scan_inclusive_sum_smem<kMaxKBlocksPow2>(sort_keys_smem, aux_smem, tid);

        const index_t n_after_scan = (num_k_blocks > 0)
            ? static_cast<index_t>(sort_keys_smem[num_k_blocks - 1]) : 0;

        // Fallback: every selection pass produced zero. Emit a single-element LUT and exit.
        if(n_after_scan == 0)
        {
            if(tid == 0)
            {
                if(causal_min_k <= causal_max_k && causal_min_k < num_k_blocks)
                {
                    lut_row[0] = static_cast<int32_t>(causal_min_k);
                    *vbn_ptr   = 1;
                }
                else
                {
                    *vbn_ptr = 0;
                }
            }
            return;
        }

        if(tid < num_k_blocks && scores_smem[tid] == kScoreSelected)
        {
            int pos = static_cast<int>(sort_keys_smem[tid]) - 1;
            sort_vals_smem[pos] = static_cast<int32_t>(tid);
        }
        __builtin_amdgcn_s_barrier();

        if(tid < n_after_scan)
        {
            int curr = sort_vals_smem[tid];
            int prev = (tid == 0) ? 0 : sort_vals_smem[tid - 1];
            lut_row[tid] = curr - prev;
        }
        if(tid == 0) *vbn_ptr = static_cast<int32_t>(n_after_scan);
    }

    CK_TILE_DEVICE void operator()(
        const float* __restrict__ k_means,
        const float* __restrict__ q_means,
        const float* __restrict__ k_sim,
        const float* __restrict__ q_sim,
        int32_t* __restrict__ lut_out,
        int32_t* __restrict__ valid_block_num_out,
        index_t nhead_q,
        index_t nhead_k,
        index_t nhead_ratio_qk,
        index_t num_q_blocks,
        index_t num_k_blocks,
        index_t hdim,
        float cdfthreshd,
        float topk,
        float simthreshold,
        const float* __restrict__ cdfthreshd_per_head,    // nullable [H_q]
        const float* __restrict__ topk_per_head,           // nullable [H_q]
        const float* __restrict__ simthreshold_per_head,  // nullable [H_q]
        index_t causal_type,
        bool attention_sink,
        index_t seqlen_q,
        index_t seqlen_k,
        index_t block_size,
        index_t window_left,
        index_t window_right,
        void* smem) const
    {
        const index_t gid = static_cast<index_t>(blockIdx.x);

        const index_t q_block = gid % num_q_blocks;
        const index_t head    = (gid / num_q_blocks) % nhead_q;
        const index_t b       = gid / (nhead_q * num_q_blocks);
        const index_t kv_head = head / nhead_ratio_qk;

        int32_t* lut_row =
            lut_out +
            (static_cast<long_index_t>(b) * nhead_q + head) * num_q_blocks * num_k_blocks +
            static_cast<long_index_t>(q_block) * num_k_blocks;
        int32_t* vbn_ptr =
            valid_block_num_out +
            (static_cast<long_index_t>(b) * nhead_q + head) * num_q_blocks +
            q_block;

        // Per-head lookups (single coalesced load per workgroup).
        const float head_cdfthreshd    = cdfthreshd_per_head   ? cdfthreshd_per_head[head]   : cdfthreshd;
        const float head_topk           = topk_per_head          ? topk_per_head[head]          : topk;
        const float head_simthreshold  = simthreshold_per_head ? simthreshold_per_head[head] : simthreshold;

        MaskRunArgs args;
        args.k_means      = k_means;
        args.q_means      = q_means;
        args.k_sim        = k_sim;
        args.q_sim        = q_sim;
        args.lut_row      = lut_row;
        args.vbn_ptr      = vbn_ptr;
        args.b            = b;
        args.head         = head;
        args.kv_head      = kv_head;
        args.q_block      = q_block;
        args.nhead_q      = nhead_q;
        args.nhead_k      = nhead_k;
        args.num_q_blocks = num_q_blocks;
        args.num_k_blocks = num_k_blocks;
        args.hdim         = hdim;
        args.head_cdfthreshd   = head_cdfthreshd;
        args.head_topk          = head_topk;
        args.head_simthreshold = head_simthreshold;
        args.causal_type    = causal_type;
        args.attention_sink = attention_sink;
        args.seqlen_q       = seqlen_q;
        args.seqlen_k       = seqlen_k;
        args.block_size     = block_size;
        args.window_left    = window_left;
        args.window_right   = window_right;
        run_with_indices(args, smem);
    }
};

} // namespace ck_tile
