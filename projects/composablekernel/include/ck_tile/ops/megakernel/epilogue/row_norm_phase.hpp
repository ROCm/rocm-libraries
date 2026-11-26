// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/megakernel/epilogue/epilogue_common.hpp"

namespace ck_tile {

/// @file
/// Whole-row phases of the target model: the norms whose rows (8192 or 4096 wide) span many GEMM
/// output tiles.
///
/// Cross-tile reduction: one workgroup owns a whole row (work item = one row), reads it once into
/// registers (kN / kBlockSize elements per thread), and reduces the sum of squares through LDS.
/// At M <= 64 this is <= 64 items of 16-32 KB each, so a row phase costs about one L2 read of
/// its inputs. The alternative, producing GEMM tiles adding their partial sum of squares to a
/// per-row fp32 atomic followed by a finalize, saves that read but needs a zeroed accumulator per
/// step, an epilogue hook (shared code) and a second pass anyway for anything after the norm
/// (residual gate, the next norm, quantisation), so it is not worth it at decode M.
///
/// Where the post-all-reduce norm sits (TP8): o_proj is K-sharded (K = 1024 per rank), so every
/// rank holds a partial [M, 8192] that item 2's all-reduce sums. MkResidualNormPhase is the
/// reduce side of that all-reduce: it takes up to kMaxInputs partial buffers (e.g. the peers'
/// IPC-mapped slots of a one-shot all-reduce), sums them in fp32, and only then applies
/// post_attention_layernorm, so the norm sees the full sum. With a reduce-scatter + all-gather
/// all-reduce, run it on the rows this rank reduces and all-gather x and h instead of the sum.
/// The unpooler output (post_feedforward side) is handled the same way.
///
/// MkResidualNormPhase, per row (r = residual; alpha, beta from the fp32 residual-gate vector g,
/// w = g / 0.3, beta = sigmoid(w), alpha = sqrt(max(sigmoid(-w) (1 + beta), 1e-3)), precomputed by
/// mk_residual_gate_coeffs):
///   s = bf16(sum of the inputs)
///   kRms:         a = bf16(RMSNorm(s, eps))                           (post_attention_layernorm)
///   kInputScaled: y = bf16(s * bf16(1 + w)); a = bf16(RMSNorm(y, eps))
///   (post_feedforward_layernorm) x = bf16(alpha * r + beta * a) (ResidualGate) h = bf16(x *
///   rsqrt(mean(x^2) + eps2) * bf16(1 + w2))               (next CenteredRMSNorm)
/// x is the new residual; h feeds the router and pooler (post-attention) or the next layer's
/// qkv projection (post-feedforward). h is skipped when h_out is null.
///
/// MkNormQuantPhase, per row: p = CenteredRMSNorm(in, eps, w) as above (pre_expert_norm on the
/// pooler output), then one of
///   kNone:   p only
///   kFp8Row: q = e4m3(p * 448 / amax), scale = amax / 448 (fp32 per row; amax = 0 -> scale 1)
///   kMxFp8:  per 32 columns: e8m0 scale e (mk_mx_scale_e8m0<8>), q = e4m3(p / 2^(e - 127))
///   kMxFp4:  per 32 columns: e = mk_mx_scale_e8m0<2>, q = e2m1(p / 2^(e - 127)),
///            two per byte, even column in the low nibble
/// Scales are row-major [M][kN / 32] bytes (unshuffled); q is row-major [M][kN] fp8 bytes or
/// [M][kN / 2] fp4 bytes.

enum class MkFirstNorm
{
    kRms,         ///< RMSNorm without weight (post_attention_layernorm)
    kInputScaled, ///< InputScaledRMSNorm (post_feedforward_layernorm)
};

enum class MkActQuant
{
    kNone,
    kFp8Row,
    kMxFp8,
    kMxFp4,
};

namespace impl {
/// Row walk shared by the row phases: chunk k of thread t is elements [8 (t + k B), +8).
template <index_t kN, index_t kNumWarps>
struct mk_row_walk
{
    static constexpr index_t kBlockSize = kNumWarps * get_warp_size();
    static constexpr index_t kChunks    = kN / (8 * kBlockSize);
    static_assert(kN % (8 * kBlockSize) == 0, "row width must be a multiple of 8 * kBlockSize");

    CK_TILE_DEVICE static index_t Col(index_t k) { return 8 * (get_thread_id() + k * kBlockSize); }
};

template <index_t kNumWarps>
struct mk_row_phase_base
{
    static constexpr index_t kSlotRows = 1;
    static constexpr index_t kWarps    = kNumWarps;
    /// As GemmPhase: waves times the wave size of the compiler pass (host passes assume 64).
    static constexpr index_t kBlockSize = kNumWarps * get_warp_size();
    using barrier_pipeline              = void;

    /// Two reductions per item, each with its own LDS words.
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return 2 * kWarps * static_cast<index_t>(sizeof(float));
    }
    CK_TILE_DEVICE static auto InitBarriers() {}
};
} // namespace impl

template <index_t kN_,
          index_t kNumWarps_,
          MkFirstNorm kFirst_,
          typename InDataType_ = bf16_t,
          index_t kMaxInputs_  = 1>
struct MkResidualNormPhase : impl::mk_row_phase_base<kNumWarps_>
{
    using Base       = impl::mk_row_phase_base<kNumWarps_>;
    using InDataType = remove_cvref_t<InDataType_>;
    using Walk       = impl::mk_row_walk<kN_, kNumWarps_>;

    static constexpr index_t kN = kN_;
    using Base::kBlockSize;
    static constexpr MkFirstNorm kFirst = kFirst_;
    static constexpr index_t kMaxInputs   = kMaxInputs_;
    static constexpr index_t kChunks      = Walk::kChunks;

    struct Args
    {
        const InDataType* in[kMaxInputs]; ///< partial sums, each [M, in_stride]
        index_t num_inputs;               ///< 1..kMaxInputs
        index_t in_stride;
        const bf16_t* first_w;   ///< kInputScaled: post_feedforward_layernorm.weight [kN]
        float first_eps;         ///< 1e-8 for both post norms
        const float* gate_alpha; ///< [kN], mk_residual_gate_coeffs of the gate vector
        const float* gate_beta;  ///< [kN]
        const bf16_t* residual;  ///< [M, residual_stride]
        index_t residual_stride;
        bf16_t* x_out; ///< new residual [M, x_stride]; may alias residual
        index_t x_stride;
        const bf16_t* next_w; ///< CenteredRMSNorm weight of the consumer [kN]
        float next_eps;       ///< 1e-5
        bf16_t* h_out;        ///< [M, h_stride], or null
        index_t h_stride;
        index_t M;
    };

    CK_TILE_HOST_DEVICE static index_t NumWork(const Args& a) { return a.M; }
    CK_TILE_HOST_DEVICE static index_t NumSlots(const Args& a) { return a.M; }
    CK_TILE_HOST_DEVICE static index_t ItemsPerSlot(const Args&, index_t) { return 1; }
    CK_TILE_DEVICE static index_t SlotOf(const Args&, index_t work) { return work; }
    CK_TILE_DEVICE static tuple<index_t, index_t> RowRange(const Args&, index_t work)
    {
        return make_tuple(work, work + 1);
    }

    CK_TILE_HOST static bool IsSupportedArgument(const Args& a)
    {
        bool ok = a.M > 0 && a.num_inputs >= 1 && a.num_inputs <= kMaxInputs && a.gate_alpha &&
                  a.gate_beta && a.residual && a.x_out && a.in_stride >= kN &&
                  a.in_stride % 8 == 0 && a.residual_stride >= kN && a.residual_stride % 8 == 0 &&
                  a.x_stride >= kN && a.x_stride % 8 == 0 &&
                  (a.h_out == nullptr || (a.next_w && a.h_stride >= kN && a.h_stride % 8 == 0)) &&
                  (kFirst != MkFirstNorm::kInputScaled || a.first_w != nullptr);
        for(index_t i = 0; i < a.num_inputs && ok; ++i)
            ok = a.in[i] != nullptr;
        return ok;
    }

    template <typename Token>
    CK_TILE_DEVICE static void Run(const Args& a, index_t row, void* smem, const Token&)
    {
        float* lds = static_cast<float*>(smem);
        float v[kChunks][8];

        // Sum of the partials, then the bf16 tensor HF sees.
        float ss = 0.f;
        static_for<0, kChunks, 1>{}([&](auto k) {
            const index_t col    = Walk::Col(k.value);
            const long_index_t o = static_cast<long_index_t>(row) * a.in_stride + col;
            mk_load_x8(a.in[0] + o, v[k.value]);
            static_for<1, kMaxInputs, 1>{}([&](auto i) {
                if(i.value < a.num_inputs)
                {
                    float t[8];
                    mk_load_x8(a.in[i.value] + o, t);
                    static_for<0, 8, 1>{}([&](auto j) { v[k.value][j.value] += t[j.value]; });
                }
            });
            float w[8];
            if constexpr(kFirst == MkFirstNorm::kInputScaled)
                mk_load_bf16x8(a.first_w + col, w);
            static_for<0, 8, 1>{}([&](auto j) {
                float y = mk_round_bf16(v[k.value][j.value]);
                if constexpr(kFirst == MkFirstNorm::kInputScaled)
                    y = mk_round_bf16(y * mk_one_plus_w(w[j.value]));
                v[k.value][j.value] = y;
                ss += y * y;
            });
        });
        ss = mk_block_reduce<Base::kWarps>(ss, lds, [](float p, float q) { return p + q; });
        const float r = rsqrtf(ss * (1.f / kN) + a.first_eps);

        // Residual gate; v becomes x.
        float ss2 = 0.f;
        static_for<0, kChunks, 1>{}([&](auto k) {
            const index_t col = Walk::Col(k.value);
            float res[8], alpha[8], beta[8];
            mk_load_bf16x8(a.residual + static_cast<long_index_t>(row) * a.residual_stride + col,
                             res);
            mk_load_f32x8(a.gate_alpha + col, alpha);
            mk_load_f32x8(a.gate_beta + col, beta);
            static_for<0, 8, 1>{}([&](auto j) {
                const float an = mk_round_bf16(v[k.value][j.value] * r);
                const float x = mk_round_bf16(alpha[j.value] * res[j.value] + beta[j.value] * an);
                v[k.value][j.value] = x;
                ss2 += x * x;
            });
            mk_store_bf16x8(a.x_out + static_cast<long_index_t>(row) * a.x_stride + col,
                              v[k.value]);
        });

        if(a.h_out == nullptr)
            return;
        ss2 = mk_block_reduce<Base::kWarps>(
            ss2, lds + Base::kWarps, [](float p, float q) { return p + q; });
        const float r2 = rsqrtf(ss2 * (1.f / kN) + a.next_eps);
        static_for<0, kChunks, 1>{}([&](auto k) {
            const index_t col = Walk::Col(k.value);
            float w[8];
            mk_load_bf16x8(a.next_w + col, w);
            static_for<0, 8, 1>{}([&](auto j) {
                v[k.value][j.value] = (v[k.value][j.value] * r2) * mk_one_plus_w(w[j.value]);
            });
            mk_store_bf16x8(a.h_out + static_cast<long_index_t>(row) * a.h_stride + col,
                              v[k.value]);
        });
    }
};

/// Arguments of MkNormQuantPhase (the same type for every kQuant).
struct MkNormQuantArgs
{
    const bf16_t* in; ///< pooler output [M, in_stride]
    index_t in_stride;
    const bf16_t* w; ///< pre_expert_norm.weight [kN]
    float eps;       ///< 1e-5
    bf16_t* p_out;   ///< normed rows [M, p_stride] bf16, or null
    index_t p_stride;
    void* q_out;      ///< quantised rows, q_stride bytes apart (unused for kNone)
    index_t q_stride; ///< bytes
    void* scale_out;  ///< kFp8Row: fp32 [M]; MX: e8m0 bytes [M, scale_stride]
    index_t scale_stride;
    index_t M;
};

template <index_t kN_, index_t kNumWarps_, MkActQuant kQuant_>
struct MkNormQuantPhase : impl::mk_row_phase_base<kNumWarps_>
{
    using Base = impl::mk_row_phase_base<kNumWarps_>;
    using Walk = impl::mk_row_walk<kN_, kNumWarps_>;

    static constexpr index_t kN = kN_;
    using Base::kBlockSize;
    static constexpr MkActQuant kQuant = kQuant_;
    static constexpr index_t kChunks     = Walk::kChunks;
    static constexpr bool kMx = kQuant == MkActQuant::kMxFp8 || kQuant == MkActQuant::kMxFp4;
    /// Bytes of q per element row.
    static constexpr index_t kQRowBytes = kQuant == MkActQuant::kMxFp4 ? kN / 2 : kN;

    using Args = MkNormQuantArgs;

    CK_TILE_HOST_DEVICE static index_t NumWork(const Args& a) { return a.M; }
    CK_TILE_HOST_DEVICE static index_t NumSlots(const Args& a) { return a.M; }
    CK_TILE_HOST_DEVICE static index_t ItemsPerSlot(const Args&, index_t) { return 1; }
    CK_TILE_DEVICE static index_t SlotOf(const Args&, index_t work) { return work; }
    CK_TILE_DEVICE static tuple<index_t, index_t> RowRange(const Args&, index_t work)
    {
        return make_tuple(work, work + 1);
    }

    CK_TILE_HOST static bool IsSupportedArgument(const Args& a)
    {
        bool ok = a.M > 0 && a.in && a.w && a.in_stride >= kN && a.in_stride % 8 == 0 &&
                  (a.p_out == nullptr || (a.p_stride >= kN && a.p_stride % 8 == 0));
        if constexpr(kQuant != MkActQuant::kNone)
            ok = ok && a.q_out && a.scale_out && a.q_stride >= kQRowBytes && a.q_stride % 16 == 0;
        if constexpr(kMx)
            ok = ok && a.scale_stride >= kN / 32;
        return ok;
    }

    template <typename Token>
    CK_TILE_DEVICE static void Run(const Args& a, index_t row, void* smem, const Token&)
    {
        float* lds = static_cast<float*>(smem);
        float v[kChunks][8];

        float ss = 0.f;
        static_for<0, kChunks, 1>{}([&](auto k) {
            mk_load_bf16x8(a.in + static_cast<long_index_t>(row) * a.in_stride +
                                 Walk::Col(k.value),
                             v[k.value]);
            static_for<0, 8, 1>{}([&](auto j) { ss += v[k.value][j.value] * v[k.value][j.value]; });
        });
        ss = mk_block_reduce<Base::kWarps>(ss, lds, [](float p, float q) { return p + q; });
        const float r = rsqrtf(ss * (1.f / kN) + a.eps);

        float amax = 0.f;
        static_for<0, kChunks, 1>{}([&](auto k) {
            const index_t col = Walk::Col(k.value);
            float w[8];
            mk_load_bf16x8(a.w + col, w);
            static_for<0, 8, 1>{}([&](auto j) {
                v[k.value][j.value] =
                    mk_round_bf16((v[k.value][j.value] * r) * mk_one_plus_w(w[j.value]));
                amax = fmaxf(amax, fabsf(v[k.value][j.value]));
            });
            if(a.p_out != nullptr)
                mk_store_bf16x8(a.p_out + static_cast<long_index_t>(row) * a.p_stride + col,
                                  v[k.value]);
        });

        if constexpr(kQuant == MkActQuant::kFp8Row)
        {
            amax = mk_block_reduce<Base::kWarps>(
                amax, lds + Base::kWarps, [](float p, float q) { return fmaxf(p, q); });
            const float inv = amax > 0.f ? 448.f / amax : 1.f;
            if(get_thread_id() == 0)
                static_cast<float*>(a.scale_out)[row] = amax > 0.f ? amax / 448.f : 1.f;
            static_for<0, kChunks, 1>{}(
                [&](auto k) { StoreFp8(a, row, k.value, v[k.value], inv); });
        }
        else if constexpr(kMx)
        {
            constexpr index_t kEmax = kQuant == MkActQuant::kMxFp8 ? 8 : 2;
            static_for<0, kChunks, 1>{}([&](auto k) {
                float m = 0.f;
                static_for<0, 8, 1>{}([&](auto j) { m = fmaxf(m, fabsf(v[k.value][j.value])); });
                m                 = mk_group_max<4>(m); // 4 lanes x 8 = one block of 32
                const uint32_t e  = mk_mx_scale_e8m0<kEmax>(m);
                const float inv   = mk_mx_inv_scale(e);
                const index_t col = Walk::Col(k.value);
                if(get_lane_id() % 4 == 0)
                    static_cast<uint8_t*>(
                        a.scale_out)[static_cast<long_index_t>(row) * a.scale_stride + col / 32] =
                        static_cast<uint8_t>(e);
                if constexpr(kQuant == MkActQuant::kMxFp8)
                    StoreFp8(a, row, k.value, v[k.value], inv);
                else
                {
                    uint32_t q = 0;
                    static_for<0, 8, 1>{}([&](auto j) {
                        q |= mk_e2m1_encode(v[k.value][j.value] * inv) << (4 * j.value);
                    });
                    *c_style_pointer_cast<uint32_t*>(static_cast<uint8_t*>(a.q_out) +
                                                     static_cast<long_index_t>(row) * a.q_stride +
                                                     col / 2) = q;
                }
            });
        }
        else
        {
            ignore = amax;
        }
    }

    private:
    CK_TILE_DEVICE static void
    StoreFp8(const Args& a, index_t row, index_t k, const float (&x)[8], float inv)
    {
        uint32x2_t q;
        q[0] = mk_pack_fp8x4(x[0] * inv, x[1] * inv, x[2] * inv, x[3] * inv);
        q[1] = mk_pack_fp8x4(x[4] * inv, x[5] * inv, x[6] * inv, x[7] * inv);
        *c_style_pointer_cast<uint32x2_t*>(static_cast<uint8_t*>(a.q_out) +
                                           static_cast<long_index_t>(row) * a.q_stride +
                                           Walk::Col(k)) = q;
    }
};

} // namespace ck_tile
