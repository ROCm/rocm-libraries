// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Standalone CPU reference for FMHA backward pass.
//
// Templated on data types so the same code works for FP32 (Phase 1),
// FP16, and BF16 (Phase 2). Uses ck_tile::reference_batched_* functions
// for the heavy lifting.
//
// Layout convention: all tensors are [nhead, seqlen, dim] (3D batched).
// The caller handles batch-loop and permutation.

#pragma once

#include <optional>

#include "ck_tile/host/host_tensor.hpp"
#include "ck_tile/host/reference/reference_batched_gemm.hpp"
#include "ck_tile/host/reference/reference_batched_softmax.hpp"

#include <cmath>
#include <thread>

namespace rocm_ck::test {

// Accumulation type: always float for reduced types, same type for float.
template <typename T>
struct AccType
{
    using type = float;
};
template <>
struct AccType<float>
{
    using type = float;
};

// -------------------------------------------------------------------------
// Forward pass reference (needed to produce O, LSE, P for backward)
// -------------------------------------------------------------------------

template <typename QType,
          typename KType,
          typename VType,
          typename OType,
          typename AccDataType = typename AccType<QType>::type>
struct FmhaFwdRefOutput
{
    ck_tile::HostTensor<AccDataType> s;   // [nhead, seqlen_q, seqlen_k]
    ck_tile::HostTensor<AccDataType> p;   // [nhead, seqlen_q, seqlen_k]
    ck_tile::HostTensor<AccDataType> lse; // [nhead, seqlen_q]
    ck_tile::HostTensor<OType> o;         // [nhead, seqlen_q, hdim_v]
};

/// Run the FMHA forward pass on CPU.
/// q: [nhead, seqlen_q, hdim_q]
/// k: [nhead, seqlen_k, hdim_q]  (note: K layout is [head, seq, dim], gemm transposes internally)
/// v: [nhead, hdim_v, seqlen_k]  (note: V layout is [head, dim, seq] for the P*V gemm)
/// scale: typically 1/sqrt(hdim_q)
template <typename QType,
          typename KType,
          typename VType,
          typename OType     = QType,
          typename GemmType  = QType,
          typename AccDataType = typename AccType<QType>::type>
FmhaFwdRefOutput<QType, KType, VType, OType, AccDataType>
fmha_fwd_cpu_ref(const ck_tile::HostTensor<QType>& q,
                 const ck_tile::HostTensor<KType>& k,
                 const ck_tile::HostTensor<VType>& v,
                 float scale)
{
    const auto nhead    = static_cast<int>(q.get_lengths()[0]);
    const auto seqlen_q = static_cast<int>(q.get_lengths()[1]);
    const auto seqlen_k = static_cast<int>(k.get_lengths()[1]);
    const auto hdim_v   = static_cast<int>(v.get_lengths()[1]);

    // S = scale * Q @ K^T
    ck_tile::HostTensor<AccDataType> s({nhead, seqlen_q, seqlen_k});
    ck_tile::reference_batched_gemm<QType, KType, AccDataType, AccDataType>(
        q, k, s, ck_tile::identity{}, ck_tile::identity{}, ck_tile::scales(scale));

    // P, LSE = softmax(S)
    ck_tile::HostTensor<AccDataType> p({nhead, seqlen_q, seqlen_k});
    ck_tile::HostTensor<AccDataType> lse({nhead, seqlen_q});
    ck_tile::reference_batched_softmax<AccDataType, AccDataType, AccDataType>(
        s, p, ck_tile::identity{}, lse);

    // O = P * V  (using low-precision P for the gemm, matching CK runner)
    ck_tile::HostTensor<GemmType> p_lp = p.template CopyAsType<GemmType>();
    ck_tile::HostTensor<OType> o({nhead, seqlen_q, hdim_v});
    ck_tile::reference_batched_gemm<GemmType, VType, AccDataType, OType>(p_lp, v, o);

    return {std::move(s), std::move(p), std::move(lse), std::move(o)};
}

// -------------------------------------------------------------------------
// Backward pass reference
// -------------------------------------------------------------------------

template <typename QGradType,
          typename KGradType,
          typename VGradType,
          typename AccDataType = typename AccType<QGradType>::type>
struct FmhaBwdRefOutput
{
    ck_tile::HostTensor<AccDataType> d;       // [nhead, seqlen_q] (OGradDotO)
    ck_tile::HostTensor<AccDataType> dq_acc;  // [nhead, seqlen_q, hdim_q] (FP32 accum)
    ck_tile::HostTensor<KGradType> dk;        // [nhead, seqlen_k, hdim_q]
    ck_tile::HostTensor<VGradType> dv;        // [nhead, seqlen_k, hdim_v]
};

/// Run the FMHA backward pass on CPU.
///
/// Inputs (from forward):
///   q:   [nhead, seqlen_q, hdim_q]
///   k:   [nhead, seqlen_k, hdim_q]
///   v:   [nhead, hdim_v, seqlen_k]
///   o:   [nhead, seqlen_q, hdim_v]
///   p:   [nhead, seqlen_q, seqlen_k] (high-precision softmax output)
///   lse: [nhead, seqlen_q]
///
/// Gradient input:
///   dO:  [nhead, seqlen_q, hdim_v]
///
/// scale: same as forward (1/sqrt(hdim_q))
template <typename QType,
          typename KType,
          typename VType,
          typename OType,
          typename OGradType,
          typename QGradType = QType,
          typename KGradType = KType,
          typename VGradType = VType,
          typename GemmType  = QType,
          typename AccDataType = typename AccType<QType>::type>
FmhaBwdRefOutput<QGradType, KGradType, VGradType, AccDataType>
fmha_bwd_cpu_ref(const ck_tile::HostTensor<QType>& q,
                 const ck_tile::HostTensor<KType>& k,
                 const ck_tile::HostTensor<VType>& v,
                 const ck_tile::HostTensor<OType>& o,
                 const ck_tile::HostTensor<AccDataType>& p,
                 const ck_tile::HostTensor<OGradType>& dO,
                 float scale)
{
    const auto nhead    = static_cast<int>(q.get_lengths()[0]);
    const auto seqlen_q = static_cast<int>(q.get_lengths()[1]);
    const auto seqlen_k = static_cast<int>(k.get_lengths()[1]);
    const auto hdim_q   = static_cast<int>(q.get_lengths()[2]);
    const auto hdim_v   = static_cast<int>(v.get_lengths()[1]);

    // -----------------------------------------------------------------
    // OGradDotO: D[h,q] = sum_d dO[h,q,d] * O[h,q,d]
    // -----------------------------------------------------------------
    ck_tile::HostTensor<AccDataType> d({nhead, seqlen_q});
    ck_tile::make_ParallelTensorFunctor(
        [&](auto i_h, auto i_q) {
            AccDataType acc = 0;
            for(int i_d = 0; i_d < hdim_v; ++i_d)
            {
                acc += ck_tile::type_convert<AccDataType>(dO(i_h, i_q, i_d)) *
                       ck_tile::type_convert<AccDataType>(o(i_h, i_q, i_d));
            }
            d(i_h, i_q) = acc;
        },
        nhead,
        seqlen_q)(std::thread::hardware_concurrency());

    // -----------------------------------------------------------------
    // DqDkDv: 5-GEMM backward
    // -----------------------------------------------------------------

    // Step 1: dP = dO @ V^T
    //   dO: [nhead, seqlen_q, hdim_v]
    //   V^T: [nhead, seqlen_k, hdim_v] (transpose of V [nhead, hdim_v, seqlen_k])
    auto v_t = v.transpose({0, 2, 1}); // [nhead, hdim_v, seqlen_k] -> [nhead, seqlen_k, hdim_v]
    ck_tile::HostTensor<AccDataType> dp({nhead, seqlen_q, seqlen_k});
    ck_tile::reference_batched_gemm<OGradType, VType, AccDataType, AccDataType>(
        dO, v_t, dp);

    // Step 2: dS[h,q,k] = P[h,q,k] * (dP[h,q,k] - D[h,q])
    ck_tile::HostTensor<AccDataType> ds({nhead, seqlen_q, seqlen_k});
    ck_tile::make_ParallelTensorFunctor(
        [&](auto i_h, auto i_q) {
            for(int i_k = 0; i_k < seqlen_k; ++i_k)
            {
                ds(i_h, i_q, i_k) = p(i_h, i_q, i_k) * (dp(i_h, i_q, i_k) - d(i_h, i_q));
            }
        },
        nhead,
        seqlen_q)(std::thread::hardware_concurrency());

    ck_tile::HostTensor<GemmType> ds_lp = ds.template CopyAsType<GemmType>();

    // Step 3: dV = P^T @ dO  (scaled by rp_undrop=1 when no dropout)
    //   P^T: [nhead, seqlen_k, seqlen_q]
    //   dO:  [nhead, seqlen_q, hdim_v]  -> need transposed as [nhead, hdim_v, seqlen_q]
    ck_tile::HostTensor<GemmType> p_lp = p.template CopyAsType<GemmType>();
    auto p_t  = p_lp.transpose({0, 2, 1}); // [nhead, seqlen_k, seqlen_q]
    auto dO_t = dO.transpose({0, 2, 1});    // [nhead, hdim_v, seqlen_q]
    ck_tile::HostTensor<VGradType> dv({nhead, seqlen_k, hdim_v});
    ck_tile::reference_batched_gemm<GemmType, OGradType, AccDataType, VGradType>(
        p_t, dO_t, dv);

    // Step 4: dQ = scale * dS @ K
    //   dS: [nhead, seqlen_q, seqlen_k]
    //   K:  [nhead, seqlen_k, hdim_q]
    auto k_t = k.transpose({0, 2, 1}); // [nhead, hdim_q, seqlen_k]
    ck_tile::HostTensor<AccDataType> dq_acc({nhead, seqlen_q, hdim_q});
    ck_tile::reference_batched_gemm<GemmType, KType, AccDataType, AccDataType>(
        ds_lp,
        k_t,
        dq_acc,
        ck_tile::identity{},
        ck_tile::identity{},
        ck_tile::scales(scale));

    // Step 5: dK = scale * dS^T @ Q
    //   dS^T: [nhead, seqlen_k, seqlen_q]
    //   Q:    [nhead, seqlen_q, hdim_q] -> need transposed as [nhead, hdim_q, seqlen_q]
    auto ds_t = ds_lp.transpose({0, 2, 1}); // [nhead, seqlen_k, seqlen_q]
    auto q_t  = q.transpose({0, 2, 1});      // [nhead, hdim_q, seqlen_q]
    ck_tile::HostTensor<KGradType> dk({nhead, seqlen_k, hdim_q});
    ck_tile::reference_batched_gemm<GemmType, QType, AccDataType, KGradType>(
        ds_t,
        q_t,
        dk,
        ck_tile::identity{},
        ck_tile::identity{},
        ck_tile::scales(scale));

    return {std::move(d), std::move(dq_acc), std::move(dk), std::move(dv)};
}

} // namespace rocm_ck::test
