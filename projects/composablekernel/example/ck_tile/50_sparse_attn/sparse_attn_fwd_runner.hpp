// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "ck_tile/host.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host/reference/reference_blocked_attention.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"

#include "sparse_attn_fwd.hpp"
#include "../01_fmha/mask.hpp"
#include "../01_fmha/bias.hpp"
#include "sparse_attention.h"
#include "ck_tile/host/reference/reference_sparge_mask_prediction.hpp"

enum class sparse_attn_result
{
    success,
    failure,
    skipped,
};

// Throws std::invalid_argument on bad input or length mismatch.
inline std::vector<float> parse_csv_floats(const std::string& csv,
                                           const char* field_name,
                                           ck_tile::index_t expected_len)
{
    std::vector<float> out;
    if(csv.empty())
        return out;
    if(expected_len <= 0)
        throw std::invalid_argument(
            std::string("parse_csv_floats: expected_len must be > 0 (field '") +
            field_name + "')");

    std::stringstream ss(csv);
    std::string token;
    while(std::getline(ss, token, ','))
    {
        const auto first = token.find_first_not_of(" \t");
        const auto last  = token.find_last_not_of(" \t");
        token = (first == std::string::npos) ? std::string{} : token.substr(first, last - first + 1);
        try
        {
            out.push_back(std::stof(token));
        }
        catch(const std::exception&)
        {
            throw std::invalid_argument(
                std::string("-") + field_name + " contains non-float token '" + token + "'");
        }
    }
    if(static_cast<ck_tile::index_t>(out.size()) != expected_len)
        throw std::invalid_argument(
            std::string("-") + field_name + " has " + std::to_string(out.size()) +
            " values but expected " + std::to_string(expected_len) + " (= nhead_q)");
    return out;
}

// Per-head hyperparam buffers: host vectors (fed to CPU reference) + RAII device buffers that must
// outlive the launch. DeviceMem members are non-movable: declare locally, pass by reference.
struct SpargePerHeadSetup
{
    std::vector<float>  h_cdf, h_topk, h_sim, h_pvthreshd;
    ck_tile::DeviceMem  buf_cdf, buf_topk, buf_sim, buf_pvthreshd;
};

// Fill hp.*_per_head_ptr from per-head CSVs or -perhead_test synth. -sparsity_per_head becomes
// 1 - sparsity[h], routed to topk/cdf by mode_is_topk. Per-field precedence: CSV > perhead_test >
// scalar. Returns false on error (perhead_test with nhead < 2).
inline bool setup_sparge_per_head(sparge_hyperparam_args& hp,
                                  SpargePerHeadSetup& out,
                                  bool perhead_test,
                                  bool mode_is_topk,
                                  ck_tile::index_t nhead,
                                  float scalar_cdf,
                                  float scalar_topk,
                                  float scalar_sim,
                                  float scalar_pvthreshd,
                                  const std::string& sparsity_per_head_csv,
                                  const std::string& sim_per_head_csv,
                                  const std::string& pvthreshd_per_head_csv)
{
    const std::vector<float> csv_sparsity_in =
        parse_csv_floats(sparsity_per_head_csv, "sparsity_per_head", nhead);
    std::vector<float> csv_cdf, csv_topk;
    if(!csv_sparsity_in.empty())
    {
        std::vector<float>& dst = mode_is_topk ? csv_topk : csv_cdf;
        dst.resize(csv_sparsity_in.size());
        for(size_t i = 0; i < csv_sparsity_in.size(); ++i)
            dst[i] = 1.0f - csv_sparsity_in[i];
    }
    const std::vector<float> csv_sim = parse_csv_floats(sim_per_head_csv, "sim_per_head", nhead);
    const std::vector<float> csv_pvthreshd =
        parse_csv_floats(pvthreshd_per_head_csv, "pvthreshd_per_head", nhead);
    const bool any_csv = !csv_sparsity_in.empty() || !csv_sim.empty() || !csv_pvthreshd.empty();

    if(!(perhead_test || any_csv))
        return true; // scalars stay in effect

    if(nhead < 2 && perhead_test)
    {
        std::cerr << "error: -perhead_test=1 requires -h >= 2 (got " << nhead
                  << "), aborting to avoid silent skip\n";
        return false;
    }

    auto fill_field = [&](std::vector<float>& dst,
                          const std::vector<float>& csv,
                          float scalar,
                          bool clamp_to_unit_interval) -> bool {
        if(!csv.empty())
        {
            dst = csv;
            return true;
        }
        if(perhead_test && (clamp_to_unit_interval ? scalar > 0.0f : scalar != 0.0f))
        {
            dst.assign(static_cast<size_t>(nhead), 0.0f);
            constexpr float clamp_lo = 0.001f, clamp_hi = 0.999f;
            for(ck_tile::index_t h = 0; h < nhead; ++h)
            {
                const float scl = (nhead < 2)
                    ? 1.0f
                    : 0.8f + 0.4f * static_cast<float>(h) / static_cast<float>(nhead - 1); // 0.8..1.2
                const float v = scalar * scl;
                dst[static_cast<size_t>(h)] =
                    clamp_to_unit_interval ? std::clamp(v, clamp_lo, clamp_hi) : v;
            }
            return true;
        }
        return false;
    };

    auto upload = [&](bool used, std::vector<float>& host, ck_tile::DeviceMem& buf,
                      const void*& hp_ptr) {
        if(!used)
            return;
        buf.Realloc(static_cast<size_t>(nhead) * sizeof(float));
        buf.ToDevice(host.data());
        hp_ptr = buf.GetDeviceBuffer();
    };

    const bool use_cdf  = fill_field(out.h_cdf,  csv_cdf,  scalar_cdf,  /*clamp=*/true);
    const bool use_topk = fill_field(out.h_topk, csv_topk, scalar_topk, /*clamp=*/true);
    const bool use_sim  = fill_field(out.h_sim,  csv_sim,  scalar_sim,  /*clamp=*/true);
    const bool use_pv   = fill_field(out.h_pvthreshd, csv_pvthreshd, scalar_pvthreshd, /*clamp=*/false);
    upload(use_cdf,  out.h_cdf,       out.buf_cdf,       hp.cdfthreshd_per_head_ptr);
    upload(use_topk, out.h_topk,      out.buf_topk,      hp.topk_per_head_ptr);
    upload(use_sim,  out.h_sim,       out.buf_sim,       hp.simthreshold_per_head_ptr);
    upload(use_pv,   out.h_pvthreshd, out.buf_pvthreshd, hp.pvthreshd_per_head_ptr);
    return true;
}

// CPU-reference mask-prediction params from the same scalars + per-head vectors as the device launch
// (so device and reference select identical blocks).
inline ck_tile::sparge_block_predict_params
make_sparge_predict_params(float scalar_cdf,
                           float scalar_topk,
                           float scalar_sim,
                           const SpargePerHeadSetup& ph,
                           bool attention_sink,
                           bool smooth_k,
                           float scale)
{
    ck_tile::sparge_block_predict_params rp;
    rp.cdfthreshd            = scalar_cdf;
    rp.topk                  = scalar_topk;
    rp.simthreshold          = scalar_sim;
    rp.cdfthreshd_per_head   = ph.h_cdf;   // empty → CPU uses the scalar
    rp.topk_per_head         = ph.h_topk;
    rp.simthreshold_per_head = ph.h_sim;
    rp.attention_sink        = attention_sink;
    rp.smooth_k              = smooth_k;
    rp.scale                 = scale;
    return rp;
}

template <typename T>
ck_tile::HostTensor<T> make_qkv_tensor(ck_tile::index_t batch,
                                       ck_tile::index_t nhead,
                                       ck_tile::index_t seqlen,
                                       ck_tile::index_t hdim,
                                       bool i_perm)
{
    if(i_perm)
        return ck_tile::HostTensor<T>({batch, nhead, seqlen, hdim});
    return ck_tile::HostTensor<T>({batch, seqlen, nhead, hdim});
}

template <typename T>
ck_tile::HostTensor<T> to_bhsd(const ck_tile::HostTensor<T>& tensor, bool is_bhsd)
{
    auto lens               = tensor.get_lengths();
    ck_tile::index_t batch  = lens[0];
    ck_tile::index_t seqlen = is_bhsd ? lens[2] : lens[1];
    ck_tile::index_t nhead  = is_bhsd ? lens[1] : lens[2];
    ck_tile::index_t hdim   = lens[3];
    ck_tile::HostTensor<T> out({batch, nhead, seqlen, hdim});
    for(ck_tile::index_t b = 0; b < batch; ++b)
        for(ck_tile::index_t h = 0; h < nhead; ++h)
            for(ck_tile::index_t s = 0; s < seqlen; ++s)
                for(ck_tile::index_t d = 0; d < hdim; ++d)
                    out(b, h, s, d) = is_bhsd ? tensor(b, h, s, d) : tensor(b, s, h, d);
    return out;
}

template <typename DataTypeConfig>
auto get_error_tolerance()
{
    using T = typename FmhaSparseFwdTypeConfig<DataTypeConfig>::QDataType;
    // fp16 atol=2.5e-1 covers worst-case alibi drift (slope*k); bf16 ceiling 3e-1.
    double rtol = 2e-2;
    double atol = 2.5e-1;
    if constexpr(std::is_same_v<T, ck_tile::bf16_t>)
    {
        atol = 3.5e-1;
        rtol = 2e-1;
    }
    return ck_tile::make_tuple(rtol, atol);
}

template <typename T>
float to_float_for_compare(T value)
{
    return static_cast<float>(value);
}

template <>
inline float to_float_for_compare<ck_tile::bf16_t>(ck_tile::bf16_t value)
{
#if CK_TILE_USE_CUSTOM_DATA_TYPE
    return static_cast<float>(value);
#else
    return ck_tile::bf16_to_float_raw(ck_tile::bit_cast<ck_tile::bf16_raw_t>(value));
#endif
}

// sageattn-style host V prequant: per-(sub-batch, kv_head, channel) absmax/fp8_max descale then fp8
// quant. v_in/v_fp8 follow i_perm; v_dequant is BHSD (reference layout); v_descale is
// [batch,nhead_k,hdim_v]. outer index ob = packed?0:b (batch V is [B,..]; group V is packed [1,..]).
template <typename T>
inline void host_prequant_v_fp8(const ck_tile::HostTensor<T>& v_in,
                                ck_tile::HostTensor<ck_tile::fp8_t>& v_fp8,
                                ck_tile::HostTensor<T>& v_dequant,
                                ck_tile::HostTensor<float>& v_descale,
                                int batch,
                                ck_tile::index_t nhead_k,
                                ck_tile::index_t hdim_v,
                                bool i_perm,
                                bool packed,
                                const std::vector<int32_t>& tok_base,
                                const std::vector<int32_t>& tok_len)
{
    const float fp8_max = ck_tile::type_convert<float>(ck_tile::numeric<ck_tile::fp8_t>::max());
    auto rd = [&](int ob, ck_tile::index_t h, ck_tile::index_t tok, ck_tile::index_t dv) {
        return to_float_for_compare(i_perm ? v_in(ob, h, tok, dv) : v_in(ob, tok, h, dv));
    };
    auto wr = [&](int ob, ck_tile::index_t h, ck_tile::index_t tok,
                  ck_tile::index_t dv) -> decltype(auto) {
        return i_perm ? v_fp8(ob, h, tok, dv) : v_fp8(ob, tok, h, dv);
    };
    for(int b = 0; b < batch; ++b)
    {
        const int ob = packed ? 0 : b;
        const int t0 = tok_base[static_cast<size_t>(b)];
        const int tl = tok_len[static_cast<size_t>(b)];
        for(ck_tile::index_t h = 0; h < nhead_k; ++h)
            for(ck_tile::index_t dv = 0; dv < hdim_v; ++dv)
            {
                float amax = 0.0f;
                for(int s = 0; s < tl; ++s)
                    amax = std::max(amax, std::abs(rd(ob, h, t0 + s, dv)));
                const float sc = (amax > 0.0f) ? amax / fp8_max : 1.0f;
                v_descale(b, h, dv) = sc;
                for(int s = 0; s < tl; ++s)
                {
                    const float vv = rd(ob, h, t0 + s, dv);
                    const auto q8  = ck_tile::type_convert<ck_tile::fp8_t>(vv / sc);
                    wr(ob, h, t0 + s, dv) = q8;
                    v_dequant(ob, h, t0 + s, dv) =
                        ck_tile::type_convert<T>(ck_tile::type_convert<float>(q8) * sc);
                }
            }
    }
}

template <typename T>
bool validate_tensors(const ck_tile::HostTensor<T>& gpu_bhsd,
                      const ck_tile::HostTensor<T>& ref_bhsd,
                      double rtol,
                      double atol,
                      [[maybe_unused]] const std::string& label = "",
                      bool print_result = true)
{
    float max_diff     = 0.0f;
    float max_rel_diff = 0.0f;
    size_t num_errors  = 0;
    constexpr size_t kMaxMismatchPrint = 5;
    std::vector<std::string> mismatch_msgs;

    for(size_t i = 0; i < gpu_bhsd.mData.size(); ++i)
    {
        float gpu_val  = to_float_for_compare(gpu_bhsd.mData[i]);
        float ref_val  = to_float_for_compare(ref_bhsd.mData[i]);
        float diff     = std::abs(gpu_val - ref_val);
        float rel_diff = (std::abs(ref_val) > 1e-6f) ? diff / std::abs(ref_val) : diff;

        max_diff     = std::max(max_diff, diff);
        max_rel_diff = std::max(max_rel_diff, rel_diff);

        // CK's standard check_err combined threshold: error iff diff exceeds
        // atol + rtol*|ref|. (The old `diff>atol && rel_diff>rtol` passed a point
        // if it was within EITHER bound, masking large-abs-near-large-ref and
        // large-rel-near-zero errors.)
        if(diff > atol + rtol * std::abs(ref_val))
        {
            num_errors++;
            if(mismatch_msgs.size() < kMaxMismatchPrint)
            {
                std::ostringstream oss;
                oss << "  Mismatch at " << i << ": GPU=" << gpu_val
                    << ", Ref=" << ref_val << ", Diff=" << diff;
                mismatch_msgs.push_back(oss.str());
            }
        }
    }

    bool pass = (num_errors == 0);
    if(print_result)
        std::cout << ", valid:" << (pass ? "y" : "n") << std::flush << std::endl;

    if(!pass)
    {
        for(const auto& msg : mismatch_msgs)
            std::cout << msg << std::endl;
        std::cout << label << ": max_abs_diff=" << max_diff
                  << ", max_rel_diff=" << max_rel_diff
                  << ", errors=" << num_errors << "/" << gpu_bhsd.mData.size() << std::endl;
    }
    return pass;
}

// When logits_soft_cap > 0, applies Gemma-style soft-cap pre-softmax (NO_BIAS only).
template <typename T, typename DataTypeConfig>
bool validate_vs_blocked_ref(
    const ck_tile::HostTensor<T>& q_host,
    const ck_tile::HostTensor<T>& k_host,
    const ck_tile::HostTensor<T>& v_host,
    const ck_tile::HostTensor<T>& output_host,
    const ck_tile::HostTensor<uint8_t>& block_mask,
    ck_tile::index_t batch,
    ck_tile::index_t nhead,
    ck_tile::index_t seqlen_q,
    ck_tile::index_t hdim_v,
    ck_tile::index_t block_size,
    float scale,
    int causal_type,
    int window_left,
    int window_right,
    bool i_perm,
    bool o_perm,
    const std::string& label,
    bool print_result     = true,
    float logits_soft_cap = 0.0f,
    float pvthreshd       = 0.0f,
    const std::vector<float>* pvthreshd_per_head = nullptr)
{
    auto q_ref   = to_bhsd(q_host, i_perm);
    auto k_ref   = to_bhsd(k_host, i_perm);
    auto v_ref   = to_bhsd(v_host, i_perm);
    auto gpu_out = to_bhsd(output_host, o_perm);
    ck_tile::HostTensor<T> ref_out({batch, nhead, seqlen_q, hdim_v});
    ck_tile::reference_blocked_attention<T, uint8_t, T>(
        q_ref, k_ref, v_ref, block_mask, ref_out, block_size, block_size, scale,
        causal_type, window_left, window_right,
        /*logits_soft_cap=*/logits_soft_cap,
        /*bias=*/static_cast<const ck_tile::HostTensor<T>*>(nullptr), /*bias_rank=*/0,
        pvthreshd, pvthreshd_per_head);
    auto [rtol, atol] = get_error_tolerance<DataTypeConfig>();
    return validate_tensors(gpu_out, ref_out, rtol, atol, label, print_result);
}

// Bias rank → shape: 0 → [1,1,sq,sk] (bcast b/h), 1 → [1,h,sq,sk] (bcast b), 2 → [b,h,sq,sk].
template <typename BiasT>
inline ck_tile::HostTensor<BiasT>
generate_random_bias(int rank,
                     ck_tile::index_t batch,
                     ck_tile::index_t nhead,
                     ck_tile::index_t seqlen_q,
                     ck_tile::index_t seqlen_k,
                     uint32_t seed)
{
    const ck_tile::index_t b0 = (rank == 2) ? batch : 1;
    const ck_tile::index_t b1 = (rank == 0) ? 1 : nhead;
    ck_tile::HostTensor<BiasT> b({b0, b1, seqlen_q, seqlen_k});
    ck_tile::FillUniformDistribution<BiasT>{-0.5f, 0.5f, seed}(b);
    return b;
}

template <typename T, typename BiasT, typename DataTypeConfig>
bool validate_vs_blocked_ref_with_bias(
    const ck_tile::HostTensor<T>& q_host,
    const ck_tile::HostTensor<T>& k_host,
    const ck_tile::HostTensor<T>& v_host,
    const ck_tile::HostTensor<T>& output_host,
    const ck_tile::HostTensor<uint8_t>& block_mask,
    const ck_tile::HostTensor<BiasT>& bias,
    int bias_rank,
    ck_tile::index_t batch,
    ck_tile::index_t nhead,
    ck_tile::index_t seqlen_q,
    ck_tile::index_t hdim_v,
    ck_tile::index_t block_size,
    float scale,
    int causal_type,
    int window_left,
    int window_right,
    bool i_perm,
    bool o_perm,
    const std::string& label,
    bool print_result     = true,
    float logits_soft_cap = 0.0f,
    float pvthreshd       = 0.0f,
    const std::vector<float>* pvthreshd_per_head = nullptr)
{
    auto q_ref   = to_bhsd(q_host, i_perm);
    auto k_ref   = to_bhsd(k_host, i_perm);
    auto v_ref   = to_bhsd(v_host, i_perm);
    auto gpu_out = to_bhsd(output_host, o_perm);
    ck_tile::HostTensor<T> ref_out({batch, nhead, seqlen_q, hdim_v});
    ck_tile::reference_blocked_attention<T, uint8_t, BiasT>(
        q_ref, k_ref, v_ref, block_mask, ref_out, block_size, block_size, scale,
        causal_type, window_left, window_right,
        logits_soft_cap, &bias, bias_rank, pvthreshd, pvthreshd_per_head);
    auto [rtol, atol] = get_error_tolerance<DataTypeConfig>();
    return validate_tensors(gpu_out, ref_out, rtol, atol, label, print_result);
}

// RAII bias buffer + alibi slopes. Caller must keep it alive across the launch (args.ptr aliases
// buf's device memory). unique_ptr keeps BiasSetup movable (DeviceMem has user-provided dtor only).
template <typename BiasT>
struct BiasSetup
{
    sparse_attn_bias_args     args;
    std::optional<ck_tile::HostTensor<BiasT>> elementwise_host;
    std::unique_ptr<ck_tile::DeviceMem> buf;
    std::vector<float>   alibi_slopes_host;
};

template <typename BiasT>
inline BiasSetup<BiasT> setup_bias(const bias_info& bi,
                                   ck_tile::index_t batch,
                                   ck_tile::index_t nhead,
                                   ck_tile::index_t seqlen_q,
                                   ck_tile::index_t seqlen_k,
                                   int causal_type,
                                   uint32_t seed,
                                   const char* api_label)
{
    BiasSetup<BiasT> s;
    if(bi.type == bias_enum::no_bias)
        return s;

    if(bi.type == bias_enum::elementwise_bias)
    {
        s.elementwise_host.emplace(generate_random_bias<BiasT>(
            bi.rank_info, batch, nhead, seqlen_q, seqlen_k, seed));
        s.buf = std::make_unique<ck_tile::DeviceMem>(
            s.elementwise_host->get_element_space_size_in_bytes());
        s.buf->ToDevice(s.elementwise_host->data());
        s.args.type = static_cast<int>(bi.type);
        s.args.rank = bi.rank_info;
        s.args.ptr  = s.buf->GetDeviceBuffer();
        s.args.stride_bias       = seqlen_k;
        s.args.nhead_stride_bias = (bi.rank_info == 0) ? 0 : seqlen_q * seqlen_k;
        s.args.batch_stride_bias = (bi.rank_info == 2) ? nhead * seqlen_q * seqlen_k : 0;
        return s;
    }

    if(causal_type == 0)
    {
        std::cerr << "[" << api_label << "] alibi requires a causal mask (-mask=t/b).\n";
        s.args.type = -1;
        return s;
    }
    const auto base_slopes = ck_tile::get_alibi_slopes<float>(nhead);
    const ck_tile::index_t outer = (bi.rank_info == 0 ? 1 : batch);
    s.alibi_slopes_host.reserve(static_cast<size_t>(outer) * nhead);
    for(ck_tile::index_t b = 0; b < outer; ++b)
        s.alibi_slopes_host.insert(s.alibi_slopes_host.end(),
                                   base_slopes.begin(), base_slopes.end());
    s.buf = std::make_unique<ck_tile::DeviceMem>(s.alibi_slopes_host.size() * sizeof(float));
    s.buf->ToDevice(s.alibi_slopes_host.data());
    s.args.type = static_cast<int>(bi.type);
    s.args.rank = bi.rank_info;
    s.args.ptr  = s.buf->GetDeviceBuffer();
    s.args.stride_bias       = (bi.rank_info == 0) ? 0 : nhead;
    s.args.nhead_stride_bias = 0;
    s.args.batch_stride_bias = 0;
    return s;
}

// Slice elementwise bias down to [1, h_or_1, sq, sk] for group-mode per-batch validation.
template <typename BiasT>
inline ck_tile::HostTensor<BiasT> slice_elementwise_bias_to_b1(
    const ck_tile::HostTensor<BiasT>& full_bias,
    int bias_rank,
    ck_tile::index_t b,
    ck_tile::index_t nhead,
    ck_tile::index_t sq,
    ck_tile::index_t sk)
{
    const ck_tile::index_t h_dim = (bias_rank == 0) ? 1 : nhead;
    const ck_tile::index_t bias_b_src = (bias_rank == 2) ? b : 0;
    ck_tile::HostTensor<BiasT> sub({1, h_dim, sq, sk});
    for(ck_tile::index_t h = 0; h < h_dim; ++h)
        for(ck_tile::index_t q = 0; q < sq; ++q)
            for(ck_tile::index_t k = 0; k < sk; ++k)
                sub(0, h, q, k) = full_bias(bias_b_src, h, q, k);
    return sub;
}

// Build the [1,h,sq,sk] dense bias replicating ck_tile::Alibi<RowMajor=true> for all mask modes.
// Mode mirrors make_alibi_from_lr_mask: is_causal (left<0 && right==0) -> VERTICAL; else causal_type.
template <typename BiasT>
inline ck_tile::HostTensor<BiasT> alibi_to_dense(
    const std::vector<float>& slopes_host,
    ck_tile::index_t nhead,
    ck_tile::index_t seqlen_q,
    ck_tile::index_t seqlen_k,
    int window_left,
    int window_right,
    int causal_type)
{
    const bool is_causal = (window_left < 0 && window_right == 0);
    const int  mode      = is_causal ? 0 : causal_type; // 0=VERTICAL,1=top-left,2=bottom-right
    const ck_tile::long_index_t srd = // shift_right_down: bottom-right only
        (mode == 2) ? std::max<ck_tile::long_index_t>(seqlen_k - seqlen_q, 0) : 0;
    const ck_tile::long_index_t slu = // shift_left_up: bottom-right only
        (mode == 2) ? std::max<ck_tile::long_index_t>(seqlen_q - seqlen_k, 0) : 0;
    ck_tile::HostTensor<BiasT> dense({1, nhead, seqlen_q, seqlen_k});
    for(ck_tile::index_t h = 0; h < nhead; ++h)
    {
        const float slope = (mode == 0) ? slopes_host[static_cast<size_t>(h)]
                                        : -slopes_host[static_cast<size_t>(h)];
        for(ck_tile::index_t i = 0; i < seqlen_q; ++i)
        {
            const ck_tile::long_index_t zp = (mode == 0) ? srd : (static_cast<ck_tile::long_index_t>(i) + srd);
            for(ck_tile::index_t j = 0; j < seqlen_k; ++j)
            {
                const ck_tile::long_index_t pos =
                    std::abs(zp - (static_cast<ck_tile::long_index_t>(j) + slu));
                dense(0, h, i, j) = static_cast<BiasT>(slope * static_cast<float>(pos));
            }
        }
    }
    return dense;
}

inline ck_tile::HostTensor<uint8_t> generate_random_block_mask(
    ck_tile::index_t batch,
    ck_tile::index_t nhead,
    ck_tile::index_t num_q_blocks,
    ck_tile::index_t num_k_blocks,
    float sparsity,
    uint32_t seed,
    bool ensure_diagonal = false)
{
    ck_tile::HostTensor<uint8_t> mask({batch, nhead, num_q_blocks, num_k_blocks});
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for(ck_tile::index_t b = 0; b < batch; ++b)
        for(ck_tile::index_t h = 0; h < nhead; ++h)
            for(ck_tile::index_t qb = 0; qb < num_q_blocks; ++qb)
                for(ck_tile::index_t kb = 0; kb < num_k_blocks; ++kb)
                {
                    bool is_diag        = ensure_diagonal && (qb == kb && qb < num_k_blocks);
                    mask(b, h, qb, kb)  = (is_diag || dist(rng) > sparsity) ? 1 : 0;
                }
    return mask;
}

inline std::vector<int32_t> to_seqstarts(const std::vector<int32_t>& seqlens)
{
    std::vector<int32_t> seqstarts(seqlens.size() + 1, 0);
    for(size_t i = 0; i < seqlens.size(); ++i)
        seqstarts[i + 1] = seqstarts[i] + seqlens[i];
    return seqstarts;
}

// Swap-walk perturbation around seqlen_avg, clamped to [seqlen_min, seqlen_max].
inline std::vector<int32_t> generate_seqlens_group(
    int32_t batch,
    int32_t seqlen_avg,
    int32_t seqlen_min,
    int32_t seqlen_max,
    uint32_t seed)
{
    seqlen_min = std::max(int32_t{1}, seqlen_min);
    if(seqlen_max <= 0)
        seqlen_max = seqlen_avg;
    seqlen_min = std::min(seqlen_min, seqlen_max);

    std::vector<int32_t> seqlens(static_cast<size_t>(batch),
                                 std::clamp(seqlen_avg, seqlen_min, seqlen_max));
    if(batch < 2)
        return seqlens;

    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> idx_dist(0, static_cast<size_t>(batch) - 1);
    std::uniform_int_distribution<size_t> step_dist(1, static_cast<size_t>(batch) - 1);
    const size_t reps = static_cast<size_t>(seqlen_avg) * (static_cast<size_t>(batch) / 2);
    for(size_t r = 0; r < reps; ++r)
    {
        const size_t dec = idx_dist(rng);
        if(seqlens[dec] == seqlen_min)
            continue;
        const size_t inc = (dec + step_dist(rng)) % static_cast<size_t>(batch);
        if(seqlens[inc] >= seqlen_max)
            continue;
        --seqlens[dec];
        ++seqlens[inc];
    }
    return seqlens;
}

inline std::vector<int32_t> seqlens_to_blocks(const std::vector<int32_t>& seqlens,
                                              ck_tile::index_t block_size)
{
    std::vector<int32_t> blocks(seqlens.size());
    for(size_t i = 0; i < seqlens.size(); ++i)
        blocks[i] = (seqlens[i] + block_size - 1) / block_size;
    return blocks;
}

// Batch-outer/head-mid packed uint8_t. linear idx = base*nhead + h*(qb_n*kb_n) + qb*kb_n + kb,
// base = mask_batch_offsets[b] (single-head pair cumulative).
inline ck_tile::HostTensor<uint8_t> generate_random_block_mask_group(
    const std::vector<int32_t>& q_blocks_per_b,
    const std::vector<int32_t>& k_blocks_per_b,
    ck_tile::index_t nhead,
    float sparsity,
    uint32_t seed,
    bool ensure_diagonal,
    std::vector<int32_t>& mask_batch_offsets_out)
{
    const auto batch = static_cast<int32_t>(q_blocks_per_b.size());
    mask_batch_offsets_out.assign(static_cast<size_t>(batch + 1), 0);
    for(int32_t b = 0; b < batch; ++b)
    {
        mask_batch_offsets_out[b + 1] =
            mask_batch_offsets_out[b] + q_blocks_per_b[b] * k_blocks_per_b[b];
    }
    const int32_t per_head_size = mask_batch_offsets_out[batch];

    // 2D alloc for sizing only; data written via the packed linear index (kernel reads flat buffer).
    ck_tile::HostTensor<uint8_t> mask({nhead, per_head_size});
    uint8_t* mask_data = mask.data();
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for(ck_tile::index_t h = 0; h < nhead; ++h)
    {
        for(int32_t b = 0; b < batch; ++b)
        {
            const int32_t qb_n  = q_blocks_per_b[b];
            const int32_t kb_n  = k_blocks_per_b[b];
            const int32_t base  = mask_batch_offsets_out[b];
            const int64_t blk0  = static_cast<int64_t>(base) * nhead +
                                  static_cast<int64_t>(h) * qb_n * kb_n;
            for(int32_t qb = 0; qb < qb_n; ++qb)
                for(int32_t kb = 0; kb < kb_n; ++kb)
                {
                    const bool is_diag = ensure_diagonal && (qb == kb);
                    mask_data[blk0 + qb * kb_n + kb] =
                        (is_diag || dist(rng) > sparsity) ? 1 : 0;
                }
        }
    }
    return mask;
}

// Slice packed Q/K/V/O back to a B=1 sub-tensor for per-batch validation.
template <typename T>
ck_tile::HostTensor<T> slice_packed_to_b1(const ck_tile::HostTensor<T>& packed,
                                          int32_t seq_start,
                                          int32_t sq,
                                          ck_tile::index_t nhead,
                                          ck_tile::index_t hdim,
                                          bool perm)
{
    auto out = perm ? ck_tile::HostTensor<T>({1, nhead, sq, hdim})
                    : ck_tile::HostTensor<T>({1, sq, nhead, hdim});
    if(perm)
    {
        for(ck_tile::index_t h = 0; h < nhead; ++h)
            for(int32_t s = 0; s < sq; ++s)
                for(ck_tile::index_t d = 0; d < hdim; ++d)
                    out(0, h, s, d) = packed(0, h, seq_start + s, d);
    }
    else
    {
        for(int32_t s = 0; s < sq; ++s)
            for(ck_tile::index_t h = 0; h < nhead; ++h)
                for(ck_tile::index_t d = 0; d < hdim; ++d)
                    out(0, s, h, d) = packed(0, seq_start + s, h, d);
    }
    return out;
}

// Slice packed jenga mask into a [1, H, q_blocks_b, k_blocks_b] sub-mask.
inline ck_tile::HostTensor<uint8_t>
slice_packed_mask_to_b1(const ck_tile::HostTensor<uint8_t>& packed_mask,
                        int32_t batch_idx,
                        const std::vector<int32_t>& q_blocks_per_b,
                        const std::vector<int32_t>& k_blocks_per_b,
                        const std::vector<int32_t>& mask_batch_offsets,
                        ck_tile::index_t nhead)
{
    const int32_t qb_n = q_blocks_per_b[batch_idx];
    const int32_t kb_n = k_blocks_per_b[batch_idx];
    const int32_t base = mask_batch_offsets[batch_idx];
    const uint8_t* mask_data = packed_mask.data();
    ck_tile::HostTensor<uint8_t> sub({1, nhead, qb_n, kb_n});
    for(ck_tile::index_t h = 0; h < nhead; ++h)
    {
        const int64_t blk0 = static_cast<int64_t>(base) * nhead +
                             static_cast<int64_t>(h) * qb_n * kb_n;
        for(int32_t qb = 0; qb < qb_n; ++qb)
            for(int32_t kb = 0; kb < kb_n; ++kb)
                sub(0, h, qb, kb) = mask_data[blk0 + qb * kb_n + kb];
    }
    return sub;
}

// Shared per-(b,h,q) row helpers for LUT delta-encode/decode. Both encoders (batch + group) and both
// decoders (batch + group) call these on a contiguous row; only the base-offset math differs upstream.

// Encode one row: block-map[num_block_k] (0/1, >0.5f test) -> delta-encoded LUT row, zero-padded to
// num_block_k, returns valid_block_num. First selected block is stored absolute (delta-from-0); each
// subsequent one is (kb - prev). Mirrors the original inner loops exactly.
inline int32_t encode_lut_row(const uint8_t* map_row, int32_t* lut_row, int32_t num_block_k)
{
    int32_t valid = 0;
    int32_t prev  = -1;
    for(int32_t kb = 0; kb < num_block_k; ++kb)
    {
        if(static_cast<float>(map_row[kb]) > 0.5f)
        {
            lut_row[valid] = (prev < 0) ? kb : (kb - prev);
            prev           = kb;
            ++valid;
        }
    }
    for(int32_t i = valid; i < num_block_k; ++i)
        lut_row[i] = 0;
    return valid;
}

// Decode one row: delta LUT row + valid_block_num -> block-map[num_block_k] set to 1 for selected
// blocks (in-range writes only). e is capped at both vbn and num_block_k; acc is a cumulative sum
// (e==0 is absolute). Mirrors the original inner loops exactly.
inline void decode_lut_row(const int32_t* lut_row, int32_t vbn, uint8_t* map_row, int32_t num_block_k)
{
    int32_t acc = 0;
    for(int32_t e = 0; e < vbn && e < num_block_k; ++e)
    {
        acc += lut_row[e];
        if(acc >= 0 && acc < num_block_k)
            map_row[acc] = 1;
    }
}

// Packed jenga mask -> VSA group (LUT, VBN), batch-outer/head-mid packed layout:
//   lut: mask_base*nhead + h*(qb_n*kb_n) + qb*kb_n + col   (delta-encoded K idx, zero-padded/row)
//   vbn: vbn_base*nhead  + h*qb_n        + qb
inline void block_map_to_lut_group(
    const ck_tile::HostTensor<uint8_t>& packed_mask,
    const std::vector<int32_t>& q_blocks_per_b,
    const std::vector<int32_t>& k_blocks_per_b,
    const std::vector<int32_t>& mask_batch_offsets,
    const std::vector<int32_t>& seqstart_q_block_host,
    ck_tile::HostTensor<int32_t>& lut_packed,
    ck_tile::HostTensor<int32_t>& vbn_packed)
{
    const auto batch = static_cast<int32_t>(q_blocks_per_b.size());
    const auto nhead = static_cast<ck_tile::index_t>(packed_mask.get_lengths()[0]);

    const uint8_t* mask_data = packed_mask.data();
    int32_t*       lut_data  = lut_packed.data();
    int32_t*       vbn_data  = vbn_packed.data();

    for(ck_tile::index_t h = 0; h < nhead; ++h)
    {
        for(int32_t b = 0; b < batch; ++b)
        {
            const int32_t qb_n      = q_blocks_per_b[b];
            const int32_t kb_n      = k_blocks_per_b[b];
            const int32_t mask_base = mask_batch_offsets[b];
            const int32_t vbn_base  = seqstart_q_block_host[b];
            const int64_t lut_blk0 = static_cast<int64_t>(mask_base) * nhead +
                                     static_cast<int64_t>(h) * qb_n * kb_n;
            const int64_t vbn_blk0 = static_cast<int64_t>(vbn_base) * nhead +
                                     static_cast<int64_t>(h) * qb_n;
            for(int32_t qb = 0; qb < qb_n; ++qb)
            {
                const int64_t row = lut_blk0 + static_cast<int64_t>(qb) * kb_n;
                vbn_data[vbn_blk0 + qb] =
                    encode_lut_row(mask_data + row, lut_data + row, kb_n);
            }
        }
    }
}

inline void block_map_to_lut(const ck_tile::HostTensor<uint8_t>& block_map,
                             ck_tile::HostTensor<int32_t>& lut,
                             ck_tile::HostTensor<int32_t>& valid_block_num)
{
    auto lens                      = block_map.get_lengths();
    ck_tile::index_t batch         = lens[0];
    ck_tile::index_t nhead         = lens[1];
    ck_tile::index_t num_q_blocks  = lens[2];
    ck_tile::index_t num_k_blocks  = lens[3];

    const uint8_t* map_data = block_map.data();
    int32_t*       lut_data = lut.data();

    for(ck_tile::index_t b = 0; b < batch; ++b)
        for(ck_tile::index_t h = 0; h < nhead; ++h)
            for(ck_tile::index_t qb = 0; qb < num_q_blocks; ++qb)
            {
                const size_t row =
                    ((static_cast<size_t>(b) * nhead + h) * num_q_blocks + qb) * num_k_blocks;
                valid_block_num(b, h, qb) = encode_lut_row(
                    map_data + row, lut_data + row, static_cast<int32_t>(num_k_blocks));
            }
}

// Decode device delta-encoded LUT + valid-block counts into a dense [b,h,qb,kb] 0/1 map, so the sage
// validation references the kernel's actual selection instead of re-predicting on the host.
inline ck_tile::HostTensor<uint8_t> device_lut_to_block_map(
    const std::vector<int32_t>& lut,
    const std::vector<int32_t>& vbn,
    ck_tile::index_t batch,
    ck_tile::index_t nhead,
    ck_tile::index_t num_q_blocks,
    ck_tile::index_t num_k_blocks)
{
    ck_tile::HostTensor<uint8_t> block_map({batch, nhead, num_q_blocks, num_k_blocks});
    for(auto& v : block_map.mData)
        v = 0;
    uint8_t* map_data = block_map.data();
    for(ck_tile::index_t b = 0; b < batch; ++b)
        for(ck_tile::index_t h = 0; h < nhead; ++h)
            for(ck_tile::index_t qb = 0; qb < num_q_blocks; ++qb)
            {
                const size_t row = (static_cast<size_t>(b) * nhead + h) * num_q_blocks + qb;
                decode_lut_row(lut.data() + row * num_k_blocks,
                               vbn[row],
                               map_data + row * num_k_blocks,
                               static_cast<int32_t>(num_k_blocks));
            }
    return block_map;
}

// Group variant of device_lut_to_block_map. Packed batch-outer/head-mid (see block_map_to_lut_group):
// mask_base = prefix sum of q_blocks*k_blocks, vbn_base = prefix sum of q_blocks. Decodes sub-batch b
// into [1, nhead, qb_b, kb_b].
inline ck_tile::HostTensor<uint8_t> device_lut_to_block_map_group(
    const std::vector<int32_t>& lut,
    const std::vector<int32_t>& vbn,
    ck_tile::index_t nhead,
    ck_tile::index_t mask_base,
    ck_tile::index_t vbn_base,
    ck_tile::index_t qb_b,
    ck_tile::index_t kb_b)
{
    ck_tile::HostTensor<uint8_t> bm({1, nhead, qb_b, kb_b});
    for(auto& v : bm.mData)
        v = 0;
    uint8_t* bm_data = bm.data();
    for(ck_tile::index_t h = 0; h < nhead; ++h)
        for(ck_tile::index_t qb = 0; qb < qb_b; ++qb)
        {
            const size_t vidx = static_cast<size_t>(vbn_base) * nhead +
                                static_cast<size_t>(h) * qb_b + qb;
            const size_t lbase = static_cast<size_t>(mask_base) * nhead +
                                 static_cast<size_t>(h) * qb_b * kb_b +
                                 static_cast<size_t>(qb) * kb_b;
            const size_t mbase = (static_cast<size_t>(h) * qb_b + qb) * kb_b;
            decode_lut_row(lut.data() + lbase,
                           vbn[vidx],
                           bm_data + mbase,
                           static_cast<int32_t>(kb_b));
        }
    return bm;
}

// Selection soft-check: device vs host predicted block-mask agreement. Near-tie score flips
// (float reduction-order, ALIBI, top-k ties) make exact match impossible, so this only HARD-FAILS
// on gross divergence; otherwise it just reports the agreement rate.
inline bool check_selection_agreement(const ck_tile::HostTensor<uint8_t>& device_mask,
                                      const ck_tile::HostTensor<uint8_t>& host_mask,
                                      const std::string& label)
{
    const size_t n = device_mask.mData.size();
    size_t agree = 0, dev_sel = 0, host_sel = 0;
    for(size_t i = 0; i < n; ++i) {
        const bool d = device_mask.mData[i] != 0, h = host_mask.mData[i] != 0;
        if(d == h) ++agree;
        dev_sel += d; host_sel += h;
    }
    const double rate = n ? double(agree) / double(n) : 1.0;
    std::cout << ", sel_agree=" << rate;
    // gross-divergence guard: tolerant of near-tie noise, catches a broken selector.
    const bool gross = (rate < 0.5)
        || (host_sel > 0 && (double(dev_sel) / double(host_sel) < 0.5
                             || double(dev_sel) / double(host_sel) > 2.0));
    if(gross)
        std::cout << " [SELECTION MISMATCH " << label << ": dev_sel=" << dev_sel
                  << " host_sel=" << host_sel << "]";
    return !gross;
}

inline std::size_t compute_sparse_attn_flop(
    ck_tile::index_t batch,
    ck_tile::index_t nhead,
    ck_tile::index_t seqlen_q,
    ck_tile::index_t seqlen_k,
    ck_tile::index_t hdim_q,
    ck_tile::index_t hdim_v,
    float sparsity,
    const mask_info& mask)
{
    std::size_t effective_area;
    if(mask.type == mask_enum::no_mask)
    {
        effective_area = static_cast<std::size_t>(
            static_cast<double>(seqlen_q) * seqlen_k * (1.0 - sparsity));
    }
    else
    {
        effective_area = static_cast<std::size_t>(
            static_cast<double>(mask.get_unmaskarea()) * (1.0 - sparsity));
    }
    return static_cast<std::size_t>(batch) * nhead *
           (2 * effective_area * hdim_q + 2 * effective_area * hdim_v);
}

template <typename T>
std::size_t compute_sparse_attn_num_byte(
    ck_tile::index_t batch,
    ck_tile::index_t nhead,
    ck_tile::index_t nhead_k,
    ck_tile::index_t seqlen_q,
    ck_tile::index_t seqlen_k,
    ck_tile::index_t hdim_q,
    ck_tile::index_t hdim_v,
    float sparsity = 0.0f)
{
    // Q load + O store dense; K + V load scaled by (1 - sparsity).
    std::size_t num_byte = static_cast<std::size_t>(batch) * nhead *
                           (sizeof(T) * seqlen_q * hdim_q + sizeof(T) * seqlen_q * hdim_v);
    const auto kv_dense = static_cast<std::size_t>(batch) * nhead_k *
                          (sizeof(T) * seqlen_k * hdim_q + sizeof(T) * seqlen_k * hdim_v);
    num_byte += static_cast<std::size_t>(static_cast<double>(kv_dense) *
                                          (1.0 - static_cast<double>(sparsity)));
    return num_byte;
}

// Group variant: sum per-seq, re-decoding mask_str per (sq,sk) so causal/window areas stay exact
// across differing sequence lengths.
inline std::size_t compute_sparse_attn_flop_group(
    ck_tile::index_t nhead,
    const std::vector<int32_t>& seqlen_qs,
    const std::vector<int32_t>& seqlen_ks,
    ck_tile::index_t hdim_q,
    ck_tile::index_t hdim_v,
    float sparsity,
    const std::string& mask_str)
{
    std::size_t total = 0;
    for(size_t b = 0; b < seqlen_qs.size(); ++b)
    {
        const auto m = mask_info::decode(mask_str, seqlen_qs[b], seqlen_ks[b]);
        const std::size_t mask_area =
            (m.type == mask_enum::no_mask)
                ? static_cast<std::size_t>(seqlen_qs[b]) * static_cast<std::size_t>(seqlen_ks[b])
                : m.get_unmaskarea();
        const auto area = static_cast<std::size_t>(
            static_cast<double>(mask_area) * (1.0 - sparsity));
        total += nhead * (2 * area * hdim_q + 2 * area * hdim_v);
    }
    return total;
}

template <typename T>
std::size_t compute_sparse_attn_num_byte_group(
    ck_tile::index_t nhead,
    ck_tile::index_t nhead_k,
    const std::vector<int32_t>& seqlen_qs,
    const std::vector<int32_t>& seqlen_ks,
    ck_tile::index_t hdim_q,
    ck_tile::index_t hdim_v,
    float sparsity)
{
    std::int64_t total_q = 0, total_k = 0;
    for(auto v : seqlen_qs) total_q += v;
    for(auto v : seqlen_ks) total_k += v;

    std::size_t num_byte = static_cast<std::size_t>(total_q) * nhead *
                           (sizeof(T) * hdim_q + sizeof(T) * hdim_v);
    const auto kv_dense = static_cast<std::size_t>(total_k) * nhead_k *
                          (sizeof(T) * hdim_q + sizeof(T) * hdim_v);
    num_byte += static_cast<std::size_t>(static_cast<double>(kv_dense) *
                                          (1.0 - static_cast<double>(sparsity)));
    return num_byte;
}

inline void print_perf(double avg_time_ms,
                       std::size_t flop,
                       std::size_t num_byte)
{
    const float tflops     = static_cast<float>(flop) / 1.E9 / avg_time_ms;
    const float gb_per_sec = static_cast<float>(num_byte) / 1.E6 / avg_time_ms;
    std::cout << std::fixed << ", " << std::setprecision(3) << avg_time_ms << " ms, "
              << std::setprecision(2) << tflops << " TFlops, "
              << std::setprecision(2) << gb_per_sec << " GB/s" << std::flush;
}

inline std::string format_int_vec(const std::vector<int32_t>& v)
{
    std::ostringstream oss;
    oss << "[";
    for(size_t i = 0; i < v.size(); ++i)
    {
        if(i > 0) oss << ",";
        oss << v[i];
    }
    oss << "]";
    return oss.str();
}

// JSON summary line: stdout if json_file empty, else appended (one line per call).
inline void emit_json_summary(const std::string& json_file,
                              const std::string& api,
                              const std::string& mode,
                              const std::string& prec,
                              ck_tile::index_t batch,
                              ck_tile::index_t nhead,
                              ck_tile::index_t nhead_k,
                              ck_tile::index_t seqlen_q,
                              ck_tile::index_t seqlen_k,
                              ck_tile::index_t hdim_q,
                              ck_tile::index_t hdim_v,
                              float sparsity,
                              const std::string& mask_str,
                              bool i_perm,
                              bool o_perm,
                              double latency_ms,
                              std::size_t flop,
                              std::size_t num_byte,
                              bool has_validation,
                              bool valid,
                              float actual_sparsity, // < 0 if not requested
                              const std::vector<int32_t>& seqlen_qs, // empty in batch
                              const std::vector<int32_t>& seqlen_ks)
{
    const float tflops     = static_cast<float>(flop) / 1.E9 / latency_ms;
    const float gb_per_sec = static_cast<float>(num_byte) / 1.E6 / latency_ms;

    std::ostringstream oss;
    oss << std::fixed;
    oss << "{\"api\":\"" << api << "\","
        << "\"mode\":\"" << mode << "\","
        << "\"prec\":\"" << prec << "\","
        << "\"batch\":" << batch << ","
        << "\"nhead\":" << nhead << ","
        << "\"nhead_k\":" << nhead_k << ","
        << "\"seqlen_q\":" << seqlen_q << ","
        << "\"seqlen_k\":" << seqlen_k << ","
        << "\"hdim_q\":" << hdim_q << ","
        << "\"hdim_v\":" << hdim_v << ","
        << "\"sparsity\":" << std::setprecision(4) << sparsity << ","
        << "\"mask_type\":\"" << mask_str << "\","
        << "\"iperm\":" << (i_perm ? 1 : 0) << ","
        << "\"operm\":" << (o_perm ? 1 : 0) << ","
        << "\"latency_ms\":" << std::setprecision(4) << latency_ms << ","
        << "\"tflops\":" << std::setprecision(2) << tflops << ","
        << "\"gb_per_s\":" << std::setprecision(2) << gb_per_sec;
    if(has_validation)
        oss << ",\"valid\":" << (valid ? "true" : "false");
    if(actual_sparsity >= 0.0f)
        oss << ",\"actual_sparsity\":" << std::setprecision(4) << actual_sparsity;
    if(!seqlen_qs.empty())
        oss << ",\"seqlen_qs\":" << format_int_vec(seqlen_qs)
            << ",\"seqlen_ks\":" << format_int_vec(seqlen_ks);
    oss << "}";

    if(json_file.empty())
    {
        std::cout << "JSON " << oss.str() << std::endl;
    }
    else
    {
        std::ofstream f(json_file, std::ios::app);
        if(f)
            f << oss.str() << "\n";
    }
}

template <typename DataTypeConfig>
sparse_attn_result sparse_attn_fwd_run(
    const std::string& api,     // jenga | vsa | sparge | sparge_sage
    ck_tile::index_t batch,
    ck_tile::index_t nhead,
    ck_tile::index_t nhead_k,
    ck_tile::index_t seqlen_q,
    ck_tile::index_t seqlen_k,
    ck_tile::index_t hdim_q,
    ck_tile::index_t hdim_v,
    float sparsity,
    float simthreshold,
    const std::string& mask_str,
    bool attention_sink,
    ck_tile::index_t block_size,
    bool i_perm,
    bool o_perm,
    bool is_v_rowmajor,
    uint32_t seed,
    int do_validation,
    float pvthreshd,
    const std::string& sparge_mode,           // sparge only: "topk" or "cdf" — block-selection algorithm
    bool perhead_test,                        // sparge only: synthesize per-head hyperparam pattern
    const std::string& sparsity_per_head_csv, // sparge only: CSV per-Q-head sparsity; routed to topk/cdf field by sparge_mode
    const std::string& sim_per_head_csv,
    const std::string& pvthreshd_per_head_csv,
    bool smooth_k,
    bool print_sparsity,
    const ck_tile::stream_config& stream_config,
    sparse_attn_mode mode = sparse_attn_mode::batch,
    bool json_out = false,
    const std::string& json_file = std::string(),
    const std::string& bias_str  = "n",
    float scale_s_user = 0.0f,        // 0 ⇒ default 1/sqrt(d)
    float logits_soft_cap_user = 0.0f, // 0 ⇒ disabled (Gemma-style: s = cap*tanh(s/cap))
    const std::string& qscale = "perwarp", // sparge_sage quant scale mode
    const std::string& qkdtype = "int8")   // sparge_sage Q/K quant dtype: int8 | fp8
{
    using T = typename FmhaSparseFwdTypeConfig<DataTypeConfig>::QDataType;
    // dispatch traits.data_type for the sparge_sage Q/K quant dtype.
    const std::string sage_data_type = (qkdtype == "fp8") ? "fp8bf16" : "i8fp8bf16";
    const bool sage_qk_fp8 = (qkdtype == "fp8");

    if(block_size != 128 || hdim_q != 128 || hdim_v != 128)
    {
        std::cout << ", not supported yet" << std::flush << std::endl;
        return sparse_attn_result::skipped;
    }

    ck_tile::index_t num_q_blocks = (seqlen_q + block_size - 1) / block_size;
    ck_tile::index_t num_k_blocks = (seqlen_k + block_size - 1) / block_size;
    // Match wrapper: scale_s 0 ⇒ default 1/sqrt(d).
    float scale                   = (scale_s_user != 0.0f)
                                        ? scale_s_user
                                        : 1.0f / std::sqrt(static_cast<float>(hdim_q));

    mask_info mask_decoded = mask_info::decode(mask_str, seqlen_q, seqlen_k);
    int causal_type        = 0;
    if(mask_decoded.type == mask_enum::mask_top_left)
        causal_type = 1;
    else if(mask_decoded.type == mask_enum::mask_bottom_right)
        causal_type = 2;

    std::cout << "[" << api << "] b=" << batch
              << ", h=" << nhead << "(" << nhead_k << ")"
              << ", s=" << seqlen_q << "x" << seqlen_k
              << ", d=" << hdim_q << "(" << hdim_v << ")"
              << ", sparsity=" << sparsity
              << ", mask=" << mask_str
              << ", sink=" << attention_sink
              << ", perm=" << i_perm << "/" << o_perm << std::flush;

    std::size_t flop     = compute_sparse_attn_flop(batch, nhead, seqlen_q, seqlen_k,
                                                     hdim_q, hdim_v, sparsity, mask_decoded);
    std::size_t num_byte = compute_sparse_attn_num_byte<T>(batch, nhead, nhead_k,
                                                            seqlen_q, seqlen_k, hdim_q, hdim_v,
                                                            sparsity);

    ck_tile::HostTensor<T> q_host = make_qkv_tensor<T>(batch, nhead, seqlen_q, hdim_q, i_perm);
    ck_tile::HostTensor<T> k_host = make_qkv_tensor<T>(batch, nhead_k, seqlen_k, hdim_q, i_perm);
    ck_tile::HostTensor<T> v_host = make_qkv_tensor<T>(batch, nhead_k, seqlen_k, hdim_v, i_perm);
    ck_tile::HostTensor<T> output_host =
        o_perm ? ck_tile::HostTensor<T>({batch, nhead, seqlen_q, hdim_v})
               : ck_tile::HostTensor<T>({batch, seqlen_q, nhead, hdim_v});

    ck_tile::FillUniformDistribution<T>{-0.5f, 0.5f, seed}(q_host);
    ck_tile::FillUniformDistribution<T>{-0.5f, 0.5f, seed + 1}(k_host);
    ck_tile::FillUniformDistribution<T>{-0.5f, 0.5f, seed + 2}(v_host);

    bool pass = true;
    float ave_time = -1.0f;

    // JSON-summary state, populated per-branch and emitted once at the end. Empty in batch mode.
    std::vector<int32_t> json_seqlen_qs;
    std::vector<int32_t> json_seqlen_ks;
    std::size_t          json_flop          = flop;
    std::size_t          json_num_byte      = num_byte;
    float                json_actual_sparsity = -1.0f;

    if(api == "jenga" && mode == sparse_attn_mode::batch)
    {
        auto block_mask = generate_random_block_mask(
            batch, nhead, num_q_blocks, num_k_blocks, sparsity, seed + 100, true);

        using BiasT = typename FmhaSparseFwdTypeConfig<DataTypeConfig>::BiasDataType;
        bias_info bi = bias_info::decode(bias_str);
        auto bs = setup_bias<BiasT>(bi, batch, nhead, seqlen_q, seqlen_k,
                                    causal_type, seed + 500, "jenga");
        if(bs.args.type < 0)
            return sparse_attn_result::failure;

        try
        {
            ave_time = jenga_sparse_attention<T>(q_host, k_host, v_host, block_mask,
                output_host, batch, nhead, nhead_k, seqlen_q, seqlen_k, hdim_q, hdim_v,
                i_perm, o_perm, is_v_rowmajor, seqlen_q, seqlen_k, stream_config, mask_str,
                bs.args, scale_s_user, logits_soft_cap_user);
        }
        catch(const std::exception& e)
        {
            std::cerr << "\nError: " << e.what() << std::endl;
            return sparse_attn_result::failure;
        }

        if(ave_time < 0)
        {
            std::cout << ", not supported yet" << std::flush << std::endl;
            return sparse_attn_result::skipped;
        }
        if(stream_config.time_kernel_)
            print_perf(static_cast<double>(ave_time), flop, num_byte);

        if(do_validation)
        {
            // jenga ignores -mask; reference must too, else false negatives.
            if(bi.type == bias_enum::elementwise_bias)
            {
                pass = validate_vs_blocked_ref_with_bias<T, BiasT, DataTypeConfig>(
                    q_host, k_host, v_host, output_host, block_mask,
                    *bs.elementwise_host, bi.rank_info,
                    batch, nhead, seqlen_q, hdim_v, block_size, scale,
                    causal_type, mask_decoded.left, mask_decoded.right,
                    i_perm, o_perm, "Jenga vs CPU ref", /*print_result=*/true,
                    logits_soft_cap_user);
            }
            else if(bi.type == bias_enum::alibi)
            {
                auto alibi_dense = alibi_to_dense<BiasT>(
                    bs.alibi_slopes_host, nhead, seqlen_q, seqlen_k,
                    mask_decoded.left, mask_decoded.right, causal_type);
                pass = validate_vs_blocked_ref_with_bias<T, BiasT, DataTypeConfig>(
                    q_host, k_host, v_host, output_host, block_mask,
                    alibi_dense, /*rank=*/1,
                    batch, nhead, seqlen_q, hdim_v, block_size, scale,
                    causal_type, mask_decoded.left, mask_decoded.right,
                    i_perm, o_perm, "Jenga vs CPU ref", /*print_result=*/true,
                    logits_soft_cap_user);
            }
            else
            {
                pass = validate_vs_blocked_ref<T, DataTypeConfig>(
                    q_host, k_host, v_host, output_host, block_mask,
                    batch, nhead, seqlen_q, hdim_v, block_size, scale,
                    causal_type, mask_decoded.left, mask_decoded.right,
                    i_perm, o_perm, "Jenga vs CPU ref",
                    /*print_result=*/true, logits_soft_cap_user);
            }
        }
    }
    else if(api == "jenga" && mode == sparse_attn_mode::group)
    {
        // Per-seq lengths in [seqlen/2, seqlen] (-s acts as max). Midpoint avg so the swap-walk has
        // both directions; avg=max would silently give uniform [max,max,...].
        const int32_t sq_min = std::max(int32_t{block_size}, seqlen_q / 2);
        const int32_t sk_min = std::max(int32_t{block_size}, seqlen_k / 2);
        const int32_t sq_avg = (sq_min + seqlen_q) / 2;
        const int32_t sk_avg = (sk_min + seqlen_k) / 2;
        auto seqlen_qs = generate_seqlens_group(batch, sq_avg, sq_min, seqlen_q, seed + 300);
        auto seqlen_ks = (seqlen_q == seqlen_k && sq_min == sk_min)
                             ? seqlen_qs
                             : generate_seqlens_group(batch, sk_avg, sk_min, seqlen_k, seed + 301);

        const auto seqstart_q_host = to_seqstarts(seqlen_qs);
        const auto seqstart_k_host = to_seqstarts(seqlen_ks);
        const int32_t total_q = seqstart_q_host.back();
        const int32_t total_k = seqstart_k_host.back();
        const auto q_blocks    = seqlens_to_blocks(seqlen_qs, block_size);
        const auto k_blocks    = seqlens_to_blocks(seqlen_ks, block_size);
        const auto seqstart_q_block_host = to_seqstarts(q_blocks);

        auto q_packed = make_qkv_tensor<T>(1, nhead,   total_q, hdim_q, i_perm);
        auto k_packed = make_qkv_tensor<T>(1, nhead_k, total_k, hdim_q, i_perm);
        auto v_packed = make_qkv_tensor<T>(1, nhead_k, total_k, hdim_v, i_perm);
        auto o_packed = o_perm ? ck_tile::HostTensor<T>({1, nhead, total_q, hdim_v})
                               : ck_tile::HostTensor<T>({1, total_q, nhead, hdim_v});
        ck_tile::FillUniformDistribution<T>{-0.5f, 0.5f, seed}(q_packed);
        ck_tile::FillUniformDistribution<T>{-0.5f, 0.5f, seed + 1}(k_packed);
        ck_tile::FillUniformDistribution<T>{-0.5f, 0.5f, seed + 2}(v_packed);

        std::vector<int32_t> mask_batch_offsets;
        auto mask_packed = generate_random_block_mask_group(
            q_blocks, k_blocks, nhead, sparsity, seed + 100, true, mask_batch_offsets);

        const std::size_t flop_g     = compute_sparse_attn_flop_group(
            nhead, seqlen_qs, seqlen_ks, hdim_q, hdim_v, sparsity, mask_str);
        const std::size_t num_byte_g = compute_sparse_attn_num_byte_group<T>(
            nhead, nhead_k, seqlen_qs, seqlen_ks, hdim_q, hdim_v, sparsity);

        using BiasT = typename FmhaSparseFwdTypeConfig<DataTypeConfig>::BiasDataType;
        bias_info bi = bias_info::decode(bias_str);
        auto bs = setup_bias<BiasT>(bi, batch, nhead, seqlen_q, seqlen_k,
                                    causal_type, seed + 500, "jenga group");
        if(bs.args.type < 0)
            return sparse_attn_result::failure;

        try
        {
            ave_time = jenga_sparse_attention<T>(
                q_packed, k_packed, v_packed, mask_packed, o_packed,
                batch, nhead, nhead_k,
                /*seqlen_q=*/0, /*seqlen_k=*/0,
                hdim_q, hdim_v,
                i_perm, o_perm, is_v_rowmajor,
                /*max_seqlen_q=*/seqlen_q, /*max_seqlen_k=*/0,
                stream_config, mask_str, bs.args, scale_s_user, logits_soft_cap_user,
                seqstart_q_host, seqstart_k_host,
                seqstart_q_block_host, mask_batch_offsets);
        }
        catch(const std::exception& e)
        {
            std::cerr << "\nError: " << e.what() << std::endl;
            return sparse_attn_result::failure;
        }
        if(ave_time < 0)
        {
            std::cout << ", not supported yet" << std::flush << std::endl;
            return sparse_attn_result::skipped;
        }
        std::cout << ", sq:" << format_int_vec(seqlen_qs)
                  << ", sk:" << format_int_vec(seqlen_ks) << std::flush;
        if(stream_config.time_kernel_)
            print_perf(static_cast<double>(ave_time), flop_g, num_byte_g);
        json_seqlen_qs = seqlen_qs;
        json_seqlen_ks = seqlen_ks;
        json_flop      = flop_g;
        json_num_byte  = num_byte_g;

        // Validate by slicing each sequence into a B=1 sub-batch; one combined valid:y/n at the end.
        if(do_validation)
        {
            pass = true;
            for(int32_t b = 0; b < batch; ++b)
            {
                const int32_t sq = seqlen_qs[b];
                const int32_t sk = seqlen_ks[b];
                auto q_b = slice_packed_to_b1(q_packed, seqstart_q_host[b], sq, nhead,   hdim_q, i_perm);
                auto k_b = slice_packed_to_b1(k_packed, seqstart_k_host[b], sk, nhead_k, hdim_q, i_perm);
                auto v_b = slice_packed_to_b1(v_packed, seqstart_k_host[b], sk, nhead_k, hdim_v, i_perm);
                auto o_b = slice_packed_to_b1(o_packed, seqstart_q_host[b], sq, nhead,   hdim_v, o_perm);
                auto mask_b = slice_packed_mask_to_b1(
                    mask_packed, b, q_blocks, k_blocks, mask_batch_offsets, nhead);
                bool sub_pass;
                if(bi.type == bias_enum::elementwise_bias)
                {
                    auto bias_b_sub = slice_elementwise_bias_to_b1<BiasT>(
                        *bs.elementwise_host, bi.rank_info, b, nhead, sq, sk);
                    sub_pass = validate_vs_blocked_ref_with_bias<T, BiasT, DataTypeConfig>(
                        q_b, k_b, v_b, o_b, mask_b,
                        bias_b_sub, bi.rank_info,
                        /*batch=*/1, nhead, sq, hdim_v, block_size, scale,
                        causal_type, mask_decoded.left, mask_decoded.right,
                        i_perm, o_perm,
                        std::string("Jenga group+bias sub-batch ") + std::to_string(b),
                        /*print_result=*/false, logits_soft_cap_user);
                }
                else if(bi.type == bias_enum::alibi)
                {
                    auto alibi_dense = alibi_to_dense<BiasT>(
                        bs.alibi_slopes_host, nhead, sq, sk,
                        mask_decoded.left, mask_decoded.right, causal_type);
                    sub_pass = validate_vs_blocked_ref_with_bias<T, BiasT, DataTypeConfig>(
                        q_b, k_b, v_b, o_b, mask_b,
                        alibi_dense, /*rank=*/1,
                        /*batch=*/1, nhead, sq, hdim_v, block_size, scale,
                        causal_type, mask_decoded.left, mask_decoded.right,
                        i_perm, o_perm,
                        std::string("Jenga group+alibi sub-batch ") + std::to_string(b),
                        /*print_result=*/false, logits_soft_cap_user);
                }
                else
                {
                    sub_pass = validate_vs_blocked_ref<T, DataTypeConfig>(
                        q_b, k_b, v_b, o_b, mask_b,
                        /*batch=*/1, nhead, sq, hdim_v, block_size, scale,
                        causal_type, mask_decoded.left, mask_decoded.right,
                        i_perm, o_perm,
                        std::string("Jenga group sub-batch ") + std::to_string(b),
                        /*print_result=*/false, logits_soft_cap_user);
                }
                pass = pass && sub_pass;
            }
            std::cout << ", valid:" << (pass ? "y" : "n") << std::flush << std::endl;
        }
    }
    else if(api == "vsa" && mode == sparse_attn_mode::batch)
    {
        auto block_mask = generate_random_block_mask(
            batch, nhead, num_q_blocks, num_k_blocks, sparsity, seed + 100, true);
        ck_tile::HostTensor<int32_t> lut({batch, nhead, num_q_blocks, num_k_blocks});
        ck_tile::HostTensor<int32_t> valid_block_num({batch, nhead, num_q_blocks});
        block_map_to_lut(block_mask, lut, valid_block_num);

        using BiasT = typename FmhaSparseFwdTypeConfig<DataTypeConfig>::BiasDataType;
        bias_info bi = bias_info::decode(bias_str);
        auto bs = setup_bias<BiasT>(bi, batch, nhead, seqlen_q, seqlen_k,
                                    causal_type, seed + 500, "vsa");
        if(bs.args.type < 0)
            return sparse_attn_result::failure;

        try
        {
            ave_time = vsa_sparse_attention<T>(q_host, k_host, v_host, lut, valid_block_num,
                output_host, batch, nhead, nhead_k, seqlen_q, seqlen_k,
                hdim_q, hdim_v, i_perm, o_perm, is_v_rowmajor, seqlen_q, seqlen_k,
                stream_config, mask_str, bs.args, scale_s_user, logits_soft_cap_user);
        }
        catch(const std::exception& e)
        {
            std::cerr << "\nError: " << e.what() << std::endl;
            return sparse_attn_result::failure;
        }

        if(ave_time < 0)
        {
            std::cout << ", not supported yet" << std::flush << std::endl;
            return sparse_attn_result::skipped;
        }
        if(stream_config.time_kernel_)
            print_perf(static_cast<double>(ave_time), flop, num_byte);

        if(do_validation)
        {
            if(bi.type == bias_enum::elementwise_bias)
            {
                pass = validate_vs_blocked_ref_with_bias<T, BiasT, DataTypeConfig>(
                    q_host, k_host, v_host, output_host, block_mask,
                    *bs.elementwise_host, bi.rank_info,
                    batch, nhead, seqlen_q, hdim_v, block_size, scale,
                    causal_type, mask_decoded.left, mask_decoded.right,
                    i_perm, o_perm, "VSA vs CPU ref", /*print_result=*/true,
                    logits_soft_cap_user);
            }
            else if(bi.type == bias_enum::alibi)
            {
                auto alibi_dense = alibi_to_dense<BiasT>(
                    bs.alibi_slopes_host, nhead, seqlen_q, seqlen_k,
                    mask_decoded.left, mask_decoded.right, causal_type);
                pass = validate_vs_blocked_ref_with_bias<T, BiasT, DataTypeConfig>(
                    q_host, k_host, v_host, output_host, block_mask,
                    alibi_dense, /*rank=*/1,
                    batch, nhead, seqlen_q, hdim_v, block_size, scale,
                    causal_type, mask_decoded.left, mask_decoded.right,
                    i_perm, o_perm, "VSA vs CPU ref", /*print_result=*/true,
                    logits_soft_cap_user);
            }
            else
            {
                pass = validate_vs_blocked_ref<T, DataTypeConfig>(
                    q_host, k_host, v_host, output_host, block_mask,
                    batch, nhead, seqlen_q, hdim_v, block_size, scale,
                    causal_type, mask_decoded.left, mask_decoded.right,
                    i_perm, o_perm, "VSA vs CPU ref",
                    /*print_result=*/true, logits_soft_cap_user);
            }
        }
    }
    else if(api == "vsa" && mode == sparse_attn_mode::group)
    {
        // VSA group: same scaffolding as jenga group, then packed mask -> packed (LUT, VBN).
        const int32_t sq_min = std::max(int32_t{block_size}, seqlen_q / 2);
        const int32_t sk_min = std::max(int32_t{block_size}, seqlen_k / 2);
        const int32_t sq_avg = (sq_min + seqlen_q) / 2;  // see jenga branch for midpoint rationale
        const int32_t sk_avg = (sk_min + seqlen_k) / 2;
        auto seqlen_qs = generate_seqlens_group(batch, sq_avg, sq_min, seqlen_q, seed + 300);
        auto seqlen_ks = (seqlen_q == seqlen_k && sq_min == sk_min)
                             ? seqlen_qs
                             : generate_seqlens_group(batch, sk_avg, sk_min, seqlen_k, seed + 301);

        const auto seqstart_q_host = to_seqstarts(seqlen_qs);
        const auto seqstart_k_host = to_seqstarts(seqlen_ks);
        const int32_t total_q = seqstart_q_host.back();
        const int32_t total_k = seqstart_k_host.back();
        const auto q_blocks    = seqlens_to_blocks(seqlen_qs, block_size);
        const auto k_blocks    = seqlens_to_blocks(seqlen_ks, block_size);
        const auto seqstart_q_block_host = to_seqstarts(q_blocks);

        auto q_packed = make_qkv_tensor<T>(1, nhead,   total_q, hdim_q, i_perm);
        auto k_packed = make_qkv_tensor<T>(1, nhead_k, total_k, hdim_q, i_perm);
        auto v_packed = make_qkv_tensor<T>(1, nhead_k, total_k, hdim_v, i_perm);
        auto o_packed = o_perm ? ck_tile::HostTensor<T>({1, nhead, total_q, hdim_v})
                               : ck_tile::HostTensor<T>({1, total_q, nhead, hdim_v});
        ck_tile::FillUniformDistribution<T>{-0.5f, 0.5f, seed}(q_packed);
        ck_tile::FillUniformDistribution<T>{-0.5f, 0.5f, seed + 1}(k_packed);
        ck_tile::FillUniformDistribution<T>{-0.5f, 0.5f, seed + 2}(v_packed);

        std::vector<int32_t> mask_batch_offsets;
        auto mask_packed = generate_random_block_mask_group(
            q_blocks, k_blocks, nhead, sparsity, seed + 100, true, mask_batch_offsets);

        // LUT shares the mask layout; VBN is packed by seqstart_q_block.
        const int32_t lut_total = mask_batch_offsets.back();
        const int32_t vbn_total = seqstart_q_block_host.back();
        ck_tile::HostTensor<int32_t> lut_packed({nhead, lut_total});
        ck_tile::HostTensor<int32_t> vbn_packed({nhead, vbn_total});
        block_map_to_lut_group(mask_packed, q_blocks, k_blocks,
                               mask_batch_offsets, seqstart_q_block_host,
                               lut_packed, vbn_packed);

        const std::size_t flop_g     = compute_sparse_attn_flop_group(
            nhead, seqlen_qs, seqlen_ks, hdim_q, hdim_v, sparsity, mask_str);
        const std::size_t num_byte_g = compute_sparse_attn_num_byte_group<T>(
            nhead, nhead_k, seqlen_qs, seqlen_ks, hdim_q, hdim_v, sparsity);

        using BiasT = typename FmhaSparseFwdTypeConfig<DataTypeConfig>::BiasDataType;
        bias_info bi = bias_info::decode(bias_str);
        auto bs = setup_bias<BiasT>(bi, batch, nhead, seqlen_q, seqlen_k,
                                    causal_type, seed + 500, "vsa group");
        if(bs.args.type < 0)
            return sparse_attn_result::failure;

        try
        {
            ave_time = vsa_sparse_attention<T>(
                q_packed, k_packed, v_packed, lut_packed, vbn_packed, o_packed,
                batch, nhead, nhead_k,
                /*seqlen_q=*/0, /*seqlen_k=*/0,
                hdim_q, hdim_v,
                i_perm, o_perm, is_v_rowmajor,
                /*max_seqlen_q=*/seqlen_q, /*max_seqlen_k=*/0,
                stream_config, mask_str, bs.args, scale_s_user, logits_soft_cap_user,
                seqstart_q_host, seqstart_k_host,
                seqstart_q_block_host, mask_batch_offsets);
        }
        catch(const std::exception& e)
        {
            std::cerr << "\nError: " << e.what() << std::endl;
            return sparse_attn_result::failure;
        }
        if(ave_time < 0)
        {
            std::cout << ", not supported yet" << std::flush << std::endl;
            return sparse_attn_result::skipped;
        }
        std::cout << ", sq:" << format_int_vec(seqlen_qs)
                  << ", sk:" << format_int_vec(seqlen_ks) << std::flush;
        if(stream_config.time_kernel_)
            print_perf(static_cast<double>(ave_time), flop_g, num_byte_g);
        json_seqlen_qs = seqlen_qs;
        json_seqlen_ks = seqlen_ks;
        json_flop      = flop_g;
        json_num_byte  = num_byte_g;

        if(do_validation)
        {
            pass = true;
            for(int32_t b = 0; b < batch; ++b)
            {
                const int32_t sq = seqlen_qs[b];
                const int32_t sk = seqlen_ks[b];
                auto q_b = slice_packed_to_b1(q_packed, seqstart_q_host[b], sq, nhead,   hdim_q, i_perm);
                auto k_b = slice_packed_to_b1(k_packed, seqstart_k_host[b], sk, nhead_k, hdim_q, i_perm);
                auto v_b = slice_packed_to_b1(v_packed, seqstart_k_host[b], sk, nhead_k, hdim_v, i_perm);
                auto o_b = slice_packed_to_b1(o_packed, seqstart_q_host[b], sq, nhead,   hdim_v, o_perm);
                auto mask_b = slice_packed_mask_to_b1(
                    mask_packed, b, q_blocks, k_blocks, mask_batch_offsets, nhead);
                bool sub_pass;
                if(bi.type == bias_enum::elementwise_bias)
                {
                    auto bias_b_sub = slice_elementwise_bias_to_b1<BiasT>(
                        *bs.elementwise_host, bi.rank_info, b, nhead, sq, sk);
                    sub_pass = validate_vs_blocked_ref_with_bias<T, BiasT, DataTypeConfig>(
                        q_b, k_b, v_b, o_b, mask_b,
                        bias_b_sub, bi.rank_info,
                        /*batch=*/1, nhead, sq, hdim_v, block_size, scale,
                        causal_type, mask_decoded.left, mask_decoded.right,
                        i_perm, o_perm,
                        std::string("VSA group+bias sub-batch ") + std::to_string(b),
                        /*print_result=*/false, logits_soft_cap_user);
                }
                else if(bi.type == bias_enum::alibi)
                {
                    auto alibi_dense = alibi_to_dense<BiasT>(
                        bs.alibi_slopes_host, nhead, sq, sk,
                        mask_decoded.left, mask_decoded.right, causal_type);
                    sub_pass = validate_vs_blocked_ref_with_bias<T, BiasT, DataTypeConfig>(
                        q_b, k_b, v_b, o_b, mask_b,
                        alibi_dense, /*rank=*/1,
                        /*batch=*/1, nhead, sq, hdim_v, block_size, scale,
                        causal_type, mask_decoded.left, mask_decoded.right,
                        i_perm, o_perm,
                        std::string("VSA group+alibi sub-batch ") + std::to_string(b),
                        /*print_result=*/false, logits_soft_cap_user);
                }
                else
                {
                    sub_pass = validate_vs_blocked_ref<T, DataTypeConfig>(
                        q_b, k_b, v_b, o_b, mask_b,
                        /*batch=*/1, nhead, sq, hdim_v, block_size, scale,
                        causal_type, mask_decoded.left, mask_decoded.right,
                        i_perm, o_perm,
                        std::string("VSA group sub-batch ") + std::to_string(b),
                        /*print_result=*/false, logits_soft_cap_user);
                }
                pass = pass && sub_pass;
            }
            std::cout << ", valid:" << (pass ? "y" : "n") << std::flush << std::endl;
        }
    }
    else if(api == "sparge" && mode == sparse_attn_mode::batch)
    {
        // topk: deterministic 1-sparsity ratio of K-blocks/Q-block. cdf: greedy until cumulative
        // softmax prob reaches 1-sparsity. The unused field stays 0.
        if(sparge_mode != "topk" && sparge_mode != "cdf")
        {
            std::cerr << "error: -sparge_mode must be 'topk' or 'cdf', got '"
                      << sparge_mode << "'\n";
            return sparse_attn_result::failure;
        }
        const bool  mode_is_topk  = (sparge_mode == "topk");
        const float scalar_topk   = mode_is_topk ? (1.0f - sparsity) : 0.0f;
        const float scalar_cdf    = mode_is_topk ? 0.0f              : (1.0f - sparsity);
        const float scalar_sim    = simthreshold;
        const float scalar_pvthreshd = pvthreshd;

        using BiasT = typename FmhaSparseFwdTypeConfig<DataTypeConfig>::BiasDataType;
        bias_info bi = bias_info::decode(bias_str);
        auto bs = setup_bias<BiasT>(bi, batch, nhead, seqlen_q, seqlen_k,
                                    causal_type, seed + 500, "sparge");
        if(bs.args.type < 0)
            return sparse_attn_result::failure;
        sparse_attn_bias_args& bias_args = bs.args;

        sparge_hyperparam_args hp;
        hp.cdfthreshd        = scalar_cdf;
        hp.topk               = scalar_topk;
        hp.simthreshold      = scalar_sim;
        hp.pvthreshd = scalar_pvthreshd;
        hp.smooth_k           = smooth_k;

        SpargePerHeadSetup ph;
        if(!setup_sparge_per_head(hp, ph, perhead_test, mode_is_topk, nhead,
                                  scalar_cdf, scalar_topk, scalar_sim, scalar_pvthreshd,
                                  sparsity_per_head_csv, sim_per_head_csv, pvthreshd_per_head_csv))
            return sparse_attn_result::failure;

        // Opt-in actual sparsity (extra hipMemcpy + host sum); else TFlops uses the -sparsity input,
        // which diverges from the realised ratio when CDF / sim / sink kick in.
        float actual_sparsity = -1.0f;
        // Device-selected block LUT + valid-block counts (consumed by the next validation task).
        std::vector<int32_t> dev_lut, dev_vbn;
        try
        {
            ave_time = sparge_sparse_attention<T>(q_host, k_host, v_host, output_host,
                batch, nhead, nhead_k, seqlen_q, seqlen_k, hdim_q, hdim_v,
                i_perm, o_perm, is_v_rowmajor, hp, mask_str, attention_sink,
                block_size, seqlen_q, seqlen_k,
                print_sparsity ? &actual_sparsity : nullptr,
                stream_config, bias_args, scale_s_user, logits_soft_cap_user,
                /*seqstart_q_host=*/{}, /*seqstart_k_host=*/{},
                /*seqstart_q_block_host=*/{}, /*seqstart_k_block_host=*/{},
                /*mask_batch_offsets=*/{}, &dev_lut, &dev_vbn);
        }
        catch(const std::exception& e)
        {
            std::cerr << "\nError: " << e.what() << std::endl;
            return sparse_attn_result::failure;
        }

        if(ave_time < 0)
        {
            std::cout << ", not supported yet" << std::flush << std::endl;
            return sparse_attn_result::skipped;
        }
        {
            const bool have_actual = (actual_sparsity >= 0.0f);
            const std::size_t flop_used = have_actual
                ? compute_sparse_attn_flop(batch, nhead, seqlen_q, seqlen_k,
                                           hdim_q, hdim_v, actual_sparsity, mask_decoded)
                : flop;
            const std::size_t num_byte_used = have_actual
                ? compute_sparse_attn_num_byte<T>(batch, nhead, nhead_k,
                                                  seqlen_q, seqlen_k, hdim_q, hdim_v,
                                                  actual_sparsity)
                : num_byte;
            if(stream_config.time_kernel_)
                print_perf(static_cast<double>(ave_time), flop_used, num_byte_used);
            json_flop            = flop_used;
            json_num_byte        = num_byte_used;
            json_actual_sparsity = actual_sparsity;
        }

        if(print_sparsity && actual_sparsity >= 0.0f)
            std::cout << ", sparsity=" << actual_sparsity << std::flush;

        if(do_validation)
        {
            auto q_ref = to_bhsd(q_host, i_perm);
            auto k_ref = to_bhsd(k_host, i_perm);

            auto rp = make_sparge_predict_params(scalar_cdf, scalar_topk, scalar_sim, ph,
                                                 attention_sink, smooth_k, scale);

            ck_tile::sparge_causal_params cp;
            cp.causal_type            = causal_type;
            cp.window_left            = mask_decoded.left;
            cp.window_right           = mask_decoded.right;

            // Numeric reference = the device's actual selection (decoded LUT); host prediction is the
            // selection soft-check oracle. See check_selection_agreement.
            const ck_tile::index_t nqb = (seqlen_q + block_size - 1) / block_size;
            const ck_tile::index_t nkb = (seqlen_k + block_size - 1) / block_size;
            auto device_mask = device_lut_to_block_map(dev_lut, dev_vbn, batch, nhead, nqb, nkb);
            auto host_mask = ck_tile::reference_sparge_mask_prediction<T>(
                q_ref, k_ref, batch, nhead, nhead_k, seqlen_q, seqlen_k,
                hdim_q, block_size, block_size, rp, cp);
            const bool sel_ok = check_selection_agreement(device_mask, host_mask, "sparge");

            if(bi.type == bias_enum::elementwise_bias)
            {
                pass = validate_vs_blocked_ref_with_bias<T, BiasT, DataTypeConfig>(
                    q_host, k_host, v_host, output_host, device_mask,
                    *bs.elementwise_host, bi.rank_info,
                    batch, nhead, seqlen_q, hdim_v, block_size, scale,
                    causal_type, mask_decoded.left, mask_decoded.right,
                    i_perm, o_perm, "Sparge+bias vs CPU ref", /*print_result=*/false,
                    logits_soft_cap_user,
                    scalar_pvthreshd, ph.h_pvthreshd.empty() ? nullptr : &ph.h_pvthreshd);
            }
            else if(bi.type == bias_enum::alibi)
            {
                // float (not bf16) so slope*pos isn't rounded (device computes it in fp32).
                auto alibi_dense = alibi_to_dense<float>(bs.alibi_slopes_host, nhead, seqlen_q, seqlen_k,
                    mask_decoded.left, mask_decoded.right, causal_type);
                pass = validate_vs_blocked_ref_with_bias<T, float, DataTypeConfig>(
                    q_host, k_host, v_host, output_host, device_mask,
                    alibi_dense, /*rank=*/1,
                    batch, nhead, seqlen_q, hdim_v, block_size, scale,
                    causal_type, mask_decoded.left, mask_decoded.right,
                    i_perm, o_perm, "Sparge+alibi vs CPU ref", /*print_result=*/false,
                    logits_soft_cap_user,
                    scalar_pvthreshd, ph.h_pvthreshd.empty() ? nullptr : &ph.h_pvthreshd);
            }
            else
            {
                pass = validate_vs_blocked_ref<T, DataTypeConfig>(
                    q_host, k_host, v_host, output_host, device_mask,
                    batch, nhead, seqlen_q, hdim_v, block_size, scale,
                    causal_type, mask_decoded.left, mask_decoded.right,
                    i_perm, o_perm, "Sparge vs CPU ref",
                    /*print_result=*/false, logits_soft_cap_user,
                    scalar_pvthreshd, ph.h_pvthreshd.empty() ? nullptr : &ph.h_pvthreshd);
            }
            pass = pass && sel_ok;
            std::cout << ", valid:" << (pass ? "y" : "n") << std::flush << std::endl;
        }
    }
    else if(api == "sparge_sage" && mode == sparse_attn_mode::batch)
    {
        if constexpr(!std::is_same_v<T, ck_tile::bf16_t>)
        {
            std::cout << ", sparge_sage requires -prec=bf16" << std::flush << std::endl;
            return sparse_attn_result::skipped;
        }
        else
        {
            if(qscale != "perwarp" && qscale != "perblock" && qscale != "perthread" &&
               qscale != "pertensor")
            {
                std::cerr << "error: -qscale must be perwarp|perblock|perthread|pertensor for "
                             "sparge_sage, got '"
                          << qscale << "'\n";
                return sparse_attn_result::failure;
            }
            // Bias (float == SaccDataType) added to the descaled fp32 s_acc; elementwise tiles load
            // into the gemm0 C distribution so the int8-MFMA layout aligns by construction.
            bias_info bi_sage = bias_info::decode(bias_str);
            if(bi_sage.type == bias_enum::alibi && causal_type == 0)
            {
                std::cout << ", sparge_sage: alibi requires a causal mask (-mask=t/b)"
                          << std::flush << std::endl;
                return sparse_attn_result::skipped;
            }
            auto bs_sage = setup_bias<float>(bi_sage, batch, nhead, seqlen_q, seqlen_k,
                                             causal_type, seed + 500, "sparge_sage");

            const bool mode_is_topk = (sparge_mode == "topk");
            const float scalar_topk = mode_is_topk ? (1.0f - sparsity) : 0.0f;
            const float scalar_cdf  = mode_is_topk ? 0.0f : (1.0f - sparsity);
            const float scalar_sim  = simthreshold;
            const float scalar_pvthreshd = pvthreshd;

            sparge_hyperparam_args hp;
            hp.cdfthreshd   = scalar_cdf;
            hp.topk         = scalar_topk;
            hp.simthreshold = scalar_sim;
            hp.pvthreshd    = scalar_pvthreshd; // Stage-2 runtime pv-skip
            hp.smooth_k     = smooth_k; // sparge_sage: gates K-quant smooth_k (km centering)

            SpargePerHeadSetup ph;
            if(!setup_sparge_per_head(hp, ph, perhead_test, mode_is_topk, nhead,
                                      scalar_cdf, scalar_topk, scalar_sim, scalar_pvthreshd,
                                      sparsity_per_head_csv, sim_per_head_csv, pvthreshd_per_head_csv))
                return sparse_attn_result::failure;

            // Host V prequant: v_fp8 follows i_perm (device reads same layout as Q/K); v_dequant BHSD.
            ck_tile::HostTensor<ck_tile::fp8_t> v_fp8 =
                make_qkv_tensor<ck_tile::fp8_t>(batch, nhead_k, seqlen_k, hdim_v, i_perm);
            ck_tile::HostTensor<float> v_descale({batch, nhead_k, hdim_v});
            ck_tile::HostTensor<T> v_dequant({batch, nhead_k, seqlen_k, hdim_v});
            host_prequant_v_fp8<T>(v_host, v_fp8, v_dequant, v_descale, batch, nhead_k, hdim_v,
                                   i_perm, /*packed=*/false,
                                   std::vector<int32_t>(static_cast<size_t>(batch), 0),
                                   std::vector<int32_t>(static_cast<size_t>(batch),
                                                        static_cast<int32_t>(seqlen_k)));

            float ave = -1.0f;
            std::vector<int32_t> dev_lut, dev_vbn; // device's selected blocks (delta-encoded)
            try
            {
                ave = sparge_sage_sparse_attention<T>(
                    q_host, k_host, v_fp8, v_descale, output_host,
                    batch, nhead, nhead_k, seqlen_q, seqlen_k, hdim_q, hdim_v,
                    i_perm, o_perm, /*is_v_rowmajor=*/true, seqlen_q, seqlen_k,
                    stream_config, mask_str, bs_sage.args, scale_s_user,
                    logits_soft_cap_user, hp, block_size, attention_sink,
                    qscale, sage_data_type,
                    {}, {}, {}, {}, {},
                    &dev_lut, &dev_vbn);
            }
            catch(const std::exception& e)
            {
                std::cerr << "\nError: " << e.what() << std::endl;
                return sparse_attn_result::failure;
            }
            if(ave < 0)
            {
                std::cout << ", not supported yet" << std::flush << std::endl;
                return sparse_attn_result::skipped;
            }
            ave_time = ave;
            if(stream_config.time_kernel_)
                print_perf(static_cast<double>(ave_time), flop, num_byte);

            if(do_validation)
            {
                auto q_ref = to_bhsd(q_host, i_perm);
                auto k_ref = to_bhsd(k_host, i_perm);
                // Reference the device's decoded LUT, not a host re-prediction: top-k diverges at
                // near-tied block scores, which ALIBI amplifies.
                const ck_tile::index_t nqb = (seqlen_q + block_size - 1) / block_size;
                const ck_tile::index_t nkb = (seqlen_k + block_size - 1) / block_size;
                auto device_mask = device_lut_to_block_map(dev_lut, dev_vbn, batch, nhead, nqb, nkb);
                // Selection soft-check: host predicts on bf16 q/k the same way the sparge path does
                // (mask kernel runs on bf16 means), then compares to the device's actual selection.
                auto sage_rp = make_sparge_predict_params(scalar_cdf, scalar_topk, scalar_sim, ph,
                                                          attention_sink, smooth_k, scale);
                ck_tile::sparge_causal_params sage_cp;
                sage_cp.causal_type  = causal_type;
                sage_cp.window_left  = mask_decoded.left;
                sage_cp.window_right = mask_decoded.right;
                auto host_mask = ck_tile::reference_sparge_mask_prediction<T>(
                    q_ref, k_ref, batch, nhead, nhead_k, seqlen_q, seqlen_k,
                    hdim_q, block_size, block_size, sage_rp, sage_cp);
                const bool sel_ok =
                    check_selection_agreement(device_mask, host_mask, "sparge_sage");

                // Dequantize Q/K at this qscale's granularity (tokens/scale Q/K: perwarp 32/64,
                // perthread 4/16, perblock 128/128, pertensor = whole (b,h)).
                const bool qs_pertensor = (qscale == "pertensor");
                const ck_tile::index_t gran_q =
                    (qscale == "perthread") ? 4 : (qscale == "perblock") ? 128 : 32;
                const ck_tile::index_t gran_k =
                    (qscale == "perthread") ? 16 : (qscale == "perblock") ? 128 : 64;
                ck_tile::HostTensor<T> q_deq({batch, nhead, seqlen_q, hdim_q});
                ck_tile::HostTensor<T> k_deq({batch, nhead_k, seqlen_k, hdim_q});
                // smooth_k: subtract per-channel global K-mean before quant (K only; Q never).
                // km==0 when disabled, matching the device km_ptr==nullptr path.
                ck_tile::HostTensor<float> k_km({batch, nhead_k, hdim_q});
                if(smooth_k)
                    k_km = ck_tile::compute_global_k_mean(
                        k_ref, batch, nhead_k, seqlen_k, hdim_q);
                if(qs_pertensor)
                {
                    ck_tile::HostTensor<float> q_sc({batch, nhead});
                    ck_tile::HostTensor<float> k_sc({batch, nhead_k});
                    if(sage_qk_fp8)
                    {
                        ck_tile::HostTensor<float> q_qf({batch, nhead, seqlen_q, hdim_q});
                        ck_tile::HostTensor<float> k_qf({batch, nhead_k, seqlen_k, hdim_q});
                        ck_tile::reference_sparge_global_quant_fp8<T>(
                            q_ref, batch, nhead, seqlen_q, hdim_q, q_qf, q_sc);
                        ck_tile::reference_sparge_global_quant_fp8<T>(
                            k_ref, batch, nhead_k, seqlen_k, hdim_q, k_qf, k_sc, &k_km);
                        for(ck_tile::index_t b = 0; b < batch; ++b)
                            for(ck_tile::index_t h = 0; h < nhead; ++h)
                                for(ck_tile::index_t s = 0; s < seqlen_q; ++s)
                                    for(ck_tile::index_t d = 0; d < hdim_q; ++d)
                                        q_deq(b, h, s, d) = ck_tile::type_convert<T>(
                                            q_qf(b, h, s, d) * q_sc(b, h));
                        for(ck_tile::index_t b = 0; b < batch; ++b)
                            for(ck_tile::index_t h = 0; h < nhead_k; ++h)
                                for(ck_tile::index_t s = 0; s < seqlen_k; ++s)
                                    for(ck_tile::index_t d = 0; d < hdim_q; ++d)
                                        k_deq(b, h, s, d) = ck_tile::type_convert<T>(
                                            k_qf(b, h, s, d) * k_sc(b, h));
                    }
                    else
                    {
                        ck_tile::HostTensor<int8_t> q_i8({batch, nhead, seqlen_q, hdim_q});
                        ck_tile::HostTensor<int8_t> k_i8({batch, nhead_k, seqlen_k, hdim_q});
                        ck_tile::reference_sparge_global_quant<T>(
                            q_ref, batch, nhead, seqlen_q, hdim_q, q_i8, q_sc);
                        ck_tile::reference_sparge_global_quant<T>(
                            k_ref, batch, nhead_k, seqlen_k, hdim_q, k_i8, k_sc, &k_km);
                        for(ck_tile::index_t b = 0; b < batch; ++b)
                            for(ck_tile::index_t h = 0; h < nhead; ++h)
                                for(ck_tile::index_t s = 0; s < seqlen_q; ++s)
                                    for(ck_tile::index_t d = 0; d < hdim_q; ++d)
                                        q_deq(b, h, s, d) = ck_tile::type_convert<T>(
                                            static_cast<float>(q_i8(b, h, s, d)) * q_sc(b, h));
                        for(ck_tile::index_t b = 0; b < batch; ++b)
                            for(ck_tile::index_t h = 0; h < nhead_k; ++h)
                                for(ck_tile::index_t s = 0; s < seqlen_k; ++s)
                                    for(ck_tile::index_t d = 0; d < hdim_q; ++d)
                                        k_deq(b, h, s, d) = ck_tile::type_convert<T>(
                                            static_cast<float>(k_i8(b, h, s, d)) * k_sc(b, h));
                    }
                }
                else
                {
                    const ck_tile::index_t nbs_q =
                        ((seqlen_q + block_size - 1) / block_size) * (block_size / gran_q);
                    const ck_tile::index_t nbs_k =
                        ((seqlen_k + block_size - 1) / block_size) * (block_size / gran_k);
                    ck_tile::HostTensor<float> q_sc({batch, nhead, nbs_q});
                    ck_tile::HostTensor<float> k_sc({batch, nhead_k, nbs_k});
                    const ck_tile::index_t scales_per_blk_q = block_size / gran_q;
                    const ck_tile::index_t scales_per_blk_k = block_size / gran_k;
                    if(sage_qk_fp8)
                    {
                        ck_tile::HostTensor<float> q_qf({batch, nhead, seqlen_q, hdim_q});
                        ck_tile::HostTensor<float> k_qf({batch, nhead_k, seqlen_k, hdim_q});
                        ck_tile::reference_sparge_rowwise_quant_fp8<T>(
                            q_ref, batch, nhead, seqlen_q, hdim_q, block_size, gran_q, q_qf, q_sc);
                        ck_tile::reference_sparge_rowwise_quant_fp8<T>(
                            k_ref, batch, nhead_k, seqlen_k, hdim_q, block_size, gran_k, k_qf, k_sc,
                            &k_km);
                        for(ck_tile::index_t b = 0; b < batch; ++b)
                            for(ck_tile::index_t h = 0; h < nhead; ++h)
                                for(ck_tile::index_t s = 0; s < seqlen_q; ++s)
                                {
                                    const ck_tile::index_t blk = s / block_size;
                                    const ck_tile::index_t g   = (s % block_size) / gran_q;
                                    const float sc = q_sc(b, h, blk * scales_per_blk_q + g);
                                    for(ck_tile::index_t d = 0; d < hdim_q; ++d)
                                        q_deq(b, h, s, d) =
                                            ck_tile::type_convert<T>(q_qf(b, h, s, d) * sc);
                                }
                        for(ck_tile::index_t b = 0; b < batch; ++b)
                            for(ck_tile::index_t h = 0; h < nhead_k; ++h)
                                for(ck_tile::index_t s = 0; s < seqlen_k; ++s)
                                {
                                    const ck_tile::index_t blk = s / block_size;
                                    const ck_tile::index_t g   = (s % block_size) / gran_k;
                                    const float sc = k_sc(b, h, blk * scales_per_blk_k + g);
                                    for(ck_tile::index_t d = 0; d < hdim_q; ++d)
                                        k_deq(b, h, s, d) =
                                            ck_tile::type_convert<T>(k_qf(b, h, s, d) * sc);
                                }
                    }
                    else
                    {
                        ck_tile::HostTensor<int8_t> q_i8({batch, nhead, seqlen_q, hdim_q});
                        ck_tile::HostTensor<int8_t> k_i8({batch, nhead_k, seqlen_k, hdim_q});
                        ck_tile::reference_sparge_rowwise_quant<T>(
                            q_ref, batch, nhead, seqlen_q, hdim_q, block_size, gran_q, q_i8, q_sc);
                        ck_tile::reference_sparge_rowwise_quant<T>(
                            k_ref, batch, nhead_k, seqlen_k, hdim_q, block_size, gran_k, k_i8, k_sc,
                            &k_km);
                        for(ck_tile::index_t b = 0; b < batch; ++b)
                            for(ck_tile::index_t h = 0; h < nhead; ++h)
                                for(ck_tile::index_t s = 0; s < seqlen_q; ++s)
                                {
                                    const ck_tile::index_t blk = s / block_size;
                                    const ck_tile::index_t g   = (s % block_size) / gran_q;
                                    const float sc = q_sc(b, h, blk * scales_per_blk_q + g);
                                    for(ck_tile::index_t d = 0; d < hdim_q; ++d)
                                        q_deq(b, h, s, d) = ck_tile::type_convert<T>(
                                            static_cast<float>(q_i8(b, h, s, d)) * sc);
                                }
                        for(ck_tile::index_t b = 0; b < batch; ++b)
                            for(ck_tile::index_t h = 0; h < nhead_k; ++h)
                                for(ck_tile::index_t s = 0; s < seqlen_k; ++s)
                                {
                                    const ck_tile::index_t blk = s / block_size;
                                    const ck_tile::index_t g   = (s % block_size) / gran_k;
                                    const float sc = k_sc(b, h, blk * scales_per_blk_k + g);
                                    for(ck_tile::index_t d = 0; d < hdim_q; ++d)
                                        k_deq(b, h, s, d) = ck_tile::type_convert<T>(
                                            static_cast<float>(k_i8(b, h, s, d)) * sc);
                                }
                    }
                }

                // Blocked attention on dequantized Q/K/V (all BHSD). ALIBI: dense [1,h,sq,sk] bias
                // added in the scaled-QK domain, same as the kernel.
                auto gpu_out = to_bhsd(output_host, o_perm);
                ck_tile::HostTensor<T> ref_out({batch, nhead, seqlen_q, hdim_v});
                if(bi_sage.type == bias_enum::alibi)
                {
                    using BiasT = float;
                    auto alibi_dense = alibi_to_dense<BiasT>(
                        bs_sage.alibi_slopes_host, nhead, seqlen_q, seqlen_k,
                        mask_decoded.left, mask_decoded.right, causal_type);
                    ck_tile::reference_blocked_attention<T, uint8_t, BiasT>(
                        q_deq, k_deq, v_dequant, device_mask, ref_out, block_size, block_size, scale,
                        causal_type, mask_decoded.left, mask_decoded.right,
                        logits_soft_cap_user, &alibi_dense, /*bias_rank=*/1,
                        scalar_pvthreshd, ph.h_pvthreshd.empty() ? nullptr : &ph.h_pvthreshd,
                        /*quant_p_fp8=*/true);
                }
                else if(bi_sage.type == bias_enum::elementwise_bias)
                {
                    using BiasT = float;
                    ck_tile::reference_blocked_attention<T, uint8_t, BiasT>(
                        q_deq, k_deq, v_dequant, device_mask, ref_out, block_size, block_size, scale,
                        causal_type, mask_decoded.left, mask_decoded.right,
                        logits_soft_cap_user, &(*bs_sage.elementwise_host), bi_sage.rank_info,
                        scalar_pvthreshd, ph.h_pvthreshd.empty() ? nullptr : &ph.h_pvthreshd,
                        /*quant_p_fp8=*/true);
                }
                else
                {
                    // NO_BIAS: model the device Stage-2 pv-skip (only wired for NO_BIAS); host diff
                    // on dequantized QK matches the device m_local on the descaled QK.
                    ck_tile::reference_blocked_attention<T, uint8_t, T>(
                        q_deq, k_deq, v_dequant, device_mask, ref_out, block_size, block_size, scale,
                        causal_type, mask_decoded.left, mask_decoded.right,
                        /*logits_soft_cap=*/logits_soft_cap_user,
                        /*bias=*/static_cast<const ck_tile::HostTensor<T>*>(nullptr),
                        /*bias_rank=*/0, scalar_pvthreshd,
                        ph.h_pvthreshd.empty() ? nullptr : &ph.h_pvthreshd,
                        /*quant_p_fp8=*/true);
                }
                // int8 QK + fp8 V -> atol ~0.07; fp8 QK (3-bit mantissa) is coarser, relax to ~0.18.
                const double q_rtol = sage_qk_fp8 ? 0.15 : 0.1;
                const double q_atol = sage_qk_fp8 ? 0.18 : 0.07;
                pass = validate_tensors(gpu_out, ref_out, q_rtol, q_atol,
                                        "sparge_sage vs dequant ref", /*print_result=*/false);
                pass = pass && sel_ok;
                std::cout << ", valid:" << (pass ? "y" : "n") << std::flush << std::endl;
            }
        }
    }
    else if(api == "sparge" && mode == sparse_attn_mode::group)
    {
        // Sparge group: packed varlen launch; validation slices each sequence into a B=1 sub-batch.
        if(sparge_mode != "topk" && sparge_mode != "cdf")
        {
            std::cerr << "error: -sparge_mode must be 'topk' or 'cdf', got '"
                      << sparge_mode << "'\n";
            return sparse_attn_result::failure;
        }
        const bool  mode_is_topk  = (sparge_mode == "topk");
        const float scalar_topk   = mode_is_topk ? (1.0f - sparsity) : 0.0f;
        const float scalar_cdf    = mode_is_topk ? 0.0f              : (1.0f - sparsity);
        const float scalar_sim    = simthreshold;
        const float scalar_pvthreshd = pvthreshd;

        sparge_hyperparam_args hp;
        hp.cdfthreshd        = scalar_cdf;
        hp.topk               = scalar_topk;
        hp.simthreshold      = scalar_sim;
        hp.pvthreshd = scalar_pvthreshd;
        hp.smooth_k           = smooth_k;

        // Per-head is head-indexed (length nhead_q), so it works in group mode too.
        SpargePerHeadSetup ph;
        if(!setup_sparge_per_head(hp, ph, perhead_test, mode_is_topk, nhead,
                                  scalar_cdf, scalar_topk, scalar_sim, scalar_pvthreshd,
                                  sparsity_per_head_csv, sim_per_head_csv, pvthreshd_per_head_csv))
            return sparse_attn_result::failure;

        const int32_t sq_min = std::max(int32_t{block_size}, seqlen_q / 2);
        const int32_t sk_min = std::max(int32_t{block_size}, seqlen_k / 2);
        const int32_t sq_avg = (sq_min + seqlen_q) / 2;  // see jenga branch for midpoint rationale
        const int32_t sk_avg = (sk_min + seqlen_k) / 2;
        auto seqlen_qs = generate_seqlens_group(batch, sq_avg, sq_min, seqlen_q, seed + 300);
        auto seqlen_ks = (seqlen_q == seqlen_k && sq_min == sk_min)
                             ? seqlen_qs
                             : generate_seqlens_group(batch, sk_avg, sk_min, seqlen_k, seed + 301);

        const auto seqstart_q_host = to_seqstarts(seqlen_qs);
        const auto seqstart_k_host = to_seqstarts(seqlen_ks);
        const int32_t total_q = seqstart_q_host.back();
        const int32_t total_k = seqstart_k_host.back();

        const auto q_blocks_g = seqlens_to_blocks(seqlen_qs, block_size);
        const auto k_blocks_g = seqlens_to_blocks(seqlen_ks, block_size);
        const auto seqstart_q_block_host = to_seqstarts(q_blocks_g);
        const auto seqstart_k_block_host = to_seqstarts(k_blocks_g);
        std::vector<int32_t> mask_batch_offsets(batch + 1, 0);
        for(int b = 0; b < batch; ++b)
            mask_batch_offsets[b + 1] = mask_batch_offsets[b] + q_blocks_g[b] * k_blocks_g[b];

        auto q_packed = make_qkv_tensor<T>(1, nhead,   total_q, hdim_q, i_perm);
        auto k_packed = make_qkv_tensor<T>(1, nhead_k, total_k, hdim_q, i_perm);
        auto v_packed = make_qkv_tensor<T>(1, nhead_k, total_k, hdim_v, i_perm);
        auto o_packed = o_perm ? ck_tile::HostTensor<T>({1, nhead, total_q, hdim_v})
                               : ck_tile::HostTensor<T>({1, total_q, nhead, hdim_v});
        ck_tile::FillUniformDistribution<T>{-0.5f, 0.5f, seed}(q_packed);
        ck_tile::FillUniformDistribution<T>{-0.5f, 0.5f, seed + 1}(k_packed);
        ck_tile::FillUniformDistribution<T>{-0.5f, 0.5f, seed + 2}(v_packed);

        // ELEMENTWISE is sized [B,h_or_1,max_sq,max_sk]; per-batch slabs use the [0:sq_b,0:sk_b]
        // sub-region. ALIBI slopes are [1*nhead]/[b*nhead], independent of seqlens.
        using BiasT = typename FmhaSparseFwdTypeConfig<DataTypeConfig>::BiasDataType;
        bias_info bi = bias_info::decode(bias_str);
        auto bs = setup_bias<BiasT>(bi, batch, nhead, seqlen_q, seqlen_k,
                                    causal_type, seed + 500, "sparge group");
        if(bs.args.type < 0)
            return sparse_attn_result::failure;

        float actual_sparsity = -1.0f;
        // Device-selected block LUT + valid-block counts (consumed by the next validation task).
        std::vector<int32_t> dev_lut, dev_vbn;
        try
        {
            ave_time = sparge_sparse_attention<T>(
                q_packed, k_packed, v_packed, o_packed,
                batch, nhead, nhead_k,
                /*seqlen_q=*/0, /*seqlen_k=*/0,
                hdim_q, hdim_v,
                i_perm, o_perm, is_v_rowmajor,
                hp, mask_str, attention_sink,
                block_size, /*max_seqlen_q=*/seqlen_q, /*max_seqlen_k=*/0,
                print_sparsity ? &actual_sparsity : nullptr,
                stream_config, bs.args, scale_s_user, logits_soft_cap_user,
                seqstart_q_host, seqstart_k_host,
                seqstart_q_block_host, seqstart_k_block_host, mask_batch_offsets,
                &dev_lut, &dev_vbn);
        }
        catch(const std::exception& e)
        {
            std::cerr << "\nError: " << e.what() << std::endl;
            return sparse_attn_result::failure;
        }
        if(ave_time < 0)
        {
            std::cout << ", not supported yet" << std::flush << std::endl;
            return sparse_attn_result::skipped;
        }
        std::cout << ", sq:" << format_int_vec(seqlen_qs)
                  << ", sk:" << format_int_vec(seqlen_ks) << std::flush;
        const auto eff_sparsity_for_metrics =
            (actual_sparsity >= 0.0f) ? actual_sparsity : sparsity;
        const std::size_t sparge_flop_g = compute_sparse_attn_flop_group(
            nhead, seqlen_qs, seqlen_ks, hdim_q, hdim_v, eff_sparsity_for_metrics, mask_str);
        const std::size_t sparge_num_byte_g = compute_sparse_attn_num_byte_group<T>(
            nhead, nhead_k, seqlen_qs, seqlen_ks, hdim_q, hdim_v, eff_sparsity_for_metrics);
        if(stream_config.time_kernel_)
            print_perf(static_cast<double>(ave_time), sparge_flop_g, sparge_num_byte_g);
        json_seqlen_qs       = seqlen_qs;
        json_seqlen_ks       = seqlen_ks;
        json_flop            = sparge_flop_g;
        json_num_byte        = sparge_num_byte_g;
        json_actual_sparsity = actual_sparsity;
        if(print_sparsity && actual_sparsity >= 0.0f)
            std::cout << ", sparsity=" << actual_sparsity << std::flush;

        if(do_validation)
        {
            pass = true;
            auto rp = make_sparge_predict_params(scalar_cdf, scalar_topk, scalar_sim, ph,
                                                 attention_sink, smooth_k, scale);

            ck_tile::sparge_causal_params cp;
            cp.causal_type     = causal_type;
            cp.window_left     = mask_decoded.left;
            cp.window_right    = mask_decoded.right;

            for(int32_t b = 0; b < batch; ++b)
            {
                const int32_t sq = seqlen_qs[b];
                const int32_t sk = seqlen_ks[b];
                auto q_b = slice_packed_to_b1(q_packed, seqstart_q_host[b], sq, nhead,   hdim_q, i_perm);
                auto k_b = slice_packed_to_b1(k_packed, seqstart_k_host[b], sk, nhead_k, hdim_q, i_perm);
                auto v_b = slice_packed_to_b1(v_packed, seqstart_k_host[b], sk, nhead_k, hdim_v, i_perm);
                auto o_b = slice_packed_to_b1(o_packed, seqstart_q_host[b], sq, nhead,   hdim_v, o_perm);

                auto q_ref_b = to_bhsd(q_b, i_perm);
                auto k_ref_b = to_bhsd(k_b, i_perm);
                // Numeric reference = device's actual selection (decoded packed LUT for this sub-batch);
                // host prediction is the selection soft-check oracle.
                auto device_mask_b = device_lut_to_block_map_group(
                    dev_lut, dev_vbn, nhead, /*mask_base=*/mask_batch_offsets[b],
                    /*vbn_base=*/seqstart_q_block_host[b], q_blocks_g[b], k_blocks_g[b]);
                auto host_mask_b = ck_tile::reference_sparge_mask_prediction<T>(
                    q_ref_b, k_ref_b, /*batch=*/1, nhead, nhead_k, sq, sk,
                    hdim_q, block_size, block_size, rp, cp);
                const bool sel_ok_b = check_selection_agreement(
                    device_mask_b, host_mask_b,
                    std::string("sparge group sub-batch ") + std::to_string(b));

                bool sub_pass;
                if(bi.type == bias_enum::elementwise_bias)
                {
                    auto bias_b_sub = slice_elementwise_bias_to_b1<BiasT>(
                        *bs.elementwise_host, bi.rank_info, b, nhead, sq, sk);
                    sub_pass = validate_vs_blocked_ref_with_bias<T, BiasT, DataTypeConfig>(
                        q_b, k_b, v_b, o_b, device_mask_b,
                        bias_b_sub, bi.rank_info,
                        /*batch=*/1, nhead, sq, hdim_v, block_size, scale,
                        causal_type, mask_decoded.left, mask_decoded.right,
                        i_perm, o_perm,
                        std::string("Sparge group+bias sub-batch ") + std::to_string(b),
                        /*print_result=*/false, logits_soft_cap_user,
                        scalar_pvthreshd, ph.h_pvthreshd.empty() ? nullptr : &ph.h_pvthreshd);
                }
                else if(bi.type == bias_enum::alibi)
                {
                    // float reference bias so slope*k isn't rounded to bf16 (device computes fp32).
                    auto alibi_dense = alibi_to_dense<float>(
                        bs.alibi_slopes_host, nhead, sq, sk,
                        mask_decoded.left, mask_decoded.right, causal_type);
                    sub_pass = validate_vs_blocked_ref_with_bias<T, float, DataTypeConfig>(
                        q_b, k_b, v_b, o_b, device_mask_b,
                        alibi_dense, /*rank=*/1,
                        /*batch=*/1, nhead, sq, hdim_v, block_size, scale,
                        causal_type, mask_decoded.left, mask_decoded.right,
                        i_perm, o_perm,
                        std::string("Sparge group+alibi sub-batch ") + std::to_string(b),
                        /*print_result=*/false, logits_soft_cap_user,
                        scalar_pvthreshd, ph.h_pvthreshd.empty() ? nullptr : &ph.h_pvthreshd);
                }
                else
                {
                    sub_pass = validate_vs_blocked_ref<T, DataTypeConfig>(
                        q_b, k_b, v_b, o_b, device_mask_b,
                        /*batch=*/1, nhead, sq, hdim_v, block_size, scale,
                        causal_type, mask_decoded.left, mask_decoded.right,
                        i_perm, o_perm,
                        std::string("Sparge group sub-batch ") + std::to_string(b),
                        /*print_result=*/false, logits_soft_cap_user,
                        scalar_pvthreshd, ph.h_pvthreshd.empty() ? nullptr : &ph.h_pvthreshd);
                }
                pass = pass && sub_pass && sel_ok_b;
            }
            std::cout << ", valid:" << (pass ? "y" : "n") << std::flush << std::endl;
        }
    }
    else if(api == "sparge_sage" && mode == sparse_attn_mode::group)
    {
        if constexpr(!std::is_same_v<T, ck_tile::bf16_t>)
        {
            std::cout << ", sparge_sage requires -prec=bf16" << std::flush << std::endl;
            return sparse_attn_result::skipped;
        }
        else
        {
            if(qscale != "perwarp" && qscale != "perblock" && qscale != "perthread" &&
               qscale != "pertensor")
            {
                std::cerr << "error: -qscale must be perwarp|perblock|perthread|pertensor for "
                             "sparge_sage, got '" << qscale << "'\n";
                return sparse_attn_result::failure;
            }
            bias_info bi_sage = bias_info::decode(bias_str);
            if(bi_sage.type == bias_enum::alibi && causal_type == 0)
            {
                std::cout << ", sparge_sage: alibi requires a causal mask (-mask=t/b)"
                          << std::flush << std::endl;
                return sparse_attn_result::skipped;
            }

            const bool mode_is_topk = (sparge_mode == "topk");
            const float scalar_topk = mode_is_topk ? (1.0f - sparsity) : 0.0f;
            const float scalar_cdf  = mode_is_topk ? 0.0f : (1.0f - sparsity);
            const float scalar_sim  = simthreshold;
            const float scalar_pvthreshd = pvthreshd;

            sparge_hyperparam_args hp;
            hp.cdfthreshd   = scalar_cdf;
            hp.topk         = scalar_topk;
            hp.simthreshold = scalar_sim;
            hp.pvthreshd    = scalar_pvthreshd; // Stage-2 runtime pv-skip
            hp.smooth_k     = smooth_k; // sparge_sage: gates K-quant smooth_k (km centering)

            SpargePerHeadSetup ph;
            if(!setup_sparge_per_head(hp, ph, perhead_test, mode_is_topk, nhead,
                                      scalar_cdf, scalar_topk, scalar_sim, scalar_pvthreshd,
                                      sparsity_per_head_csv, sim_per_head_csv, pvthreshd_per_head_csv))
                return sparse_attn_result::failure;

            const int32_t sq_min = std::max(int32_t{block_size}, seqlen_q / 2);
            const int32_t sk_min = std::max(int32_t{block_size}, seqlen_k / 2);
            const int32_t sq_avg = (sq_min + seqlen_q) / 2;
            const int32_t sk_avg = (sk_min + seqlen_k) / 2;
            auto seqlen_qs = generate_seqlens_group(batch, sq_avg, sq_min, seqlen_q, seed + 300);
            auto seqlen_ks = (seqlen_q == seqlen_k && sq_min == sk_min)
                                 ? seqlen_qs
                                 : generate_seqlens_group(batch, sk_avg, sk_min, seqlen_k,
                                                          seed + 301);
            const auto seqstart_q_host = to_seqstarts(seqlen_qs);
            const auto seqstart_k_host = to_seqstarts(seqlen_ks);
            const int32_t total_q = seqstart_q_host.back();
            const int32_t total_k = seqstart_k_host.back();

            auto q_packed = make_qkv_tensor<T>(1, nhead,   total_q, hdim_q, i_perm);
            auto k_packed = make_qkv_tensor<T>(1, nhead_k, total_k, hdim_q, i_perm);
            auto v_packed = make_qkv_tensor<T>(1, nhead_k, total_k, hdim_v, i_perm);
            auto o_packed = make_qkv_tensor<T>(1, nhead, total_q, hdim_v, o_perm);
            ck_tile::FillUniformDistribution<T>{-0.5f, 0.5f, seed}(q_packed);
            ck_tile::FillUniformDistribution<T>{-0.5f, 0.5f, seed + 1}(k_packed);
            ck_tile::FillUniformDistribution<T>{-0.5f, 0.5f, seed + 2}(v_packed);

            // Host V prequant, packed varlen. v_fp8 follows i_perm; v_dequant BHSD (ref slices it).
            ck_tile::HostTensor<ck_tile::fp8_t> v_fp8 =
                make_qkv_tensor<ck_tile::fp8_t>(1, nhead_k, total_k, hdim_v, i_perm);
            ck_tile::HostTensor<float> v_descale({batch, nhead_k, hdim_v});
            ck_tile::HostTensor<T> v_dequant({1, nhead_k, total_k, hdim_v});
            host_prequant_v_fp8<T>(
                v_packed, v_fp8, v_dequant, v_descale, batch, nhead_k, hdim_v, i_perm,
                /*packed=*/true,
                std::vector<int32_t>(seqstart_k_host.begin(), seqstart_k_host.begin() + batch),
                seqlen_ks);

            // Bias as float. ALIBI: rank-1 [nhead] slopes shared across sub-batches. ELEMENTWISE:
            // dense [1,1,maxSq,maxSk] plane; each sub-batch validates its own [sq,sk] slice.
            auto bs_sage = setup_bias<float>(bi_sage, batch, nhead, seqlen_q, seqlen_k,
                                             causal_type, seed + 500, "sparge_sage");

            const auto q_blocks_call = seqlens_to_blocks(seqlen_qs, block_size);
            const auto k_blocks_call = seqlens_to_blocks(seqlen_ks, block_size);
            const auto seqstart_q_block_host = to_seqstarts(q_blocks_call);
            const auto seqstart_k_block_host = to_seqstarts(k_blocks_call);
            std::vector<int32_t> mask_batch_offsets(batch + 1, 0);
            for(int b = 0; b < batch; ++b)
                mask_batch_offsets[b + 1] =
                    mask_batch_offsets[b] + q_blocks_call[b] * k_blocks_call[b];

            float ave = -1.0f;
            std::vector<int32_t> dev_lut, dev_vbn; // device's selected blocks (delta-encoded)
            try
            {
                ave = sparge_sage_sparse_attention<T>(
                    q_packed, k_packed, v_fp8, v_descale, o_packed,
                    batch, nhead, nhead_k, seqlen_q, seqlen_k, hdim_q, hdim_v,
                    i_perm, o_perm, /*is_v_rowmajor=*/true, seqlen_q, seqlen_k,
                    stream_config, mask_str, bs_sage.args, scale_s_user,
                    logits_soft_cap_user, hp, block_size, attention_sink,
                    qscale, sage_data_type,
                    seqstart_q_host, seqstart_k_host,
                    seqstart_q_block_host, seqstart_k_block_host, mask_batch_offsets,
                    &dev_lut, &dev_vbn);
            }
            catch(const std::exception& e)
            {
                std::cerr << "\nError: " << e.what() << std::endl;
                return sparse_attn_result::failure;
            }
            if(ave < 0)
            {
                std::cout << ", not supported yet" << std::flush << std::endl;
                return sparse_attn_result::skipped;
            }
            ave_time = ave;
            std::cout << ", sq:" << format_int_vec(seqlen_qs)
                      << ", sk:" << format_int_vec(seqlen_ks) << std::flush;

            if(do_validation)
            {
                pass = true;
                // Per-qscale granularity (tokens/scale Q/K): perwarp 32/64, perthread 4/16,
                // perblock 128/128, pertensor = whole sequence.
                const bool qs_pertensor = (qscale == "pertensor");
                const ck_tile::index_t gran_q =
                    (qscale == "perthread") ? 4 : (qscale == "perblock") ? 128 : 32;
                const ck_tile::index_t gran_k =
                    (qscale == "perthread") ? 16 : (qscale == "perblock") ? 128 : 64;
                const ck_tile::index_t scales_per_blk_q = block_size / gran_q;
                const ck_tile::index_t scales_per_blk_k = block_size / gran_k;
                // Packed-LUT geometry for decoding the device selection per sub-batch: offsets are
                // prefix sums of per-batch q_blocks*k_blocks (lut) and q_blocks (vbn).
                std::vector<ck_tile::index_t> q_blocks_g(batch), k_blocks_g(batch),
                    q_block_off_g(batch), lut_off_g(batch);
                ck_tile::index_t total_q_blocks_g = 0, total_qk_blocks_g = 0;
                for(int32_t b = 0; b < batch; ++b)
                {
                    q_block_off_g[b] = total_q_blocks_g;
                    lut_off_g[b]     = total_qk_blocks_g;
                    q_blocks_g[b]    = (seqlen_qs[b] + block_size - 1) / block_size;
                    k_blocks_g[b]    = (seqlen_ks[b] + block_size - 1) / block_size;
                    total_q_blocks_g += q_blocks_g[b];
                    total_qk_blocks_g += q_blocks_g[b] * k_blocks_g[b];
                }
                for(int32_t b = 0; b < batch; ++b)
                {
                    const int32_t sq = seqlen_qs[b];
                    const int32_t sk = seqlen_ks[b];
                    auto q_b = slice_packed_to_b1(q_packed, seqstart_q_host[b], sq, nhead,
                                                  hdim_q, i_perm);
                    auto k_b = slice_packed_to_b1(k_packed, seqstart_k_host[b], sk, nhead_k,
                                                  hdim_q, i_perm);
                    auto o_b = slice_packed_to_b1(o_packed, seqstart_q_host[b], sq, nhead,
                                                  hdim_v, o_perm);

                    auto q_ref_b = to_bhsd(q_b, i_perm);
                    auto k_ref_b = to_bhsd(k_b, i_perm);
                    // Numeric reference = the kernel's decoded LUT (truth); top-k diverges at near-tied
                    // block scores, corrupting ALIBI rows (see batch path).
                    auto device_mask_b = device_lut_to_block_map_group(
                        dev_lut, dev_vbn, nhead, /*mask_base=*/lut_off_g[b],
                        /*vbn_base=*/q_block_off_g[b], q_blocks_g[b], k_blocks_g[b]);
                    // Selection soft-check: host predicts on bf16 q/k the same way the sparge path
                    // does, then compares to the device's actual selection.
                    auto sage_rp = make_sparge_predict_params(
                        scalar_cdf, scalar_topk, scalar_sim, ph, attention_sink, smooth_k, scale);
                    ck_tile::sparge_causal_params sage_cp;
                    sage_cp.causal_type  = causal_type;
                    sage_cp.window_left  = mask_decoded.left;
                    sage_cp.window_right = mask_decoded.right;
                    auto host_mask_b = ck_tile::reference_sparge_mask_prediction<T>(
                        q_ref_b, k_ref_b, /*batch=*/1, nhead, nhead_k, sq, sk,
                        hdim_q, block_size, block_size, sage_rp, sage_cp);
                    const bool sel_ok_b = check_selection_agreement(
                        device_mask_b, host_mask_b,
                        std::string("sparge_sage group sub-batch ") + std::to_string(b));

                    // Dequantize this sub-batch's Q/K at the qscale granularity. PERTENSOR uses one
                    // global scale per (seq,head).
                    ck_tile::HostTensor<T> q_deq({1, nhead, sq, hdim_q});
                    ck_tile::HostTensor<T> k_deq({1, nhead_k, sk, hdim_q});
                    // smooth_k: per-channel K-mean over this sub-batch's sk (K only). Zero km when
                    // disabled, bit-identical to the device km_ptr==nullptr path.
                    ck_tile::HostTensor<float> k_km_b({1, nhead_k, hdim_q});
                    if(smooth_k)
                        k_km_b = ck_tile::compute_global_k_mean(
                            k_ref_b, 1, nhead_k, sk, hdim_q);
                    if(qs_pertensor)
                    {
                        ck_tile::HostTensor<float> q_sc({1, nhead});
                        ck_tile::HostTensor<float> k_sc({1, nhead_k});
                        if(sage_qk_fp8)
                        {
                            ck_tile::HostTensor<float> q_qf({1, nhead, sq, hdim_q});
                            ck_tile::HostTensor<float> k_qf({1, nhead_k, sk, hdim_q});
                            ck_tile::reference_sparge_global_quant_fp8<T>(
                                q_ref_b, 1, nhead, sq, hdim_q, q_qf, q_sc);
                            ck_tile::reference_sparge_global_quant_fp8<T>(
                                k_ref_b, 1, nhead_k, sk, hdim_q, k_qf, k_sc, &k_km_b);
                            for(ck_tile::index_t h = 0; h < nhead; ++h)
                                for(int32_t s = 0; s < sq; ++s)
                                    for(ck_tile::index_t d = 0; d < hdim_q; ++d)
                                        q_deq(0, h, s, d) = ck_tile::type_convert<T>(
                                            q_qf(0, h, s, d) * q_sc(0, h));
                            for(ck_tile::index_t h = 0; h < nhead_k; ++h)
                                for(int32_t s = 0; s < sk; ++s)
                                    for(ck_tile::index_t d = 0; d < hdim_q; ++d)
                                        k_deq(0, h, s, d) = ck_tile::type_convert<T>(
                                            k_qf(0, h, s, d) * k_sc(0, h));
                        }
                        else
                        {
                            ck_tile::HostTensor<int8_t> q_i8({1, nhead, sq, hdim_q});
                            ck_tile::HostTensor<int8_t> k_i8({1, nhead_k, sk, hdim_q});
                            ck_tile::reference_sparge_global_quant<T>(
                                q_ref_b, 1, nhead, sq, hdim_q, q_i8, q_sc);
                            ck_tile::reference_sparge_global_quant<T>(
                                k_ref_b, 1, nhead_k, sk, hdim_q, k_i8, k_sc, &k_km_b);
                            for(ck_tile::index_t h = 0; h < nhead; ++h)
                                for(int32_t s = 0; s < sq; ++s)
                                    for(ck_tile::index_t d = 0; d < hdim_q; ++d)
                                        q_deq(0, h, s, d) = ck_tile::type_convert<T>(
                                            static_cast<float>(q_i8(0, h, s, d)) * q_sc(0, h));
                            for(ck_tile::index_t h = 0; h < nhead_k; ++h)
                                for(int32_t s = 0; s < sk; ++s)
                                    for(ck_tile::index_t d = 0; d < hdim_q; ++d)
                                        k_deq(0, h, s, d) = ck_tile::type_convert<T>(
                                            static_cast<float>(k_i8(0, h, s, d)) * k_sc(0, h));
                        }
                    }
                    else
                    {
                        const ck_tile::index_t nbs_q =
                            ((sq + block_size - 1) / block_size) * scales_per_blk_q;
                        const ck_tile::index_t nbs_k =
                            ((sk + block_size - 1) / block_size) * scales_per_blk_k;
                        ck_tile::HostTensor<float> q_sc({1, nhead, nbs_q});
                        ck_tile::HostTensor<float> k_sc({1, nhead_k, nbs_k});
                        if(sage_qk_fp8)
                        {
                            ck_tile::HostTensor<float> q_qf({1, nhead, sq, hdim_q});
                            ck_tile::HostTensor<float> k_qf({1, nhead_k, sk, hdim_q});
                            ck_tile::reference_sparge_rowwise_quant_fp8<T>(
                                q_ref_b, 1, nhead, sq, hdim_q, block_size, gran_q, q_qf, q_sc);
                            ck_tile::reference_sparge_rowwise_quant_fp8<T>(
                                k_ref_b, 1, nhead_k, sk, hdim_q, block_size, gran_k, k_qf, k_sc,
                                &k_km_b);
                            for(ck_tile::index_t h = 0; h < nhead; ++h)
                                for(int32_t s = 0; s < sq; ++s)
                                {
                                    const ck_tile::index_t blk = s / block_size;
                                    const ck_tile::index_t g   = (s % block_size) / gran_q;
                                    const float sc = q_sc(0, h, blk * scales_per_blk_q + g);
                                    for(ck_tile::index_t d = 0; d < hdim_q; ++d)
                                        q_deq(0, h, s, d) =
                                            ck_tile::type_convert<T>(q_qf(0, h, s, d) * sc);
                                }
                            for(ck_tile::index_t h = 0; h < nhead_k; ++h)
                                for(int32_t s = 0; s < sk; ++s)
                                {
                                    const ck_tile::index_t blk = s / block_size;
                                    const ck_tile::index_t g   = (s % block_size) / gran_k;
                                    const float sc = k_sc(0, h, blk * scales_per_blk_k + g);
                                    for(ck_tile::index_t d = 0; d < hdim_q; ++d)
                                        k_deq(0, h, s, d) =
                                            ck_tile::type_convert<T>(k_qf(0, h, s, d) * sc);
                                }
                        }
                        else
                        {
                            ck_tile::HostTensor<int8_t> q_i8({1, nhead, sq, hdim_q});
                            ck_tile::HostTensor<int8_t> k_i8({1, nhead_k, sk, hdim_q});
                            ck_tile::reference_sparge_rowwise_quant<T>(
                                q_ref_b, 1, nhead, sq, hdim_q, block_size, gran_q, q_i8, q_sc);
                            ck_tile::reference_sparge_rowwise_quant<T>(
                                k_ref_b, 1, nhead_k, sk, hdim_q, block_size, gran_k, k_i8, k_sc,
                                &k_km_b);
                            for(ck_tile::index_t h = 0; h < nhead; ++h)
                                for(int32_t s = 0; s < sq; ++s)
                                {
                                    const ck_tile::index_t blk = s / block_size;
                                    const ck_tile::index_t g   = (s % block_size) / gran_q;
                                    const float sc = q_sc(0, h, blk * scales_per_blk_q + g);
                                    for(ck_tile::index_t d = 0; d < hdim_q; ++d)
                                        q_deq(0, h, s, d) = ck_tile::type_convert<T>(
                                            static_cast<float>(q_i8(0, h, s, d)) * sc);
                                }
                            for(ck_tile::index_t h = 0; h < nhead_k; ++h)
                                for(int32_t s = 0; s < sk; ++s)
                                {
                                    const ck_tile::index_t blk = s / block_size;
                                    const ck_tile::index_t g   = (s % block_size) / gran_k;
                                    const float sc = k_sc(0, h, blk * scales_per_blk_k + g);
                                    for(ck_tile::index_t d = 0; d < hdim_q; ++d)
                                        k_deq(0, h, s, d) = ck_tile::type_convert<T>(
                                            static_cast<float>(k_i8(0, h, s, d)) * sc);
                                }
                        }
                    }

                    ck_tile::HostTensor<T> v_deq_b({1, nhead_k, sk, hdim_v});
                    for(ck_tile::index_t h = 0; h < nhead_k; ++h)
                        for(int32_t s = 0; s < sk; ++s)
                            for(ck_tile::index_t dv = 0; dv < hdim_v; ++dv)
                                v_deq_b(0, h, s, dv) =
                                    v_dequant(0, h, seqstart_k_host[b] + s, dv);

                    auto gpu_out_b = to_bhsd(o_b, o_perm);
                    ck_tile::HostTensor<T> ref_out({1, nhead, sq, hdim_v});
                    if(bi_sage.type == bias_enum::alibi)
                    {
                        using BiasT = float;
                        auto alibi_dense = alibi_to_dense<BiasT>(
                            bs_sage.alibi_slopes_host, nhead, sq, sk,
                            mask_decoded.left, mask_decoded.right, causal_type);
                        ck_tile::reference_blocked_attention<T, uint8_t, BiasT>(
                            q_deq, k_deq, v_deq_b, device_mask_b, ref_out, block_size, block_size,
                            scale, causal_type, mask_decoded.left, mask_decoded.right,
                            logits_soft_cap_user, &alibi_dense, /*bias_rank=*/1,
                            scalar_pvthreshd, ph.h_pvthreshd.empty() ? nullptr : &ph.h_pvthreshd,
                            /*quant_p_fp8=*/true);
                    }
                    else if(bi_sage.type == bias_enum::elementwise_bias)
                    {
                        using BiasT = float;
                        auto bias_b = slice_elementwise_bias_to_b1<BiasT>(
                            *bs_sage.elementwise_host, bi_sage.rank_info, b, nhead, sq, sk);
                        ck_tile::reference_blocked_attention<T, uint8_t, BiasT>(
                            q_deq, k_deq, v_deq_b, device_mask_b, ref_out, block_size, block_size,
                            scale, causal_type, mask_decoded.left, mask_decoded.right,
                            logits_soft_cap_user, &bias_b, bi_sage.rank_info,
                            scalar_pvthreshd, ph.h_pvthreshd.empty() ? nullptr : &ph.h_pvthreshd,
                            /*quant_p_fp8=*/true);
                    }
                    else
                    {
                        // NO_BIAS: model the device Stage-2 pv-skip (scalar + per-head pvthreshd).
                        ck_tile::reference_blocked_attention<T, uint8_t, T>(
                            q_deq, k_deq, v_deq_b, device_mask_b, ref_out, block_size, block_size,
                            scale, causal_type, mask_decoded.left, mask_decoded.right,
                            /*logits_soft_cap=*/logits_soft_cap_user,
                            /*bias=*/static_cast<const ck_tile::HostTensor<T>*>(nullptr),
                            /*bias_rank=*/0, scalar_pvthreshd,
                            ph.h_pvthreshd.empty() ? nullptr : &ph.h_pvthreshd,
                            /*quant_p_fp8=*/true);
                    }
                    const double g_rtol = sage_qk_fp8 ? 0.15 : 0.1;
                    const double g_atol = sage_qk_fp8 ? 0.18 : 0.07;
                    bool sub_pass = validate_tensors(
                        gpu_out_b, ref_out, g_rtol, g_atol,
                        std::string("sparge_sage group sub-batch ") + std::to_string(b));
                    pass = pass && sub_pass && sel_ok_b;
                }
                std::cout << ", valid:" << (pass ? "y" : "n") << std::flush << std::endl;
            }
        }
    }
    else
    {
        std::cerr << "Unknown API: " << api << std::endl;
        return sparse_attn_result::failure;
    }

    if(!do_validation)
        std::cout << std::flush << std::endl;

    // Optional JSON-Lines summary, emitted regardless of validation outcome.
    if(json_out || !json_file.empty())
    {
        const std::string prec = std::is_same_v<T, ck_tile::bf16_t> ? "bf16" : "fp16";
        emit_json_summary(json_file,
                          api,
                          mode == sparse_attn_mode::group ? "group" : "batch",
                          prec,
                          batch, nhead, nhead_k,
                          seqlen_q, seqlen_k, hdim_q, hdim_v,
                          sparsity, mask_str, i_perm, o_perm,
                          static_cast<double>(ave_time),
                          json_flop, json_num_byte,
                          /*has_validation=*/static_cast<bool>(do_validation),
                          pass,
                          json_actual_sparsity,
                          json_seqlen_qs, json_seqlen_ks);
    }

    return pass ? sparse_attn_result::success : sparse_attn_result::failure;
}
