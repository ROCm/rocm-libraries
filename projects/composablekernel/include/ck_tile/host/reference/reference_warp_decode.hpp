// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"

namespace ck_tile {
namespace reference {

enum class RefScaleMode { None, PerTensor, PerToken, Block2D };

template <typename ScaleDataType>
float lookup_scale(const ScaleDataType* p_scale,
                   RefScaleMode mode,
                   index_t row_idx,
                   index_t col_idx,
                   index_t max_cols,
                   index_t block_n,
                   index_t block_k)
{
    if(!p_scale || mode == RefScaleMode::None)
        return 1.0f;

    if(mode == RefScaleMode::PerTensor)
        return static_cast<float>(p_scale[0]);
    else if(mode == RefScaleMode::PerToken)
        return static_cast<float>(p_scale[row_idx]);
    else if(mode == RefScaleMode::Block2D)
    {
        index_t r = row_idx / block_n;
        index_t c = col_idx / block_k;
        return static_cast<float>(p_scale[r * (max_cols / block_k) + c]);
    }

    return 1.0f;
}

template <typename WDataType>
float unpack_weight(const WDataType& val, [[maybe_unused]] index_t sub_idx)
{
    return type_convert<float>(val);
}

template <>
inline float unpack_weight<pk_fp4_t>(const pk_fp4_t& val, index_t sub_idx)
{
    static constexpr float lut[16] = {
        0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
        -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
    uint8_t raw = static_cast<uint8_t>(val);
    uint8_t nib = sub_idx ? (raw >> 4) : (raw & 0x0F);
    return lut[nib];
}

template <typename WDataType>
constexpr index_t weight_pack_factor()
{
    if constexpr(std::is_same_v<WDataType, pk_fp4_t>)
        return 2;
    else
        return 1;
}

template <typename XDataType,
          typename WDataType,
          typename ComputeDataType,
          typename IntermediateDataType,
          typename XScaleDataType = float,
          typename WScaleDataType = float>
void reference_warp_decode_gate_up(
    const HostTensor<XDataType>& x,
    const HostTensor<WDataType>& w_gate,
    const HostTensor<WDataType>& w_up,
    const HostTensor<int32_t>& router_ids,
    HostTensor<IntermediateDataType>& intermediate,
    const XScaleDataType* p_x_scale = nullptr,
    const WScaleDataType* p_w_gate_scale = nullptr,
    const WScaleDataType* p_w_up_scale = nullptr,
    RefScaleMode x_scale_mode = RefScaleMode::None,
    RefScaleMode w_scale_mode = RefScaleMode::None,
    index_t w_scale_block_n = 1,
    index_t w_scale_block_k = 1)
{
    const index_t B = x.get_lengths()[0];
    const index_t HIDDEN = x.get_lengths()[1];
    const index_t INTER = w_gate.get_lengths()[1];
    const index_t TOP_K = router_ids.get_lengths()[1];

    constexpr index_t PACK = weight_pack_factor<WDataType>();

    for(index_t b = 0; b < B; ++b)
    {
        for(index_t k = 0; k < TOP_K; ++k)
        {
            index_t e = router_ids(b, k);
            for(index_t j = 0; j < INTER; ++j)
            {
                ComputeDataType gate_acc = 0;
                ComputeDataType up_acc = 0;

                for(index_t i = 0; i < HIDDEN; ++i)
                {
                    ComputeDataType x_val = type_convert<ComputeDataType>(x(b, i));

                    ComputeDataType g_val = type_convert<ComputeDataType>(
                        unpack_weight(w_gate(e, j, i), i % PACK));
                    ComputeDataType u_val = type_convert<ComputeDataType>(
                        unpack_weight(w_up(e, j, i), i % PACK));

                    float xs = lookup_scale(p_x_scale, x_scale_mode,
                                            b, i, HIDDEN, 1, 1);
                    float gs = lookup_scale(p_w_gate_scale, w_scale_mode,
                                            e * INTER + j, i, HIDDEN,
                                            w_scale_block_n, w_scale_block_k);
                    float us = lookup_scale(p_w_up_scale, w_scale_mode,
                                            e * INTER + j, i, HIDDEN,
                                            w_scale_block_n, w_scale_block_k);

                    gate_acc += (x_val * type_convert<ComputeDataType>(xs)) *
                                (g_val * type_convert<ComputeDataType>(gs));
                    up_acc   += (x_val * type_convert<ComputeDataType>(xs)) *
                                (u_val * type_convert<ComputeDataType>(us));
                }

                ComputeDataType silu_gate = gate_acc /
                    (type_convert<ComputeDataType>(1.0f) + std::exp(-gate_acc));
                ComputeDataType result = silu_gate * up_acc;

                intermediate(b, k, j) = type_convert<IntermediateDataType>(result);
            }
        }
    }
}

template <typename IntermediateDataType,
          typename WDataType,
          typename ComputeDataType,
          typename YDataType,
          typename WScaleDataType = float>
void reference_warp_decode_down_reduce(
    const HostTensor<IntermediateDataType>& intermediate,
    const HostTensor<WDataType>& w_down,
    const HostTensor<int32_t>& router_ids,
    const HostTensor<float>& router_wts,
    HostTensor<YDataType>& y,
    const WScaleDataType* p_w_down_scale = nullptr,
    RefScaleMode w_scale_mode = RefScaleMode::None,
    index_t w_scale_block_n = 1,
    index_t w_scale_block_k = 1)
{
    const index_t B = intermediate.get_lengths()[0];
    const index_t TOP_K = intermediate.get_lengths()[1];
    const index_t INTER = intermediate.get_lengths()[2];
    const index_t HIDDEN = w_down.get_lengths()[1];

    constexpr index_t PACK = weight_pack_factor<WDataType>();

    for(index_t b = 0; b < B; ++b)
    {
        for(index_t out_j = 0; out_j < HIDDEN; ++out_j)
        {
            ComputeDataType acc = 0;

            for(index_t k = 0; k < TOP_K; ++k)
            {
                index_t e = router_ids(b, k);
                float w = router_wts(b, k);

                for(index_t i = 0; i < INTER; ++i)
                {
                    ComputeDataType act_val = type_convert<ComputeDataType>(intermediate(b, k, i));

                    ComputeDataType d_val = type_convert<ComputeDataType>(
                        unpack_weight(w_down(e, out_j, i), i % PACK));

                    float ds = lookup_scale(p_w_down_scale, w_scale_mode,
                                            e * HIDDEN + out_j, i, INTER,
                                            w_scale_block_n, w_scale_block_k);

                    acc += type_convert<ComputeDataType>(w) * act_val *
                           (d_val * type_convert<ComputeDataType>(ds));
                }
            }

            y(b, out_j) = type_convert<YDataType>(acc);
        }
    }
}

} // namespace reference
} // namespace ck_tile
