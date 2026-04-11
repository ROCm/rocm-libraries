// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"

namespace ck_tile {
namespace reference {

template <typename XDataType,
          typename WDataType,
          typename ComputeDataType,
          typename IntermediateDataType>
void reference_warp_decode_gate_up(
    const HostTensor<XDataType>& x,                    // [B, HIDDEN]
    const HostTensor<WDataType>& w_gate,               // [E, INTER, HIDDEN]
    const HostTensor<WDataType>& w_up,                 // [E, INTER, HIDDEN]
    const HostTensor<int32_t>& router_ids,             // [B, TOP_K]
    HostTensor<IntermediateDataType>& intermediate)    // [B, TOP_K, INTER]
{
    const index_t B = x.get_lengths()[0];
    const index_t HIDDEN = x.get_lengths()[1];
    const index_t E = w_gate.get_lengths()[0];
    const index_t INTER = w_gate.get_lengths()[1];
    const index_t TOP_K = router_ids.get_lengths()[1];

    for(index_t b = 0; b < B; ++b) {
        for(index_t k = 0; k < TOP_K; ++k) {
            index_t e = router_ids(b, k);
            for(index_t j = 0; j < INTER; ++j) {
                ComputeDataType gate_acc = 0;
                ComputeDataType up_acc = 0;

                for(index_t i = 0; i < HIDDEN; ++i) {
                    ComputeDataType x_val = type_convert<ComputeDataType>(x(b, i));
                    ComputeDataType g_val = type_convert<ComputeDataType>(w_gate(e, j, i));
                    ComputeDataType u_val = type_convert<ComputeDataType>(w_up(e, j, i));

                    gate_acc += x_val * g_val;
                    up_acc += x_val * u_val;
                }

                // SwiGLU
                ComputeDataType silu_gate = gate_acc / (type_convert<ComputeDataType>(1.0f) + std::exp(-gate_acc));
                ComputeDataType result = silu_gate * up_acc;

                intermediate(b, k, j) = type_convert<IntermediateDataType>(result);
            }
        }
    }
}

template <typename IntermediateDataType,
          typename WDataType,
          typename ComputeDataType,
          typename YDataType>
void reference_warp_decode_down_reduce(
    const HostTensor<IntermediateDataType>& intermediate, // [B, TOP_K, INTER]
    const HostTensor<WDataType>& w_down,                  // [E, HIDDEN, INTER]
    const HostTensor<int32_t>& router_ids,                // [B, TOP_K]
    const HostTensor<float>& router_wts,                  // [B, TOP_K]
    HostTensor<YDataType>& y)                             // [B, HIDDEN]
{
    const index_t B = intermediate.get_lengths()[0];
    const index_t TOP_K = intermediate.get_lengths()[1];
    const index_t INTER = intermediate.get_lengths()[2];
    const index_t E = w_down.get_lengths()[0];
    const index_t HIDDEN = w_down.get_lengths()[1];

    for(index_t b = 0; b < B; ++b) {
        for(index_t out_j = 0; out_j < HIDDEN; ++out_j) {
            ComputeDataType acc = 0;

            for(index_t k = 0; k < TOP_K; ++k) {
                index_t e = router_ids(b, k);
                float w = router_wts(b, k);

                for(index_t i = 0; i < INTER; ++i) {
                    ComputeDataType act_val = type_convert<ComputeDataType>(intermediate(b, k, i));
                    ComputeDataType d_val = type_convert<ComputeDataType>(w_down(e, out_j, i));
                    
                    acc += type_convert<ComputeDataType>(w) * act_val * d_val;
                }
            }

            y(b, out_j) = type_convert<YDataType>(acc);
        }
    }
}

} // namespace reference
} // namespace ck_tile
