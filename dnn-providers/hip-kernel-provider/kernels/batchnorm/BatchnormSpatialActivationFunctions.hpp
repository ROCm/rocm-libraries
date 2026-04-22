// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// TODO: I don't think we need a separate file for this, we should move this into
// batchnorm_functions.hpp or activation functions.hpp
#pragma once

#include "Configuration.hpp"
#include "HipKernelMath.hpp"

namespace hip_kernel_provider
{
namespace batchnorm
{

template <typename FpPrecType,
          hip_kernel_provider::neuron_op_type NrnOpType = hip_kernel_provider::config::neuron_op,
          typename std::enable_if<NrnOpType == neuron_op_type::pasthru>::type* = nullptr>
__forceinline__ __host__ __device__ FpPrecType activation_op(FpPrecType const& tmp,
                                                             FpPrecType const&,
                                                             FpPrecType const&)
{
    return tmp;
}

template <typename FpPrecType,
          hip_kernel_provider::neuron_op_type NrnOpType = hip_kernel_provider::config::neuron_op,
          typename std::enable_if<NrnOpType == neuron_op_type::pasthru>::type* = nullptr>
__forceinline__ __host__ __device__ FpPrecType bwd_activation_op(FpPrecType const& dy,
                                                                 FpPrecType const&,
                                                                 FpPrecType const&,
                                                                 FpPrecType const&,
                                                                 FpPrecType const&,
                                                                 FpPrecType const&)
{
    return dy;
}

template <typename FpPrecType,
          hip_kernel_provider::neuron_op_type NrnOpType = hip_kernel_provider::config::neuron_op,
          typename std::enable_if<NrnOpType == neuron_op_type::relu>::type* = nullptr>
__forceinline__ __host__ __device__ FpPrecType activation_op(FpPrecType const& tmp,
                                                             FpPrecType const&,
                                                             FpPrecType const&)
{
    return max(tmp, hip_kernel_provider::cast<FpPrecType>(0.));
}

template <typename FpPrecType,
          hip_kernel_provider::neuron_op_type NrnOpType = hip_kernel_provider::config::neuron_op,
          typename std::enable_if<NrnOpType == neuron_op_type::relu>::type* = nullptr>
__forceinline__ __host__ __device__ FpPrecType bwd_activation_op(FpPrecType const& dy,
                                                                 FpPrecType const& xnorm,
                                                                 FpPrecType const& scale,
                                                                 FpPrecType const& bias,
                                                                 FpPrecType const&,
                                                                 FpPrecType const&)
{
    FpPrecType macro_tmp = scale * xnorm + bias;
    return (macro_tmp > 0) ? dy : 0;
}

template <typename FpPrecType,
          hip_kernel_provider::neuron_op_type NrnOpType = hip_kernel_provider::config::neuron_op,
          typename std::enable_if<NrnOpType == neuron_op_type::clipped_relu>::type* = nullptr>
__forceinline__ __host__ __device__ FpPrecType activation_op(FpPrecType const& tmp,
                                                             FpPrecType const& _alpha,
                                                             FpPrecType const&)
{
    return min(_alpha, max(tmp, hip_kernel_provider::cast<FpPrecType>(0.)));
}

template <typename FpPrecType,
          hip_kernel_provider::neuron_op_type NrnOpType = hip_kernel_provider::config::neuron_op,
          typename std::enable_if<NrnOpType == neuron_op_type::clipped_relu>::type* = nullptr>
__forceinline__ __host__ __device__ FpPrecType bwd_activation_op(FpPrecType const& dy,
                                                                 FpPrecType const& xnorm,
                                                                 FpPrecType const& scale,
                                                                 FpPrecType const& bias,
                                                                 FpPrecType const& alpha,
                                                                 FpPrecType const&)
{
    FpPrecType macro_tmp = scale * xnorm + bias;
    return (macro_tmp > 0 && macro_tmp <= alpha) ? dy : 0;
}

template <typename FpPrecType,
          hip_kernel_provider::neuron_op_type NrnOpType = hip_kernel_provider::config::neuron_op,
          typename std::enable_if<NrnOpType == neuron_op_type::clamp>::type* = nullptr>
__forceinline__ __host__ __device__ FpPrecType activation_op(FpPrecType const& tmp,
                                                             FpPrecType const& _alpha,
                                                             FpPrecType const& _beta)
{
    return hip_kernel_provider::max(_alpha, hip_kernel_provider::min(_beta, tmp));
}

template <typename FpPrecType,
          hip_kernel_provider::neuron_op_type NrnOpType = hip_kernel_provider::config::neuron_op,
          typename std::enable_if<NrnOpType == neuron_op_type::clamp>::type* = nullptr>
__forceinline__ __host__ __device__ FpPrecType bwd_activation_op(FpPrecType const& dy,
                                                                 FpPrecType const& xnorm,
                                                                 FpPrecType const& scale,
                                                                 FpPrecType const& bias,
                                                                 FpPrecType const& alpha,
                                                                 FpPrecType const& beta)
{
    FpPrecType macro_tmp = scale * xnorm + bias;
    return (macro_tmp > alpha && macro_tmp <= beta) ? dy : 0;
}

} // namespace batchnorm
} // namespace hip_kernel_provider
