// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/testing/testing.hpp"
#include "ck_tile/builder/testing/conv/bwd_data.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_elementwise_op.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_tensor_type.hpp"
#include <type_traits>
#include <array>

/// This file contains the implementation details for invoking/testing
/// bwd data grouped convolution operations in old CK. The main item is
/// the `run()` function, which is the main implementation used to invoke
/// CK grouped backward data convolution kernels.

namespace ck_tile::builder::test {

namespace detail {

/// @brief Concept for checking whether a bwd data convolution is invoked like old CK.
///
/// This is the same as `::ck_tile::builder::test::CkConvBwdDataInstance`, except
/// with some utility aliases. For that reason, its moved to this detail
/// namespace.
template <typename Conv,
          auto SIGNATURE,
          size_t SPATIAL_DIM = SIGNATURE.spatial_dim,
          // TODO: We shouldn't need to call into an internal namespace here.
          typename Types = factory::internal::ConvTensorDataTypes<SIGNATURE>,
          typename Ops   = factory::internal::ConvElementwiseOps<SIGNATURE>>
concept CkConvBwdDataInstance =
    requires(Conv& conv,
             const Types::OutDataType* p_output, // A (output image)
             const Types::WeiDataType* p_weight, // B (weight)
             std::array<const void*, 0> p_ds,    // empty Ds (bias)
             Types::InDataType* p_input,         // E (input image)
             std::array<index_t, SPATIAL_DIM + 3> out_lengths,
             std::array<index_t, SPATIAL_DIM + 3> out_strides,
             std::array<index_t, SPATIAL_DIM + 3> wei_lengths,
             std::array<index_t, SPATIAL_DIM + 3> wei_strides,
             std::array<std::array<index_t, SPATIAL_DIM + 3>, 0> ds_lengths,
             std::array<std::array<index_t, SPATIAL_DIM + 3>, 0> ds_strides,
             std::array<index_t, SPATIAL_DIM + 3> in_lengths,
             std::array<index_t, SPATIAL_DIM + 3> in_strides,
             std::array<index_t, SPATIAL_DIM> filter,
             Ops::OutElementwiseOp elementwise_a,  // output image op
             Ops::WeiElementwiseOp elementwise_b,  // weight op
             Ops::InElementwiseOp elementwise_cde, // input image op
             ck::index_t split_k) {
        requires ValidConvSignature<SIGNATURE>;
        requires ConvDirectionIsBackwardData<SIGNATURE>;

        {
            // backward-data CK kernels: output (A), weight (B), bias (Ds, empty), input (E)
            conv.MakeArgument(p_output,
                              p_weight,
                              p_ds,
                              p_input,
                              // output image lengths/strides (A)
                              out_lengths,
                              out_strides,
                              // weight lengths/strides (B)
                              wei_lengths,
                              wei_strides,
                              // bias lengths/strides (Ds - empty)
                              ds_lengths,
                              ds_strides,
                              // input image lengths/strides (E)
                              in_lengths,
                              in_strides,
                              // convolution strides/dilations/pads
                              filter,
                              filter,
                              filter,
                              filter,
                              // element-wise operations
                              elementwise_a,
                              elementwise_b,
                              elementwise_cde,
                              split_k)
        };
    };

} // namespace detail

/// @brief Concept for checking whether a bwd data convolution is invoked like old CK.
///
/// - Conv The convolution type.
/// - SIGNATURE The convolution signature.
///
/// @see detail::CkConvBwdDataInstance
template <typename Conv, auto SIGNATURE>
concept CkConvBwdDataInstance = detail::CkConvBwdDataInstance<Conv, SIGNATURE>;

/// @brief `run()` specialization for backward data convolution and old CK.
///
/// @tparam SIGNATURE Backward data convolution signature.
/// @returns RunResult about how the operation completed (or not).
///
/// @see run()
template <auto SIGNATURE>
    requires ValidConvSignature<SIGNATURE> && ConvDirectionIsBackwardData<SIGNATURE>
[[nodiscard]] RunResult run(CkConvBwdDataInstance<SIGNATURE> auto& conv,
                            const Args<SIGNATURE>& args,
                            const Inputs<SIGNATURE>& inputs,
                            const Outputs<SIGNATURE>& outputs)
{
    using Types = factory::internal::ConvTensorDataTypes<SIGNATURE>;

    constexpr auto spatial_dim = SIGNATURE.spatial_dim;

    const auto copy = [](const auto& src, auto& dst) {
        std::copy(src.begin(), src.end(), dst.begin());
    };

    const auto to_ck_extent = [&](const auto& extent) {
        std::array<ck::index_t, spatial_dim> result;
        copy(extent, result);
        return result;
    };

    const auto param = args.to_ck_conv_param();

    const auto input_desc  = args.make_input_descriptor();
    const auto weight_desc = args.make_weight_descriptor();
    const auto output_desc = args.make_output_descriptor();

    // For backward data: convert descriptor data to CK format
    auto to_ck_lengths_from_vec = [](const auto& vec) {
        std::array<ck::index_t, spatial_dim + 3> result;
        std::copy(vec.begin(), vec.end(), result.begin());
        return result;
    };

    // Create empty Ds array - for backward data, no bias tensors
    const std::array<std::array<ck::index_t, spatial_dim + 3>, 0> ds_lengths{};
    const std::array<std::array<ck::index_t, spatial_dim + 3>, 0> ds_strides{};

    auto ck_args = conv.MakeArgument(
        static_cast<const Types::OutDataType*>(inputs.output), // p_a (output image A)
        static_cast<const Types::WeiDataType*>(inputs.weight), // p_b (weight B)
        std::array<const void*, 0>{},                          // p_ds (empty - no bias)
        static_cast<Types::InDataType*>(outputs.input),        // p_e (input image E - result)
        to_ck_lengths_from_vec(output_desc.get_lengths()),     // a_g_n_k_wos_lengths
        to_ck_lengths_from_vec(output_desc.get_strides()),     // a_g_n_k_wos_strides
        to_ck_lengths_from_vec(weight_desc.get_lengths()),     // b_g_k_c_xs_lengths
        to_ck_lengths_from_vec(weight_desc.get_strides()),     // b_g_k_c_xs_strides
        ds_lengths,                                            // ds_g_n_c_wis_lengths (empty)
        ds_strides,                                            // ds_g_n_c_wis_strides (empty)
        to_ck_lengths_from_vec(input_desc.get_lengths()),      // e_g_n_c_wis_lengths
        to_ck_lengths_from_vec(input_desc.get_strides()),      // e_g_n_c_wis_strides
        to_ck_extent(param.conv_filter_strides_),
        to_ck_extent(param.conv_filter_dilations_),
        to_ck_extent(param.input_left_pads_),
        to_ck_extent(param.input_right_pads_),
        args.a_elementwise_op,
        args.b_elementwise_op,
        args.cde_elementwise_op,
        args.k_batch);

    if(!conv.IsSupportedArgument(ck_args))
        return RunResult::not_supported("invalid ck arguments");

    return RunResult::from_runtime(conv.MakeInvoker().Run(ck_args));
}

} // namespace ck_tile::builder::test
