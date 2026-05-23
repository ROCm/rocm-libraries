// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <string>
#include <tuple>

#include "ck_tile/core.hpp"
#include "ck_tile/core/tensor/tile_elementwise.hpp"
#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/host/concat.hpp"
#include "ck_tile/core/utility/env.hpp"
#include "ck_tile/host/convolution_parameter.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"
#include "ck_tile/ops/gemm/kernel/batched_gemm_kernel.hpp"
#include "ck_tile/ops/grouped_convolution/utils/transform_conv_fwd_to_gemm.hpp"
#include "ck_tile/ops/grouped_convolution/utils/transform_conv_fwd_to_wcnn.hpp"
#include "ck_tile/ops/grouped_convolution/utils/grouped_convolution_utils.hpp"

#ifdef CK_EXPERIMENTAL_BUILDER
#include "ck_tile/builder/reflect/instance_traits_tile_grouped_convolution_forward.hpp"
#endif

namespace ck_tile {

/// @brief Kernel arguments for WCNN forward convolution.
///
/// Follows the same per-layout constructor pattern as GroupedConvFwdKernelArgs, but builds
/// HWC-style descriptors (via TransformConvFwdToHWCWcnn) instead of implicit-GEMM descriptors.
template <typename WcnnFwdTraitsType_, typename CDElementwise_>
struct WcnnFwdKernelArgs
{
    using ConvToWcnnFwdTransformer =
        TransformConvFwdToHWCWcnn<WcnnFwdTraitsType_::NDimSpatial,
                                  WcnnFwdTraitsType_::ConvSpecialization,
                                  WcnnFwdTraitsType_::VectorSizeA,
                                  WcnnFwdTraitsType_::VectorSizeB,
                                  WcnnFwdTraitsType_::VectorSizeC>;
    using CDElementwise                     = CDElementwise_;
    static constexpr index_t NDimSpatial    = WcnnFwdTraitsType_::NDimSpatial;
    static constexpr index_t NonSpatialDims = 3; // G, N, C/K
    static constexpr index_t NumDTensor     = WcnnFwdTraitsType_::NumDTensor;

    // ======================== 1D: NWGC / GKXC / NWGK ========================
    template <
        typename InLay                      = typename WcnnFwdTraitsType_::InLayout,
        typename WeiLay                     = typename WcnnFwdTraitsType_::WeiLayout,
        typename OutLay                     = typename WcnnFwdTraitsType_::OutLayout,
        typename std::enable_if<std::is_same_v<InLay, tensor_layout::convolution::NWGC> &&
                                    std::is_same_v<WeiLay, tensor_layout::convolution::GKXC> &&
                                    std::is_same_v<OutLay, tensor_layout::convolution::NWGK>,
                                bool>::type = false>
    CK_TILE_HOST WcnnFwdKernelArgs(const GroupedConvFwdHostArgs<CDElementwise>& args)
        : elfunc(args.elfunc)
    {
        G = static_cast<index_t>(args.G_);
        N = static_cast<index_t>(args.N_);
        C = static_cast<index_t>(args.C_);
        K = static_cast<index_t>(args.K_);

        input_spatial_lengths  = {static_cast<index_t>(args.input_spatial_lengths_[0])};
        output_spatial_lengths = {static_cast<index_t>(args.output_spatial_lengths_[0])};
        filter_spatial_lengths = {static_cast<index_t>(args.filter_spatial_lengths_[0])};
        conv_filter_strides    = {static_cast<index_t>(args.conv_filter_strides_[0])};
        conv_filter_dilations  = {static_cast<index_t>(args.conv_filter_dilations_[0])};
        input_left_pads        = {static_cast<index_t>(args.input_left_pads_[0])};
        input_right_pads       = {static_cast<index_t>(args.input_right_pads_[0])};

        // NWGC: N, W, G, C (innermost)
        in_g_n_c_wis_lengths = {G, N, C, input_spatial_lengths[0]};
        in_g_n_c_wis_strides = {
            C,                                // G stride
            input_spatial_lengths[0] * G * C, // N stride
            1,                                // C stride (innermost)
            G * C                             // W stride
        };

        // GKXC: G, K, X, C (innermost)
        wei_g_k_c_xs_lengths = {G, K, C, filter_spatial_lengths[0]};
        wei_g_k_c_xs_strides = {
            K * filter_spatial_lengths[0] * C, // G stride
            filter_spatial_lengths[0] * C,     // K stride
            1,                                 // C stride
            C                                  // X stride
        };

        // NWGK: N, W, G, K (innermost)
        out_g_n_k_wos_lengths = {G, N, K, output_spatial_lengths[0]};
        out_g_n_k_wos_strides = {
            K,                                 // G stride
            output_spatial_lengths[0] * G * K, // N stride
            1,                                 // K stride (innermost)
            G * K                              // W stride
        };

        in_ptr  = args.in_ptr;
        wei_ptr = args.wei_ptr;
        for(index_t d = 0; d < NumDTensor; d++)
        {
            ds_ptr[d] = args.ds_ptr[d];
        }
        out_ptr = args.out_ptr;

        transformer_ = ConvToWcnnFwdTransformer{in_g_n_c_wis_lengths,
                                                wei_g_k_c_xs_lengths,
                                                out_g_n_k_wos_lengths,
                                                conv_filter_strides,
                                                conv_filter_dilations,
                                                input_left_pads,
                                                input_right_pads};

        a_grid_desc_h_w_c =
            transformer_.template MakeADescriptor_H_W_C<typename WcnnFwdTraitsType_::InLayout>();
        b_grid_desc_k_yx_c =
            transformer_.template MakeBDescriptor_K_YX_C<typename WcnnFwdTraitsType_::WeiLayout>();
        c_grid_desc_h_w_k =
            transformer_.template MakeCDescriptor_H_W_K<typename WcnnFwdTraitsType_::OutLayout>();

        ComputeDerivedFields();
    }

    // ======================== 2D: NHWGC / GKYXC / NHWGK ========================
    template <
        typename InLay                      = typename WcnnFwdTraitsType_::InLayout,
        typename WeiLay                     = typename WcnnFwdTraitsType_::WeiLayout,
        typename OutLay                     = typename WcnnFwdTraitsType_::OutLayout,
        typename std::enable_if<std::is_same_v<InLay, tensor_layout::convolution::NHWGC> &&
                                    std::is_same_v<WeiLay, tensor_layout::convolution::GKYXC> &&
                                    std::is_same_v<OutLay, tensor_layout::convolution::NHWGK>,
                                bool>::type = false>
    CK_TILE_HOST WcnnFwdKernelArgs(const GroupedConvFwdHostArgs<CDElementwise>& args)
        : elfunc(args.elfunc)
    {
        G = static_cast<index_t>(args.G_);
        N = static_cast<index_t>(args.N_);
        C = static_cast<index_t>(args.C_);
        K = static_cast<index_t>(args.K_);

        input_spatial_lengths  = {static_cast<index_t>(args.input_spatial_lengths_[0]),
                                  static_cast<index_t>(args.input_spatial_lengths_[1])};
        output_spatial_lengths = {static_cast<index_t>(args.output_spatial_lengths_[0]),
                                  static_cast<index_t>(args.output_spatial_lengths_[1])};
        filter_spatial_lengths = {static_cast<index_t>(args.filter_spatial_lengths_[0]),
                                  static_cast<index_t>(args.filter_spatial_lengths_[1])};
        conv_filter_strides    = {static_cast<index_t>(args.conv_filter_strides_[0]),
                                  static_cast<index_t>(args.conv_filter_strides_[1])};
        conv_filter_dilations  = {static_cast<index_t>(args.conv_filter_dilations_[0]),
                                  static_cast<index_t>(args.conv_filter_dilations_[1])};
        input_left_pads        = {static_cast<index_t>(args.input_left_pads_[0]),
                                  static_cast<index_t>(args.input_left_pads_[1])};
        input_right_pads       = {static_cast<index_t>(args.input_right_pads_[0]),
                                  static_cast<index_t>(args.input_right_pads_[1])};

        const index_t Hi = input_spatial_lengths[0];
        const index_t Wi = input_spatial_lengths[1];
        const index_t Ho = output_spatial_lengths[0];
        const index_t Wo = output_spatial_lengths[1];

        // NHWGC: N, H, W, G, C (innermost)
        in_g_n_c_wis_lengths = {G, N, C, Hi, Wi};
        in_g_n_c_wis_strides = {
            C,               // G stride
            Hi * Wi * G * C, // N stride
            1,               // C stride (innermost)
            Wi * G * C,      // H stride
            G * C            // W stride
        };

        // GKYXC: G, K, Y, X, C (innermost)
        const index_t Y      = filter_spatial_lengths[0];
        const index_t X      = filter_spatial_lengths[1];
        wei_g_k_c_xs_lengths = {G, K, C, Y, X};
        wei_g_k_c_xs_strides = {
            K * Y * X * C, // G stride
            Y * X * C,     // K stride
            1,             // C stride (innermost)
            X * C,         // Y stride
            C              // X stride
        };

        // NHWGK: N, H, W, G, K (innermost)
        out_g_n_k_wos_lengths = {G, N, K, Ho, Wo};
        out_g_n_k_wos_strides = {
            K,               // G stride
            Ho * Wo * G * K, // N stride
            1,               // K stride (innermost)
            Wo * G * K,      // H stride
            G * K            // W stride
        };

        in_ptr  = args.in_ptr;
        wei_ptr = args.wei_ptr;
        for(index_t d = 0; d < NumDTensor; d++)
        {
            ds_ptr[d] = args.ds_ptr[d];
        }
        out_ptr = args.out_ptr;

        transformer_ = ConvToWcnnFwdTransformer{in_g_n_c_wis_lengths,
                                                wei_g_k_c_xs_lengths,
                                                out_g_n_k_wos_lengths,
                                                conv_filter_strides,
                                                conv_filter_dilations,
                                                input_left_pads,
                                                input_right_pads};

        a_grid_desc_h_w_c =
            transformer_.template MakeADescriptor_H_W_C<typename WcnnFwdTraitsType_::InLayout>();
        b_grid_desc_k_yx_c =
            transformer_.template MakeBDescriptor_K_YX_C<typename WcnnFwdTraitsType_::WeiLayout>();
        c_grid_desc_h_w_k =
            transformer_.template MakeCDescriptor_H_W_K<typename WcnnFwdTraitsType_::OutLayout>();

        ComputeDerivedFields();
    }

    // ======================== Common data members ========================
    index_t G, N, C, K;
    array<index_t, NDimSpatial> input_spatial_lengths;
    array<index_t, NDimSpatial> output_spatial_lengths;
    array<index_t, NDimSpatial> filter_spatial_lengths;
    array<index_t, NDimSpatial> conv_filter_strides;
    array<index_t, NDimSpatial> conv_filter_dilations;
    array<index_t, NDimSpatial> input_left_pads;
    array<index_t, NDimSpatial> input_right_pads;

    // Full tensor lengths and strides (for building HWC descriptors)
    array<index_t, NonSpatialDims + NDimSpatial> in_g_n_c_wis_lengths;
    array<index_t, NonSpatialDims + NDimSpatial> in_g_n_c_wis_strides;
    array<index_t, NonSpatialDims + NDimSpatial> wei_g_k_c_xs_lengths;
    array<index_t, NonSpatialDims + NDimSpatial> wei_g_k_c_xs_strides;
    array<index_t, NonSpatialDims + NDimSpatial> out_g_n_k_wos_lengths;
    array<index_t, NonSpatialDims + NDimSpatial> out_g_n_k_wos_strides;

    index_t ConvH; // = N * product(output spatial dims except last W)

    // Per-group strides for pointer offset computation
    long_index_t group_stride_in;
    long_index_t group_stride_wei;
    long_index_t group_stride_out;

    const void* in_ptr;
    const void* wei_ptr;
    const void* ds_ptr[NumDTensor > 0 ? NumDTensor : 1];
    void* out_ptr;

    const CDElementwise elfunc;

    ConvToWcnnFwdTransformer transformer_;

    using AGridDescHWC = remove_cvref_t<
        decltype(ConvToWcnnFwdTransformer{}
                     .template MakeADescriptor_H_W_C<typename WcnnFwdTraitsType_::InLayout>())>;
    using BGridDescKYXC = remove_cvref_t<
        decltype(ConvToWcnnFwdTransformer{}
                     .template MakeBDescriptor_K_YX_C<typename WcnnFwdTraitsType_::WeiLayout>())>;
    using CGridDescHWK = remove_cvref_t<
        decltype(ConvToWcnnFwdTransformer{}
                     .template MakeCDescriptor_H_W_K<typename WcnnFwdTraitsType_::OutLayout>())>;

    AGridDescHWC a_grid_desc_h_w_c;
    BGridDescKYXC b_grid_desc_k_yx_c;
    CGridDescHWK c_grid_desc_h_w_k;

    private:
    CK_TILE_HOST void ComputeDerivedFields()
    {
        ConvH = N;
        for(index_t i = 0; i < NDimSpatial - 1; ++i)
        {
            ConvH *= output_spatial_lengths[i];
        }

        group_stride_in  = in_g_n_c_wis_strides[0];
        group_stride_wei = wei_g_k_c_xs_strides[0];
        group_stride_out = out_g_n_k_wos_strides[0];

        if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
        {
            std::cout << "WcnnFwd: G=" << G << ", N=" << N << ", C=" << C << ", K=" << K
                      << ", ConvH=" << ConvH << std::endl;
        }
    }
};

/// @brief The Grouped Convolution kernel device arguments.
template <typename GroupedConvTraitsType_, typename CDElementwise_>
struct GroupedConvFwdKernelArgs
{
    using ConvToGemmFwdTransformer =
        TransformConvFwdToGemm<GroupedConvTraitsType_::NDimSpatial,
                               GroupedConvTraitsType_::ConvSpecialization,
                               GroupedConvTraitsType_::VectorSizeA,
                               GroupedConvTraitsType_::VectorSizeB,
                               GroupedConvTraitsType_::VectorSizeC,
                               GroupedConvTraitsType_::NumGroupsToMerge,
                               true>; // Split N enabled
    using CDElementwise                 = CDElementwise_;
    static constexpr index_t NumDTensor = GroupedConvTraitsType_::NumDTensor;

    static_assert(!GroupedConvTraitsType_::ExplicitGemm ||
                      GroupedConvTraitsType_::NumGroupsToMerge == 1,
                  "Explicit GEMM does not support merging convolution groups!");

    template <
        typename InLay                      = typename GroupedConvTraitsType_::InLayout,
        typename WeiLay                     = typename GroupedConvTraitsType_::WeiLayout,
        typename OutLay                     = typename GroupedConvTraitsType_::OutLayout,
        typename std::enable_if<std::is_same_v<InLay, tensor_layout::convolution::NWGC> &&
                                    std::is_same_v<WeiLay, tensor_layout::convolution::GKXC> &&
                                    std::is_same_v<OutLay, tensor_layout::convolution::NWGK>,
                                bool>::type = false>
    CK_TILE_HOST GroupedConvFwdKernelArgs(const GroupedConvFwdHostArgs<CDElementwise>& args)
        : elfunc(args.elfunc)
    {
        in_g_n_c_wis_lengths  = {static_cast<index_t>(args.G_),
                                 static_cast<index_t>(args.N_),
                                 static_cast<index_t>(args.C_),
                                 static_cast<index_t>(args.input_spatial_lengths_[0])};
        wei_g_k_c_xs_lengths  = {static_cast<index_t>(args.G_),
                                 static_cast<index_t>(args.K_),
                                 static_cast<index_t>(args.C_),
                                 static_cast<index_t>(args.filter_spatial_lengths_[0])};
        out_g_n_k_wos_lengths = {static_cast<index_t>(args.G_),
                                 static_cast<index_t>(args.N_),
                                 static_cast<index_t>(args.K_),
                                 static_cast<index_t>(args.output_spatial_lengths_[0])};

        conv_filter_strides   = {static_cast<index_t>(args.conv_filter_strides_[0])};
        conv_filter_dilations = {static_cast<index_t>(args.conv_filter_dilations_[0])};
        input_left_pads       = {static_cast<index_t>(args.input_left_pads_[0])};
        input_right_pads      = {static_cast<index_t>(args.input_right_pads_[0])};

        k_batch = args.k_batch;

        in_ptr  = args.in_ptr;
        wei_ptr = args.wei_ptr;
        for(index_t d = 0; d < NumDTensor; d++)
        {
            ds_ptr[d] = args.ds_ptr[d];
        }
        out_ptr = args.out_ptr;

        // Create and STORE transformer (for split-image support)
        transformer_ = ConvToGemmFwdTransformer{in_g_n_c_wis_lengths,
                                                wei_g_k_c_xs_lengths,
                                                out_g_n_k_wos_lengths,
                                                conv_filter_strides,
                                                conv_filter_dilations,
                                                input_left_pads,
                                                input_right_pads};

        a_grid_desc_m_k =
            transformer_.template MakeADescriptor_M_K<typename GroupedConvTraitsType_::InLayout>();
        b_grid_desc_n_k =
            transformer_.template MakeBDescriptor_N_K<typename GroupedConvTraitsType_::WeiLayout>();
        c_grid_desc_m_n =
            transformer_.template MakeCDescriptor_M_N<typename GroupedConvTraitsType_::OutLayout>();

        NumGroupsToMerge = GroupedConvTraitsType_::NumGroupsToMerge;
        group_stride_a   = args.C_ * NumGroupsToMerge;
        group_stride_b   = args.K_ * args.C_ * NumGroupsToMerge *
                         std::accumulate(args.filter_spatial_lengths_.begin(),
                                         args.filter_spatial_lengths_.end(),
                                         1,
                                         std::multiplies<index_t>());
        group_stride_c = args.K_ * NumGroupsToMerge;

        // Initialize Split-N support fields for 1D convolution (NWGC layout)
        // Get the actual split N from transformer
        n_per_split = transformer_.GetN();
        original_n  = transformer_.GetOriginalN();
        n_splits    = ck_tile::integer_divide_ceil(original_n, n_per_split);

        // Calculate batch strides using the original argument dimensions.
        // These are the original dimensions passed to the constructor, not modified by the invoker
        // yet. (The invoker modifies args after calling MakeKernelArgs.) VERIFIED: G_ MUST be
        // included - NWGC layout has all groups within each batch
        input_batch_stride  = args.G_ * args.C_ * args.input_spatial_lengths_[0];
        output_batch_stride = args.G_ * args.K_ * args.output_spatial_lengths_[0];

        GemmM     = a_grid_desc_m_k.get_length(number<0>{});
        GemmN     = b_grid_desc_n_k.get_length(number<0>{});
        GemmK     = a_grid_desc_m_k.get_length(number<1>{});
        GemmBatch = integer_divide_ceil(args.G_, NumGroupsToMerge);

        if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
        {
            std::cout << "GemmM: " << GemmM << ", GemmN: " << GemmN << ", GemmK: " << GemmK
                      << ", GemmBatch: " << GemmBatch << ", N per split: " << n_per_split
                      << ", number of N splits: " << n_splits
                      << ", input_batch_stride: " << input_batch_stride
                      << ", output_batch_stride: " << output_batch_stride
                      << ", NumGroupsToMerge: " << NumGroupsToMerge << std::endl;
        }
    }

    template <
        typename InLay                      = typename GroupedConvTraitsType_::InLayout,
        typename WeiLay                     = typename GroupedConvTraitsType_::WeiLayout,
        typename OutLay                     = typename GroupedConvTraitsType_::OutLayout,
        typename std::enable_if<std::is_same_v<InLay, tensor_layout::convolution::NHWGC> &&
                                    std::is_same_v<WeiLay, tensor_layout::convolution::GKYXC> &&
                                    std::is_same_v<OutLay, tensor_layout::convolution::NHWGK>,
                                bool>::type = false>
    CK_TILE_HOST GroupedConvFwdKernelArgs(const GroupedConvFwdHostArgs<CDElementwise>& args)
        : elfunc(args.elfunc)
    {
        in_g_n_c_wis_lengths  = {static_cast<index_t>(args.G_),
                                 static_cast<index_t>(args.N_),
                                 static_cast<index_t>(args.C_),
                                 static_cast<index_t>(args.input_spatial_lengths_[0]),
                                 static_cast<index_t>(args.input_spatial_lengths_[1])};
        wei_g_k_c_xs_lengths  = {static_cast<index_t>(args.G_),
                                 static_cast<index_t>(args.K_),
                                 static_cast<index_t>(args.C_),
                                 static_cast<index_t>(args.filter_spatial_lengths_[0]),
                                 static_cast<index_t>(args.filter_spatial_lengths_[1])};
        out_g_n_k_wos_lengths = {static_cast<index_t>(args.G_),
                                 static_cast<index_t>(args.N_),
                                 static_cast<index_t>(args.K_),
                                 static_cast<index_t>(args.output_spatial_lengths_[0]),
                                 static_cast<index_t>(args.output_spatial_lengths_[1])};

        conv_filter_strides   = {static_cast<index_t>(args.conv_filter_strides_[0]),
                                 static_cast<index_t>(args.conv_filter_strides_[1])};
        conv_filter_dilations = {static_cast<index_t>(args.conv_filter_dilations_[0]),
                                 static_cast<index_t>(args.conv_filter_dilations_[1])};
        input_left_pads       = {static_cast<index_t>(args.input_left_pads_[0]),
                                 static_cast<index_t>(args.input_left_pads_[1])};
        input_right_pads      = {static_cast<index_t>(args.input_right_pads_[0]),
                                 static_cast<index_t>(args.input_right_pads_[1])};

        k_batch = args.k_batch;

        in_ptr  = args.in_ptr;
        wei_ptr = args.wei_ptr;
        for(index_t d = 0; d < NumDTensor; d++)
        {
            ds_ptr[d] = args.ds_ptr[d];
        }
        out_ptr = args.out_ptr;

        // Create and STORE transformer (for split-image support)
        transformer_ = ConvToGemmFwdTransformer{in_g_n_c_wis_lengths,
                                                wei_g_k_c_xs_lengths,
                                                out_g_n_k_wos_lengths,
                                                conv_filter_strides,
                                                conv_filter_dilations,
                                                input_left_pads,
                                                input_right_pads};

        a_grid_desc_m_k =
            transformer_.template MakeADescriptor_M_K<typename GroupedConvTraitsType_::InLayout>();
        b_grid_desc_n_k =
            transformer_.template MakeBDescriptor_N_K<typename GroupedConvTraitsType_::WeiLayout>();
        c_grid_desc_m_n =
            transformer_.template MakeCDescriptor_M_N<typename GroupedConvTraitsType_::OutLayout>();

        NumGroupsToMerge = GroupedConvTraitsType_::NumGroupsToMerge;
        group_stride_a   = args.C_ * NumGroupsToMerge;
        group_stride_b   = args.K_ * args.C_ * NumGroupsToMerge *
                         std::accumulate(args.filter_spatial_lengths_.begin(),
                                         args.filter_spatial_lengths_.end(),
                                         1,
                                         std::multiplies<index_t>());
        group_stride_c = args.K_ * NumGroupsToMerge;

        // Initialize Split-N support fields for 2D convolution (NHWGC layout)
        // Get the actual split N from transformer
        n_per_split = transformer_.GetN();
        original_n  = transformer_.GetOriginalN();
        n_splits    = ck_tile::integer_divide_ceil(original_n, n_per_split);

        // Calculate batch strides for NHWGC layout
        // VERIFIED: G_ MUST be included - NHWGC layout has all groups within each batch
        input_batch_stride =
            args.G_ * args.C_ * args.input_spatial_lengths_[0] * args.input_spatial_lengths_[1];
        output_batch_stride =
            args.G_ * args.K_ * args.output_spatial_lengths_[0] * args.output_spatial_lengths_[1];

        GemmM     = a_grid_desc_m_k.get_length(number<0>{});
        GemmN     = b_grid_desc_n_k.get_length(number<0>{});
        GemmK     = a_grid_desc_m_k.get_length(number<1>{});
        GemmBatch = integer_divide_ceil(args.G_, NumGroupsToMerge);

        if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
        {
            std::cout << "GemmM: " << GemmM << ", GemmN: " << GemmN << ", GemmK: " << GemmK
                      << ", GemmBatch: " << GemmBatch << ", N per split: " << n_per_split
                      << ", number of N splits: " << n_splits
                      << ", input_batch_stride: " << input_batch_stride
                      << ", output_batch_stride: " << output_batch_stride
                      << ", NumGroupsToMerge: " << NumGroupsToMerge << std::endl;
        }
    }

    template <
        typename InLay                      = typename GroupedConvTraitsType_::InLayout,
        typename WeiLay                     = typename GroupedConvTraitsType_::WeiLayout,
        typename OutLay                     = typename GroupedConvTraitsType_::OutLayout,
        typename std::enable_if<std::is_same_v<InLay, tensor_layout::convolution::NDHWGC> &&
                                    std::is_same_v<WeiLay, tensor_layout::convolution::GKZYXC> &&
                                    std::is_same_v<OutLay, tensor_layout::convolution::NDHWGK>,
                                bool>::type = false>
    CK_TILE_HOST GroupedConvFwdKernelArgs(const GroupedConvFwdHostArgs<CDElementwise>& args)
        : elfunc(args.elfunc)
    {
        in_g_n_c_wis_lengths  = {static_cast<index_t>(args.G_),
                                 static_cast<index_t>(args.N_),
                                 static_cast<index_t>(args.C_),
                                 static_cast<index_t>(args.input_spatial_lengths_[0]),
                                 static_cast<index_t>(args.input_spatial_lengths_[1]),
                                 static_cast<index_t>(args.input_spatial_lengths_[2])};
        wei_g_k_c_xs_lengths  = {static_cast<index_t>(args.G_),
                                 static_cast<index_t>(args.K_),
                                 static_cast<index_t>(args.C_),
                                 static_cast<index_t>(args.filter_spatial_lengths_[0]),
                                 static_cast<index_t>(args.filter_spatial_lengths_[1]),
                                 static_cast<index_t>(args.filter_spatial_lengths_[2])};
        out_g_n_k_wos_lengths = {static_cast<index_t>(args.G_),
                                 static_cast<index_t>(args.N_),
                                 static_cast<index_t>(args.K_),
                                 static_cast<index_t>(args.output_spatial_lengths_[0]),
                                 static_cast<index_t>(args.output_spatial_lengths_[1]),
                                 static_cast<index_t>(args.output_spatial_lengths_[2])};

        conv_filter_strides   = {static_cast<index_t>(args.conv_filter_strides_[0]),
                                 static_cast<index_t>(args.conv_filter_strides_[1]),
                                 static_cast<index_t>(args.conv_filter_strides_[2])};
        conv_filter_dilations = {static_cast<index_t>(args.conv_filter_dilations_[0]),
                                 static_cast<index_t>(args.conv_filter_dilations_[1]),
                                 static_cast<index_t>(args.conv_filter_dilations_[2])};
        input_left_pads       = {static_cast<index_t>(args.input_left_pads_[0]),
                                 static_cast<index_t>(args.input_left_pads_[1]),
                                 static_cast<index_t>(args.input_left_pads_[2])};
        input_right_pads      = {static_cast<index_t>(args.input_right_pads_[0]),
                                 static_cast<index_t>(args.input_right_pads_[1]),
                                 static_cast<index_t>(args.input_right_pads_[2])};

        k_batch = args.k_batch;

        in_ptr  = args.in_ptr;
        wei_ptr = args.wei_ptr;
        for(index_t d = 0; d < NumDTensor; d++)
        {
            ds_ptr[d] = args.ds_ptr[d];
        }
        out_ptr = args.out_ptr;

        // Create and STORE transformer (for split-image support)
        transformer_ = ConvToGemmFwdTransformer{in_g_n_c_wis_lengths,
                                                wei_g_k_c_xs_lengths,
                                                out_g_n_k_wos_lengths,
                                                conv_filter_strides,
                                                conv_filter_dilations,
                                                input_left_pads,
                                                input_right_pads};

        a_grid_desc_m_k =
            transformer_.template MakeADescriptor_M_K<typename GroupedConvTraitsType_::InLayout>();
        b_grid_desc_n_k =
            transformer_.template MakeBDescriptor_N_K<typename GroupedConvTraitsType_::WeiLayout>();
        c_grid_desc_m_n =
            transformer_.template MakeCDescriptor_M_N<typename GroupedConvTraitsType_::OutLayout>();

        NumGroupsToMerge = GroupedConvTraitsType_::NumGroupsToMerge;
        group_stride_a   = args.C_ * NumGroupsToMerge;
        group_stride_b   = args.K_ * args.C_ * NumGroupsToMerge *
                         std::accumulate(args.filter_spatial_lengths_.begin(),
                                         args.filter_spatial_lengths_.end(),
                                         1,
                                         std::multiplies<index_t>());
        group_stride_c = args.K_ * NumGroupsToMerge;

        // Initialize Split-N support fields for 3D convolution (NDHWGC layout)
        // Get the actual split N from transformer
        n_per_split = transformer_.GetN();
        original_n  = transformer_.GetOriginalN();
        n_splits    = ck_tile::integer_divide_ceil(original_n, n_per_split);

        // Calculate batch strides for NDHWGC layout
        // VERIFIED: G_ MUST be included - NDHWGC layout has all groups within each batch
        input_batch_stride = args.G_ * args.C_ * args.input_spatial_lengths_[0] *
                             args.input_spatial_lengths_[1] * args.input_spatial_lengths_[2];
        output_batch_stride = args.G_ * args.K_ * args.output_spatial_lengths_[0] *
                              args.output_spatial_lengths_[1] * args.output_spatial_lengths_[2];

        GemmM     = a_grid_desc_m_k.get_length(number<0>{});
        GemmN     = b_grid_desc_n_k.get_length(number<0>{});
        GemmK     = a_grid_desc_m_k.get_length(number<1>{});
        GemmBatch = integer_divide_ceil(args.G_, NumGroupsToMerge);

        if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
        {
            std::cout << "GemmM: " << GemmM << ", GemmN: " << GemmN << ", GemmK: " << GemmK
                      << ", GemmBatch: " << GemmBatch << ", N per split: " << n_per_split
                      << ", number of N splits: " << n_splits
                      << ", input_batch_stride: " << input_batch_stride
                      << ", output_batch_stride: " << output_batch_stride
                      << ", NumGroupsToMerge: " << NumGroupsToMerge << std::endl;
        }
    }
    using AGridDescMK = remove_cvref_t<
        decltype(ConvToGemmFwdTransformer{}
                     .template MakeADescriptor_M_K<typename GroupedConvTraitsType_::InLayout>())>;
    using BGridDescNK = remove_cvref_t<
        decltype(ConvToGemmFwdTransformer{}
                     .template MakeBDescriptor_N_K<typename GroupedConvTraitsType_::WeiLayout>())>;
    using CGridDescMN = remove_cvref_t<
        decltype(ConvToGemmFwdTransformer{}
                     .template MakeCDescriptor_M_N<typename GroupedConvTraitsType_::OutLayout>())>;

    static constexpr index_t NonSpatialDims = 3;
    array<index_t, NonSpatialDims + GroupedConvTraitsType_::NDimSpatial> in_g_n_c_wis_lengths;
    array<index_t, NonSpatialDims + GroupedConvTraitsType_::NDimSpatial> wei_g_k_c_xs_lengths;
    array<index_t, NonSpatialDims + GroupedConvTraitsType_::NDimSpatial> out_g_n_k_wos_lengths;

    array<index_t, GroupedConvTraitsType_::NDimSpatial> conv_filter_strides;
    array<index_t, GroupedConvTraitsType_::NDimSpatial> conv_filter_dilations;
    array<index_t, GroupedConvTraitsType_::NDimSpatial> input_left_pads;
    array<index_t, GroupedConvTraitsType_::NDimSpatial> input_right_pads;

    index_t k_batch;
    index_t GemmM;
    index_t GemmN;
    index_t GemmK;
    index_t GemmBatch;
    index_t NumGroupsToMerge;

    const void* in_ptr;
    const void* wei_ptr;
    std::array<const void*, NumDTensor> ds_ptr;
    const CDElementwise elfunc;
    void* out_ptr;

    AGridDescMK a_grid_desc_m_k;
    BGridDescNK b_grid_desc_n_k;
    CGridDescMN c_grid_desc_m_n;

    long_index_t group_stride_a;
    long_index_t group_stride_b;
    long_index_t group_stride_c;

    // Split-N support fields - initialize to safe defaults
    index_t n_splits            = 1; // Number of batch splits (e.g., 2 for 128→64×2)
    index_t n_per_split         = 1; // Batches per split (N_ from transformer)
    index_t original_n          = 1; // Original batch size before splitting
    index_t input_batch_stride  = 0; // Stride to next batch in input tensor
    index_t output_batch_stride = 0; // Stride to next batch in output tensor

    // Split-image support - spatial offsets (applied per-batch in operator())
    long_index_t spatial_offset_in  = 0; // Spatial offset for input (e.g., W/2 for 1D split)
    long_index_t spatial_offset_out = 0; // Spatial offset for output (e.g., W/2 for 1D split)

    // Split-image support - transformer instance
    ConvToGemmFwdTransformer transformer_;

    // Forward declare descriptor types (will be defined after using declarations)
    using ConvToGemmFwdTransformer_t = ConvToGemmFwdTransformer;
    using AGridDescMK_t              = AGridDescMK;
    using CGridDescMN_t              = CGridDescMN;

    // Split-image support: Common data for all pieces
    struct SplitImageInfo
    {
        // Common dimensions (same for all pieces)
        index_t total_d = 1, total_h = 1, total_w = 1; // Total tensor dimensions
        index_t total_spatial = 1; // Pre-calculated: total_d * total_h * total_w
        index_t num_d_pieces = 1, num_h_pieces = 1, num_w_pieces = 1; // Split factors

        // Minimal per-piece data (only unique values)
        struct PieceInfo
        {
            index_t block_start;               // Starting block index for this piece
            index_t block_end;                 // Ending block index (exclusive)
            index_t d_start, h_start, w_start; // Piece starting position in OUTPUT space
            index_t d_size, h_size, w_size;    // Piece size in OUTPUT space
        };

        static constexpr index_t MaxPieces = 64; // Max pieces: 4 (1D), 16 (2D), 64 (3D)
        std::array<PieceInfo, MaxPieces> pieces; // Array of minimal piece descriptors
    };

    index_t num_spatial_pieces = 1; // Number of spatial pieces (1 = no split)
    SplitImageInfo split_image;     // Nested structure with common + per-piece data
};

/// @brief The Grouped Convolution Forward kernel template.
///
/// @paragraph Overview Overview
///            This class provides a unified grouped convolution forward kernel template
///            supporting both Implicit GEMM and WCNN paths.
///            The path is selected at compile time via the EnableWcnn trait.
///
///            For the GEMM path, the algorithm is divided into:
///            @li @b Prolog - The start of GEMM kernel implementation in @ref operator()
///                function call operator which determines the work scope of each workgroup.
///            @li @b GemmPipeline - The core part of matrix multiplication algorithm.
///                This is the place where each workgroup is loading data from global memory and
///                carrying out dot products.
///            @li @b Epilogue - The final part of matrix multiplication implementation
///                 responsible for storing results to global memory. This is also the place where
///                 any additional operator fusion may take place.
///
///            For the WCNN path, the pipeline operates on spatially-local tiles (H, W, K, C)
///            instead of GEMM tiles (M, N, K), using warp-level convolution instructions.
///
/// @tparam EnableWcnn                  Compile-time switch: false for Implicit GEMM path,
///                                     true for WCNN path.
/// @tparam GroupedConvTraitsType_       The type of class providing traits for grouped convolution.
///                                     EnableWcnn trait selects between GEMM and WCNN paths.
/// @tparam TilePartitioner_            The type of class providing mapping of workgroup index into
///                                     the output data tile to be calculated.
/// @tparam GemmPipeline_               For GEMM path: the core matrix multiplication pipeline.
///                                     For WCNN path: the warp convolution pipeline.
/// @tparam EpiloguePipeline_           The type of class providing the final part of the
///                                     computation, responsible for storing results to global
///                                     memory.
template <bool EnableWcnn,
          typename GroupedConvTraitsType_,
          typename TilePartitioner_,
          typename GemmPipeline_,
          typename EpiloguePipeline_>
struct GroupedConvForwardKernelConfig;

template <typename GroupedConvTraitsType_,
          typename TilePartitioner_,
          typename GemmPipeline_,
          typename EpiloguePipeline_>
struct GroupedConvForwardKernelConfig<false,
                                      GroupedConvTraitsType_,
                                      TilePartitioner_,
                                      GemmPipeline_,
                                      EpiloguePipeline_>
{
    using GemmALayout  = remove_cvref_t<typename GemmPipeline_::ALayout>;
    using GemmBLayout  = remove_cvref_t<typename GemmPipeline_::BLayout>;
    using GemmCLayout  = remove_cvref_t<typename GemmPipeline_::CLayout>;
    using GemmDsLayout = remove_cvref_t<typename EpiloguePipeline_::DsLayout>;

    using InDataType  = remove_cvref_t<typename GemmPipeline_::ADataType>;
    using WeiDataType = remove_cvref_t<typename GemmPipeline_::BDataType>;
    using DsDataType  = remove_cvref_t<typename EpiloguePipeline_::DsDataType>;
    using OutDataType = remove_cvref_t<typename EpiloguePipeline_::ODataType>;
    using AccDataType = OutDataType;

    using CDElementwise = typename EpiloguePipeline_::CDElementwise;
    using KernelArgs    = GroupedConvFwdKernelArgs<GroupedConvTraitsType_, CDElementwise>;

    static constexpr index_t kBlockSize = GemmPipeline_::BlockSize;

    static constexpr index_t HPerBlock = 1;
    static constexpr index_t WPerBlock = 1;
    static constexpr index_t KPerBlock = 1;
    static constexpr index_t CPerBlock = 1;
    static constexpr index_t HPerWcnn  = 1;
    static constexpr index_t WPerWcnn  = 1;
    static constexpr index_t FilterY   = 1;
    static constexpr index_t FilterX   = 1;

    // Static configuration validation (checked at class instantiation time)
    static_assert(GemmPipeline_::kPadM && GemmPipeline_::kPadN && GemmPipeline_::kPadK,
                  "Not supported!");
    static_assert(std::is_same_v<GemmALayout, tensor_layout::gemm::RowMajor> ||
                      GroupedConvTraitsType_::NumGroupsToMerge > 1,
                  "Not supported!");
    static_assert(std::is_same_v<GemmBLayout, tensor_layout::gemm::ColumnMajor>, "Not supported!");
    static_assert(std::is_same_v<GemmCLayout, tensor_layout::gemm::RowMajor>, "Not supported!");
    static_assert(GroupedConvTraitsType_::ExplicitGemm == false ||
                      GroupedConvTraitsType_::NumGroupsToMerge == 1,
                  "Not supported!");
};

template <typename GroupedConvTraitsType_,
          typename TilePartitioner_,
          typename WcnnPipeline_,
          typename EpiloguePipeline_>
struct GroupedConvForwardKernelConfig<true,
                                      GroupedConvTraitsType_,
                                      TilePartitioner_,
                                      WcnnPipeline_,
                                      EpiloguePipeline_>
{
    using GemmALayout  = void;
    using GemmBLayout  = void;
    using GemmCLayout  = void;
    using GemmDsLayout = void;

    using InDataType  = remove_cvref_t<typename WcnnPipeline_::ADataType>;
    using WeiDataType = remove_cvref_t<typename WcnnPipeline_::BDataType>;
    using AccDataType = remove_cvref_t<typename WcnnPipeline_::AccDataType>;
    using DsDataType  = tuple<>;
    using OutDataType = remove_cvref_t<typename EpiloguePipeline_::ODataType>;

    using CDElementwise = element_wise::PassThrough;
    using KernelArgs    = WcnnFwdKernelArgs<GroupedConvTraitsType_, CDElementwise>;

    static constexpr index_t kBlockSize = WcnnPipeline_::BlockSize;

    static constexpr index_t HPerBlock = WcnnPipeline_::HPerBlock;
    static constexpr index_t WPerBlock = WcnnPipeline_::WPerBlock;
    static constexpr index_t KPerBlock = WcnnPipeline_::KPerBlock;
    static constexpr index_t CPerBlock = WcnnPipeline_::CPerBlock;
    static constexpr index_t HPerWcnn  = WcnnPipeline_::HPerWcnn;
    static constexpr index_t WPerWcnn  = WcnnPipeline_::WPerWcnn;
    static constexpr index_t FilterY   = WcnnPipeline_::FilterY;
    static constexpr index_t FilterX   = WcnnPipeline_::FilterX;
};

template <typename GroupedConvTraitsType_,
          typename TilePartitioner_,
          typename GemmPipeline_,
          typename EpiloguePipeline_>
struct GroupedConvolutionForwardKernel
{
    static constexpr bool EnableWcnn       = GroupedConvTraitsType_::EnableWcnn;
    static constexpr bool EnableSplitImage = GroupedConvTraitsType_::EnableSplitImage;
    static constexpr index_t NDimSpatial   = GroupedConvTraitsType_::NDimSpatial;
    static constexpr ConvolutionSpecialization ConvSpecialization =
        GroupedConvTraitsType_::ConvSpecialization;
    using TilePartitioner  = remove_cvref_t<TilePartitioner_>;
    using GemmPipeline     = remove_cvref_t<GemmPipeline_>;
    using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;
    using KernelConfig     = GroupedConvForwardKernelConfig<EnableWcnn,
                                                            GroupedConvTraitsType_,
                                                            TilePartitioner_,
                                                            GemmPipeline_,
                                                            EpiloguePipeline_>;
    using GemmALayout      = typename KernelConfig::GemmALayout;
    using GemmBLayout      = typename KernelConfig::GemmBLayout;
    using GemmCLayout      = typename KernelConfig::GemmCLayout;

    using InLayout  = remove_cvref_t<typename GroupedConvTraitsType_::InLayout>;
    using WeiLayout = remove_cvref_t<typename GroupedConvTraitsType_::WeiLayout>;
    using OutLayout = remove_cvref_t<typename GroupedConvTraitsType_::OutLayout>;
    using DsLayout  = remove_cvref_t<typename GroupedConvTraitsType_::DsLayout>;

    using GemmDsLayout                  = typename KernelConfig::GemmDsLayout;
    static constexpr index_t NumDTensor = GroupedConvTraitsType_::NumDTensor;

    static constexpr index_t kBlockSize = KernelConfig::kBlockSize;

    using InDataType  = typename KernelConfig::InDataType;
    using WeiDataType = typename KernelConfig::WeiDataType;
    using DsDataType  = typename KernelConfig::DsDataType;
    // Below type is actually accumulation data type - the output of block GEMM.
    using OutDataType = typename KernelConfig::OutDataType;
    using AccDataType = typename KernelConfig::AccDataType;

    using CDElementwise = typename KernelConfig::CDElementwise;

    using GroupedConvFwdKernelArgsSpecialized = typename KernelConfig::KernelArgs;

    static constexpr bool IsSplitKSupported = false;

    static constexpr auto I0 = number<0>();
    static constexpr auto I1 = number<1>();
    static constexpr auto I2 = number<2>();
    static constexpr auto I3 = number<3>();
    static constexpr auto I5 = number<5>();

    static constexpr index_t HPerBlock = KernelConfig::HPerBlock;
    static constexpr index_t WPerBlock = KernelConfig::WPerBlock;
    static constexpr index_t KPerBlock = KernelConfig::KPerBlock;
    static constexpr index_t CPerBlock = KernelConfig::CPerBlock;
    static constexpr index_t HPerWcnn  = KernelConfig::HPerWcnn;
    static constexpr index_t WPerWcnn  = KernelConfig::WPerWcnn;
    static constexpr index_t FilterY   = KernelConfig::FilterY;
    static constexpr index_t FilterX   = KernelConfig::FilterX;

    // Helper struct for spatial coordinates
    struct SpatialCoords
    {
        index_t d, h, w;
    };

    // Helper: Convert flat spatial index to (d,h,w) coordinates
    CK_TILE_DEVICE static SpatialCoords
    UnflattenSpatial(index_t flat, index_t h_size, index_t w_size)
    {
        if constexpr(NDimSpatial == 1)
        {
            return SpatialCoords{0, 0, flat};
        }
        else if constexpr(NDimSpatial == 2)
        {
            return SpatialCoords{0, flat / w_size, flat % w_size};
        }
        else // NDimSpatial == 3
        {
            const index_t hw        = h_size * w_size;
            const index_t d         = flat / hw;
            const index_t remainder = flat % hw;
            return SpatialCoords{d, remainder / w_size, remainder % w_size};
        }
    }

    // Helper: Convert (d,h,w) to flat spatial index
    CK_TILE_DEVICE static index_t
    FlattenSpatial(index_t d, index_t h, index_t w, index_t total_h, index_t total_w)
    {
        if constexpr(NDimSpatial == 1)
        {
            return w;
        }
        else if constexpr(NDimSpatial == 2)
        {
            return h * total_w + w;
        }
        else // NDimSpatial == 3
        {
            return (d * total_h + h) * total_w + w;
        }
    }

    // Helper: Find which piece owns a block using binary search
    template <typename SplitImageInfo>
    CK_TILE_DEVICE static index_t
    FindPieceId(index_t block_id, const SplitImageInfo& split_info, index_t num_pieces)
    {
        index_t left     = 0;
        index_t right    = num_pieces - 1;
        index_t piece_id = (left + right) / 2;

        while(!(block_id >= split_info.pieces[piece_id].block_start &&
                block_id < split_info.pieces[piece_id].block_end) &&
              left <= right)
        {
            if(block_id < split_info.pieces[piece_id].block_start)
            {
                right = piece_id - 1;
            }
            else
            {
                left = piece_id + 1;
            }
            piece_id = (left + right) / 2;
        }
        return piece_id;
    }

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        if constexpr(EnableWcnn)
        {
            // clang-format off
            return concat('_', "grouped_convolution_forward",
                "wcnn",
                gemm_prec_str<InDataType, WeiDataType>(),
                InLayout::name,
                WeiLayout::name,
                OutLayout::name,
                "tile", HPerBlock, WPerBlock, CPerBlock, KPerBlock
            );
            // clang-format on
        }
        else
        {
            constexpr auto NumGroupsToMerge = GroupedConvTraitsType_::NumGroupsToMerge;
            // clang-format off
            return concat('_', "grouped_convolution_forward",
                gemm_prec_str<InDataType, WeiDataType>(),
                InLayout::name,
                WeiLayout::name,
                OutLayout::name,
                "gemm",
                GemmPipeline::GetName(),
                "epilogue",
                EpiloguePipeline::GetName(),
                getConvSpecializationString(ConvSpecialization),
                "MergedGroups",
                NumGroupsToMerge,
                "SplitImage",
                EnableSplitImage,
                "ExplicitGemm",
                GroupedConvTraitsType_::ExplicitGemm
            );
            // clang-format on
        }
    }

    [[nodiscard]] CK_TILE_HOST static const std::string GetTypeString() { return GetName(); }

#ifdef CK_EXPERIMENTAL_BUILDER
    CK_TILE_HOST std::string GetInstanceString() const
    {
        if constexpr(EnableWcnn)
        {
            return GetName();
        }
        else
        {
            using Kernel = GroupedConvolutionForwardKernel<GroupedConvTraitsType_,
                                                           TilePartitioner_,
                                                           GemmPipeline_,
                                                           EpiloguePipeline_>;
            static_assert(ck_tile::reflect::HasInstanceTraits<Kernel>,
                          "Specialization of instance_traits not found. Please check that a "
                          "specialization exists in file "
                          "ck_tile/builder/reflect/"
                          "instance_traits_tile_grouped_convolution_forward.hpp "
                          "for the given template parameters.");
            return ck_tile::reflect::instance_string<Kernel>();
        }
    }
#endif

    CK_TILE_HOST static auto GridSize(const GroupedConvFwdKernelArgsSpecialized& kargs)
    {
        if constexpr(EnableWcnn)
        {
            const index_t grid_hw = TilePartitioner::GridSize(
                kargs.ConvH, kargs.output_spatial_lengths[NDimSpatial - 1]);
            const index_t grid_size_k = integer_divide_ceil(kargs.K, KPerBlock);
            return dim3(grid_hw * grid_size_k, kargs.G);
        }
        else
        {
            return dim3(TilePartitioner::GridSize(kargs.GemmM, kargs.GemmN),
                        kargs.GemmBatch,
                        kargs.n_splits);
        }
    }

    CK_TILE_HOST static auto BlockSize()
    {
        return is_wave32() ? dim3(kBlockSize / 2) : dim3(kBlockSize);
    }

    CK_TILE_HOST static constexpr GroupedConvFwdKernelArgsSpecialized
    MakeKernelArgs(const GroupedConvFwdHostArgs<CDElementwise>& hostArgs)
    {
        auto kargs = GroupedConvFwdKernelArgsSpecialized(hostArgs);
        return kargs;
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return max(GemmPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    CK_TILE_HOST static bool IsSupportedArgument(const GroupedConvFwdKernelArgsSpecialized& kargs)
    {
        if constexpr(EnableWcnn)
        {
            if constexpr(!(is_any_of<InDataType, fp16_t, bf16_t>::value))
            {
                return false;
            }

            if constexpr(!(is_any_of<AccDataType, float, fp16_t>::value))
            {
                return false;
            }

            namespace ctc = tensor_layout::convolution;
            if constexpr(NDimSpatial == 1)
            {
                static_assert(std::is_same_v<InLayout, ctc::NWGC>,
                              "Unsupported 1D input layout for WCNN");
                static_assert(std::is_same_v<WeiLayout, ctc::GKXC>,
                              "Unsupported 1D weight layout for WCNN");
                static_assert(std::is_same_v<OutLayout, ctc::NWGK>,
                              "Unsupported 1D output layout for WCNN");
            }
            else if constexpr(NDimSpatial == 2)
            {
                static_assert(std::is_same_v<InLayout, ctc::NHWGC>,
                              "Unsupported 2D input layout for WCNN");
                static_assert(std::is_same_v<WeiLayout, ctc::GKYXC>,
                              "Unsupported 2D weight layout for WCNN");
                static_assert(std::is_same_v<OutLayout, ctc::NHWGK>,
                              "Unsupported 2D output layout for WCNN");
            }

            for(index_t i = 0; i < NDimSpatial; ++i)
            {
                if(kargs.filter_spatial_lengths[i] < 1)
                {
                    return false;
                }
            }

            if(kargs.C % CPerBlock != 0)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    std::cout << "WcnnFwd: C=" << kargs.C
                              << " not aligned to CPerBlock=" << CPerBlock << std::endl;
                }
                return false;
            }

            return true;
        }
        else
        {
            if constexpr(GemmPipeline_::Async)
            {
                if(get_device_name() != "gfx950")
                {
                    return false;
                }
            }

            if constexpr((GroupedConvTraitsType_::VectorSizeC % 2 != 0 &&
                          is_any_of<OutDataType, fp16_t, bf16_t>::value) ||
                         !IsSplitKSupported)
            {
                if(kargs.k_batch != 1)
                {
                    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                    {
                        CK_TILE_ERROR("Conditions not met for Kbatch >1 !");
                    }
                    return false;
                }
            }

            const index_t ConvK = kargs.wei_g_k_c_xs_lengths[number<1>{}];
            const index_t ConvC = kargs.wei_g_k_c_xs_lengths[number<2>{}];

            // check ConvolutionSpecialization
            if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter1x1Stride1Pad0)
            {
                // check if it's 1x1, stride=1 conv
                for(index_t i = 0; i < NDimSpatial; ++i)
                {
                    const index_t SpatialDim = kargs.wei_g_k_c_xs_lengths[i + 3];
                    const index_t ConvStride = kargs.conv_filter_strides[i];
                    const index_t LeftPad    = kargs.input_left_pads[i];
                    const index_t RightPad   = kargs.input_right_pads[i];

                    if(!(SpatialDim == 1 && ConvStride == 1 && LeftPad == 0 && RightPad == 0))
                    {
                        return false;
                    }
                }
            }
            else if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter1x1Pad0)
            {
                // check if it's 1x1 conv
                for(index_t i = 0; i < NDimSpatial; ++i)
                {
                    const index_t SpatialDim = kargs.wei_g_k_c_xs_lengths[i + 3];
                    const index_t LeftPad    = kargs.input_left_pads[i];
                    const index_t RightPad   = kargs.input_right_pads[i];

                    if(!(SpatialDim == 1 && LeftPad == 0 && RightPad == 0))
                    {
                        return false;
                    }
                }
            }
            else if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter3x3)
            {
                if(ConvC != 1)
                {
                    return false;
                }
                for(index_t i = 0; i < NDimSpatial; ++i)
                {
                    const index_t filter_spatial_dim = kargs.wei_g_k_c_xs_lengths[i + I3];

                    if(filter_spatial_dim != I3)
                    {
                        return false;
                    }
                }
            }

            if constexpr(GroupedConvTraitsType_::ExplicitGemm &&
                         ConvSpecialization != ConvolutionSpecialization::Filter1x1Stride1Pad0)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR(
                        "Explicit Gemm is supported only for Filter1x1Stride1Pad0 specialization!");
                }
                return false;
            }

            namespace ctc = tensor_layout::convolution;

            if constexpr(std::is_same_v<InLayout, ctc::NWGC> ||
                         std::is_same_v<InLayout, ctc::NHWGC> ||
                         std::is_same_v<InLayout, ctc::NDHWGC>)
            {
                // Check access for A tensor
                if(ConvC % GroupedConvTraitsType_::VectorSizeA != 0 &&
                   GroupedConvTraitsType_::NumGroupsToMerge == 1)
                {
                    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                    {
                        CK_TILE_ERROR(
                            "Conv C is not a multiple of vector load size for input image!");
                    }
                    return false;
                }
                else if(GroupedConvTraitsType_::NumGroupsToMerge > 1)
                {
                    if(ConvC != 1)
                    {
                        if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                        {
                            CK_TILE_ERROR(
                                "ConvC must be equal to 1 for NumGroupsToMerge > 1 to allow "
                                "vector reads on group dimension!");
                        }
                        return false;
                    }

                    const index_t ConvG = kargs.wei_g_k_c_xs_lengths[number<0>{}];
                    if(ConvG % GroupedConvTraitsType_::NumGroupsToMerge != 0)
                    {
                        if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                        {
                            CK_TILE_ERROR("ConvG must be a multiple of NumGroupsToMerge!");
                        }
                        return false;
                    }
                }
            }
            else
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("Not supported input layout!");
                }
                return false;
            }

            // check vector access of B
            // FIXME: layout
            if constexpr(std::is_same_v<WeiLayout, ctc::GKXC> ||
                         std::is_same_v<WeiLayout, ctc::GKYXC> ||
                         std::is_same_v<WeiLayout, ctc::GKZYXC>)
            {
                if(ConvC % GroupedConvTraitsType_::VectorSizeB != 0)
                {
                    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                    {
                        CK_TILE_ERROR("Conv C is not a multiple of vector load size for weight!");
                    }
                    return false;
                }
            }
            else
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("Not supported weight layout!");
                }
                return false;
            }

            // check vector access of E
            if constexpr(std::is_same_v<OutLayout, ctc::NWGK> ||
                         std::is_same_v<OutLayout, ctc::NHWGK> ||
                         std::is_same_v<OutLayout, ctc::NDHWGK>)
            {
                if(ConvK % GroupedConvTraitsType_::VectorSizeC != 0)
                {
                    // Try to read over G
                    if(GroupedConvTraitsType_::NumGroupsToMerge > 1)
                    {
                        const index_t ConvG = kargs.wei_g_k_c_xs_lengths[number<0>{}];
                        if(ConvG % GroupedConvTraitsType_::NumGroupsToMerge != 0 ||
                           ConvG % GroupedConvTraitsType_::VectorSizeC != 0)
                        {
                            if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                            {
                                CK_TILE_ERROR(
                                    "ConvG must be a multiple of NumGroupsToMerge to allow "
                                    "writing over G dimension");
                            }
                            return false;
                        }
                    }
                    else
                    {
                        if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                        {
                            CK_TILE_ERROR(
                                "ConvK is not a multiple of vector store size for output image!");
                        }
                        return false;
                    }
                }
            }
            else
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("Not supported output layout!");
                }
                return false;
            }

            if constexpr(GroupedConvTraitsType_::NumGroupsToMerge > 1)
            {
                // currently group merging works only for C == 1 due to tensor transformation
                // limitations
                if(ConvC != 1)
                {
                    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                    {
                        CK_TILE_ERROR("ConvC must be equal to 1 for NumGroupsToMerge > 1 to allow "
                                      "vector reads on group dimension!");
                    }
                    return false;
                }

                const index_t ConvG = kargs.wei_g_k_c_xs_lengths[number<0>{}];
                if(ConvG % GroupedConvTraitsType_::NumGroupsToMerge != 0)
                {
                    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                    {
                        CK_TILE_ERROR("ConvG must be a multiple of NumGroupsToMerge!");
                    }
                    return false;
                }
            }

            return true;
        }
    }

    /// @brief Creates the A (input) tensor block window.
    /// @param a_ptr       Pointer to input tensor in global memory.
    /// @param a_desc      Tensor descriptor (AGridDescHWC for WCNN, AGridDescMK for GEMM).
    /// @param block_idx_0 WCNN: h_block_start; GEMM: block_idx_m.
    /// @param block_idx_1 WCNN: w_block_start; GEMM: unused (default 0).
    template <typename ADescType>
    CK_TILE_DEVICE static auto MakeABlockWindow(const InDataType* a_ptr,
                                                const ADescType& a_desc,
                                                const index_t block_idx_0,
                                                const index_t block_idx_1 = 0)
    {
        if constexpr(EnableWcnn)
        {
            // block_idx_0 = h_block_start, block_idx_1 = w_block_start
            const auto a_tensor_view = make_tensor_view<address_space_enum::global>(a_ptr, a_desc);

            const auto a_pad_view = pad_tensor_view(
                a_tensor_view,
                make_tuple(number<HPerBlock>{}, number<WPerBlock>{}, number<CPerBlock>{}),
                sequence<GemmPipeline::kPadH, GemmPipeline::kPadW, false>{});

            const auto padded_h = a_pad_view.get_tensor_descriptor().get_length(number<0>{});
            const auto padded_w = a_pad_view.get_tensor_descriptor().get_length(number<1>{});
            const auto padded_c = a_pad_view.get_tensor_descriptor().get_length(number<2>{});

            const auto num_h_tiles = padded_h / HPerBlock;
            const auto num_w_tiles = padded_w / WPerBlock;

            const auto a_7d_view = transform_tensor_view(
                a_pad_view,
                make_tuple(make_unmerge_transform(make_tuple(
                               num_h_tiles, number<HPerBlock / HPerWcnn>{}, number<HPerWcnn>{})),
                           make_unmerge_transform(make_tuple(
                               num_w_tiles, number<WPerBlock / WPerWcnn>{}, number<WPerWcnn>{})),
                           make_pass_through_transform(padded_c)),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                make_tuple(sequence<0, 1, 2>{}, sequence<3, 4, 5>{}, sequence<6>{}));

            const auto a_2d_view = transform_tensor_view(
                a_7d_view,
                make_tuple(make_merge_transform(make_tuple(num_h_tiles,
                                                           num_w_tiles,
                                                           number<HPerBlock / HPerWcnn>{},
                                                           number<WPerBlock / WPerWcnn>{},
                                                           number<HPerWcnn>{},
                                                           number<WPerWcnn>{})),
                           make_pass_through_transform(padded_c)),
                make_tuple(sequence<0, 3, 1, 4, 2, 5>{}, sequence<6>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            const index_t h_tile_idx = block_idx_0 / HPerBlock;
            const index_t w_tile_idx = block_idx_1 / WPerBlock;
            const index_t hw_block_start =
                (h_tile_idx * num_w_tiles + w_tile_idx) * (HPerBlock * WPerBlock);

            return make_tile_window(
                a_2d_view,
                make_tuple(number<HPerBlock * WPerBlock>{}, number<CPerBlock>{}),
                {hw_block_start, 0});
        }
        else if constexpr(GroupedConvTraitsType_::NumGroupsToMerge == 1)
        {
            // Access by K
            // Step 1: Create tensor view
            const auto& a_tensor_view = make_tensor_view<address_space_enum::global>(a_ptr, a_desc);

            // Step 2: Create padded view
            const auto& a_pad_view =
                pad_tensor_view(a_tensor_view,
                                make_tuple(number<TilePartitioner::MPerBlock>{},
                                           number<TilePartitioner::KPerBlock>{}),
                                sequence<true, true>{});

            // Step 3: Create tile window
            return make_tile_window(a_pad_view,
                                    make_tuple(number<TilePartitioner::MPerBlock>{},
                                               number<TilePartitioner::KPerBlock>{}),
                                    {block_idx_0, 0});
        }
        else
        {
            // Access by M
            const auto a_desc_reversed = transform_tensor_descriptor(
                a_desc,
                make_tuple(make_pass_through_transform(a_desc.get_length(I0)),
                           make_pass_through_transform(a_desc.get_length(I1))),
                make_tuple(sequence<0>{}, sequence<1>{}),
                make_tuple(sequence<1>{}, sequence<0>{}));
            // Step 1: Create tensor view
            const auto& a_tensor_view =
                make_tensor_view<address_space_enum::global>(a_ptr, a_desc_reversed);

            // Step 2: Create padded view
            const auto& a_pad_view =
                pad_tensor_view(a_tensor_view,
                                make_tuple(number<TilePartitioner::KPerBlock>{},
                                           number<TilePartitioner::MPerBlock>{}),
                                sequence<true, true>{});

            // Step 3: Create tile window
            return make_tile_window(a_pad_view,
                                    make_tuple(number<TilePartitioner::KPerBlock>{},
                                               number<TilePartitioner::MPerBlock>{}),
                                    {0, block_idx_0});
        }
    }

    /// @brief Creates the B (weight) tensor block window.
    /// @param b_ptr    Pointer to weight tensor in global memory.
    /// @param b_desc   Tensor descriptor (BGridDescKYXC for WCNN, BGridDescNK for GEMM).
    /// @param block_idx WCNN: k_block_start; GEMM: block_idx_n.
    template <typename BDescType>
    CK_TILE_DEVICE static auto
    MakeBBlockWindow(const WeiDataType* b_ptr, const BDescType& b_desc, const index_t block_idx)
    {
        if constexpr(EnableWcnn)
        {
            // block_idx = k_block_start
            const auto b_tensor_view = make_tensor_view<address_space_enum::global>(b_ptr, b_desc);

            const auto b_pad_view = pad_tensor_view(
                b_tensor_view,
                make_tuple(number<KPerBlock>{}, number<FilterY * FilterX>{}, number<CPerBlock>{}),
                sequence<GemmPipeline::kPadK, false, false>{});

            const auto b_2d_view = transform_tensor_view(
                b_pad_view,
                make_tuple(make_merge_transform(make_tuple(
                               b_pad_view.get_tensor_descriptor().get_length(number<0>{}),
                               b_pad_view.get_tensor_descriptor().get_length(number<1>{}))),
                           make_pass_through_transform(
                               b_pad_view.get_tensor_descriptor().get_length(number<2>{}))),
                make_tuple(sequence<0, 1>{}, sequence<2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            const index_t kyx_block_start =
                block_idx * b_pad_view.get_tensor_descriptor().get_length(number<1>{});

            return make_tile_window(
                b_2d_view,
                make_tuple(number<KPerBlock * FilterY * FilterX>{}, number<CPerBlock>{}),
                {kyx_block_start, 0});
        }
        else
        {
            // block_idx = block_idx_n
            // Step 1: Create tensor view
            const auto& b_tensor_view = make_tensor_view<address_space_enum::global>(b_ptr, b_desc);

            // Step 2: Create padded view
            const auto& b_pad_view =
                pad_tensor_view(b_tensor_view,
                                make_tuple(number<TilePartitioner::NPerBlock>{},
                                           number<TilePartitioner::KPerBlock>{}),
                                sequence<true, true>{});

            // Step 3: Create tile window
            return make_tile_window(b_pad_view,
                                    make_tuple(number<TilePartitioner::NPerBlock>{},
                                               number<TilePartitioner::KPerBlock>{}),
                                    {block_idx, 0});
        }
    }

    template <typename CDescType>
    CK_TILE_DEVICE static auto MakeDBlockWindows(const std::array<const void*, NumDTensor>& ds_ptr,
                                                 const CDescType& c_desc,
                                                 const index_t block_idx_m,
                                                 const index_t block_idx_n)
    {
        // Step 1: Create tensor views
        const auto& ds_tensor_view = generate_tuple(
            [&](auto i) {
                static_assert(std::is_same_v<std::tuple_element_t<i, DsLayout>, OutLayout>,
                              "Not supported!");
                static_assert(std::is_same_v<GemmCLayout, tensor_layout::gemm::RowMajor>,
                              "Not supported!");
                static_assert(std::is_same_v<std::tuple_element_t<i, DsDataType>, OutDataType>,
                              "Not supported!");

                return make_tensor_view<address_space_enum::global>(
                    static_cast<const OutDataType*>(ds_ptr[i]), c_desc);
            },
            number<NumDTensor>{});

        // Step 2: Create padded views
        const auto& ds_pad_view = generate_tuple(
            [&](auto i) {
                return pad_tensor_view(ds_tensor_view[i],
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::NPerBlock>{}),
                                       sequence<true, true>{});
            },
            number<NumDTensor>{});

        // Step 3: Create tile windows
        return generate_tuple(
            [&](auto i) {
                return make_tile_window(ds_pad_view[i],
                                        make_tuple(number<TilePartitioner::MPerBlock>{},
                                                   number<TilePartitioner::NPerBlock>{}),
                                        {block_idx_m, block_idx_n});
            },
            number<NumDTensor>{});
    }

    /// @brief Creates the C (output) tensor block window.
    /// @param c_ptr       Pointer to output tensor in global memory.
    /// @param c_desc      Tensor descriptor (CGridDescHWK for WCNN, CGridDescMN for GEMM).
    /// @param block_idx_0 WCNN: h_block_start; GEMM: block_idx_m.
    /// @param block_idx_1 WCNN: w_block_start; GEMM: block_idx_n.
    /// @param block_idx_2 WCNN: k_block_start; GEMM: unused (default 0).
    template <memory_operation_enum DstInMemOp = memory_operation_enum::set, typename CDescType>
    CK_TILE_DEVICE static auto MakeCBlockWindow(OutDataType* c_ptr,
                                                const CDescType& c_desc,
                                                const index_t block_idx_0,
                                                const index_t block_idx_1,
                                                const index_t block_idx_2 = 0)
    {
        if constexpr(EnableWcnn)
        {
            // block_idx_0 = h_block_start, block_idx_1 = w_block_start, block_idx_2 = k_block_start
            const auto c_tensor_view =
                make_tensor_view<address_space_enum::global, DstInMemOp>(c_ptr, c_desc);

            const auto c_pad_view = pad_tensor_view(
                c_tensor_view,
                make_tuple(number<HPerBlock>{}, number<WPerBlock>{}, number<KPerBlock>{}),
                sequence<GemmPipeline::kPadH, GemmPipeline::kPadW, GemmPipeline::kPadK>{});

            return make_tile_window(
                c_pad_view,
                make_tuple(number<HPerBlock>{}, number<WPerBlock>{}, number<KPerBlock>{}),
                {block_idx_0, block_idx_1, block_idx_2});
        }
        else
        {
            // block_idx_0 = block_idx_m, block_idx_1 = block_idx_n
            const auto& c_tensor_view =
                make_tensor_view<address_space_enum::global, DstInMemOp>(c_ptr, c_desc);

            const auto& c_pad_view =
                pad_tensor_view(c_tensor_view,
                                make_tuple(number<TilePartitioner::MPerBlock>{},
                                           number<TilePartitioner::NPerBlock>{}),
                                sequence<true, true>{});

            return make_tile_window(c_pad_view,
                                    make_tuple(number<TilePartitioner::MPerBlock>{},
                                               number<TilePartitioner::NPerBlock>{}),
                                    {block_idx_0, block_idx_1});
        }
    }

    /**
     * @brief Runs single Implicit GEMM problem cooperatively by whole workgroup.
     *        Used only for the non-WCNN (Implicit GEMM) path.
     *
     * @param a_ptr input A pointer (with group and batch offsets applied)
     * @param b_ptr input B pointer (with group offset applied)
     * @param ds_ptr input D tensors pointer array
     * @param c_ptr output C pointer (with group and batch offsets applied)
     * @param smem_ptr_0 The start memory pointer of the shared memory block.
     * @param a_desc Input tensor A descriptor (M x K)
     * @param b_desc Weight tensor B descriptor (N x K)
     * @param c_desc Output tensor C descriptor (M x N)
     * @param gemm_k The GEMM K dimension
     * @param k_batch The K batch parameter for split-K
     * @param block_idx_m The GEMM's output M dimension tile index processed by this workgroup.
     * @param block_idx_n The GEMM's output N dimension tile index processed by this workgroup.
     * @param elfunc The elementwise function for fused operations.
     */
    template <typename ADescType, typename BDescType, typename CDescType>
    CK_TILE_DEVICE static void RunGemm(const InDataType* a_ptr,
                                       const WeiDataType* b_ptr,
                                       const std::array<const void*, NumDTensor>& ds_ptr,
                                       OutDataType* c_ptr,
                                       void* smem_ptr_0,
                                       const ADescType& a_desc,
                                       const BDescType& b_desc,
                                       const CDescType& c_desc,
                                       const index_t gemm_k,
                                       const index_t k_batch,
                                       const index_t block_idx_m,
                                       const index_t block_idx_n,
                                       const CDElementwise& elfunc)
    {
        // Create block windows using specialized methods
        const auto& a_block_window  = MakeABlockWindow(a_ptr, a_desc, block_idx_m);
        const auto& b_block_window  = MakeBBlockWindow(b_ptr, b_desc, block_idx_n);
        const auto& ds_block_window = MakeDBlockWindows(ds_ptr, c_desc, block_idx_m, block_idx_n);

        const index_t num_loop = amd_wave_read_first_lane(TilePartitioner::GetLoopNum(gemm_k));

        // Run GEMM cooperatively by whole workgroup.
        const auto& c_block_tile = GemmPipeline{}.template operator()(
            a_block_window, b_block_window, num_loop, smem_ptr_0);

        // Run Epilogue Pipeline with k_batch dispatching
        if(k_batch == 1)
        {
            auto c_block_window = MakeCBlockWindow<memory_operation_enum::set>(
                c_ptr, c_desc, block_idx_m, block_idx_n);

            EpiloguePipeline{elfunc}
                .template operator()<decltype(c_block_window), decltype(c_block_tile)>(
                    c_block_window, c_block_tile, ds_block_window, smem_ptr_0);
        }
        else
        {
            if constexpr(!(GroupedConvTraitsType_::VectorSizeC % 2 != 0 &&
                           is_any_of<OutDataType, fp16_t, bf16_t>::value) &&
                         IsSplitKSupported)
            {
                auto c_block_window = MakeCBlockWindow<memory_operation_enum::atomic_add>(
                    c_ptr, c_desc, block_idx_m, block_idx_n);

                EpiloguePipeline{elfunc}
                    .template operator()<decltype(c_block_window), decltype(c_block_tile)>(
                        c_block_window, c_block_tile, ds_block_window, smem_ptr_0);
            }
        }
    }

    CK_TILE_DEVICE void CallExplicitGemm(GroupedConvFwdKernelArgsSpecialized& kargs) const
    {
        static_assert(NumDTensor == 0, "Not supported!");
        using ExplicitBatchedGemmKernel =
            BatchedGemmKernel<TilePartitioner, GemmPipeline, EpiloguePipeline>;
        const auto batched_gemm_kargs = typename ExplicitBatchedGemmKernel::BatchedGemmKernelArgs{
            {{kargs.in_ptr},
             {kargs.wei_ptr},
             {},
             kargs.out_ptr,
             kargs.GemmM,
             kargs.GemmN,
             kargs.GemmK,
             {kargs.GemmK * kargs.GemmBatch},
             {kargs.GemmK},
             {},
             kargs.GemmBatch * kargs.GemmN,
             kargs.k_batch},
            kargs.GemmK,
            kargs.GemmN * kargs.GemmK,
            kargs.GemmN,
            kargs.GemmBatch};
        ExplicitBatchedGemmKernel{}(batched_gemm_kargs);
    }

    CK_TILE_DEVICE void RunWcnn(GroupedConvFwdKernelArgsSpecialized& kargs) const
    {
        const auto blockIdX = amd_wave_read_first_lane(blockIdx.x);
        const auto blockIdY = amd_wave_read_first_lane(blockIdx.y);

        const auto group_offset_in =
            amd_wave_read_first_lane(static_cast<long_index_t>(blockIdY) * kargs.group_stride_in);
        const auto group_offset_wei =
            amd_wave_read_first_lane(static_cast<long_index_t>(blockIdY) * kargs.group_stride_wei);
        const auto group_offset_out =
            amd_wave_read_first_lane(static_cast<long_index_t>(blockIdY) * kargs.group_stride_out);

        const InDataType* in_ptr = static_cast<const InDataType*>(kargs.in_ptr) + group_offset_in;
        const WeiDataType* wei_ptr =
            static_cast<const WeiDataType*>(kargs.wei_ptr) + group_offset_wei;
        OutDataType* out_ptr = static_cast<OutDataType*>(kargs.out_ptr) + group_offset_out;

        const index_t grid_size_k = integer_divide_ceil(kargs.K, KPerBlock);

        const index_t block_hw_k  = blockIdX;
        const index_t block_idx_k = block_hw_k % grid_size_k;
        const index_t block_hw    = block_hw_k / grid_size_k;
        const auto [block_idx_h, block_idx_w] =
            TilePartitioner{kargs.ConvH, kargs.output_spatial_lengths[NDimSpatial - 1]}
                .GetOutputTileIndex(block_hw);

        const index_t h_block_start = amd_wave_read_first_lane(block_idx_h * HPerBlock);
        const index_t w_block_start = amd_wave_read_first_lane(block_idx_w * WPerBlock);
        const index_t k_block_start = amd_wave_read_first_lane(block_idx_k * KPerBlock);

        const auto a_block_window =
            MakeABlockWindow(in_ptr, kargs.a_grid_desc_h_w_c, h_block_start, w_block_start);
        const auto b_block_window =
            MakeBBlockWindow(wei_ptr, kargs.b_grid_desc_k_yx_c, k_block_start);
        auto c_block_window = MakeCBlockWindow(
            out_ptr, kargs.c_grid_desc_h_w_k, h_block_start, w_block_start, k_block_start);

        const index_t num_loop = amd_wave_read_first_lane(TilePartitioner::GetLoopNum(kargs.C));

        __shared__ char smem_ptr[GetSmemSize()];

        const auto c_block_tile =
            GemmPipeline{}(a_block_window, b_block_window, num_loop, smem_ptr);

        EpiloguePipeline{}(c_block_window, c_block_tile, nullptr, smem_ptr);
    }

    CK_TILE_DEVICE void operator()(GroupedConvFwdKernelArgsSpecialized& kargs) const
    {
        if constexpr(EnableWcnn)
        {
            RunWcnn(kargs);
        }
        else if constexpr(GroupedConvTraitsType_::ExplicitGemm)
        {
            CallExplicitGemm(kargs);
        }
        else
        {
            const auto blockIdX = amd_wave_read_first_lane(blockIdx.x);
            const auto blockIdY = amd_wave_read_first_lane(blockIdx.y);

            const auto group_offset_a = amd_wave_read_first_lane(kargs.group_stride_a * blockIdY);
            const auto group_offset_b = amd_wave_read_first_lane(kargs.group_stride_b * blockIdY);
            const auto group_offset_c = amd_wave_read_first_lane(kargs.group_stride_c * blockIdY);

            // Split-N handling: Get which split this workgroup handles
            const auto blockIdZ = amd_wave_read_first_lane(blockIdx.z);

            // Calculate batch offset for this split
            const index_t batch_offset = amd_wave_read_first_lane(blockIdZ * kargs.n_per_split);

            // Calculate memory offsets for this split
            const long_index_t input_batch_offset =
                static_cast<long_index_t>(batch_offset) *
                static_cast<long_index_t>(kargs.input_batch_stride);
            const long_index_t output_batch_offset =
                static_cast<long_index_t>(batch_offset) *
                static_cast<long_index_t>(kargs.output_batch_stride);

            // Calculate base pointers with group and batch offsets
            const InDataType* base_a_ptr =
                static_cast<const InDataType*>(kargs.in_ptr) + group_offset_a + input_batch_offset;
            const WeiDataType* b_ptr = static_cast<const WeiDataType*>(kargs.wei_ptr) +
                                       group_offset_b; // No batch offset for weights!
            OutDataType* base_c_ptr =
                static_cast<OutDataType*>(kargs.out_ptr) + group_offset_c + output_batch_offset;

            // Apply group offsets to D tensors
            std::array<const void*, NumDTensor> ds_ptr_with_offsets;
            static_for<0, NumDTensor, 1>{}([&](auto d) {
                using DType            = std::tuple_element_t<d, DsDataType>;
                ds_ptr_with_offsets[d] = static_cast<const DType*>(kargs.ds_ptr[d]) +
                                         group_offset_c + output_batch_offset;
            });

            // =====================================================================
            // Split-image: Map local block to global tile index (if enabled)
            // =====================================================================
            const InDataType* a_ptr;
            OutDataType* c_ptr;
            index_t i_m = 0;
            index_t i_n = 0;

            // Pre-calculate block_id (used in both split-image and non-split paths)
            const index_t block_id = static_cast<index_t>(blockIdX);

            if constexpr(EnableSplitImage)
            {
                // Add spatial offsets for split-image (constexpr optimization)
                a_ptr = base_a_ptr + kargs.spatial_offset_in;
                c_ptr = base_c_ptr + kargs.spatial_offset_out;

                // Find which piece owns this block using binary search
                // Reference: device_grouped_conv_fwd_multiple_d_xdl_large_tensor_cshuffle.hpp
                const index_t piece_id =
                    FindPieceId(block_id, kargs.split_image, kargs.num_spatial_pieces);
                const auto& piece      = kargs.split_image.pieces[piece_id];
                const auto& split_info = kargs.split_image;

                // Calculate local block ID and tile indices
                const index_t local_block_id = block_id - piece.block_start;
                const index_t local_gemm_m =
                    kargs.n_per_split * piece.d_size * piece.h_size * piece.w_size;
                const auto [local_tile_m, local_tile_n] =
                    TilePartitioner{local_gemm_m, kargs.GemmN}.GetOutputTileIndex(local_block_id);

                // Extract batch and spatial coordinates from local tile
                const index_t local_m_start      = local_tile_m * TilePartitioner::MPerBlock;
                const index_t spatial_per_batch  = piece.d_size * piece.h_size * piece.w_size;
                const index_t local_n            = local_m_start / spatial_per_batch;
                const index_t local_spatial_flat = local_m_start % spatial_per_batch;

                // Convert to local spatial coordinates
                const auto local_coords =
                    UnflattenSpatial(local_spatial_flat, piece.h_size, piece.w_size);

                // Convert to global spatial coordinates
                const index_t global_n = local_n;
                const index_t global_d = piece.d_start + local_coords.d;
                const index_t global_h = piece.h_start + local_coords.h;
                const index_t global_w = piece.w_start + local_coords.w;

                // Convert to global M index
                const index_t global_spatial_per_batch = split_info.total_spatial; // Pre-calculated
                const index_t global_spatial_flat      = FlattenSpatial(
                    global_d, global_h, global_w, split_info.total_h, split_info.total_w);
                const index_t global_m = global_n * global_spatial_per_batch + global_spatial_flat;

                // Set tile indices for GEMM operation
                i_m = amd_wave_read_first_lane(global_m);
                i_n = amd_wave_read_first_lane(local_tile_n * TilePartitioner::NPerBlock);
            }
            else
            {
                // No spatial offsets needed for regular path
                a_ptr = base_a_ptr;
                c_ptr = base_c_ptr;

                // No split-image: use standard tile partitioning
                const auto [iM, iN] =
                    TilePartitioner{kargs.GemmM, kargs.GemmN}.GetOutputTileIndex(block_id);
                i_m = amd_wave_read_first_lane(iM * TilePartitioner::MPerBlock);
                i_n = amd_wave_read_first_lane(iN * TilePartitioner::NPerBlock);
            }

            // Use global descriptors for all cases
            const auto& a_desc = kargs.a_grid_desc_m_k;
            const auto& b_desc = kargs.b_grid_desc_n_k;
            const auto& c_desc = kargs.c_grid_desc_m_n;

            // allocate LDS
            __shared__ char smem_ptr[GetSmemSize()];

            // Disable Async for other archs than gfx950
            if constexpr(GemmPipeline_::Async)
            {
#if defined(__gfx950__)
                RunGemm(a_ptr,
                        b_ptr,
                        ds_ptr_with_offsets,
                        c_ptr,
                        smem_ptr,
                        a_desc,
                        b_desc,
                        c_desc,
                        kargs.GemmK,
                        kargs.k_batch,
                        i_m,
                        i_n,
                        kargs.elfunc);
#endif
            }
            else
            {
                RunGemm(a_ptr,
                        b_ptr,
                        ds_ptr_with_offsets,
                        c_ptr,
                        smem_ptr,
                        a_desc,
                        b_desc,
                        c_desc,
                        kargs.GemmK,
                        kargs.k_batch,
                        i_m,
                        i_n,
                        kargs.elfunc);
            }
        }
    }
};

} // namespace ck_tile
