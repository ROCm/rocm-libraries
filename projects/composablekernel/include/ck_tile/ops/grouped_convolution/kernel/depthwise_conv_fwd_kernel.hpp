// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/concat.hpp"
#include "ck_tile/host/convolution_parameter.hpp"
#include "ck_tile/ops/grouped_convolution/kernel/depthwise_conv_fwd_traits.hpp"

namespace ck_tile {

/// @brief Host-side arguments for depthwise convolution forward pass.
struct DepthwiseConvFwdHostArgs : public conv::ConvParam
{
    CK_TILE_HOST DepthwiseConvFwdHostArgs() = delete;

    CK_TILE_HOST DepthwiseConvFwdHostArgs(conv::ConvParam conv_param,
                                          const void* p_in_,
                                          const void* p_wei_,
                                          void* p_out_,
                                          std::array<index_t, 5> in_strides_,
                                          std::array<index_t, 5> wei_strides_,
                                          std::array<index_t, 5> out_strides_)
        : conv::ConvParam(conv_param),
          p_in(p_in_),
          p_wei(p_wei_),
          p_out(p_out_),
          in_strides(in_strides_),
          wei_strides(wei_strides_),
          out_strides(out_strides_)
    {
    }

    const void* p_in;
    const void* p_wei;
    void* p_out;

    // Stride-parametric: any layout with w_stride==1 (NGCHW, GNCHW, etc.).
    std::array<index_t, 5> in_strides;  // [g, n, c, h, w]
    std::array<index_t, 5> wei_strides; // [g, k, c, y, x]
    std::array<index_t, 5> out_strides; // [g, n, k, h, w]
};

/// @brief Device-side kernel arguments for depthwise convolution.
template <typename Traits_>
struct DepthwiseConvFwdKernelArgs
{
    using Traits      = Traits_;
    using InDataType  = typename Traits::InDataType;
    using WeiDataType = typename Traits::WeiDataType;
    using OutDataType = typename Traits::OutDataType;

    const InDataType* p_in;
    const WeiDataType* p_wei;
    OutDataType* p_out;

    index_t G;
    index_t N;
    index_t Hi;
    index_t Wi;
    index_t Ho;
    index_t Wo;

    index_t in_g_stride;
    index_t in_n_stride;
    index_t in_h_stride;
    index_t in_w_stride;

    index_t wei_g_stride;
    index_t wei_y_stride;
    index_t wei_x_stride;

    index_t out_g_stride;
    index_t out_n_stride;
    index_t out_h_stride;
    index_t out_w_stride;
};

/// @brief Depthwise convolution forward kernel.
template <typename Traits_, typename Pipeline_>
struct DepthwiseConvFwdKernel
{
    using Traits      = Traits_;
    using Pipeline    = Pipeline_;
    using InDataType  = typename Traits::InDataType;
    using WeiDataType = typename Traits::WeiDataType;
    using AccDataType = typename Traits::AccDataType;
    using OutDataType = typename Traits::OutDataType;

    using KernelArgs = DepthwiseConvFwdKernelArgs<Traits>;

    // Stride array dimension indices; TODO: derive from NDimSpatial for 1D/3D
    static constexpr index_t kDimG = 0;
    static constexpr index_t kDimN = 1;
    static constexpr index_t kDimC = 2;
    static constexpr index_t kDimH = 3;
    static constexpr index_t kDimW = 4;

    static constexpr index_t kSpatialH = 0;
    static constexpr index_t kSpatialW = 1;

    static constexpr index_t kBlockSize  = Traits::BlockSize;
    static constexpr index_t TileOutH    = Traits::TileOutH;
    static constexpr index_t TileOutW    = Traits::TileOutW;
    static constexpr index_t NBatch      = Traits::NBatch;
    static constexpr index_t TilePerWave = Traits::TilePerWave;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return Pipeline::GetSmemSize(); }

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "depthwise_conv_fwd",
            kBlockSize,
            concat('x', Traits::FilterH, Traits::FilterW),
            concat('x', Traits::StrideH, Traits::StrideW),
            concat('x', Traits::DilationH, Traits::DilationW),
            concat('x', Traits::PadH, Traits::PadW),
            NBatch,
            concat('x', Traits::SubTileH, Traits::SubTileW),
            concat('x', Traits::InVectorSize, Traits::OutVectorSize),
            concat('x', TileOutH, TileOutW));
        // clang-format on
    }

    CK_TILE_HOST static constexpr auto BlockSize() { return dim3(kBlockSize); }

    // grid.x = G, grid.y = ceil(N / NBatch)
    CK_TILE_HOST static auto GridSize(index_t G, index_t N)
    {
        const index_t num_batch_groups = integer_divide_ceil(N, NBatch);
        return dim3(G, num_batch_groups, 1);
    }

    CK_TILE_HOST static constexpr KernelArgs MakeKernelArgs(const DepthwiseConvFwdHostArgs& args)
    {
        KernelArgs kargs;

        kargs.p_in  = static_cast<const InDataType*>(args.p_in);
        kargs.p_wei = static_cast<const WeiDataType*>(args.p_wei);
        kargs.p_out = static_cast<OutDataType*>(args.p_out);

        kargs.G  = static_cast<index_t>(args.G_);
        kargs.N  = static_cast<index_t>(args.N_);
        kargs.Hi = static_cast<index_t>(args.input_spatial_lengths_[kSpatialH]);
        kargs.Wi = static_cast<index_t>(args.input_spatial_lengths_[kSpatialW]);
        kargs.Ho = static_cast<index_t>(args.output_spatial_lengths_[kSpatialH]);
        kargs.Wo = static_cast<index_t>(args.output_spatial_lengths_[kSpatialW]);

        kargs.in_g_stride = args.in_strides[kDimG];
        kargs.in_n_stride = args.in_strides[kDimN];
        kargs.in_h_stride = args.in_strides[kDimH];
        kargs.in_w_stride = args.in_strides[kDimW];

        kargs.wei_g_stride = args.wei_strides[kDimG];
        kargs.wei_y_stride = args.wei_strides[kDimH];
        kargs.wei_x_stride = args.wei_strides[kDimW];

        kargs.out_g_stride = args.out_strides[kDimG];
        kargs.out_n_stride = args.out_strides[kDimN];
        kargs.out_h_stride = args.out_strides[kDimH];
        kargs.out_w_stride = args.out_strides[kDimW];

        return kargs;
    }

    CK_TILE_HOST static bool IsSupportedArgument(const DepthwiseConvFwdHostArgs& args)
    {
        if constexpr(NBatch % Traits::TilePerWave != 0)
        {
            return false;
        }
        if constexpr(Traits::SubTileW * Traits::StrideW % Traits::InVectorSizeInternal != 0)
        {
            return false;
        }
        // TODO: support PadW==0 (valid convolution)
        if constexpr(Traits::PadW == 0)
        {
            return false;
        }
        if constexpr(integer_divide_ceil(Traits::LdsTileW, Traits::InVectorSize) >
                     Traits::BlockSize)
        {
            return false;
        }
        if constexpr(GetSmemSize() > get_smem_capacity())
        {
            return false;
        }

        // FIXME: known crash cases due to WriteDataToLds boundary issue
        static constexpr index_t kCrashBypassG1 = 192;
        static constexpr index_t kCrashBypassN1 = 1024;
        static constexpr index_t kCrashBypassG2 = 2048;
        static constexpr index_t kCrashBypassN2 = 32;
        if((args.G_ == kCrashBypassG1 && args.N_ == kCrashBypassN1) ||
           (args.G_ == kCrashBypassG2 && args.N_ == kCrashBypassN2))
        {
            return false;
        }
        if(args.C_ != 1 || args.K_ != 1)
        {
            return false;
        }
        if(static_cast<index_t>(args.filter_spatial_lengths_[kSpatialH]) != Traits::FilterH ||
           static_cast<index_t>(args.filter_spatial_lengths_[kSpatialW]) != Traits::FilterW)
        {
            return false;
        }
        if(static_cast<index_t>(args.conv_filter_strides_[kSpatialH]) != Traits::StrideH ||
           static_cast<index_t>(args.conv_filter_strides_[kSpatialW]) != Traits::StrideW)
        {
            return false;
        }
        if(static_cast<index_t>(args.conv_filter_dilations_[kSpatialH]) != Traits::DilationH ||
           static_cast<index_t>(args.conv_filter_dilations_[kSpatialW]) != Traits::DilationW)
        {
            return false;
        }
        if(static_cast<index_t>(args.input_left_pads_[kSpatialH]) != Traits::PadH ||
           static_cast<index_t>(args.input_left_pads_[kSpatialW]) != Traits::PadW)
        {
            return false;
        }
        // TODO: support asymmetric padding
        if(static_cast<index_t>(args.input_right_pads_[kSpatialH]) != Traits::PadH ||
           static_cast<index_t>(args.input_right_pads_[kSpatialW]) != Traits::PadW)
        {
            return false;
        }

        if(static_cast<index_t>(args.N_) % NBatch != 0)
        {
            return false;
        }
        // TilePerWave > 1 requires entire spatial output fits in one tile
        if constexpr(Traits::TilePerWave != 1)
        {
            if(static_cast<index_t>(args.output_spatial_lengths_[kSpatialH]) > Traits::TileOutH ||
               static_cast<index_t>(args.output_spatial_lengths_[kSpatialW]) > Traits::TileOutW)
            {
                return false;
            }
        }
        if(args.out_strides[kDimW] != 1)
        {
            return false;
        }
        if(args.input_spatial_lengths_[kSpatialH] < args.filter_spatial_lengths_[kSpatialH] ||
           args.input_spatial_lengths_[kSpatialW] < args.filter_spatial_lengths_[kSpatialW])
        {
            return false;
        }

        return true;
    }

    CK_TILE_DEVICE void operator()(KernelArgs& kargs) const
    {
        const index_t g_idx       = __builtin_amdgcn_readfirstlane(blockIdx.x);
        const index_t batch_group = __builtin_amdgcn_readfirstlane(blockIdx.y);

        const auto* p_in_base = kargs.p_in + static_cast<long_index_t>(g_idx) * kargs.in_g_stride +
                                static_cast<long_index_t>(batch_group * NBatch) * kargs.in_n_stride;

        const auto* p_wei_base =
            kargs.p_wei + static_cast<long_index_t>(g_idx) * kargs.wei_g_stride;

        auto* p_out_base = kargs.p_out + static_cast<long_index_t>(g_idx) * kargs.out_g_stride +
                           static_cast<long_index_t>(batch_group * NBatch) * kargs.out_n_stride;

        __shared__ char smem[GetSmemSize()];

        Pipeline{}(p_in_base,
                   p_wei_base,
                   p_out_base,
                   smem,
                   kargs.Hi,
                   kargs.Wi,
                   kargs.Ho,
                   kargs.Wo,
                   kargs.in_h_stride,
                   kargs.in_w_stride,
                   kargs.in_n_stride,
                   kargs.wei_y_stride,
                   kargs.wei_x_stride,
                   kargs.out_h_stride,
                   kargs.out_w_stride,
                   kargs.out_n_stride);
    }
};

} // namespace ck_tile
