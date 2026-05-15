// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <tuple>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/pooling.hpp"
#include "ck_tile/ops/pooling_bwd.hpp"
#include "ck_tile/host/reference/reference_pool_bwd.hpp"

template <typename Tuple>
class TestCkTilePoolingBwd : public ::testing::Test
{
    protected:
    using DInDataType     = std::tuple_element_t<0, Tuple>;
    using DOutDataType    = std::tuple_element_t<1, Tuple>;
    using IndexDataType   = std::tuple_element_t<2, Tuple>;
    using ComputeDataType = std::tuple_element_t<3, Tuple>;

    static constexpr ck_tile::index_t kBlockSize  = 256;
    static constexpr ck_tile::index_t kVectorSize = 4;

    using BwdShape = ck_tile::PoolBwdShape<kBlockSize, kVectorSize>;

    using FwdBlockWarps = ck_tile::sequence<1, 1>;
    using FwdBlockTile  = ck_tile::sequence<128, 1>;
    using FwdWarpTile   = ck_tile::sequence<128, 1>;
    using FwdThreadTile = ck_tile::sequence<2, 1>;
    using FwdShape = ck_tile::PoolShape<FwdBlockWarps, FwdBlockTile, FwdWarpTile, FwdThreadTile>;

    struct Config2D
    {
        ck_tile::index_t N, H, W, C;
        ck_tile::index_t Y, X;
        ck_tile::index_t Sy, Sx;
        ck_tile::index_t Dy, Dx;
        ck_tile::index_t LeftPy, LeftPx;
        ck_tile::index_t RightPy, RightPx;
        std::string name;
    };

    struct Config3D
    {
        ck_tile::index_t N, D, H, W, C;
        ck_tile::index_t Z, Y, X;
        ck_tile::index_t Sz, Sy, Sx;
        ck_tile::index_t Dz, Dy, Dx;
        ck_tile::index_t LeftPz, LeftPy, LeftPx;
        ck_tile::index_t RightPz, RightPy, RightPx;
        std::string name;
    };

    static double rtol_for_dtype()
    {
        if constexpr(std::is_same_v<DInDataType, float>)
            return 1e-5;
        else if constexpr(std::is_same_v<DInDataType, ck_tile::half_t> ||
                          std::is_same_v<DInDataType, ck_tile::bf16_t>)
            return 1e-2;
        else
            return 1e-1;
    }

    static double atol_for_dtype()
    {
        if constexpr(std::is_same_v<DInDataType, float>)
            return 1e-5;
        else if constexpr(std::is_same_v<DInDataType, ck_tile::half_t> ||
                          std::is_same_v<DInDataType, ck_tile::bf16_t>)
            return 1e-2;
        else
            return 1e-1;
    }

    template <bool kHasOverlap>
    bool RunPool2D_HasOverlap(const Config2D& cfg)
    {
        using InDataType  = DInDataType;
        using OutDataType = DOutDataType;

        const ck_tile::index_t Ys = (cfg.Y - 1) * cfg.Dy + 1;
        const ck_tile::index_t Xs = (cfg.X - 1) * cfg.Dx + 1;
        const ck_tile::index_t Ho = (cfg.H + cfg.LeftPy + cfg.RightPy - Ys) / cfg.Sy + 1;
        const ck_tile::index_t Wo = (cfg.W + cfg.LeftPx + cfg.RightPx - Xs) / cfg.Sx + 1;

        const auto input_shape  = ck_tile::make_tuple(cfg.N, cfg.H, cfg.W, cfg.C);
        const auto output_shape = ck_tile::make_tuple(cfg.N, Ho, Wo, cfg.C);
        const auto input_strides =
            ck_tile::make_tuple(cfg.H * cfg.W * cfg.C, cfg.W * cfg.C, cfg.C, 1);
        const auto output_strides = ck_tile::make_tuple(Ho * Wo * cfg.C, Wo * cfg.C, cfg.C, 1);
        const auto window_lengths = ck_tile::make_tuple(cfg.Y, cfg.X);
        const auto window_strides = ck_tile::make_tuple(cfg.Sy, cfg.Sx);
        const auto window_dils    = ck_tile::make_tuple(cfg.Dy, cfg.Dx);
        const auto left_pads      = ck_tile::make_tuple(cfg.LeftPy, cfg.LeftPx);
        const auto right_pads     = ck_tile::make_tuple(cfg.RightPy, cfg.RightPx);

        ck_tile::HostTensor<InDataType> h_in({cfg.N, cfg.H, cfg.W, cfg.C});
        ck_tile::HostTensor<OutDataType> h_out({cfg.N, Ho, Wo, cfg.C});
        ck_tile::HostTensor<IndexDataType> h_indices({cfg.N, Ho, Wo, cfg.C});
        ck_tile::HostTensor<DOutDataType> h_dout({cfg.N, Ho, Wo, cfg.C});
        ck_tile::HostTensor<DInDataType> h_din({cfg.N, cfg.H, cfg.W, cfg.C});
        ck_tile::HostTensor<DInDataType> h_din_ref({cfg.N, cfg.H, cfg.W, cfg.C});

        ck_tile::FillUniformDistribution<InDataType>{-5.f, 5.f}(h_in);
        ck_tile::FillUniformDistribution<DOutDataType>{-1.f, 1.f}(h_dout);

        ck_tile::DeviceMem in_buf(h_in.get_element_space_size_in_bytes());
        ck_tile::DeviceMem out_buf(h_out.get_element_space_size_in_bytes());
        ck_tile::DeviceMem indices_buf(h_indices.get_element_space_size_in_bytes());
        ck_tile::DeviceMem dout_buf(h_dout.get_element_space_size_in_bytes());
        ck_tile::DeviceMem din_buf(h_din.get_element_space_size_in_bytes());

        in_buf.ToDevice(h_in.data());
        dout_buf.ToDevice(h_dout.data());

        constexpr bool kPropagateNan = false;
        using ReduceOp               = ck_tile::ReduceOp::Max;
        using FwdProblem             = ck_tile::PoolProblem<InDataType,
                                                            OutDataType,
                                                            ComputeDataType,
                                                            IndexDataType,
                                                            ReduceOp,
                                                            true,
                                                            kPropagateNan,
                                                            FwdShape>;
        using FwdKernel              = ck_tile::PoolKernel<FwdProblem>;

        auto fwd_host_args = ck_tile::PoolHostArgs<decltype(input_shape), decltype(window_lengths)>{
            static_cast<InDataType*>(in_buf.GetDeviceBuffer()),
            static_cast<OutDataType*>(out_buf.GetDeviceBuffer()),
            static_cast<IndexDataType*>(indices_buf.GetDeviceBuffer()),
            input_shape,
            output_shape,
            input_strides,
            output_strides,
            window_lengths,
            window_strides,
            window_dils,
            left_pads,
            right_pads};
        auto fwd_kargs = FwdKernel::MakeKernelArgs(fwd_host_args);

        if(!FwdKernel::IsSupportedArgument(fwd_kargs))
        {
            return true;
        }

        constexpr ck_tile::index_t kFwdBlockPerCu = 1;
        const ck_tile::index_t fwd_grid_size      = FwdKernel::CalculateGridSize(fwd_kargs);
        const ck_tile::index_t fwd_block_size     = FwdKernel::BlockSize();

        ck_tile::launch_kernel(ck_tile::stream_config{nullptr, false, 0},
                               ck_tile::make_kernel<kFwdBlockPerCu>(
                                   FwdKernel{}, fwd_grid_size, fwd_block_size, 0, fwd_kargs));

        using BwdProblem = ck_tile::
            PoolBwdProblem<DOutDataType, IndexDataType, DInDataType, BwdShape, kHasOverlap>;
        using BwdKernel = ck_tile::PoolBwdKernel<BwdProblem>;

        const ck_tile::long_index_t din_length =
            static_cast<ck_tile::long_index_t>(h_din.get_element_space_size());
        const ck_tile::long_index_t dout_length =
            static_cast<ck_tile::long_index_t>(h_dout.get_element_space_size());

        const std::size_t workspace_bytes = BwdKernel::GetWorkSpaceSize(ck_tile::PoolBwdHostArgs{
            nullptr, nullptr, nullptr, nullptr, dout_length, din_length, kHasOverlap});

        ck_tile::DeviceMem workspace_buf(workspace_bytes);

        auto bwd_host_args = ck_tile::PoolBwdHostArgs{
            static_cast<DOutDataType*>(dout_buf.GetDeviceBuffer()),
            static_cast<IndexDataType*>(indices_buf.GetDeviceBuffer()),
            static_cast<DInDataType*>(din_buf.GetDeviceBuffer()),
            workspace_bytes > 0 ? workspace_buf.GetDeviceBuffer() : nullptr,
            dout_length,
            din_length,
            kHasOverlap};

        if(!BwdKernel::IsSupportedArgument(bwd_host_args))
        {
            return true;
        }

        auto bwd_kargs = BwdKernel::MakeKernelArgs(bwd_host_args);

        constexpr ck_tile::index_t kBwdBlockPerCu = 1;
        const ck_tile::index_t bwd_block_size     = BwdKernel::BlockSize();
        auto stream                               = ck_tile::stream_config{nullptr, false, 0};
        const ck_tile::index_t bwd_grid_size = BwdKernel::CalculateGridSize(stream, dout_length);

        auto memset_din = [&](const ck_tile::stream_config& s) {
            HIP_CHECK_ERROR(
                hipMemsetAsync(din_buf.GetDeviceBuffer(),
                               0,
                               static_cast<std::size_t>(din_length) * sizeof(DInDataType),
                               s.stream_id_));
        };

        if constexpr(BwdProblem::kNeedFp32Workspace)
        {
            auto memset_workspace = [&](const ck_tile::stream_config& s) {
                HIP_CHECK_ERROR(hipMemsetAsync(workspace_buf.GetDeviceBuffer(),
                                               0,
                                               static_cast<std::size_t>(din_length) * sizeof(float),
                                               s.stream_id_));
            };

            using CastKernel =
                ck_tile::PoolBwdCastKernel<float, DInDataType, kBlockSize, kVectorSize>;
            auto cast_host_args = ck_tile::PoolBwdCastHostArgs{
                workspace_buf.GetDeviceBuffer(), din_buf.GetDeviceBuffer(), din_length};
            if(!CastKernel::IsSupportedArgument(cast_host_args))
            {
                return true;
            }
            auto cast_kargs = CastKernel::MakeKernelArgs(cast_host_args);
            const ck_tile::index_t cast_grid_size =
                CastKernel::CalculateGridSize(stream, din_length);

            ck_tile::launch_kernel(
                stream,
                memset_workspace,
                ck_tile::make_kernel<kBwdBlockPerCu>(
                    BwdKernel{}, bwd_grid_size, bwd_block_size, 0, bwd_kargs),
                ck_tile::make_kernel<kBwdBlockPerCu>(
                    CastKernel{}, cast_grid_size, CastKernel::BlockSize(), 0, cast_kargs));
        }
        else
        {
            ck_tile::launch_kernel(stream,
                                   memset_din,
                                   ck_tile::make_kernel<kBwdBlockPerCu>(
                                       BwdKernel{}, bwd_grid_size, bwd_block_size, 0, bwd_kargs));
        }

        indices_buf.FromDevice(h_indices.data());
        din_buf.FromDevice(h_din.data());

        ck_tile::reference_pool_bwd<DOutDataType, IndexDataType, ComputeDataType, DInDataType>(
            h_dout, h_indices, h_din_ref);

        const bool pass = ck_tile::check_err(
            h_din, h_din_ref, "Error: incorrect dx", rtol_for_dtype(), atol_for_dtype());
        std::cout << "[2D " << cfg.name << "] " << (pass ? "PASS" : "FAIL") << std::endl;
        return pass;
    }

    template <bool kHasOverlap>
    bool RunPool3D_HasOverlap(const Config3D& cfg)
    {
        using InDataType  = DInDataType;
        using OutDataType = DOutDataType;

        const ck_tile::index_t Zs = (cfg.Z - 1) * cfg.Dz + 1;
        const ck_tile::index_t Ys = (cfg.Y - 1) * cfg.Dy + 1;
        const ck_tile::index_t Xs = (cfg.X - 1) * cfg.Dx + 1;
        const ck_tile::index_t Do = (cfg.D + cfg.LeftPz + cfg.RightPz - Zs) / cfg.Sz + 1;
        const ck_tile::index_t Ho = (cfg.H + cfg.LeftPy + cfg.RightPy - Ys) / cfg.Sy + 1;
        const ck_tile::index_t Wo = (cfg.W + cfg.LeftPx + cfg.RightPx - Xs) / cfg.Sx + 1;

        const auto input_shape   = ck_tile::make_tuple(cfg.N, cfg.D, cfg.H, cfg.W, cfg.C);
        const auto output_shape  = ck_tile::make_tuple(cfg.N, Do, Ho, Wo, cfg.C);
        const auto input_strides = ck_tile::make_tuple(
            cfg.D * cfg.H * cfg.W * cfg.C, cfg.H * cfg.W * cfg.C, cfg.W * cfg.C, cfg.C, 1);
        const auto output_strides =
            ck_tile::make_tuple(Do * Ho * Wo * cfg.C, Ho * Wo * cfg.C, Wo * cfg.C, cfg.C, 1);
        const auto window_lengths = ck_tile::make_tuple(cfg.Z, cfg.Y, cfg.X);
        const auto window_strides = ck_tile::make_tuple(cfg.Sz, cfg.Sy, cfg.Sx);
        const auto window_dils    = ck_tile::make_tuple(cfg.Dz, cfg.Dy, cfg.Dx);
        const auto left_pads      = ck_tile::make_tuple(cfg.LeftPz, cfg.LeftPy, cfg.LeftPx);
        const auto right_pads     = ck_tile::make_tuple(cfg.RightPz, cfg.RightPy, cfg.RightPx);

        ck_tile::HostTensor<InDataType> h_in(
            {cfg.N, cfg.D, cfg.H, cfg.W, cfg.C},
            {cfg.D * cfg.H * cfg.W * cfg.C, cfg.H * cfg.W * cfg.C, cfg.W * cfg.C, cfg.C, 1});
        ck_tile::HostTensor<OutDataType> h_out(
            {cfg.N, Do, Ho, Wo, cfg.C},
            {Do * Ho * Wo * cfg.C, Ho * Wo * cfg.C, Wo * cfg.C, cfg.C, 1});
        ck_tile::HostTensor<IndexDataType> h_indices(
            {cfg.N, Do, Ho, Wo, cfg.C},
            {Do * Ho * Wo * cfg.C, Ho * Wo * cfg.C, Wo * cfg.C, cfg.C, 1});
        ck_tile::HostTensor<DOutDataType> h_dout(
            {cfg.N, Do, Ho, Wo, cfg.C},
            {Do * Ho * Wo * cfg.C, Ho * Wo * cfg.C, Wo * cfg.C, cfg.C, 1});
        ck_tile::HostTensor<DInDataType> h_din(
            {cfg.N, cfg.D, cfg.H, cfg.W, cfg.C},
            {cfg.D * cfg.H * cfg.W * cfg.C, cfg.H * cfg.W * cfg.C, cfg.W * cfg.C, cfg.C, 1});
        ck_tile::HostTensor<DInDataType> h_din_ref(
            {cfg.N, cfg.D, cfg.H, cfg.W, cfg.C},
            {cfg.D * cfg.H * cfg.W * cfg.C, cfg.H * cfg.W * cfg.C, cfg.W * cfg.C, cfg.C, 1});

        ck_tile::FillUniformDistribution<InDataType>{-5.f, 5.f}(h_in);
        ck_tile::FillUniformDistribution<DOutDataType>{-1.f, 1.f}(h_dout);

        ck_tile::DeviceMem in_buf(h_in.get_element_space_size_in_bytes());
        ck_tile::DeviceMem out_buf(h_out.get_element_space_size_in_bytes());
        ck_tile::DeviceMem indices_buf(h_indices.get_element_space_size_in_bytes());
        ck_tile::DeviceMem dout_buf(h_dout.get_element_space_size_in_bytes());
        ck_tile::DeviceMem din_buf(h_din.get_element_space_size_in_bytes());

        in_buf.ToDevice(h_in.data());
        dout_buf.ToDevice(h_dout.data());

        constexpr bool kPropagateNan = false;
        using ReduceOp               = ck_tile::ReduceOp::Max;
        using FwdProblem             = ck_tile::PoolProblem<InDataType,
                                                            OutDataType,
                                                            ComputeDataType,
                                                            IndexDataType,
                                                            ReduceOp,
                                                            true,
                                                            kPropagateNan,
                                                            FwdShape>;
        using FwdKernel              = ck_tile::PoolKernel<FwdProblem>;

        auto fwd_host_args = ck_tile::PoolHostArgs<decltype(input_shape), decltype(window_lengths)>{
            static_cast<InDataType*>(in_buf.GetDeviceBuffer()),
            static_cast<OutDataType*>(out_buf.GetDeviceBuffer()),
            static_cast<IndexDataType*>(indices_buf.GetDeviceBuffer()),
            input_shape,
            output_shape,
            input_strides,
            output_strides,
            window_lengths,
            window_strides,
            window_dils,
            left_pads,
            right_pads};
        auto fwd_kargs = FwdKernel::MakeKernelArgs(fwd_host_args);

        if(!FwdKernel::IsSupportedArgument(fwd_kargs))
        {
            return true;
        }

        constexpr ck_tile::index_t kFwdBlockPerCu = 1;
        const ck_tile::index_t fwd_grid_size      = FwdKernel::CalculateGridSize(fwd_kargs);
        const ck_tile::index_t fwd_block_size     = FwdKernel::BlockSize();

        ck_tile::launch_kernel(ck_tile::stream_config{nullptr, false, 0},
                               ck_tile::make_kernel<kFwdBlockPerCu>(
                                   FwdKernel{}, fwd_grid_size, fwd_block_size, 0, fwd_kargs));

        using BwdProblem = ck_tile::
            PoolBwdProblem<DOutDataType, IndexDataType, DInDataType, BwdShape, kHasOverlap>;
        using BwdKernel = ck_tile::PoolBwdKernel<BwdProblem>;

        const ck_tile::long_index_t din_length =
            static_cast<ck_tile::long_index_t>(h_din.get_element_space_size());
        const ck_tile::long_index_t dout_length =
            static_cast<ck_tile::long_index_t>(h_dout.get_element_space_size());

        const std::size_t workspace_bytes = BwdKernel::GetWorkSpaceSize(ck_tile::PoolBwdHostArgs{
            nullptr, nullptr, nullptr, nullptr, dout_length, din_length, kHasOverlap});

        ck_tile::DeviceMem workspace_buf(workspace_bytes);

        auto bwd_host_args = ck_tile::PoolBwdHostArgs{
            static_cast<DOutDataType*>(dout_buf.GetDeviceBuffer()),
            static_cast<IndexDataType*>(indices_buf.GetDeviceBuffer()),
            static_cast<DInDataType*>(din_buf.GetDeviceBuffer()),
            workspace_bytes > 0 ? workspace_buf.GetDeviceBuffer() : nullptr,
            dout_length,
            din_length,
            kHasOverlap};

        if(!BwdKernel::IsSupportedArgument(bwd_host_args))
        {
            return true;
        }

        auto bwd_kargs = BwdKernel::MakeKernelArgs(bwd_host_args);

        constexpr ck_tile::index_t kBwdBlockPerCu = 1;
        const ck_tile::index_t bwd_block_size     = BwdKernel::BlockSize();
        auto stream                               = ck_tile::stream_config{nullptr, false, 0};
        const ck_tile::index_t bwd_grid_size = BwdKernel::CalculateGridSize(stream, dout_length);

        auto memset_din = [&](const ck_tile::stream_config& s) {
            HIP_CHECK_ERROR(
                hipMemsetAsync(din_buf.GetDeviceBuffer(),
                               0,
                               static_cast<std::size_t>(din_length) * sizeof(DInDataType),
                               s.stream_id_));
        };

        if constexpr(BwdProblem::kNeedFp32Workspace)
        {
            auto memset_workspace = [&](const ck_tile::stream_config& s) {
                HIP_CHECK_ERROR(hipMemsetAsync(workspace_buf.GetDeviceBuffer(),
                                               0,
                                               static_cast<std::size_t>(din_length) * sizeof(float),
                                               s.stream_id_));
            };

            using CastKernel =
                ck_tile::PoolBwdCastKernel<float, DInDataType, kBlockSize, kVectorSize>;
            auto cast_host_args = ck_tile::PoolBwdCastHostArgs{
                workspace_buf.GetDeviceBuffer(), din_buf.GetDeviceBuffer(), din_length};
            if(!CastKernel::IsSupportedArgument(cast_host_args))
            {
                return true;
            }
            auto cast_kargs = CastKernel::MakeKernelArgs(cast_host_args);
            const ck_tile::index_t cast_grid_size =
                CastKernel::CalculateGridSize(stream, din_length);

            ck_tile::launch_kernel(
                stream,
                memset_workspace,
                ck_tile::make_kernel<kBwdBlockPerCu>(
                    BwdKernel{}, bwd_grid_size, bwd_block_size, 0, bwd_kargs),
                ck_tile::make_kernel<kBwdBlockPerCu>(
                    CastKernel{}, cast_grid_size, CastKernel::BlockSize(), 0, cast_kargs));
        }
        else
        {
            ck_tile::launch_kernel(stream,
                                   memset_din,
                                   ck_tile::make_kernel<kBwdBlockPerCu>(
                                       BwdKernel{}, bwd_grid_size, bwd_block_size, 0, bwd_kargs));
        }

        indices_buf.FromDevice(h_indices.data());
        din_buf.FromDevice(h_din.data());

        ck_tile::reference_pool_bwd<DOutDataType, IndexDataType, ComputeDataType, DInDataType>(
            h_dout, h_indices, h_din_ref);

        const bool pass = ck_tile::check_err(
            h_din, h_din_ref, "Error: incorrect dx", rtol_for_dtype(), atol_for_dtype());
        std::cout << "[3D " << cfg.name << "] " << (pass ? "PASS" : "FAIL") << std::endl;
        return pass;
    }
};

using TestConfig_F32  = std::tuple<float, float, ck_tile::index_t, float>;
using TestConfig_F16  = std::tuple<ck_tile::half_t, ck_tile::half_t, ck_tile::index_t, float>;
using TestConfig_BF16 = std::tuple<ck_tile::bf16_t, ck_tile::bf16_t, ck_tile::index_t, float>;

using TestTypes = ::testing::Types<TestConfig_F32, TestConfig_F16, TestConfig_BF16>;

TYPED_TEST_SUITE(TestCkTilePoolingBwd, TestTypes);

TYPED_TEST(TestCkTilePoolingBwd, Pool2D_NoOverlap_2x2)
{
    typename TestFixture::Config2D cfg = {
        1, 16, 16, 32, 2, 2, 2, 2, 1, 1, 0, 0, 0, 0, "no_overlap_2x2"};
    EXPECT_TRUE((this->template RunPool2D_HasOverlap<false>(cfg)));
}

TYPED_TEST(TestCkTilePoolingBwd, Pool2D_Overlap_3x3_Stride1)
{
    typename TestFixture::Config2D cfg = {
        1, 16, 16, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, "overlap_3x3_s1_pad1"};
    EXPECT_TRUE((this->template RunPool2D_HasOverlap<true>(cfg)));
}

TYPED_TEST(TestCkTilePoolingBwd, Pool2D_Overlap_3x3_Stride2_Dilation2)
{
    typename TestFixture::Config2D cfg = {
        2, 16, 16, 32, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, "overlap_3x3_s2_d2_pad2"};
    EXPECT_TRUE((this->template RunPool2D_HasOverlap<true>(cfg)));
}

TYPED_TEST(TestCkTilePoolingBwd, Pool3D_NoOverlap_2x2x2)
{
    typename TestFixture::Config3D cfg = {
        1, 4, 4, 4, 32, 2, 2, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, "no_overlap_2x2x2"};
    EXPECT_TRUE((this->template RunPool3D_HasOverlap<false>(cfg)));
}

TYPED_TEST(TestCkTilePoolingBwd, Pool3D_Overlap_3x3x3_Stride2_Pad1)
{
    typename TestFixture::Config3D cfg = {
        1, 8, 8, 8, 32, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, "overlap_3x3x3_s2_pad1"};
    EXPECT_TRUE((this->template RunPool3D_HasOverlap<true>(cfg)));
}
