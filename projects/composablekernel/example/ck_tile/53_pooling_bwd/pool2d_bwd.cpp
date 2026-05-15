// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstring>
#include <iostream>

#include "ck_tile/host.hpp"
#include "ck_tile/ops/pooling.hpp"
#include "ck_tile/ops/pooling_bwd.hpp"
#include "ck_tile/host/reference/reference_pool.hpp"
#include "ck_tile/host/reference/reference_pool_bwd.hpp"

template <typename DOutDataType,
          typename DInDataType,
          typename ComputeDataType,
          typename IndexDataType,
          ck_tile::index_t Y_  = 2,
          ck_tile::index_t X_  = 2,
          ck_tile::index_t Sy_ = 2,
          ck_tile::index_t Sx_ = 2,
          ck_tile::index_t Py_ = 0,
          ck_tile::index_t Px_ = 0>
bool run()
{
    constexpr ck_tile::index_t N  = 2;
    constexpr ck_tile::index_t H  = 32;
    constexpr ck_tile::index_t W  = 32;
    constexpr ck_tile::index_t C  = 32;
    constexpr ck_tile::index_t Y  = Y_;
    constexpr ck_tile::index_t X  = X_;
    constexpr ck_tile::index_t Sy = Sy_;
    constexpr ck_tile::index_t Sx = Sx_;
    constexpr ck_tile::index_t Dy = 1;
    constexpr ck_tile::index_t Dx = 1;

    constexpr ck_tile::index_t LeftPy  = Py_;
    constexpr ck_tile::index_t LeftPx  = Px_;
    constexpr ck_tile::index_t RightPy = Py_;
    constexpr ck_tile::index_t RightPx = Px_;

    constexpr ck_tile::index_t Ys = (Y - 1) * Dy + 1;
    constexpr ck_tile::index_t Xs = (X - 1) * Dx + 1;
    constexpr ck_tile::index_t Ho = (H + LeftPy + RightPy - Ys) / Sy + 1;
    constexpr ck_tile::index_t Wo = (W + LeftPx + RightPx - Xs) / Sx + 1;

    using InDataType  = DInDataType;
    using OutDataType = DOutDataType;

    constexpr bool kOutputIndex  = true;
    constexpr bool kPropagateNan = false;

    const auto input_shape    = ck_tile::make_tuple(N, H, W, C);
    const auto output_shape   = ck_tile::make_tuple(N, Ho, Wo, C);
    const auto input_strides  = ck_tile::make_tuple(H * W * C, W * C, C, 1);
    const auto output_strides = ck_tile::make_tuple(Ho * Wo * C, Wo * C, C, 1);
    const auto window_lengths = ck_tile::make_tuple(Y, X);
    const auto window_strides = ck_tile::make_tuple(Sy, Sx);
    const auto window_dils    = ck_tile::make_tuple(Dy, Dx);
    const auto left_pads      = ck_tile::make_tuple(LeftPy, LeftPx);
    const auto right_pads     = ck_tile::make_tuple(RightPy, RightPx);

    constexpr bool kHasOverlap = ((Y - 1) * Dy + 1) > Sy || ((X - 1) * Dx + 1) > Sx;

    ck_tile::HostTensor<InDataType> h_in({N, H, W, C}, {H * W * C, W * C, C, 1});
    ck_tile::HostTensor<OutDataType> h_out({N, Ho, Wo, C}, {Ho * Wo * C, Wo * C, C, 1});
    ck_tile::HostTensor<IndexDataType> h_indices({N, Ho, Wo, C}, {Ho * Wo * C, Wo * C, C, 1});
    ck_tile::HostTensor<DOutDataType> h_dout({N, Ho, Wo, C}, {Ho * Wo * C, Wo * C, C, 1});

    ck_tile::HostTensor<DInDataType> h_din({N, H, W, C}, {H * W * C, W * C, C, 1});
    ck_tile::HostTensor<DInDataType> h_din_ref({N, H, W, C}, {H * W * C, W * C, C, 1});

    ck_tile::FillUniformDistribution<InDataType>{-5.f, 5.f}(h_in);
    ck_tile::FillUniformDistribution<DOutDataType>{-1.f, 1.f}(h_dout);

    ck_tile::DeviceMem in_buf(h_in.get_element_space_size_in_bytes());
    ck_tile::DeviceMem out_buf(h_out.get_element_space_size_in_bytes());
    ck_tile::DeviceMem indices_buf(h_indices.get_element_space_size_in_bytes());
    ck_tile::DeviceMem dout_buf(h_dout.get_element_space_size_in_bytes());
    ck_tile::DeviceMem din_buf(h_din.get_element_space_size_in_bytes());

    in_buf.ToDevice(h_in.data());
    dout_buf.ToDevice(h_dout.data());

    using ReduceOp   = ck_tile::ReduceOp::Max;
    using BlockWarps = ck_tile::sequence<1, 1>;
    using BlockTile  = ck_tile::sequence<128, 1>;
    using WarpTile   = ck_tile::sequence<128, 1>;
    using ThreadTile = ck_tile::sequence<2, 1>;

    using FwdShape   = ck_tile::PoolShape<BlockWarps, BlockTile, WarpTile, ThreadTile>;
    using FwdProblem = ck_tile::PoolProblem<InDataType,
                                            OutDataType,
                                            ComputeDataType,
                                            IndexDataType,
                                            ReduceOp,
                                            kOutputIndex,
                                            kPropagateNan,
                                            FwdShape>;
    using FwdKernel  = ck_tile::PoolKernel<FwdProblem>;

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
        std::cerr << "ERROR: forward pool kernel arguments are not supported." << std::endl;
        return false;
    }

    constexpr ck_tile::index_t kFwdBlockPerCu = 1;
    const ck_tile::index_t fwd_block_size     = FwdKernel::BlockSize();
    const ck_tile::index_t fwd_grid_size      = FwdKernel::CalculateGridSize(fwd_kargs);

    ck_tile::launch_kernel(ck_tile::stream_config{nullptr, false, 0},
                           ck_tile::make_kernel<kFwdBlockPerCu>(
                               FwdKernel{}, fwd_grid_size, fwd_block_size, 0, fwd_kargs));

    using BwdShape = ck_tile::PoolBwdShape<256, 4>;
    using BwdProblem =
        ck_tile::PoolBwdProblem<DOutDataType, IndexDataType, DInDataType, BwdShape, kHasOverlap>;
    using BwdKernel = ck_tile::PoolBwdKernel<BwdProblem>;

    const ck_tile::long_index_t din_length =
        static_cast<ck_tile::long_index_t>(h_din.get_element_space_size());
    const ck_tile::long_index_t dout_length =
        static_cast<ck_tile::long_index_t>(h_dout.get_element_space_size());

    ck_tile::DeviceMem workspace_buf(BwdKernel::GetWorkSpaceSize(ck_tile::PoolBwdHostArgs{
        nullptr, nullptr, nullptr, nullptr, dout_length, din_length, kHasOverlap}));

    auto bwd_host_args = ck_tile::PoolBwdHostArgs{
        static_cast<DOutDataType*>(dout_buf.GetDeviceBuffer()),
        static_cast<IndexDataType*>(indices_buf.GetDeviceBuffer()),
        static_cast<DInDataType*>(din_buf.GetDeviceBuffer()),
        BwdKernel::GetWorkSpaceSize(ck_tile::PoolBwdHostArgs{
            nullptr, nullptr, nullptr, nullptr, dout_length, din_length, kHasOverlap}) > 0
            ? workspace_buf.GetDeviceBuffer()
            : nullptr,
        dout_length,
        din_length,
        kHasOverlap};

    if(!BwdKernel::IsSupportedArgument(bwd_host_args))
    {
        std::cerr << "ERROR: backward pool kernel arguments are not supported." << std::endl;
        return false;
    }

    auto bwd_kargs = BwdKernel::MakeKernelArgs(bwd_host_args);

    constexpr ck_tile::index_t kBwdBlockPerCu = 1;
    const ck_tile::index_t bwd_block_size     = BwdKernel::BlockSize();

    auto stream              = ck_tile::stream_config{nullptr, false, 0};
    const auto bwd_grid_size = BwdKernel::CalculateGridSize(stream, dout_length);

    auto memset_din = [&](const ck_tile::stream_config& s) {
        HIP_CHECK_ERROR(hipMemsetAsync(din_buf.GetDeviceBuffer(),
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

        // Match the bwd kernel's element-grouping so divisibility constraints
        // and grid sizing stay in lockstep across the two kernels.
        using CastKernel = ck_tile::PoolBwdCastKernel<float,
                                                      DInDataType,
                                                      BwdKernel::kBlockSize,
                                                      BwdKernel::kVectorSize>;
        auto cast_host_args = ck_tile::PoolBwdCastHostArgs{
            workspace_buf.GetDeviceBuffer(), din_buf.GetDeviceBuffer(), din_length};

        if(!CastKernel::IsSupportedArgument(cast_host_args))
        {
            std::cerr << "ERROR: pool bwd cast kernel arguments are not supported." << std::endl;
            return false;
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

    // Tolerances follow the dtype: fp32 path is bit-exact w.r.t. reference,
    // fp16/bf16 cast at the epilogue introduces ~1e-2 absolute error.
    const double rtol = std::is_same_v<DInDataType, float> ? 1e-5 : 1e-2;
    const double atol = std::is_same_v<DInDataType, float> ? 1e-5 : 1e-2;
    bool pass         = ck_tile::check_err(h_din, h_din_ref, "Error: incorrect dx", rtol, atol);

    std::cout << "N=" << N << " H=" << H << " W=" << W << " C=" << C << " Y=" << Y << " X=" << X
              << " Sy=" << Sy << " Sx=" << Sx << " has_overlap=" << kHasOverlap
              << " valid=" << (pass ? "y" : "n") << std::endl;

    return pass;
}

int main()
{
    bool ok = true;
    // fp32 / no-overlap: scalar set path (Y=2, Sy=2, no padding).
    ok &= run<float, float, float, ck_tile::index_t>();
    // fp16 / overlap: exercises the fp32 workspace + atomic add + cast kernel
    // path, which is the most subtle code path in the backward pipeline.
    // Y=3, Sy=1, pad=1 gives a real window overlap.
    ok &= run<ck_tile::half_t, ck_tile::half_t, float, ck_tile::index_t, 3, 3, 1, 1, 1, 1>();
    return ok ? 0 : -1;
}
