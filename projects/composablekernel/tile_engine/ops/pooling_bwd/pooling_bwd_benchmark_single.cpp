// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file pooling_bwd_benchmark_single.cpp
 * @brief Single-kernel benchmark for max-pool backward (2D and 3D).
 *
 * The generated header (included via -include) provides:
 *   - SelectedKernel             (struct with ::launch())
 *   - KERNEL_NAME                (constexpr const char*)
 *   - POOLING_BWD_DIM            (constexpr int, 2 or 3)
 *   - DInDataType, DOutDataType, ComputeDataType, IndexDataType
 *   - TensorShape, WindowShape
 */

#include <iostream>
#include <string>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/pooling.hpp"
#include "ck_tile/ops/pooling_bwd.hpp"
#include "ck_tile/host/reference/reference_pool.hpp"
#include "ck_tile/host/reference/reference_pool_bwd.hpp"

template <int PoolDim>
static int benchmark_pooling_bwd(int argc, char* argv[])
{
    if constexpr(PoolDim == 2)
    {
        ck_tile::ArgParser arg_parser;
        arg_parser.insert("n", "1", "Batch size (N)")
            .insert("h", "16", "Input height (H)")
            .insert("w", "16", "Input width (W)")
            .insert("c", "32", "Channels (C)")
            .insert("wy", "2", "Window height (Y)")
            .insert("wx", "2", "Window width (X)")
            .insert("sy", "2", "Window stride height")
            .insert("sx", "2", "Window stride width")
            .insert("dy", "1", "Window dilation height")
            .insert("dx", "1", "Window dilation width")
            .insert("phy", "0", "Padding height left")
            .insert("phyr", "0", "Padding height right")
            .insert("pwx", "0", "Padding width left")
            .insert("pwxr", "0", "Padding width right")
            .insert("verify", "1", "Verify results (0/1)")
            .insert("warmup", "5", "Warmup iterations")
            .insert("repeat", "20", "Repeat iterations")
            .insert("log", "1", "Log level");

        if(!arg_parser.parse(argc, argv))
            return -1;

        const ck_tile::index_t N       = arg_parser.get_int("n");
        const ck_tile::index_t H       = arg_parser.get_int("h");
        const ck_tile::index_t W       = arg_parser.get_int("w");
        const ck_tile::index_t C       = arg_parser.get_int("c");
        const ck_tile::index_t Y       = arg_parser.get_int("wy");
        const ck_tile::index_t X       = arg_parser.get_int("wx");
        const ck_tile::index_t Sy      = arg_parser.get_int("sy");
        const ck_tile::index_t Sx      = arg_parser.get_int("sx");
        const ck_tile::index_t Dy      = arg_parser.get_int("dy");
        const ck_tile::index_t Dx      = arg_parser.get_int("dx");
        const ck_tile::index_t LeftPy  = arg_parser.get_int("phy");
        const ck_tile::index_t RightPy = arg_parser.get_int("phyr");
        const ck_tile::index_t LeftPx  = arg_parser.get_int("pwx");
        const ck_tile::index_t RightPx = arg_parser.get_int("pwxr");

        const bool verify   = arg_parser.get_int("verify") != 0;
        const int warmup    = arg_parser.get_int("warmup");
        const int repeat    = arg_parser.get_int("repeat");
        const int log_level = arg_parser.get_int("log");

        const ck_tile::index_t Ys = (Y - 1) * Dy + 1;
        const ck_tile::index_t Xs = (X - 1) * Dx + 1;
        const ck_tile::index_t Ho = (H + LeftPy + RightPy - Ys) / Sy + 1;
        const ck_tile::index_t Wo = (W + LeftPx + RightPx - Xs) / Sx + 1;

        std::cout << "Pooling bwd 2D benchmark: " << KERNEL_NAME << std::endl;
        std::cout << "  Input:  NHWC = " << N << "x" << H << "x" << W << "x" << C << std::endl;
        std::cout << "  Output: NHWC = " << N << "x" << Ho << "x" << Wo << "x" << C << std::endl;
        std::cout << "  Window: " << Y << "x" << X << ", stride: " << Sy << "x" << Sx
                  << ", dilation: " << Dy << "x" << Dx << std::endl;

        using InDataType  = DInDataType;
        using OutDataType = DOutDataType;

        ck_tile::HostTensor<InDataType> h_in({N, H, W, C});
        ck_tile::HostTensor<OutDataType> h_out({N, Ho, Wo, C});
        ck_tile::HostTensor<IndexDataType> h_indices({N, Ho, Wo, C});
        ck_tile::HostTensor<DOutDataType> h_dout({N, Ho, Wo, C});
        ck_tile::HostTensor<DInDataType> h_din({N, H, W, C});
        ck_tile::HostTensor<DInDataType> h_din_ref({N, H, W, C});

        ck_tile::FillUniformDistribution<InDataType>{-5.f, 5.f}(h_in);
        ck_tile::FillUniformDistribution<DOutDataType>{-1.f, 1.f}(h_dout);

        ck_tile::DeviceMem d_in(h_in.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d_out(h_out.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d_indices(h_indices.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d_dout(h_dout.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d_din(h_din.get_element_space_size_in_bytes());

        d_in.ToDevice(h_in.data());
        d_dout.ToDevice(h_dout.data());

        const auto input_shape   = ck_tile::make_tuple(N, H, W, C);
        const auto output_shape  = ck_tile::make_tuple(N, Ho, Wo, C);
        const auto input_strides = ck_tile::make_tuple(H * W * C, W * C, C, ck_tile::index_t{1});
        const auto output_strides =
            ck_tile::make_tuple(Ho * Wo * C, Wo * C, C, ck_tile::index_t{1});
        const auto window_lengths = ck_tile::make_tuple(Y, X);
        const auto window_strides = ck_tile::make_tuple(Sy, Sx);
        const auto window_dils    = ck_tile::make_tuple(Dy, Dx);
        const auto left_pads      = ck_tile::make_tuple(LeftPy, LeftPx);
        const auto right_pads     = ck_tile::make_tuple(RightPy, RightPx);

        // Run forward to obtain indices (with kOutputIndex=true).
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
                                                true,
                                                false,
                                                FwdShape>;
        using FwdKernel  = ck_tile::PoolKernel<FwdProblem>;

        auto fwd_host_args = ck_tile::PoolHostArgs<decltype(input_shape), decltype(window_lengths)>{
            d_in.GetDeviceBuffer(),
            d_out.GetDeviceBuffer(),
            d_indices.GetDeviceBuffer(),
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
            return -1;
        }

        constexpr ck_tile::index_t kFwdBlockPerCu = 1;
        const ck_tile::index_t fwd_grid_size      = FwdKernel::CalculateGridSize(fwd_kargs);
        const ck_tile::index_t fwd_block_size     = FwdKernel::BlockSize();

        ck_tile::launch_kernel(ck_tile::stream_config{nullptr, false, 0},
                               ck_tile::make_kernel<kFwdBlockPerCu>(
                                   FwdKernel{}, fwd_grid_size, fwd_block_size, 0, fwd_kargs));

        const ck_tile::long_index_t din_length =
            static_cast<ck_tile::long_index_t>(h_din.get_element_space_size());
        const ck_tile::long_index_t dout_length =
            static_cast<ck_tile::long_index_t>(h_dout.get_element_space_size());

        const bool has_overlap = ((Y - 1) * Dy + 1) > Sy || ((X - 1) * Dx + 1) > Sx;

        // Allow the SelectedKernel to report an unsupported configuration
        // (e.g. wrong overlap flavor) by skipping verification instead of
        // launching with mismatched semantics.
        if(SelectedKernel::kHasOverlap != has_overlap)
        {
            std::cerr << "Skipping benchmark: kernel has_overlap=" << SelectedKernel::kHasOverlap
                      << " does not match problem has_overlap=" << has_overlap << std::endl;
            return 0;
        }

        const std::size_t workspace_bytes =
            SelectedKernel::BwdKernel::GetWorkSpaceSize(ck_tile::PoolBwdHostArgs{
                nullptr, nullptr, nullptr, nullptr, dout_length, din_length, has_overlap});
        ck_tile::DeviceMem d_workspace(workspace_bytes);

        ck_tile::PoolBwdHostArgs bwd_host_args{d_dout.GetDeviceBuffer(),
                                               d_indices.GetDeviceBuffer(),
                                               d_din.GetDeviceBuffer(),
                                               workspace_bytes > 0 ? d_workspace.GetDeviceBuffer()
                                                                   : nullptr,
                                               dout_length,
                                               din_length,
                                               has_overlap};

        ck_tile::stream_config stream{nullptr, true, log_level, warmup, repeat};

        float latency = 0;
        try
        {
            latency = SelectedKernel::launch(bwd_host_args, stream);
        }
        catch(const std::exception& e)
        {
            std::cerr << "Kernel launch failed: " << e.what() << std::endl;
            return -1;
        }

        const std::size_t bytes_read = static_cast<std::size_t>(N) * Ho * Wo * C *
                                       (sizeof(DOutDataType) + sizeof(IndexDataType));
        const std::size_t bytes_written =
            static_cast<std::size_t>(N) * H * W * C * sizeof(DInDataType);
        const float bandwidth = (bytes_read + bytes_written) / (latency * 1e-3f) / 1e9f;

        std::cout << "  Latency: " << latency << " ms" << std::endl;
        std::cout << "  Bandwidth: " << bandwidth << " GB/s" << std::endl;

        if(verify)
        {
            d_indices.FromDevice(h_indices.data());
            d_din.FromDevice(h_din.data());

            ck_tile::reference_pool_bwd<DOutDataType, IndexDataType, ComputeDataType, DInDataType>(
                h_dout, h_indices, h_din_ref);

            const double rtol = std::is_same_v<DInDataType, float> ? 1e-5 : 1e-2;
            const double atol = std::is_same_v<DInDataType, float> ? 1e-5 : 1e-2;
            const bool pass =
                ck_tile::check_err(h_din, h_din_ref, "Error: Incorrect dx", rtol, atol);
            std::cout << "  Verification: " << (pass ? "PASS" : "FAIL") << std::endl;
        }

        return 0;
    }
    else // PoolDim == 3
    {
        ck_tile::ArgParser arg_parser;
        arg_parser.insert("n", "1", "Batch size (N)")
            .insert("d", "4", "Input depth (D)")
            .insert("h", "16", "Input height (H)")
            .insert("w", "16", "Input width (W)")
            .insert("c", "32", "Channels (C)")
            .insert("wz", "2", "Window depth (Z)")
            .insert("wy", "2", "Window height (Y)")
            .insert("wx", "2", "Window width (X)")
            .insert("sz", "2", "Window stride depth")
            .insert("sy", "2", "Window stride height")
            .insert("sx", "2", "Window stride width")
            .insert("dz", "1", "Window dilation depth")
            .insert("dy", "1", "Window dilation height")
            .insert("dx", "1", "Window dilation width")
            .insert("pdz", "0", "Padding depth left")
            .insert("pdzr", "0", "Padding depth right")
            .insert("phy", "0", "Padding height left")
            .insert("phyr", "0", "Padding height right")
            .insert("pwx", "0", "Padding width left")
            .insert("pwxr", "0", "Padding width right")
            .insert("verify", "1", "Verify results (0/1)")
            .insert("warmup", "5", "Warmup iterations")
            .insert("repeat", "20", "Repeat iterations")
            .insert("log", "1", "Log level");

        if(!arg_parser.parse(argc, argv))
            return -1;

        const ck_tile::index_t N       = arg_parser.get_int("n");
        const ck_tile::index_t D       = arg_parser.get_int("d");
        const ck_tile::index_t H       = arg_parser.get_int("h");
        const ck_tile::index_t W       = arg_parser.get_int("w");
        const ck_tile::index_t C       = arg_parser.get_int("c");
        const ck_tile::index_t Z       = arg_parser.get_int("wz");
        const ck_tile::index_t Y       = arg_parser.get_int("wy");
        const ck_tile::index_t X       = arg_parser.get_int("wx");
        const ck_tile::index_t Sz      = arg_parser.get_int("sz");
        const ck_tile::index_t Sy      = arg_parser.get_int("sy");
        const ck_tile::index_t Sx      = arg_parser.get_int("sx");
        const ck_tile::index_t Dz      = arg_parser.get_int("dz");
        const ck_tile::index_t Dy      = arg_parser.get_int("dy");
        const ck_tile::index_t Dx      = arg_parser.get_int("dx");
        const ck_tile::index_t LeftPz  = arg_parser.get_int("pdz");
        const ck_tile::index_t RightPz = arg_parser.get_int("pdzr");
        const ck_tile::index_t LeftPy  = arg_parser.get_int("phy");
        const ck_tile::index_t RightPy = arg_parser.get_int("phyr");
        const ck_tile::index_t LeftPx  = arg_parser.get_int("pwx");
        const ck_tile::index_t RightPx = arg_parser.get_int("pwxr");

        const bool verify   = arg_parser.get_int("verify") != 0;
        const int warmup    = arg_parser.get_int("warmup");
        const int repeat    = arg_parser.get_int("repeat");
        const int log_level = arg_parser.get_int("log");

        const ck_tile::index_t Zs = (Z - 1) * Dz + 1;
        const ck_tile::index_t Ys = (Y - 1) * Dy + 1;
        const ck_tile::index_t Xs = (X - 1) * Dx + 1;
        const ck_tile::index_t Do = (D + LeftPz + RightPz - Zs) / Sz + 1;
        const ck_tile::index_t Ho = (H + LeftPy + RightPy - Ys) / Sy + 1;
        const ck_tile::index_t Wo = (W + LeftPx + RightPx - Xs) / Sx + 1;

        std::cout << "Pooling bwd 3D benchmark: " << KERNEL_NAME << std::endl;
        std::cout << "  Input:  NDHWC = " << N << "x" << D << "x" << H << "x" << W << "x" << C
                  << std::endl;
        std::cout << "  Output: NDHWC = " << N << "x" << Do << "x" << Ho << "x" << Wo << "x" << C
                  << std::endl;
        std::cout << "  Window: " << Z << "x" << Y << "x" << X << ", stride: " << Sz << "x" << Sy
                  << "x" << Sx << ", dilation: " << Dz << "x" << Dy << "x" << Dx << std::endl;

        using InDataType  = DInDataType;
        using OutDataType = DOutDataType;

        ck_tile::HostTensor<InDataType> h_in({N, D, H, W, C});
        ck_tile::HostTensor<OutDataType> h_out({N, Do, Ho, Wo, C});
        ck_tile::HostTensor<IndexDataType> h_indices({N, Do, Ho, Wo, C});
        ck_tile::HostTensor<DOutDataType> h_dout({N, Do, Ho, Wo, C});
        ck_tile::HostTensor<DInDataType> h_din({N, D, H, W, C});
        ck_tile::HostTensor<DInDataType> h_din_ref({N, D, H, W, C});

        ck_tile::FillUniformDistribution<InDataType>{-5.f, 5.f}(h_in);
        ck_tile::FillUniformDistribution<DOutDataType>{-1.f, 1.f}(h_dout);

        ck_tile::DeviceMem d_in(h_in.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d_out(h_out.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d_indices(h_indices.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d_dout(h_dout.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d_din(h_din.get_element_space_size_in_bytes());

        d_in.ToDevice(h_in.data());
        d_dout.ToDevice(h_dout.data());

        const auto input_shape  = ck_tile::make_tuple(N, D, H, W, C);
        const auto output_shape = ck_tile::make_tuple(N, Do, Ho, Wo, C);
        const auto input_strides =
            ck_tile::make_tuple(D * H * W * C, H * W * C, W * C, C, ck_tile::index_t{1});
        const auto output_strides =
            ck_tile::make_tuple(Do * Ho * Wo * C, Ho * Wo * C, Wo * C, C, ck_tile::index_t{1});
        const auto window_lengths = ck_tile::make_tuple(Z, Y, X);
        const auto window_strides = ck_tile::make_tuple(Sz, Sy, Sx);
        const auto window_dils    = ck_tile::make_tuple(Dz, Dy, Dx);
        const auto left_pads      = ck_tile::make_tuple(LeftPz, LeftPy, LeftPx);
        const auto right_pads     = ck_tile::make_tuple(RightPz, RightPy, RightPx);

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
                                                true,
                                                false,
                                                FwdShape>;
        using FwdKernel  = ck_tile::PoolKernel<FwdProblem>;

        auto fwd_host_args = ck_tile::PoolHostArgs<decltype(input_shape), decltype(window_lengths)>{
            d_in.GetDeviceBuffer(),
            d_out.GetDeviceBuffer(),
            d_indices.GetDeviceBuffer(),
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
            return -1;
        }

        constexpr ck_tile::index_t kFwdBlockPerCu = 1;
        const ck_tile::index_t fwd_grid_size      = FwdKernel::CalculateGridSize(fwd_kargs);
        const ck_tile::index_t fwd_block_size     = FwdKernel::BlockSize();

        ck_tile::launch_kernel(ck_tile::stream_config{nullptr, false, 0},
                               ck_tile::make_kernel<kFwdBlockPerCu>(
                                   FwdKernel{}, fwd_grid_size, fwd_block_size, 0, fwd_kargs));

        const ck_tile::long_index_t din_length =
            static_cast<ck_tile::long_index_t>(h_din.get_element_space_size());
        const ck_tile::long_index_t dout_length =
            static_cast<ck_tile::long_index_t>(h_dout.get_element_space_size());

        const bool has_overlap =
            ((Z - 1) * Dz + 1) > Sz || ((Y - 1) * Dy + 1) > Sy || ((X - 1) * Dx + 1) > Sx;

        if(SelectedKernel::kHasOverlap != has_overlap)
        {
            std::cerr << "Skipping benchmark: kernel has_overlap=" << SelectedKernel::kHasOverlap
                      << " does not match problem has_overlap=" << has_overlap << std::endl;
            return 0;
        }

        const std::size_t workspace_bytes =
            SelectedKernel::BwdKernel::GetWorkSpaceSize(ck_tile::PoolBwdHostArgs{
                nullptr, nullptr, nullptr, nullptr, dout_length, din_length, has_overlap});
        ck_tile::DeviceMem d_workspace(workspace_bytes);

        ck_tile::PoolBwdHostArgs bwd_host_args{d_dout.GetDeviceBuffer(),
                                               d_indices.GetDeviceBuffer(),
                                               d_din.GetDeviceBuffer(),
                                               workspace_bytes > 0 ? d_workspace.GetDeviceBuffer()
                                                                   : nullptr,
                                               dout_length,
                                               din_length,
                                               has_overlap};

        ck_tile::stream_config stream{nullptr, true, log_level, warmup, repeat};

        float latency = 0;
        try
        {
            latency = SelectedKernel::launch(bwd_host_args, stream);
        }
        catch(const std::exception& e)
        {
            std::cerr << "Kernel launch failed: " << e.what() << std::endl;
            return -1;
        }

        const std::size_t bytes_read = static_cast<std::size_t>(N) * Do * Ho * Wo * C *
                                       (sizeof(DOutDataType) + sizeof(IndexDataType));
        const std::size_t bytes_written =
            static_cast<std::size_t>(N) * D * H * W * C * sizeof(DInDataType);
        const float bandwidth = (bytes_read + bytes_written) / (latency * 1e-3f) / 1e9f;

        std::cout << "  Latency: " << latency << " ms" << std::endl;
        std::cout << "  Bandwidth: " << bandwidth << " GB/s" << std::endl;

        if(verify)
        {
            d_indices.FromDevice(h_indices.data());
            d_din.FromDevice(h_din.data());

            ck_tile::reference_pool_bwd<DOutDataType, IndexDataType, ComputeDataType, DInDataType>(
                h_dout, h_indices, h_din_ref);

            const double rtol = std::is_same_v<DInDataType, float> ? 1e-5 : 1e-2;
            const double atol = std::is_same_v<DInDataType, float> ? 1e-5 : 1e-2;
            const bool pass =
                ck_tile::check_err(h_din, h_din_ref, "Error: Incorrect dx", rtol, atol);
            std::cout << "  Verification: " << (pass ? "PASS" : "FAIL") << std::endl;
        }

        return 0;
    }
}

int main(int argc, char* argv[]) { return benchmark_pooling_bwd<POOLING_BWD_DIM>(argc, argv); }
