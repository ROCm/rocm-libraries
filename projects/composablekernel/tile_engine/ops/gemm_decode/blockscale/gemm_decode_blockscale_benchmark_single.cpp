// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm_decode/gemm_decode.hpp"

#include "gemm_decode_common.hpp"

#if !defined(GEMM_DECODE_BLOCKSCALE_KERNEL_DEFINED)
#include "blockscale/gemm_decode_blockscale_single_dsv3.hpp"
#endif

using namespace ck_tile;

namespace {

template <typename ADataType, typename BDataType>
void FillRandom(HostTensor<ADataType>& a, HostTensor<BDataType>& b, int init)
{
    if(init == 0)
    {
        FillUniformDistribution<ADataType>{-1.0f, 1.0f}(a);
        FillUniformDistribution<BDataType>{-1.0f, 1.0f}(b);
    }
    else if(init == 1)
    {
        FillMonotonicSeq<ADataType>{}(a);
        FillMonotonicSeq<BDataType>{}(b);
    }
    else
    {
        FillConstant<ADataType>{type_convert<ADataType>(1.0f)}(a);
        FillConstant<BDataType>{type_convert<BDataType>(1.0f)}(b);
    }
}

template <typename ADataType, typename BDataType, typename CDataType>
void RunBenchmark(const gemm_decode_tile_engine::DecodeProblem& p,
                  int warmup,
                  int repeat,
                  int verify,
                  int init,
                  int metric_kind)
{
    using Kernel = SelectedGemmDecodeBlockscaleKernel;

    constexpr index_t kXBlockN = Kernel::kXScaleBlockN;
    constexpr index_t kXBlockK = Kernel::kXScaleBlockK;
    constexpr index_t kWBlockN = Kernel::kWScaleBlockN;
    constexpr index_t kWBlockK = Kernel::kWScaleBlockK;

    if((p.M % kXBlockN) != 0 || (p.N % kWBlockN) != 0 ||
       (p.K % kXBlockK) != 0 || (p.K % kWBlockK) != 0)
    {
        throw std::invalid_argument(
            "shape not divisible by the blockscale block sizes (X=Block2D<1,128>, "
            "W=Block2D<128,128>)");
    }

    const index_t aqn = p.M / kXBlockN;
    const index_t aqk = p.K / kXBlockK;
    const index_t bqn = p.N / kWBlockN;
    const index_t bqk = p.K / kWBlockK;

    HostTensor<ADataType> a({p.M, p.K});
    HostTensor<BDataType> b({p.N, p.K});
    HostTensor<float> x_scale({aqn, aqk});
    HostTensor<float> w_scale({bqn, bqk});
    HostTensor<CDataType> c_dev({p.M, p.N});

    FillRandom<ADataType, BDataType>(a, b, init);
    FillUniformDistribution<float>{0.05f, 0.25f}(x_scale);
    FillUniformDistribution<float>{0.05f, 0.25f}(w_scale);

    DeviceMem a_buf(a.get_element_space_size_in_bytes());
    DeviceMem b_buf(b.get_element_space_size_in_bytes());
    DeviceMem c_buf(c_dev.get_element_space_size_in_bytes());
    DeviceMem xs_buf(x_scale.get_element_space_size_in_bytes());
    DeviceMem ws_buf(w_scale.get_element_space_size_in_bytes());

    a_buf.ToDevice(a.mData.data());
    b_buf.ToDevice(b.mData.data());
    xs_buf.ToDevice(x_scale.mData.data());
    ws_buf.ToDevice(w_scale.mData.data());
    c_buf.SetZero();

    auto kargs = Kernel::MakeKernelArgs(a_buf.GetDeviceBuffer(),
                                        b_buf.GetDeviceBuffer(),
                                        c_buf.GetDeviceBuffer(),
                                        xs_buf.GetDeviceBuffer(),
                                        ws_buf.GetDeviceBuffer(),
                                        /*p_bias=*/nullptr,
                                        p.M,
                                        p.N,
                                        p.K,
                                        p.stride_a == 0 ? p.K : p.stride_a,
                                        p.stride_b == 0 ? p.K : p.stride_b,
                                        p.stride_c == 0 ? p.N : p.stride_c,
                                        p.k_batch);

    if(!Kernel::IsSupportedArgument(kargs))
    {
        throw std::invalid_argument("benchmark Kargs rejected by IsSupportedArgument()");
    }

    const stream_config s_warmup{nullptr, /*time_kernel=*/false};
    for(int i = 0; i < warmup; ++i)
    {
        c_buf.SetZero();
        launch_gemm_decode_blockscale<Kernel>(kargs, s_warmup);
    }

    const stream_config s_timed{nullptr,
                                /*time_kernel=*/true,
                                /*log_level=*/0,
                                /*cold_niters=*/0,
                                /*nrepeat=*/repeat};
    c_buf.SetZero();
    const float ms = launch_gemm_decode_blockscale<Kernel>(kargs, s_timed);

    const double flops  = gemm_decode_tile_engine::DecodeFlops<ADataType, BDataType>(p);
    const double bytes  = gemm_decode_tile_engine::DecodeBytes<ADataType, BDataType, CDataType>(p);
    const double tflops = flops / (static_cast<double>(ms) * 1.0e9);
    const double gbs    = bytes / (static_cast<double>(ms) * 1.0e6);

    const char* metric_name = metric_kind == 1 ? "TFLOPS" : metric_kind == 2 ? "GB/s" : "ms";
    const double metric_val = metric_kind == 1 ? tflops : metric_kind == 2 ? gbs : ms;

    std::cout << "gemm_decode_blockscale M=" << p.M << " N=" << p.N << " K=" << p.K
              << " k_batch=" << p.k_batch << " | " << ms << " ms, " << tflops << " TFLOP/s, "
              << gbs << " GB/s | " << metric_name << " = " << metric_val << std::endl;

    if(verify > 0)
    {
        HostTensor<CDataType> c_host({p.M, p.N});
        for(index_t m = 0; m < p.M; ++m)
        {
            for(index_t n = 0; n < p.N; ++n)
            {
                float acc = 0.0f;
                for(index_t k = 0; k < p.K; ++k)
                {
                    const float xs = x_scale(m / kXBlockN, k / kXBlockK);
                    const float ws = w_scale(n / kWBlockN, k / kWBlockK);
                    acc += type_convert<float>(a(m, k)) * type_convert<float>(b(n, k)) * xs * ws;
                }
                c_host(m, n) = type_convert<CDataType>(acc);
            }
        }
        c_buf.SetZero();
        launch_gemm_decode_blockscale<Kernel>(kargs, s_warmup);
        c_buf.FromDevice(c_dev.mData.data());

        float max_diff = 0.0f;
        for(index_t i = 0; i < static_cast<index_t>(c_host.get_element_space_size()); ++i)
        {
            const float h = type_convert<float>(c_host.mData[i]);
            const float d = type_convert<float>(c_dev.mData[i]);
            max_diff      = std::max(max_diff, std::abs(h - d));
        }
        const float atol = 1.5f * std::sqrt(static_cast<float>(p.K));
        std::cout << "verify: max_abs_diff = " << max_diff << " (atol = " << atol << ")"
                  << ((max_diff > atol) ? " FAIL" : " PASS") << std::endl;
    }
}

} // namespace

int main(int argc, char* argv[])
{
    try
    {
        auto parser = gemm_decode_tile_engine::create_decode_arg_parser();
        if(!parser.parse(argc, argv))
            return EXIT_FAILURE;

        gemm_decode_tile_engine::DecodeProblem problem;
        problem.M        = parser.get_int("m");
        problem.N        = parser.get_int("n");
        problem.K        = parser.get_int("k");
        problem.stride_a = parser.get_int("stride_a");
        problem.stride_b = parser.get_int("stride_b");
        problem.stride_c = parser.get_int("stride_c");
        problem.k_batch  = parser.get_int("split_k");

        const int warmup      = parser.get_int("warmup");
        const int repeat      = parser.get_int("repeat");
        const int verify      = parser.get_int("verify");
        const int init        = parser.get_int("init");
        const int metric_kind = parser.get_int("metric");

        RunBenchmark<SelectedADataType, SelectedBDataType, SelectedCDataType>(
            problem, warmup, repeat, verify, init, metric_kind);
        return 0;
    }
    catch(const std::exception& e)
    {
        std::cerr << "benchmark error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
