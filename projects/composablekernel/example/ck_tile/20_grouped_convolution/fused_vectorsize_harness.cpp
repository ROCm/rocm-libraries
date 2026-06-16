// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// AICK-1303: runtime harness for the fused-VectorSize conv kernels (vanilla path).
// Sets up a real fp16 2D NHWGC conv, builds kargs via the real ck_tile types, and
// launches the requested kernel with <<<>>> (compiler-handled kernarg ABI). The
// kernels are compiled in (kernels.inc), so this measures the as-compiled (static
// VGPR) kernels: each solo at its own occupancy, and fused_conv pinned to the VS1
// budget. Comparing fused path i vs solo i quantifies the occupancy cost of plain
// fusion - the cost dynamic VGPR would remove (dynamic VGPR itself is unsupported on
// this ROCm; see dvgpr/README.md).
//
// Usage: fused_vectorsize_harness <kernel> <sel> [reps]
//   kernel: solo1|solo2|solo4|solo8|fused_conv ; sel: fused path 0..3 (ignored for solo)
// Prints: kernel,sel,avg_us,checksum
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "fused_vectorsize_probe.hpp"
#include "fused_vectorsize_kernels.inc"

#define HIP_CHECK(x)                                                                        \
    do                                                                                      \
    {                                                                                       \
        hipError_t e = (x);                                                                 \
        if(e != hipSuccess)                                                                 \
        {                                                                                   \
            std::fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, \
                         __LINE__);                                                          \
            std::exit(1);                                                                   \
        }                                                                                   \
    } while(0)

int main(int argc, char** argv)
{
    if(argc < 3)
    {
        std::fprintf(stderr, "usage: %s <kernel> <sel> [reps]\n", argv[0]);
        return 2;
    }
    const std::string kernel = argv[1];
    const int sel            = std::atoi(argv[2]);
    const int reps           = argc > 3 ? std::atoi(argv[3]) : 100;

    // fp16 2D conv, shape from env (defaults = small baseline). C,K must be multiples
    // of 8 so VectorSize 1/2/4/8 stay valid. Override via CONV_N/K/C/HI/WI/FY/FX/STRIDE/PAD.
    using Bld   = FusedConvKernelBuilder<8, 8, 8>;
    auto envi   = [](const char* k, int d) { const char* v = getenv(k); return v ? std::atoi(v) : d; };
    const int N = envi("CONV_N", 64), Kc = envi("CONV_K", 128), Cc = envi("CONV_C", 128);
    const int HI = envi("CONV_HI", 28), WI = envi("CONV_WI", 28);
    const int FY = envi("CONV_FY", 3), FX = envi("CONV_FX", 3);
    const int ST = envi("CONV_STRIDE", 1), PD = envi("CONV_PAD", 1);
    ck_tile::conv::ConvParam conv_param{2,  1,        N,        Kc,       Cc, {FY, FX},
                                        {HI, WI}, {ST, ST}, {1, 1}, {PD, PD}, {PD, PD}};
    std::fprintf(stderr, "shape: G1 N%d K%d C%d %dx%d filt%dx%d s%d p%d\n", N, Kc, Cc, HI, WI, FY, FX, ST, PD);

    const auto in_desc =
        ck_tile::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<Bld::InLayout>(conv_param);
    const auto wei_desc =
        ck_tile::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<Bld::WeiLayout>(conv_param);
    const auto out_desc =
        ck_tile::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<Bld::OutLayout>(conv_param);

    ck_tile::HostTensor<Bld::InDataType> input(in_desc);
    ck_tile::HostTensor<Bld::WeiDataType> weight(wei_desc);
    ck_tile::HostTensor<Bld::OutDataType> output(out_desc);
    ck_tile::FillUniformDistribution<Bld::InDataType>{-1.f, 1.f}(input);
    ck_tile::FillUniformDistribution<Bld::WeiDataType>{-1.f, 1.f}(weight);

    ck_tile::DeviceMem in_buf(input.get_element_space_size_in_bytes());
    ck_tile::DeviceMem wei_buf(weight.get_element_space_size_in_bytes());
    ck_tile::DeviceMem out_buf(output.get_element_space_size_in_bytes());
    in_buf.ToDevice(input.data());
    wei_buf.ToDevice(weight.data());
    out_buf.SetZero();

    ck_tile::GroupedConvFwdHostArgs<> hargs(conv_param, in_buf.GetDeviceBuffer(),
                                            wei_buf.GetDeviceBuffer(), {},
                                            out_buf.GetDeviceBuffer(), 1 /*kbatch*/);

    auto a1 = K1::MakeKernelArgs(hargs);
    auto a2 = K2::MakeKernelArgs(hargs);
    auto a4 = K4::MakeKernelArgs(hargs);
    auto a8 = K8::MakeKernelArgs(hargs);
    if(!K1::IsSupportedArgument(a1))
    {
        std::fprintf(stderr, "error: conv shape unsupported by the kernel; edit the shape.\n");
        return 3;
    }

    const dim3 grid  = K1::GridSize(a1);
    const dim3 block = K1::BlockSize();

    auto launch = [&]() {
        if(kernel == "fused_conv")
            fused_conv<<<grid, block>>>(a1, a2, a4, a8, sel);
        else if(kernel == "solo1")
            solo1<<<grid, block>>>(a1);
        else if(kernel == "solo2")
            solo2<<<grid, block>>>(a2);
        else if(kernel == "solo4")
            solo4<<<grid, block>>>(a4);
        else if(kernel == "solo8")
            solo8<<<grid, block>>>(a8);
        else
        {
            std::fprintf(stderr, "unknown kernel: %s\n", kernel.c_str());
            std::exit(2);
        }
    };

    for(int i = 0; i < 5; ++i) launch();  // warmup
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t t0, t1;
    HIP_CHECK(hipEventCreate(&t0));
    HIP_CHECK(hipEventCreate(&t1));
    HIP_CHECK(hipEventRecord(t0));
    for(int i = 0; i < reps; ++i) launch();
    HIP_CHECK(hipEventRecord(t1));
    HIP_CHECK(hipEventSynchronize(t1));
    float ms = 0.f;
    HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));

    out_buf.FromDevice(output.data());
    double checksum = 0.0;
    for(auto v : output.mData) checksum += static_cast<double>(ck_tile::type_convert<float>(v));

    std::printf("%s,%d,%.3f,%.6e\n", kernel.c_str(), sel, (ms * 1000.0) / reps, checksum);
    return 0;
}
