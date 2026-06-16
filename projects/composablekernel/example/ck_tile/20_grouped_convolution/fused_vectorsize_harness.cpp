// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// AICK-1303: runtime harness for the fused-VectorSize dynamic-VGPR experiment.
// Sets up a real fp16 2D NHWGC conv problem, builds the kernel args via the real
// ck_tile types, loads a code object (.hsaco) at runtime, and launches the requested
// kernel via hipModuleLaunchKernel. The SAME harness runs the vanilla or the patched
// (dynamic-VGPR) .hsaco, so the only variable is the device code.
//
// Usage: fused_vectorsize_harness <code_object.hsaco> <kernel> <sel> [reps]
//   kernel: solo1|solo2|solo4|solo8|fused_conv
//   sel:    fused path 0..3 (VS1/VS2/VS4/VS8); ignored for solo kernels
// Prints: kernel,sel,avg_us,checksum  (checksum = output sum, a cheap correctness proxy)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "fused_vectorsize_probe.hpp"

#define HIP_CHECK(x)                                                                       \
    do                                                                                     \
    {                                                                                      \
        hipError_t e = (x);                                                                \
        if(e != hipSuccess)                                                                \
        {                                                                                  \
            std::fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e),          \
                         __FILE__, __LINE__);                                              \
            std::exit(1);                                                                  \
        }                                                                                  \
    } while(0)

// Kernarg buffer for fused_conv: mirrors the kernel signature (4 kargs + sel). For
// solo kernels the buffer is just the single kargs struct.
struct FusedArgs
{
    K1::GroupedConvFwdKernelArgsSpecialized a1;
    K2::GroupedConvFwdKernelArgsSpecialized a2;
    K4::GroupedConvFwdKernelArgsSpecialized a4;
    K8::GroupedConvFwdKernelArgsSpecialized a8;
    int sel;
};

int main(int argc, char** argv)
{
    if(argc < 4)
    {
        std::fprintf(stderr,
                     "usage: %s <code_object.hsaco> <kernel> <sel> [reps]\n", argv[0]);
        return 2;
    }
    const std::string hsaco  = argv[1];
    const std::string kernel = argv[2];
    const int sel            = std::atoi(argv[3]);
    const int reps           = argc > 4 ? std::atoi(argv[4]) : 50;

    // Representative fp16 2D conv: G=1 N=64 K=128 C=128, 3x3 filter, 28x28, stride1 pad1.
    // C and K are multiples of 8 so VectorSize 1/2/4/8 are all valid.
    using B = FusedConvKernelBuilder<8, 8, 8>;
    ck_tile::conv::ConvParam conv_param{2, 1, 64, 128, 128, {3, 3}, {28, 28},
                                        {1, 1}, {1, 1}, {1, 1}, {1, 1}};

    const auto in_desc =
        ck_tile::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<B::InLayout>(conv_param);
    const auto wei_desc =
        ck_tile::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<B::WeiLayout>(conv_param);
    const auto out_desc =
        ck_tile::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<B::OutLayout>(conv_param);

    ck_tile::HostTensor<B::InDataType> input(in_desc);
    ck_tile::HostTensor<B::WeiDataType> weight(wei_desc);
    ck_tile::HostTensor<B::OutDataType> output(out_desc);
    ck_tile::FillUniformDistribution<B::InDataType>{-1.f, 1.f}(input);
    ck_tile::FillUniformDistribution<B::WeiDataType>{-1.f, 1.f}(weight);

    ck_tile::DeviceMem in_buf(input.get_element_space_size_in_bytes());
    ck_tile::DeviceMem wei_buf(weight.get_element_space_size_in_bytes());
    ck_tile::DeviceMem out_buf(output.get_element_space_size_in_bytes());
    in_buf.ToDevice(input.data());
    wei_buf.ToDevice(weight.data());
    out_buf.SetZero();

    ck_tile::GroupedConvFwdHostArgs<> hargs(conv_param,
                                            in_buf.GetDeviceBuffer(),
                                            wei_buf.GetDeviceBuffer(),
                                            {},
                                            out_buf.GetDeviceBuffer(),
                                            1 /*kbatch*/);

    // Build kargs (identical content across VectorSize; the kernel picks the path).
    auto k1 = K1::MakeKernelArgs(hargs);
    if(!K1::IsSupportedArgument(k1))
        std::fprintf(stderr, "warning: argument reported unsupported for this shape\n");

    FusedArgs fused{K1::MakeKernelArgs(hargs), K2::MakeKernelArgs(hargs),
                    K4::MakeKernelArgs(hargs), K8::MakeKernelArgs(hargs), sel};

    void*  argbuf  = (kernel == "fused_conv") ? static_cast<void*>(&fused)
                                              : static_cast<void*>(&k1);
    size_t argsize = (kernel == "fused_conv") ? sizeof(fused) : sizeof(k1);

    const dim3 grid  = K1::GridSize(k1);
    const dim3 block = K1::BlockSize();

    hipModule_t mod;
    hipFunction_t fn;
    HIP_CHECK(hipModuleLoad(&mod, hsaco.c_str()));
    HIP_CHECK(hipModuleGetFunction(&fn, mod, kernel.c_str()));

    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, argbuf,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE, &argsize,
                      HIP_LAUNCH_PARAM_END};

    auto launch = [&]() {
        HIP_CHECK(hipModuleLaunchKernel(fn, grid.x, grid.y, grid.z, block.x, block.y, block.z,
                                        0, nullptr, nullptr, config));
    };

    for(int i = 0; i < 5; ++i) launch();  // warmup
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
    HIP_CHECK(hipModuleUnload(mod));
    return 0;
}
