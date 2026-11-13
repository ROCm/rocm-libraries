// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#pragma once

#include "ck/host_utility/hip_check_error.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/stream_config.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_dl.hpp"

namespace dl_gemm_test {

using Row         = ck::tensor_layout::gemm::RowMajor;
using Col         = ck::tensor_layout::gemm::ColumnMajor;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

// Match the 128x128 factory instances: contiguous M/N transfers for Col A/Row B,
// contiguous K transfers for Row A/Col B. K1=1 retains the original FP32 example.
template <bool ContiguousK, ck::index_t K1>
struct Transfer
{
    using Slice   = S<2, 1, 4, K1>;
    using Cluster = S<8, 1, 32, 1>;
    using Order   = S<0, 3, 1, 2>;
    using Src     = S<1, 1, 4, 1>;
    using Dst     = S<1, 1, 4, K1>;
};

template <ck::index_t K1>
struct Transfer<true, K1>
{
    using Slice   = S<8, 1, 1, K1>;
    using Cluster = S<2, 1, 128, 1>;
    using Order   = S<1, 2, 0, 3>;
    using Src     = S<4, 1, 1, K1>;
    using Dst     = S<1, 1, 1, K1>;
};

template <typename A       = float,
          typename B       = A,
          typename C       = A,
          typename Acc     = float,
          typename ALayout = Col,
          typename BLayout = Row,
          typename CLayout = Row,
          ck::index_t K1   = 2,
          ck::tensor_operation::device::GemmSpecialization Specialization =
              ck::tensor_operation::device::GemmSpecialization::Default,
          typename AT = Transfer<ck::is_same_v<ALayout, Row>, K1>,
          typename BT = Transfer<ck::is_same_v<BLayout, Col>, K1>>
using Gemm = ck::tensor_operation::device::DeviceGemmDl<A,
                                                        B,
                                                        C,
                                                        Acc,
                                                        ALayout,
                                                        BLayout,
                                                        CLayout,
                                                        PassThrough,
                                                        PassThrough,
                                                        PassThrough,
                                                        Specialization,
                                                        256,
                                                        128,
                                                        128,
                                                        16,
                                                        K1,
                                                        4,
                                                        4,
                                                        1,
                                                        S<8, 2>,
                                                        S<8, 2>,
                                                        typename AT::Slice,
                                                        typename AT::Cluster,
                                                        typename AT::Order,
                                                        typename AT::Order,
                                                        typename AT::Src,
                                                        typename AT::Order,
                                                        typename AT::Dst,
                                                        typename BT::Slice,
                                                        typename BT::Cluster,
                                                        typename BT::Order,
                                                        typename BT::Order,
                                                        typename BT::Src,
                                                        typename BT::Order,
                                                        typename BT::Dst,
                                                        S<0, 1, 2, 3, 4, 5>,
                                                        5,
                                                        4>;

template <typename Layout>
ck::index_t offset(ck::index_t row, ck::index_t col, ck::index_t stride)
{
    return ck::is_same_v<Layout, Row> ? row * stride + col : row + col * stride;
}

struct Problem
{
    ck::index_t m;
    ck::index_t n;
    ck::index_t k;
    ck::index_t stride_a;
    ck::index_t stride_b;
    ck::index_t stride_c;
};

template <typename Operation, typename Data, typename ALayout, typename BLayout>
bool run_problem(const Problem& p)
{
    const auto admission = Operation::MakeArgument(nullptr,
                                                   nullptr,
                                                   nullptr,
                                                   p.m,
                                                   p.n,
                                                   p.k,
                                                   p.stride_a,
                                                   p.stride_b,
                                                   p.stride_c,
                                                   PassThrough{},
                                                   PassThrough{},
                                                   PassThrough{});
    if(!Operation::IsSupportedArgument(admission))
    {
        std::cerr << "FAIL: valid problem rejected" << std::endl;
        return false;
    }

    constexpr float sentinel = 123.75f;
    const auto poison        = ck::type_convert<Data>(std::numeric_limits<float>::quiet_NaN());
    std::vector<Data> a((ck::is_same_v<ALayout, Row> ? p.m : p.k) * p.stride_a,
                        ck::type_convert<Data>(sentinel));
    std::vector<Data> b((ck::is_same_v<BLayout, Row> ? p.k : p.n) * p.stride_b,
                        ck::type_convert<Data>(sentinel));
    std::vector<Data> c(p.m * p.stride_c, ck::type_convert<Data>(sentinel));
    for(ck::index_t m = 0; m < p.m; ++m)
    {
        for(ck::index_t k = 0; k < p.k; ++k)
        {
            a[offset<ALayout>(m, k, p.stride_a)] =
                ck::type_convert<Data>(((5 * m + 3 * k) % 17 - 8) / 8.0f);
        }
        for(ck::index_t n = 0; n < p.n; ++n)
        {
            c[m * p.stride_c + n] = poison;
        }
    }
    for(ck::index_t k = 0; k < p.k; ++k)
    {
        for(ck::index_t n = 0; n < p.n; ++n)
        {
            b[offset<BLayout>(k, n, p.stride_b)] =
                ck::type_convert<Data>(((7 * k + 11 * n) % 19 - 9) / 8.0f);
        }
    }

    ck::DeviceMem device_a(a.size() * sizeof(Data));
    ck::DeviceMem device_b(b.size() * sizeof(Data));
    ck::DeviceMem device_c(c.size() * sizeof(Data));
    device_a.ToDevice(a.data());
    device_b.ToDevice(b.data());
    device_c.ToDevice(c.data());
    auto argument = Operation::MakeArgument(static_cast<const Data*>(device_a.GetDeviceBuffer()),
                                            static_cast<const Data*>(device_b.GetDeviceBuffer()),
                                            static_cast<Data*>(device_c.GetDeviceBuffer()),
                                            p.m,
                                            p.n,
                                            p.k,
                                            p.stride_a,
                                            p.stride_b,
                                            p.stride_c,
                                            PassThrough{},
                                            PassThrough{},
                                            PassThrough{});
    Operation::MakeInvoker().Run(argument, StreamConfig{nullptr, false});
    HIP_CHECK_ERROR(hipGetLastError());
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    device_c.FromDevice(c.data());

    // Inputs and products are exact dyadics. K <= 128 keeps all partial sums exact
    // in FP32; convert the final sum to Data to match the FP16 output rounding.
    // Keep exact checks on gfx90c and for FP16. Other FP32 targets need the
    // standalone test's relative/absolute tolerance (including gfx10.3).
    // Padding stays exact, and active C starts as NaN to catch a no-op kernel.
    const bool exact_output = !ck::is_same_v<Data, float> || ck::get_device_name() == "gfx90c";
    for(ck::index_t m = 0; m < p.m; ++m)
    {
        for(ck::index_t n = 0; n < p.stride_c; ++n)
        {
            float expected = sentinel;
            if(n < p.n)
            {
                expected = 0;
                for(ck::index_t k = 0; k < p.k; ++k)
                {
                    expected += ck::type_convert<float>(a[offset<ALayout>(m, k, p.stride_a)]) *
                                ck::type_convert<float>(b[offset<BLayout>(k, n, p.stride_b)]);
                }
            }
            expected            = ck::type_convert<float>(ck::type_convert<Data>(expected));
            const float actual  = ck::type_convert<float>(c[m * p.stride_c + n]);
            const bool mismatch = exact_output || n >= p.n ? actual != expected
                                                           : std::abs(actual - expected) >
                                                                 1e-3f * std::abs(expected) + 1e-6f;
            if(!std::isfinite(actual) || mismatch)
            {
                std::cerr << "FAIL: C[" << m << ", " << n << "] = " << actual << ", expected "
                          << expected << std::endl;
                return false;
            }
        }
    }
    std::cout << "PASS" << std::endl;
    return true;
}

template <typename Operation = Gemm<>>
bool expect_rejection(const char* name,
                      ck::index_t m        = 128,
                      ck::index_t n        = 128,
                      ck::index_t k        = 64,
                      ck::index_t stride_a = 256,
                      ck::index_t stride_b = 256,
                      ck::index_t stride_c = 256)
{
    // Admission checks do not launch kernels or dereference the data pointers.
    auto argument       = Operation::MakeArgument(nullptr,
                                            nullptr,
                                            nullptr,
                                            m,
                                            n,
                                            k,
                                            stride_a,
                                            stride_b,
                                            stride_c,
                                            PassThrough{},
                                            PassThrough{},
                                            PassThrough{});
    const bool rejected = !Operation::IsSupportedArgument(argument);
    std::cout << name << ": " << (rejected ? "PASS (rejected)" : "FAIL (accepted)") << std::endl;
    return rejected;
}

template <typename Operation, typename Data, typename ALayout, typename BLayout, ck::index_t KBlock>
bool run_suite(const char* name)
{
    // Four K-block counts exercise main-loop absent/present and single/double tail.
    // Nonsquare cases exercise workgroup indexing; padding preserves vector alignment.
    bool pass = true;
    for(int repeat = 0; repeat < 10; ++repeat)
    {
        for(const auto& shape : std::vector<Problem>{{128, 128, KBlock, 0, 0, 0},
                                                     {128, 128, 2 * KBlock, 0, 0, 0},
                                                     {128, 128, 3 * KBlock, 0, 0, 0},
                                                     {128, 128, 4 * KBlock, 0, 0, 0},
                                                     {256, 128, 4 * KBlock, 0, 0, 0},
                                                     {128, 256, 4 * KBlock, 0, 0, 0},
                                                     {256, 128, 4 * KBlock, 8, 8, 8}})
        {
            const Problem p{shape.m,
                            shape.n,
                            shape.k,
                            (ck::is_same_v<ALayout, Row> ? shape.k : shape.m) + shape.stride_a,
                            (ck::is_same_v<BLayout, Row> ? shape.n : shape.k) + shape.stride_b,
                            shape.n + shape.stride_c};
            std::cout << name << " run " << repeat + 1 << "/10 M=" << p.m << " N=" << p.n
                      << " K=" << p.k << " strides=" << p.stride_a << "," << p.stride_b << ","
                      << p.stride_c << ": " << std::flush;
            pass = run_problem<Operation, Data, ALayout, BLayout>(p) && pass;
        }
    }
    pass = expect_rejection<Operation>("M vector/block alignment", 130) && pass;
    pass = expect_rejection<Operation>("N vector/block alignment", 128, 130) && pass;
    pass = expect_rejection<Operation>("M block alignment", 132) && pass;
    pass = expect_rejection<Operation>("N block alignment", 128, 132) && pass;
    // Respect the constructor's K % K1 precondition while violating block alignment.
    pass = expect_rejection<Operation>(
               "K block/vector alignment", 128, 128, KBlock + Operation::K1Number.value) &&
           pass;
    if constexpr(Operation::K1Number.value > 1)
    {
        if(ck::get_device_name() == "gfx90c")
        {
            // Construct valid descriptors before modeling a release-mode truncated K.
            // This checks raw admission without violating debug constructor assertions.
            auto argument       = Operation::MakeArgument(nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    128,
                                                    128,
                                                    KBlock,
                                                    256,
                                                    256,
                                                    256,
                                                    PassThrough{},
                                                    PassThrough{},
                                                    PassThrough{});
            argument.K_raw_     = KBlock + 1;
            const bool rejected = !Operation::IsSupportedArgument(argument);
            std::cout << "Raw K1 alignment: " << (rejected ? "PASS (rejected)" : "FAIL (accepted)")
                      << std::endl;
            pass = rejected && pass;
        }
    }
    return pass;
}

template <typename Test>
int run_on_native_device(Test test)
{
    try
    {
        int device = 0;
        hipDeviceProp_t properties{};
        HIP_CHECK_ERROR(hipGetDevice(&device));
        HIP_CHECK_ERROR(hipGetDeviceProperties(&properties, device));
        int runtime_version = 0;
        int driver_version  = 0;
        HIP_CHECK_ERROR(hipRuntimeGetVersion(&runtime_version));
        HIP_CHECK_ERROR(hipDriverGetVersion(&driver_version));
        const auto arch = ck::get_device_name();
        std::cout << "Device " << device << ": " << properties.name << ", native architecture "
                  << properties.gcnArchName << ", HIP runtime version " << runtime_version
                  << ", HIP driver version " << driver_version << std::endl;
        if(arch != "gfx90c" && arch != "gfx906" && !ck::is_gfx103_supported() &&
           !ck::is_gfx11_supported() && !ck::is_gfx117_supported() && !ck::is_gfx12_supported())
        {
            std::cout << "SKIP: native device does not support this DL GEMM test" << std::endl;
            return 77;
        }

        return test(arch);
    }
    catch(const std::exception& error)
    {
        std::cerr << "FAIL: " << error.what() << std::endl;
        return 1;
    }
}

} // namespace dl_gemm_test
