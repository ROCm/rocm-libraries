// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "ck/host_utility/hip_check_error.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/stream_config.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_multiple_d_dl.hpp"

namespace {

using Row         = ck::tensor_layout::gemm::RowMajor;
using Col         = ck::tensor_layout::gemm::ColumnMajor;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Half        = ck::half_t;
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

// The first existing 64x64 FP16 Col/Row/Row batched DL instance (K1 = 2).
template <typename A               = Half,
          typename B               = Half,
          typename E               = Half,
          typename Acc             = float,
          typename ALayout         = Col,
          typename BLayout         = Row,
          typename ELayout         = Row,
          typename DsLayout        = ck::Tuple<>,
          typename Ds              = ck::Tuple<>,
          typename Epilogue        = PassThrough,
          ck::index_t OutputVector = 4>
using Gemm = ck::tensor_operation::device::DeviceBatchedGemmMultipleD_Dl<
    ALayout,
    BLayout,
    DsLayout,
    ELayout,
    A,
    B,
    Acc,
    Ds,
    E,
    PassThrough,
    PassThrough,
    Epilogue,
    ck::tensor_operation::device::GemmSpecialization::Default,
    64,
    64,
    64,
    8,
    2,
    4,
    4,
    1,
    S<4, 2>,
    S<4, 2>,
    S<2, 1, 4, 2>,
    S<4, 1, 16, 1>,
    S<0, 3, 1, 2>,
    S<0, 3, 1, 2>,
    S<1, 1, 4, 1>,
    S<0, 3, 1, 2>,
    S<1, 1, 4, 2>,
    S<2, 1, 4, 2>,
    S<4, 1, 16, 1>,
    S<0, 3, 1, 2>,
    S<0, 3, 1, 2>,
    S<1, 1, 4, 1>,
    S<0, 3, 1, 2>,
    S<1, 1, 4, 2>,
    S<0, 1, 2, 3, 4, 5>,
    5,
    OutputVector>;

struct Problem
{
    ck::index_t m       = 64;
    ck::index_t n       = 64;
    ck::index_t k       = 64;
    ck::index_t batch   = 2;
    ck::index_t lda     = 68;
    ck::index_t ldb     = 68;
    ck::index_t lde     = 68;
    ck::index_t batch_a = 64 * 68 + 16;
    ck::index_t batch_b = 64 * 68 + 16;
    ck::index_t batch_e = 64 * 68 + 16;
};

template <typename Operation = Gemm<>>
auto argument(const Problem& p, const void* a = nullptr, const void* b = nullptr, void* e = nullptr)
{
    return Operation::MakeArgument(a,
                                   b,
                                   {},
                                   e,
                                   p.m,
                                   p.n,
                                   p.k,
                                   p.batch,
                                   p.lda,
                                   p.ldb,
                                   {},
                                   p.lde,
                                   p.batch_a,
                                   p.batch_b,
                                   {},
                                   p.batch_e,
                                   PassThrough{},
                                   PassThrough{},
                                   {});
}

bool run_problem(const Problem& p)
{
    // Check support before any allocation: gfx90c rejecting a valid case is a failure.
    if(!Gemm<>::IsSupportedArgument(argument(p)))
    {
        std::cerr << "FAIL: valid problem rejected: " << p.m << 'x' << p.n << 'x' << p.k
                  << " batch=" << p.batch << std::endl;
        return false;
    }

    const Half sentinel = ck::type_convert<Half>(-123.5f);
    std::vector<Half> a(p.batch * p.batch_a, sentinel);
    std::vector<Half> b(p.batch * p.batch_b, sentinel);
    std::vector<Half> e(p.batch * p.batch_e, sentinel);
    for(int g = 0; g < p.batch; ++g)
    {
        for(int m = 0; m < p.m; ++m)
        {
            for(int k = 0; k < p.k; ++k)
                a[g * p.batch_a + m + k * p.lda] =
                    ck::type_convert<Half>(((5 * m + 3 * k + 7 * g) % 17 - 8) / 8.0f);
            for(int n = 0; n < p.n; ++n)
                e[g * p.batch_e + m * p.lde + n] =
                    ck::type_convert<Half>(std::numeric_limits<float>::quiet_NaN());
        }
        for(int k = 0; k < p.k; ++k)
            for(int n = 0; n < p.n; ++n)
                b[g * p.batch_b + k * p.ldb + n] =
                    ck::type_convert<Half>(((7 * k + 11 * n + 5 * g) % 19 - 9) / 8.0f);
    }

    ck::DeviceMem device_a(a.size() * sizeof(Half));
    ck::DeviceMem device_b(b.size() * sizeof(Half));
    ck::DeviceMem device_e(e.size() * sizeof(Half));
    device_a.ToDevice(a.data());
    device_b.ToDevice(b.data());
    device_e.ToDevice(e.data());
    auto arg = argument(
        p, device_a.GetDeviceBuffer(), device_b.GetDeviceBuffer(), device_e.GetDeviceBuffer());
    Gemm<>::MakeInvoker().Run(arg, StreamConfig{nullptr, false});
    HIP_CHECK_ERROR(hipGetLastError());
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    device_e.FromDevice(e.data());
    std::vector<Half> actual_a(a.size());
    std::vector<Half> actual_b(b.size());
    device_a.FromDevice(actual_a.data());
    device_b.FromDevice(actual_b.data());
    if(a != actual_a || b != actual_b)
    {
        std::cerr << "FAIL: input or input padding modified" << std::endl;
        return false;
    }

    // Products are multiples of 1/64 and K <= 64, so FP32 partial sums are exact.
    // Compare after the final half conversion; NaN poisoning catches an empty kernel.
    // Inspect the entire allocation, including row padding and inter-batch gaps.
    for(int g = 0; g < p.batch; ++g)
    {
        for(int offset = 0; offset < p.batch_e; ++offset)
        {
            const int m    = offset / p.lde;
            const int n    = offset % p.lde;
            float expected = ck::type_convert<float>(sentinel);
            if(m < p.m && n < p.n)
            {
                float sum = 0;
                for(int k = 0; k < p.k; ++k)
                    sum += ck::type_convert<float>(a[g * p.batch_a + m + k * p.lda]) *
                           ck::type_convert<float>(b[g * p.batch_b + k * p.ldb + n]);
                expected = ck::type_convert<float>(ck::type_convert<Half>(sum));
            }
            const float actual = ck::type_convert<float>(e[g * p.batch_e + offset]);
            if(!std::isfinite(expected) || !std::isfinite(actual) || actual != expected)
            {
                std::cerr << "FAIL: batch " << g << " offset " << offset << " actual=" << actual
                          << " expected=" << expected << std::endl;
                return false;
            }
        }
    }
    std::cout << "PASS: " << p.m << 'x' << p.n << 'x' << p.k << " batch=" << p.batch
              << " lda/ldb/lde=" << p.lda << '/' << p.ldb << '/' << p.lde << std::endl;
    return true;
}

template <typename Operation = Gemm<>>
bool reject(const char* name, const Problem& p = {})
{
    if(Operation::IsSupportedArgument(argument<Operation>(p)))
    {
        std::cerr << "FAIL: accepted " << name << std::endl;
        return false;
    }
    std::cout << "PASS: rejected " << name << std::endl;
    return true;
}

bool offset_boundary_case()
{
    // Admission only: these aligned offsets need no allocation or kernel launch.
    Problem p;
    p.batch   = 2;
    p.batch_a = p.batch_b = p.batch_e = std::numeric_limits<ck::index_t>::max() / 4 * 4;

    const bool accepted = Gemm<>::IsSupportedArgument(argument(p));
    std::cout << (accepted ? "PASS" : "FAIL") << ": accepted aligned batch offset boundary"
              << std::endl;
    return accepted;
}

bool negative_cases()
{
    bool pass = true;
    pass &= reject<Gemm<float, float, float>>("FP32");
    pass &= reject<Gemm<Half, Half, Half, Half>>("half accumulation");
    pass &= reject<Gemm<Half, Half, float>>("float output");
    pass &= reject<Gemm<Half, float>>("mixed inputs");
    pass &= reject<Gemm<Half, Half, Half, float, Row>>("row-major A");
    pass &= reject<Gemm<Half, Half, Half, float, Col, Col>>("column-major B");
    pass &= reject<Gemm<Half, Half, Half, float, Col, Row, Col>>("column-major E");
    pass &= reject<Gemm<Half,
                        Half,
                        Half,
                        float,
                        Col,
                        Row,
                        Row,
                        ck::Tuple<Row>,
                        ck::Tuple<Half>,
                        ck::tensor_operation::element_wise::Add>>("nonempty Ds");
    pass &= reject<Gemm<Half,
                        Half,
                        Half,
                        float,
                        Col,
                        Row,
                        Row,
                        ck::Tuple<>,
                        ck::Tuple<>,
                        ck::tensor_operation::element_wise::Scale>>("scaled epilogue");
    pass &= reject<
        Gemm<Half, Half, Half, float, Col, Row, Row, ck::Tuple<>, ck::Tuple<>, PassThrough, 2>>(
        "unvalidated tuning");
    for(int which = 0; which < 3; ++which)
    {
        // Support checks inspect alignment without dereferencing these pointers.
        auto* misaligned    = reinterpret_cast<void*>(2);
        const auto arg      = argument(Problem{},
                                  which == 0 ? misaligned : nullptr,
                                  which == 1 ? misaligned : nullptr,
                                  which == 2 ? misaligned : nullptr);
        const bool rejected = !Gemm<>::IsSupportedArgument(arg);
        pass &= rejected;
        std::cout << (rejected ? "PASS" : "FAIL") << ": misaligned pointer " << which << std::endl;
    }
    const auto check = [&](const char* name, auto change) {
        Problem p;
        change(p);
        pass &= reject(name, p);
    };
    check("zero M", [](auto& p) { p.m = 0; });
    check("zero N", [](auto& p) { p.n = 0; });
    check("zero K", [](auto& p) { p.k = 0; });
    check("M tile remainder", [](auto& p) { p.m = 63; });
    check("N tile remainder", [](auto& p) { p.n = 63; });
    check("K1 remainder", [](auto& p) { p.k = 17; });
    check("K0 tile remainder", [](auto& p) { p.k = 62; });
    check("A batch offset overflow", [](auto& p) {
        p.batch   = 3;
        p.batch_a = 1 << 30;
    });
    check("B batch offset overflow", [](auto& p) {
        p.batch   = 3;
        p.batch_b = 1 << 30;
    });
    check("E batch offset overflow", [](auto& p) {
        p.batch   = 3;
        p.batch_e = 1 << 30;
    });
    check("zero batch", [](auto& p) { p.batch = 0; });
    check("negative batch", [](auto& p) { p.batch = -1; });
    check("short lda", [](auto& p) { p.lda = 60; });
    check("short ldb", [](auto& p) { p.ldb = 60; });
    check("short lde", [](auto& p) { p.lde = 60; });
    check("unaligned lda", [](auto& p) { p.lda = 65; });
    check("unaligned ldb", [](auto& p) { p.ldb = 65; });
    check("unaligned lde", [](auto& p) { p.lde = 65; });
    check("negative lda", [](auto& p) { p.lda = -4; });
    check("negative ldb", [](auto& p) { p.ldb = -4; });
    check("negative lde", [](auto& p) { p.lde = -4; });
    check("overlapping A batches", [](auto& p) { p.batch_a = p.k * p.lda - 4; });
    check("overlapping B batches", [](auto& p) { p.batch_b = p.k * p.ldb - 4; });
    check("overlapping E batches", [](auto& p) { p.batch_e = p.m * p.lde - 4; });
    check("unaligned A batch", [](auto& p) { ++p.batch_a; });
    check("unaligned B batch", [](auto& p) { ++p.batch_b; });
    check("unaligned E batch", [](auto& p) { ++p.batch_e; });
    check("negative A batch", [](auto& p) { p.batch_a = -4; });
    check("negative B batch", [](auto& p) { p.batch_b = -4; });
    check("negative E batch", [](auto& p) { p.batch_e = -4; });
    return pass;
}

} // namespace

int main()
{
    try
    {
        int device_count = 0;
        HIP_CHECK_ERROR(hipGetDeviceCount(&device_count));
        if(device_count == 0)
        {
            std::cout << "SKIP: no HIP device" << std::endl;
            return 77;
        }
        // This regression covers only the new gfx90c admission contract.
        if(ck::get_device_name() != "gfx90c")
        {
            std::cout << "SKIP: requires gfx90c" << std::endl;
            return 77;
        }
        bool pass = negative_cases();
        pass &= offset_boundary_case();
        for(int batch : {1, 2, 5})
            for(int k :
                {16, 32, 48, 64}) // no-main/single, no-main/double, main/single, main/double
                for(int padding : {0, 4, 12})
                {
                    Problem p;
                    p.m       = 128;
                    p.n       = 192; // six workgroups per batch
                    p.k       = k;
                    p.batch   = batch;
                    p.lda     = p.m + padding;
                    p.ldb     = p.n + padding;
                    p.lde     = p.n + padding;
                    p.batch_a = p.k * p.lda + 16;
                    p.batch_b = p.k * p.ldb + 24;
                    p.batch_e = p.m * p.lde + 32;
                    pass &= run_problem(p);
                }
        return pass ? 0 : 1;
    }
    catch(const std::exception& e)
    {
        std::cerr << "FAIL: " << e.what() << std::endl;
        return 1;
    }
}
