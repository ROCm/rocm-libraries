// MIT License
//
// Copyright (c) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "../common_test_header.hpp"

#include <rocprim/device/config_types.hpp>

#include <hip/hip_runtime.h>

#include "../../common/utils_device_ptr.hpp"

using rocprim::detail::target_arch;

__global__
void write_target_arch([[maybe_unused]] target_arch host_arch, int* __restrict__ result)
{
#if !defined(ROCPRIM_TARGET_SPIRV)
    static constexpr auto arch = rocprim::detail::device_target_arch();

    if constexpr(!ROCPRIM_IS_GENERIC())
    {
        *result = arch == host_arch;
    }
    else
    {
        *result = -2;
    }
#else
    *result = -1;
#endif
}

// If this compile then
TEST(RocprimConfigDispatchTests, StrEqualN)
{
    using rocprim::detail::prefix_equals;

    static_assert(prefix_equals("", "", 0), "");
    ASSERT_TRUE(prefix_equals("", "", 0));
    static_assert(prefix_equals("a", "ab", 1), "");
    ASSERT_TRUE(prefix_equals("a", "ab", 1));
    static_assert(prefix_equals("prefix", "prefixaaa", 6), "");
    ASSERT_TRUE(prefix_equals("prefix", "prefixaaa", 6));

    static_assert(!prefix_equals("a", "b", 0), "");
    ASSERT_FALSE(prefix_equals("a", "b", 0));
    static_assert(!prefix_equals("a", "b", 1), "");
    ASSERT_FALSE(prefix_equals("a", "b", 1));
    static_assert(!prefix_equals("a", "", 1), "");
    ASSERT_FALSE(prefix_equals("a", "", 1));
    static_assert(!prefix_equals("", "b", 100), "");
    ASSERT_FALSE(prefix_equals("", "b", 100));
    static_assert(!prefix_equals("prefix", "prefixaaa", 7), "");
    ASSERT_FALSE(prefix_equals("prefix", "prefixaaa", 7));
    static_assert(!prefix_equals("hasprefix", "hasp", 4), "");
    ASSERT_FALSE(prefix_equals("hasprefix", "hasp", 4));
}

TEST(RocprimConfigDispatchTests, GetTargetArch)
{
    using rocprim::detail::get_target_arch_from_name;

    static_assert(get_target_arch_from_name("gfx_nonexisting", 7) == target_arch::unknown);
    ASSERT_TRUE(get_target_arch_from_name("gfx_nonexisting", 7) == target_arch::unknown);
    static_assert(get_target_arch_from_name("gfx803", 6) == target_arch::gfx803);
    ASSERT_TRUE(get_target_arch_from_name("gfx803", 6) == target_arch::gfx803);
}

#if !defined(ROCPRIM_EXPERIMENTAL_SPIRV) // This macro disables the config_dispatching

TEST(RocprimConfigDispatchTests, HostMatchesDevice)
{
    const int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    // Test with both the default stream (0) and in hipStreamLegacy,
    // since they take different code paths through rocprim::detail::get_device_from_stream.
    for(const hipStream_t stream : {static_cast<hipStream_t>(hipStreamDefault), hipStreamLegacy})
    {
        target_arch host_arch;
        HIP_CHECK(rocprim::detail::host_target_arch(stream, host_arch));

        int* result_ptr = nullptr;
        HIP_CHECK(common::hipMallocHelper(&result_ptr, sizeof(result_ptr)));

        hipLaunchKernelGGL(write_target_arch, dim3(1), dim3(1), 0, stream, host_arch, result_ptr);
        HIP_CHECK(hipGetLastError());

        int result = -1;
        HIP_CHECK(hipMemcpy(&result, result_ptr, sizeof(result), hipMemcpyDeviceToHost));

        if(result != -1)
        {
            if(result != -2)
            {
                ASSERT_NE(host_arch, target_arch::invalid);
                ASSERT_EQ(result, 1);
            }
            else
            {
                GTEST_SKIP() << "Generic build: result is null; skipping arch match assertion.";
            }
        }
        else
        {
            GTEST_SKIP() << "SPIR-V build: result is null; skipping arch match assertion.";
        }

        HIP_CHECK(hipFree(result_ptr));
    }
}

TEST(RocprimConfigDispatchTests, ParseCommonArches)
{
    using rocprim::detail::parse_gcn_arch;
    using rocprim::detail::target_arch;

    ASSERT_EQ(parse_gcn_arch(""), target_arch::unknown);
    ASSERT_EQ(parse_gcn_arch("not a gfx arch"), target_arch::unknown);
    ASSERT_EQ(parse_gcn_arch(":"), target_arch::unknown);
    ASSERT_EQ(parse_gcn_arch("g:"), target_arch::unknown);

    ASSERT_EQ(parse_gcn_arch("gfx803"), target_arch::gfx803);
    ASSERT_EQ(parse_gcn_arch("gfx900:sramecc+"), target_arch::gfx900);
    ASSERT_EQ(parse_gcn_arch("gfx906:::"), target_arch::gfx906);
    ASSERT_EQ(parse_gcn_arch("gfx908:"), target_arch::gfx908);
    ASSERT_EQ(parse_gcn_arch("gfx90a:sramecc+:xnack-"), target_arch::gfx90a);
}

#endif // ROCPRIM_EXPERIMENTAL_SPIRV

#ifndef _WIN32
TEST(RocprimConfigDispatchTests, DeviceIdFromStream)
{
    using rocprim::detail::get_device_from_stream;

    const int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    int                          result;
    static constexpr hipStream_t default_stream = 0;
    HIP_CHECK(get_device_from_stream(default_stream, result));
    ASSERT_EQ(result, device_id);

    HIP_CHECK(get_device_from_stream(hipStreamPerThread, result));
    ASSERT_EQ(result, device_id);

    // hipStreamLegacy support was added in ROCm 6.2.0
    #if(HIP_VERSION_MAJOR > 6 || (HIP_VERSION_MAJOR == 6 && HIP_VERSION_MINOR >= 2))
    HIP_CHECK(get_device_from_stream(hipStreamLegacy, result));
    ASSERT_EQ(result, device_id);
    #endif

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(get_device_from_stream(stream, result));
    HIP_CHECK(hipStreamDestroy(stream));
    ASSERT_EQ(result, device_id);

    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    HIP_CHECK(get_device_from_stream(stream, result));
    HIP_CHECK(hipStreamDestroy(stream));
    ASSERT_EQ(result, device_id);
}
#endif

namespace rocprim
{
namespace detail
{

using Targets
    = comp_targets<comp_target<gen::rdna2, target_arch::gfx1030, gpu::rx6900, rep::amdgcn>,
                   comp_target<gen::rdna3, target_arch::gfx1100, gpu::rx7900, rep::amdgcn>,
                   comp_target<gen::cdna1, target_arch::gfx908, gpu::mi100, rep::amdgcn>,
                   comp_target<gen::cdna2, target_arch::gfx90a, gpu::mi210, rep::amdgcn>,
                   comp_target<gen::cdna3, target_arch::gfx942, gpu::mi300x, rep::amdgcn>,
                   comp_target<gen::cdna3, target_arch::gfx942, gpu::mi300a, rep::amdgcn>,
                   comp_target<gen::rdna4, target_arch::gfx1200, gpu::rx9060, rep::amdgcn>,
                   comp_target<gen::rdna4, target_arch::gfx1201, gpu::rx9070, rep::amdgcn>,
                   comp_target<gen::cdna4, target_arch::gfx950, gpu::mi350x, rep::amdgcn>,
                   comp_target<gen::unknown, target_arch::unknown, gpu::generic, rep::amdgcn>>;

struct test_config_params
{
    kernel_config_params kernel_config{};

    __host__ __device__
    inline bool
        operator==(test_config_params other) const
    {
        return (kernel_config.block_size == other.kernel_config.block_size)
               && (kernel_config.items_per_thread == other.kernel_config.items_per_thread);
    }
};

template<unsigned int BlockSize, unsigned int ItemsPerThread, class TargetsInput>
struct TestSelector
{
    using targets    = TargetsInput;
    using param_type = test_config_params;

    param_type params;

    template<class Target>
    constexpr TestSelector(Target)
        : params(test_config_params{
            {BlockSize, ItemsPerThread}
    })
    {}
};

template<class TargetsInput>
struct TestSelector2
{
    using targets    = TargetsInput;
    using param_type = test_config_params;

    param_type params;

    template<class Target>
    constexpr param_type picker_helper()
    {
        constexpr auto arch = Target::i;
        constexpr auto gen  = Target::g;
        constexpr auto gpu  = Target::s;

        // Assign unique configs per architecture/gen/GPU
        if constexpr(arch == target_arch::gfx1030 && gen == gen::rdna2 && gpu == gpu::rx6900)
        {
            return param_type{
                {64, 1}
            };
        }
        else if constexpr(arch == target_arch::gfx1100 && gen == gen::rdna3 && gpu == gpu::rx7900)
        {
            return param_type{
                {128, 2}
            };
        }
        else if constexpr(arch == target_arch::gfx908 && gen == gen::cdna1 && gpu == gpu::mi100)
        {
            return param_type{
                {192, 3}
            };
        }
        else if constexpr(arch == target_arch::gfx90a && gen == gen::cdna2 && gpu == gpu::mi210)
        {
            return param_type{
                {256, 4}
            };
        }
        else if constexpr(arch == target_arch::gfx942 && gen == gen::cdna3 && gpu == gpu::mi300x)
        {
            return param_type{
                {320, 5}
            };
        }
        else if constexpr(arch == target_arch::gfx942 && gen == gen::cdna3 && gpu == gpu::mi300a)
        {
            return param_type{
                {384, 6}
            };
        }
        else if constexpr(arch == target_arch::gfx1200 && gen == gen::rdna4 && gpu == gpu::rx9060)
        {
            return param_type{
                {448, 7}
            };
        }
        else if constexpr(arch == target_arch::gfx1201 && gen == gen::rdna4 && gpu == gpu::rx9070)
        {
            return param_type{
                {512, 8}
            };
        }
        else if constexpr(arch == target_arch::gfx950 && gen == gen::cdna4 && gpu == gpu::mi350x)
        {
            return param_type{
                {576, 9}
            };
        }
        else
        {
            // Default fallback for unknown targets
            return param_type{
                {1, 1}
            };
        }
    }

    template<class Target>
    constexpr TestSelector2(Target) : params(picker_helper<Target>())
    {}
};

} // namespace detail
} // namespace rocprim

// This test needs to be changed once the selection logic changes for most_common_config.
TEST(RocprimConfigDispatchTests, MostCommonConfig)
{
    using namespace rocprim::detail;

    // 1. Default-constructed target (all unknowns) should return itself.
    ASSERT_EQ(most_common_config<Targets>(target()), target());
    // 2. Exact match (same arch & gpu) should return that target.
    ASSERT_EQ(most_common_config<Targets>(target(target_arch::gfx1200, gpu::rx9060)),
              target(target_arch::gfx1200, gpu::rx9060));
    // 3. Unknown arch/gpu combination should return default unknown.
    ASSERT_EQ(most_common_config<Targets>(target(target_arch::gfx906, gpu::mi50)), target());
    // 4. Multiple entries with same arch — latest wins (mi300a over mi300x).
    ASSERT_EQ(most_common_config<Targets>(target(target_arch::gfx942, gpu::mi308x)),
              target(target_arch::gfx942, gpu::mi300a));
    // 5. Target with same generation but unknown arch — picks matching gen target.
    ASSERT_EQ(most_common_config<Targets>(target(target_arch::gfx1102)),
              target(target_arch::gfx1100, gpu::rx7900));
    // 6. Generation-only match (no arch/gpu known).
    ASSERT_EQ(most_common_config<Targets>(target(gen::rdna2)),
              target(target_arch::gfx1030, gpu::rx6900));
    // 7. Arch-only match (gpu::generic, known arch).
    ASSERT_EQ(most_common_config<Targets>(target(target_arch::gfx90a)),
              target(target_arch::gfx90a, gpu::mi210));
    // 8. If gen has multiple matching targets (cdna3), latest listed is chosen.
    ASSERT_EQ(most_common_config<Targets>(target(gen::cdna3)),
              target(target_arch::gfx942, gpu::mi300a));
}

// This test needs to be changed once the selection logic changes for most_common_config.
TEST(RocprimConfigDispatchTests, DefaultSelectConfig)
{
    using namespace rocprim::detail;

    using Selector = TestSelector2<Targets>;
    using Params   = typename Selector::param_type;

    auto cfg = [](target t) { return default_select_config<Selector>(t); };

    // 1. Default target (unknown)
    ASSERT_EQ(cfg(target()),
              (Params{
                  {1, 1}
    }));

    // 2. Exact match (gfx1200, rx9060)
    ASSERT_EQ(cfg(target(target_arch::gfx1200, gpu::rx9060)),
              (Params{
                  {448, 7}
    }));

    // 3. Unknown arch/gpu
    ASSERT_EQ(cfg(target(target_arch::gfx906, gpu::mi50)),
              (Params{
                  {1, 1}
    }));

    // 4. Multiple entries same arch (gfx942) — latest wins (mi300a)
    ASSERT_EQ(cfg(target(target_arch::gfx942, gpu::mi308x)),
              (Params{
                  {384, 6}
    }));

    // 5. Same gen but unknown arch (gfx1102)
    ASSERT_EQ(cfg(target(target_arch::gfx1102)),
              (Params{
                  {128, 2}
    }));

    // 6. Generation-only match (rdna2)
    ASSERT_EQ(cfg(target(gen::rdna2)),
              (Params{
                  {64, 1}
    }));

    // 7. Arch-only match (gfx90a)
    ASSERT_EQ(cfg(target(target_arch::gfx90a)),
              (Params{
                  {256, 4}
    }));

    // 8. Generation with multiple matches (cdna3 -> gfx942/mi300a)
    ASSERT_EQ(cfg(target(gen::cdna3)),
              (Params{
                  {384, 6}
    }));

    // 9. Other targets in the list
    ASSERT_EQ(cfg(target(target_arch::gfx1201, gpu::rx9070)),
              (Params{
                  {512, 8}
    }));
    ASSERT_EQ(cfg(target(target_arch::gfx950, gpu::mi350x)),
              (Params{
                  {576, 9}
    }));
}

TEST(RocprimConfigDispatchTests, ExecuteLaunchPlan)
{
    using namespace rocprim::detail;
    using EmptyTargets
        = comp_targets<comp_target<gen::unknown, target_arch::unknown, gpu::generic, rep::amdgcn>>;
    using Config = rocprim::default_config;

    constexpr unsigned int block_size = 256;
    constexpr unsigned int ipt        = 2;

    hipStream_t stream = 0;

    const target current_target(stream);

    target* d_output;
    HIP_CHECK(hipMalloc(&d_output, sizeof(target)));

    auto kernel = [=](auto arch_config) { *d_output = decltype(arch_config)::config_target; };

    HIP_CHECK((execute_launch_plan<Config,
                                   TestSelector<block_size, ipt, EmptyTargets>,
                                   default_config_static_selector>(current_target,
                                                                   kernel,
                                                                   dim3(1),
                                                                   dim3(block_size),
                                                                   0,
                                                                   stream)));

    // Impossible target
    target h_output{gen::cdna4, target_arch::gfx942, gpu::rx6900, rep::spirv};
    HIP_CHECK(hipMemcpy(&h_output, d_output, sizeof(target), hipMemcpyDeviceToHost));
    // Compared to targets with only unknown inside.
    ASSERT_EQ(target(), h_output);

    HIP_CHECK((execute_launch_plan<Config,
                                   TestSelector<block_size, ipt, Targets>,
                                   default_config_static_selector>(current_target,
                                                                   kernel,
                                                                   dim3(1),
                                                                   dim3(block_size),
                                                                   0,
                                                                   stream)));

    // Impossible target
    h_output = target{gen::cdna4, target_arch::gfx942, gpu::rx6900, rep::spirv};
    HIP_CHECK(hipMemcpy(&h_output, d_output, sizeof(target), hipMemcpyDeviceToHost));
    // Should have the same targets as most_common_config.
    ASSERT_EQ(most_common_config<Targets>(current_target), h_output);

	// Clean up
	HIP_CHECK(hipFree(d_output));
}
