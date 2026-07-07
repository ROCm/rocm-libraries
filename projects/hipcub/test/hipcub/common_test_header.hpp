// MIT License
//
// Copyright (c) 2020-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// Google Test
#include <gtest/gtest.h>

// HIP API
#include <hip/hip_runtime.h>

// These are used by the TempDisablement class, and can be removed
// once we have proper fixes for the issues it is working around.
#if defined(HIPCUB_ROCPRIM_API)
#include <rocprim/device/config_types.hpp>
#include <set>
#include <algorithm>
#endif

// test_utils.hpp should only be included by this header.
// The following definition is used as guard in test_utils.hpp
// Including test_utils.hpp by itself will cause a compile error.
#define TEST_UTILS_INCLUDE_GAURD
#include "test_utils.hpp"

#if defined(__SANITIZE_ADDRESS__) || (defined(__has_feature) && __has_feature(address_sanitizer))
    #define GTEST_SKIP_ASAN()                           \
        do                                              \
        {                                               \
            GTEST_SKIP() << "Skipping test under ASan"; \
        }                                               \
        while(0)
#else
    #define GTEST_SKIP_ASAN()
#endif

#define HIP_CHECK(condition)         \
{                                    \
    hipError_t error = condition;    \
    if(error != hipSuccess){         \
        std::cout << "HIP error: " << hipGetErrorString(error) << " line: " << __LINE__ << std::endl; \
        exit(error); \
    } \
}

#define INSTANTIATE_TYPED_TEST_EXPANDED_1(line, test_suite_name, ...) \
    namespace Id##line {                                              \
        using test_type = __VA_ARGS__;                                \
        INSTANTIATE_TYPED_TEST_SUITE_P(                               \
            Id##line, test_suite_name, test_type);                    \
    }

#define INSTANTIATE_TYPED_TEST_EXPANDED(line, test_suite_name, ...) \
    INSTANTIATE_TYPED_TEST_EXPANDED_1(line, test_suite_name, __VA_ARGS__)

// Used in input file for hipcub_test_add_parallel.
// Instantiate a typed test suite with a unique name based on line number.
// Do not call this macro twice on the same line.
#define INSTANTIATE_TYPED_TEST(test_suite_name, ...) \
    INSTANTIATE_TYPED_TEST_EXPANDED(__LINE__, test_suite_name, __VA_ARGS__)

namespace test_common_utils
{

inline char* __get_env(const char* name)
{
    char* env;
#ifdef _MSC_VER
    size_t  len;
    errno_t err = _dupenv_s(&env, &len, name);
    if(err)
    {
        return nullptr;
    }
#else
    env = std::getenv(name);
#endif
    return env;
}

inline void clean_env(char* env)
{
#ifdef _MSC_VER
    if(env != nullptr)
    {
        free(env);
    }
#endif
    (void)env;
}

inline int obtain_device_from_ctest()
{
    static const std::string rg0    = "CTEST_RESOURCE_GROUP_0";
    char*                    env    = __get_env(rg0.c_str());
    int                      device = 0;
    if(env != nullptr)
    {
        std::string amdgpu_target(env);
        std::transform(
            amdgpu_target.cbegin(),
            amdgpu_target.cend(),
            amdgpu_target.begin(),
            // Feeding std::toupper plainly results in implicitly truncating conversions between int and char triggering warnings.
            [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
        char*       env_reqs = __get_env((rg0 + "_" + amdgpu_target).c_str());
        std::string reqs(env_reqs);
        device = std::atoi(
            reqs.substr(reqs.find(':') + 1, reqs.find(',') - (reqs.find(':') + 1)).c_str());
        clean_env(env_reqs);
    }
    clean_env(env);
    return device;
}

inline bool use_hmm()
{

    char*      env = __get_env("HIPCUB_USE_HMM");
    const bool hmm = (env != nullptr) && (strcmp(env, "1") == 0);
    clean_env(env);
    return hmm;
}

// Helper for HMM allocations: HMM is requested through HIPCUB_USE_HMM environment variable
template <class T>
hipError_t hipMallocHelper(T** devPtr, size_t size)
{
    if (use_hmm())
    {
        return hipMallocManaged((void**)devPtr, size);
    }
    else
    {
        return hipMalloc((void**)devPtr, size);
    }
    return hipSuccess;
}

// Temporary functions to disable larger-sized tests on gfx115x devices.
// TODO: remove this once it's replaced with a more robust fix.
class TempDisablement
{
public:
	static bool is_arch_disabled()
	{
#if defined(HIPCUB_ROCPRIM_API)
		rocprim::detail::target_arch arch = rocprim::detail::target_arch::unknown;
		if (rocprim::detail::host_target_arch(hipStreamDefault, arch) != HIP_SUCCESS)
		{
			std::cerr << "Warning: unable to fetch target architecture for disablement check." << std::endl;
		}

		const std::set<rocprim::detail::target_arch> disabled_arches = {
			rocprim::detail::target_arch::gfx1150,
			rocprim::detail::target_arch::gfx1151,
			rocprim::detail::target_arch::gfx1152,
			rocprim::detail::target_arch::gfx1153
		};

		return disabled_arches.find(arch) != disabled_arches.end();
#else
		return false;
#endif
	}

	static std::vector<size_t> filter_sizes(std::vector<size_t> sizes)
	{
#if defined(HIPCUB_ROCPRIM_API)
		if (TempDisablement::is_arch_disabled())
		{
			auto it = std::remove_if(sizes.begin(), sizes.end(), [](const size_t& size) {
				return size > TempDisablement::size_limit;
			});
			sizes.erase(it, sizes.end());
		}
#endif

		return sizes;
	}
	
	static std::vector<std::tuple<size_t, size_t>> filter_sizes(std::vector<std::tuple<size_t, size_t>> sizes)
	{
#if defined(HIPCUB_ROCPRIM_API)
		if (TempDisablement::is_arch_disabled())
		{
			auto it = std::remove_if(sizes.begin(), sizes.end(), [](const std::tuple<size_t, size_t>& size) {
				return std::get<0>(size) + std::get<1>(size) > TempDisablement::size_limit;
			});
			sizes.erase(it, sizes.end());
		}
#endif

		return sizes;
	}
	
private:
	static inline const size_t size_limit = 10000;
};

}
