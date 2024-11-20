/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include "../testing/unittest/random.h"
#include "bitwise_repro/bwr_db.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <list>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

// HIP API
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

// GoogleTest-compatible HIP_CHECK macro. FAIL is called to log the Google Test trace.
// The lambda is invoked immediately as assertions that generate a fatal failure can
// only be used in void-returning functions.
#ifndef HIP_CHECK
#define HIP_CHECK(condition)                                                                      \
    do                                                                                            \
    {                                                                                             \
        hipError_t error = condition;                                                             \
        if(error != hipSuccess)                                                                   \
        {                                                                                         \
            [error]() { FAIL() << "HIP error " << error << ": " << hipGetErrorString(error); }(); \
            exit(error);                                                                          \
        }                                                                                         \
    } while(0)
#endif

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HIP

#include "test_assertions.hpp"
#include "test_utils.hpp"

#include <cstdlib>
#include <string>
#include <cctype>

namespace test
{

inline char* get_env(const char* name)
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

inline void clean_env(char* name)
{
#ifdef _MSC_VER
    if(name != nullptr)
    {
        free(name);
    }
#endif
    (void)name;
}

inline int set_device_from_ctest()
{
    static const std::string rg0    = "CTEST_RESOURCE_GROUP_0";
    char*                    env    = get_env(rg0.c_str());
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
        char*       env_reqs = get_env((rg0 + "_" + amdgpu_target).c_str());
        std::string reqs(env_reqs);
        device = std::atoi(
            reqs.substr(reqs.find(':') + 1, reqs.find(',') - (reqs.find(':') + 1)).c_str());
        clean_env(env_reqs);
        HIP_CHECK(hipSetDevice(device));
    }
    clean_env(env);
    return device;
}
}

// If enabled, set up the database for inter-run bitwise reproducibility testing.
// Inter-run testing is enabled through the following environment variables:
// ROCTHRUST_BWR_PATH - path to the database (or where it should be created)
// ROCTHRUST_BWR_GENERATE - if set to 1, info about any function calls not
// found in the database will be inserted. No errors will be reported in this mode.
namespace inter_run_bwr
{
    // Disable this testing by default.
    bool enabled = false;

    // This code doesn't need to be visible outside this file.
    namespace
    {
        const static std::string path_env = "ROCTHRUST_BWR_PATH";
        const static std::string generate_env = "ROCTHRUST_BWR_GENERATE";

        // Check the environment variables to see if the database should be
        // instantiated, and if so, what mode it should be in.
        std::unique_ptr<BitwiseReproDB> create_db()
        {
            // Get the path to the database from an environment variable.
            const char* db_path = std::getenv(path_env.c_str());
            const char* db_mode = std::getenv(generate_env.c_str());
            if (db_path)
            {
                // Check if we are allowed to insert rows into the database if
                // we encounter calls that aren't already recorded.
                BitwiseReproDB::Mode mode = BitwiseReproDB::Mode::test_mode;
                if (db_mode && std::stoi(db_mode) > 0)
                    mode = BitwiseReproDB::Mode::generate_mode;

                enabled = true;
                return std::make_unique<BitwiseReproDB>(db_path, mode);
            }
            else if (db_mode)
            {
                throw std::runtime_error("ROCTHRUST_BWR_GENERATE is defined, but no database path was given.\n"
                    "Please set ROCTHRUST_BWR_PATH to the database path.");
            }

            return nullptr;
        }
    }

    // Create/open the run-to-run bitwise reproducibility database.
    std::unique_ptr<BitwiseReproDB> db = create_db();
}

// Input type parameter
template <class InputType, class ExecutionPolicy = std::decay_t<decltype(thrust::hip::par)>>
struct Params
{
    using input_type = InputType;
    using execution_policy = thrust::detail::host_t;
};

template <class T, class ExecutionPolicy>
struct Params<thrust::device_vector<T>, ExecutionPolicy>
{
    using input_type = thrust::device_vector<T>;
    using execution_policy = ExecutionPolicy;
};

// Definition of typed test cases with given parameter type
#define TESTS_DEFINE(x, y)                                          \
    template <class Params>                                         \
    class x : public ::testing::Test                                \
    {                                                               \
    public:                                                         \
        using input_type = typename Params::input_type;             \
        using execution_policy = typename Params::execution_policy; \
    };                                                              \
                                                                    \
    TYPED_TEST_SUITE(x, y);

// Set of test parameter types

class large_data
{
public:
    __host__ __device__ large_data()
    {
        data[0] = 0;
    }
    __host__ __device__ large_data(large_data const & val)
    {
        data[0] = val.data[0];
    }
    __host__ __device__ large_data(int n)
    {
        data[0] = static_cast<int8_t>(n);
    }
    large_data& __host__ __device__ operator=(large_data const & val)
    {
        data[0] = val.data[0];
        return *this;
    }
    bool __host__ __device__ operator==(large_data const & val) const
    {
        return data[0] == val.data[0];
    }
    large_data& __host__ __device__ operator++()
    {
        ++data[0];
        return *this;
    }
    __host__ __device__ operator int() const
    {
        return static_cast<int>(data[0]);
    }

    int8_t data[512];
};

template<class T>
bool __host__ __device__ operator==(T const & lhs, large_data const & rhs)
{
    return static_cast<large_data>(lhs).data[0] == rhs.data[0];
}

// Host and device vectors of all type as a test parameter
typedef ::testing::Types<
    Params<thrust::host_vector<short>>,
    Params<thrust::host_vector<int>>,
    Params<thrust::host_vector<long long>>,
    Params<thrust::host_vector<unsigned short>>,
    Params<thrust::host_vector<unsigned int>>,
    Params<thrust::host_vector<unsigned long long>>,
    Params<thrust::host_vector<float>>,
    Params<thrust::host_vector<double>>,
    Params<thrust::device_vector<short>>,
    Params<thrust::device_vector<int>>,
    Params<thrust::device_vector<int>, std::decay_t<decltype(thrust::hip::par_nosync)>>,
    Params<thrust::device_vector<long long>>,
    Params<thrust::device_vector<unsigned short>>,
    Params<thrust::device_vector<unsigned int>>,
    Params<thrust::device_vector<unsigned long long>>,
    Params<thrust::device_vector<float>>,
    Params<thrust::device_vector<float>, std::decay_t<decltype(thrust::hip::par_det)>>,
    Params<thrust::device_vector<float>, std::decay_t<decltype(thrust::hip::par_det_nosync)>>,
    Params<thrust::device_vector<double>>>
    FullTestsParams;

typedef ::testing::Types<
    Params<thrust::host_vector<short>>,
    Params<thrust::host_vector<int>>,
    Params<thrust::host_vector<long long>>,
    Params<thrust::host_vector<unsigned short>>,
    Params<thrust::host_vector<unsigned int>>,
    Params<thrust::host_vector<unsigned long long>>,
    Params<thrust::host_vector<float>>,
    Params<thrust::host_vector<double>>,
    Params<thrust::device_vector<short>>,
    Params<thrust::device_vector<int>>,
    Params<thrust::device_vector<int>, std::decay_t<decltype(thrust::hip::par_nosync)>>,
    Params<thrust::device_vector<long long>>,
    Params<thrust::device_vector<unsigned short>>,
    Params<thrust::device_vector<unsigned int>>,
    Params<thrust::device_vector<unsigned long long>>,
    Params<thrust::device_vector<float>>,
    Params<thrust::device_vector<float>, std::decay_t<decltype(thrust::hip::par_det)>>,
    Params<thrust::device_vector<float>, std::decay_t<decltype(thrust::hip::par_det_nosync)>>,
    Params<thrust::device_vector<double>>,
    Params<thrust::device_vector<large_data>>>
    FullWithLargeTypesTestsParams;

// Host and device vectors of signed type
typedef ::testing::Types<Params<thrust::host_vector<short>>,
                         Params<thrust::host_vector<int>>,
                         Params<thrust::host_vector<long long>>,
                         Params<thrust::host_vector<float>>,
                         Params<thrust::host_vector<double>>,
                         Params<thrust::device_vector<short>>,
                         Params<thrust::device_vector<int>>,
                         Params<thrust::device_vector<long long>>,
                         Params<thrust::device_vector<float>>,
                         Params<thrust::device_vector<double>>>
    VectorSignedTestsParams;

// Host and device vectors of integer types as a test parameter
typedef ::testing::Types<Params<thrust::host_vector<short>>,
                         Params<thrust::host_vector<int>>,
                         Params<thrust::host_vector<long long>>,
                         Params<thrust::host_vector<unsigned short>>,
                         Params<thrust::host_vector<unsigned int>>,
                         Params<thrust::host_vector<unsigned long long>>,
                         Params<thrust::device_vector<short>>,
                         Params<thrust::device_vector<int>>,
                         Params<thrust::device_vector<long long>>,
                         Params<thrust::device_vector<unsigned short>>,
                         Params<thrust::device_vector<unsigned int>>,
                         Params<thrust::device_vector<unsigned long long>>>
    VectorIntegerTestsParams;

// Host and device vectors of signed integer types as a test parameter
typedef ::testing::Types<Params<thrust::host_vector<short>>,
                         Params<thrust::host_vector<int>>,
                         Params<thrust::host_vector<long long>>,
                         Params<thrust::device_vector<short>>,
                         Params<thrust::device_vector<int>>,
                         Params<thrust::device_vector<long long>>>
    VectorSignedIntegerTestsParams;

// Host vectors of numerical types as a test parameter
typedef ::testing::Types<Params<thrust::host_vector<short>>,
                         Params<thrust::host_vector<int>>,
                         Params<thrust::host_vector<long long>>,
                         Params<thrust::host_vector<unsigned short>>,
                         Params<thrust::host_vector<unsigned int>>,
                         Params<thrust::host_vector<unsigned long long>>,
                         Params<thrust::host_vector<float>>,
                         Params<thrust::host_vector<double>>>
    HostVectorTestsParams;

// Host vectors of integer types as a test parameter
typedef ::testing::Types<Params<thrust::host_vector<short>>,
                         Params<thrust::host_vector<int>>,
                         Params<thrust::host_vector<long long>>,
                         Params<thrust::host_vector<unsigned short>>,
                         Params<thrust::host_vector<unsigned int>>,
                         Params<thrust::host_vector<unsigned long long>>>
    HostVectorIntegerTestsParams;

// Scalar numerical types
typedef ::testing::Types<Params<short>,
                         Params<int>,
                         Params<long long>,
                         Params<unsigned short>,
                         Params<unsigned int>,
                         Params<unsigned long long>,
                         Params<float>,
                         Params<double>>
    NumericalTestsParams;

// Scalar integer types
typedef ::testing::Types<Params<short>,
                         Params<int>,
                         Params<long long>,
                         Params<unsigned short>,
                         Params<unsigned int>,
                         Params<unsigned long long>>
    IntegerTestsParams;

// Scalar signed integer types
typedef ::testing::Types<Params<short>, Params<int>, Params<long long>> SignedIntegerTestsParams;

#if defined(_WIN32) && defined(__HIP__)
// Scalar unsigned integer types of all lengths
typedef ::testing::Types<Params<thrust::detail::uint16_t>,
                         Params<thrust::detail::uint32_t>,
                         Params<thrust::detail::uint64_t>>
    UnsignedIntegerTestsParams;

typedef ::testing::Types<Params<short>,
                         Params<int>,
                         Params<long long>,
                         Params<unsigned short>,
                         Params<unsigned int>,
                         Params<unsigned long long>,
                         Params<thrust::detail::uint16_t>,
                         Params<thrust::detail::uint32_t>,
                         Params<thrust::detail::uint64_t>>
    AllIntegerTestsParams;
#else
// Scalar unsigned integer types of all lengths
typedef ::testing::Types<Params<thrust::detail::uint8_t>,
                         Params<thrust::detail::uint16_t>,
                         Params<thrust::detail::uint32_t>,
                         Params<thrust::detail::uint64_t>>
    UnsignedIntegerTestsParams;

// Scalar all integer types
typedef ::testing::Types<Params<short>,
                         Params<int>,
                         Params<long long>,
                         Params<unsigned short>,
                         Params<unsigned int>,
                         Params<unsigned long long>,
                         Params<thrust::detail::uint8_t>,
                         Params<thrust::detail::uint16_t>,
                         Params<thrust::detail::uint32_t>,
                         Params<thrust::detail::uint64_t>>
    AllIntegerTestsParams;
#endif


// Scalar float types
typedef ::testing::Types<Params<float>, Params<double>> FloatTestsParams;

// --------------------Input Output test parameters--------
template <class Input, class Output = Input>
struct ParamsInOut
{
    using input_type  = Input;
    using output_type = Output;
};

// Definition of typed test cases with given parameter type
#define TESTS_INOUT_DEFINE(x, y)                               \
    template <class ParamsInOut>                               \
    class x : public ::testing::Test                           \
    {                                                          \
    public:                                                    \
        using input_type  = typename ParamsInOut::input_type;  \
        using output_type = typename ParamsInOut::output_type; \
    };                                                         \
                                                               \
    TYPED_TEST_SUITE(x, y);

typedef ::testing::Types<ParamsInOut<short>,
                         ParamsInOut<int>,
                         ParamsInOut<long long>,
                         ParamsInOut<unsigned short>,
                         ParamsInOut<unsigned int>,
                         ParamsInOut<unsigned long long>,
                         ParamsInOut<float>,
                         ParamsInOut<double>,
                         ParamsInOut<int, long long>,
                         ParamsInOut<unsigned int, unsigned long long>,
                         ParamsInOut<float, double>>
    AllInOutTestsParams;
