// Copyright (c) 2017-2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <gtest/gtest.h>
#include <stdio.h>

#include <cmath>
#include <numeric>
#include <random>
#include <type_traits>

#include "../test_common.hpp"

#undef ROCRAND_DETAIL_BM_NOT_IN_STATE
#include <rocrand/rocrand_log_normal.h>

#define ROCRAND_CHECK(cmd)                                                                \
    do                                                                                    \
    {                                                                                     \
        auto status = cmd;                                                                \
        if(status != 0)                                                                   \
        {                                                                                 \
            std::cerr << "Encountered ROCRAND error: " << status << "at line" << __LINE__ \
                      << " in file " << __FILE__ << "\n";                                 \
            exit(status);                                                                 \
        }                                                                                 \
    }                                                                                     \
    while(0)

// If x is small then get withing 0.001 otherwise 5%
#define GET_EPS(x) x < 0.01 ? 0.001 : x * 0.05

struct GlobalSizes
{
    static constexpr size_t items_per_thread = 10000;
    static constexpr size_t block_size       = 8;
    static constexpr size_t items_per_block  = items_per_thread * block_size;
    static constexpr size_t grid_size        = 8;
    static constexpr size_t size             = grid_size * items_per_block;
};

template<class RocrandPRNGType>
inline void GetRocrandDeviceState(RocrandPRNGType* device_state)
{

    RocrandPRNGType* host_state
        = new RocrandPRNGType[GlobalSizes::block_size * GlobalSizes::grid_size];

    for(size_t i = 0; i < GlobalSizes::block_size * GlobalSizes::grid_size; i++)
    {
        if constexpr(std::is_same_v<RocrandPRNGType, rocrand_state_sobol32>)
        {
            const unsigned int* directions;
            ROCRAND_CHECK(
                rocrand_get_direction_vectors32(&directions, ROCRAND_DIRECTION_VECTORS_32_JOEKUO6));
            rocrand_init(directions, 123456 ^ i, host_state + i);
        }
        // scrambled sobol32 case
        else if constexpr(std::is_same_v<RocrandPRNGType, rocrand_state_scrambled_sobol32>)
        {
            const unsigned int* directions;
            ROCRAND_CHECK(
                rocrand_get_direction_vectors32(&directions, ROCRAND_DIRECTION_VECTORS_32_JOEKUO6));
            rocrand_init(directions, 123456 ^ i, 654321 ^ i, host_state + i);
        }
        // sobol64 case
        else if constexpr(std::is_same_v<RocrandPRNGType, rocrand_state_sobol64>)
        {
            const unsigned long long* directions;
            ROCRAND_CHECK(
                rocrand_get_direction_vectors64(&directions, ROCRAND_DIRECTION_VECTORS_64_JOEKUO6));
            rocrand_init(directions, 123456 ^ i, host_state + i);
        }
        // scrambled sobol64 case
        else if constexpr(std::is_same_v<RocrandPRNGType, rocrand_state_scrambled_sobol64>)
        {
            const unsigned long long* directions;
            ROCRAND_CHECK(
                rocrand_get_direction_vectors64(&directions, ROCRAND_DIRECTION_VECTORS_64_JOEKUO6));
            rocrand_init(directions, 123456 ^ i, 654321 ^ i, host_state + i);
        }
        // lfsr113 case
        else if constexpr(std::is_same_v<RocrandPRNGType, rocrand_state_lfsr113>)
        {
            rocrand_init({0xabcd, 0xdabc, 0xcdab, 0xbcda}, 0, 0, host_state + i);
        }
        else
        {
            rocrand_init(123456 ^ i, 654321 ^ i, 0, host_state + i);
        }
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(device_state + i,
                            host_state + i,
                            sizeof(RocrandPRNGType),
                            hipMemcpyHostToDevice));
    }

    delete[] host_state;
}

template<typename ReturnType, typename OutputType, class StartIt, class EndIt, class ReadMeanFunc>
ReturnType get_actual_mean(const size_t        test_size,
                           const size_t        out_size,
                           StartIt             begin,
                           EndIt               end,
                           const ReadMeanFunc& rmf)
{
    ReturnType actual_mean = std::accumulate(begin,
                                             end,
                                             (ReturnType)0,
                                             [=](ReturnType acc, OutputType x)
                                             { return acc + static_cast<ReturnType>(rmf(x)); })
                             / static_cast<ReturnType>(test_size * out_size);
    return actual_mean;
}

template<typename ReturnType, typename OutputType, class StartIt, class EndIt, class ReadStdFunc>
ReturnType get_actual_std_dev(const size_t       test_size,
                              const size_t       out_size,
                              StartIt            begin,
                              EndIt              end,
                              ReturnType         actual_mean,
                              const ReadStdFunc& rsf)
{
    ReturnType actual_std_dev
        = std::accumulate(begin,
                          end,
                          (ReturnType)0,
                          [=](ReturnType acc, OutputType x)
                          { return acc + static_cast<ReturnType>(rsf(x, actual_mean)); });
    actual_std_dev = std::sqrt(actual_std_dev / static_cast<ReturnType>(test_size * out_size - 1));
    return actual_std_dev;
}

template<typename OutputType, class RocrandPRNGType, size_t OutSize>
struct ParamsHolder
{
    using out_type                   = OutputType;
    using rng                        = RocrandPRNGType;
    static constexpr size_t out_size = OutSize;
};

using logNormalParams = ::testing::Types<ParamsHolder<float, rocrand_state_philox4x32_10, 1>,
                                         ParamsHolder<double, rocrand_state_philox4x32_10, 1>,
                                         ParamsHolder<float, rocrand_state_mrg31k3p, 1>,
                                         ParamsHolder<double, rocrand_state_mrg31k3p, 1>,
                                         ParamsHolder<float, rocrand_state_mrg32k3a, 1>,
                                         ParamsHolder<double, rocrand_state_mrg32k3a, 1>,
                                         ParamsHolder<float, rocrand_state_xorwow, 1>,
                                         ParamsHolder<double, rocrand_state_xorwow, 1>,
                                         ParamsHolder<float, rocrand_state_sobol32, 1>,
                                         ParamsHolder<float, rocrand_state_scrambled_sobol32, 1>,
                                         ParamsHolder<float, rocrand_state_sobol64, 1>,
                                         ParamsHolder<float, rocrand_state_scrambled_sobol64, 1>,
                                         ParamsHolder<float, rocrand_state_lfsr113, 1>,
                                         ParamsHolder<float, rocrand_state_threefry2x32_20, 1>,
                                         ParamsHolder<float, rocrand_state_threefry2x64_20, 1>,
                                         ParamsHolder<float, rocrand_state_threefry4x32_20, 1>,
                                         ParamsHolder<float, rocrand_state_threefry4x64_20, 1>,
                                         ParamsHolder<double, rocrand_state_sobol32, 1>,
                                         ParamsHolder<double, rocrand_state_scrambled_sobol32, 1>,
                                         ParamsHolder<double, rocrand_state_sobol64, 1>,
                                         ParamsHolder<double, rocrand_state_scrambled_sobol64, 1>,
                                         ParamsHolder<double, rocrand_state_lfsr113, 1>,
                                         ParamsHolder<double, rocrand_state_threefry2x32_20, 1>,
                                         ParamsHolder<double, rocrand_state_threefry2x64_20, 1>,
                                         ParamsHolder<double, rocrand_state_threefry4x32_20, 1>,
                                         ParamsHolder<double, rocrand_state_threefry4x64_20, 1>,
                                         ParamsHolder<float2, rocrand_state_philox4x32_10, 2>,
                                         ParamsHolder<float2, rocrand_state_mrg31k3p, 2>,
                                         ParamsHolder<float2, rocrand_state_mrg32k3a, 2>,
                                         ParamsHolder<float2, rocrand_state_xorwow, 2>,
                                         ParamsHolder<float2, rocrand_state_lfsr113, 2>,
                                         ParamsHolder<float2, rocrand_state_threefry2x32_20, 2>,
                                         ParamsHolder<float2, rocrand_state_threefry2x64_20, 2>,
                                         ParamsHolder<float2, rocrand_state_threefry4x32_20, 2>,
                                         ParamsHolder<float2, rocrand_state_threefry4x64_20, 2>,
                                         ParamsHolder<double2, rocrand_state_philox4x32_10, 2>,
                                         ParamsHolder<double2, rocrand_state_mrg31k3p, 2>,
                                         ParamsHolder<double2, rocrand_state_mrg32k3a, 2>,
                                         ParamsHolder<double2, rocrand_state_xorwow, 2>,
                                         ParamsHolder<double2, rocrand_state_lfsr113, 2>,
                                         ParamsHolder<double2, rocrand_state_threefry2x32_20, 2>,
                                         ParamsHolder<double2, rocrand_state_threefry2x64_20, 2>,
                                         ParamsHolder<double2, rocrand_state_threefry4x32_20, 2>,
                                         ParamsHolder<double2, rocrand_state_threefry4x64_20, 2>,
                                         ParamsHolder<float4, rocrand_state_philox4x32_10, 4>,
                                         ParamsHolder<double4, rocrand_state_philox4x32_10, 4>>;

template<class ParamsHolder>
class LogNormalTest : public ::testing::Test
{
public:
    using out_type                   = typename ParamsHolder::out_type;
    using prng_type                  = typename ParamsHolder::rng;
    static constexpr size_t out_size = ParamsHolder::out_size;
};

TYPED_TEST_SUITE(LogNormalTest, logNormalParams);

template<typename OutputType, typename InputType, class RocrandPRNGType, class LogNormalFunc>
void __global__ log_normal_kernel(OutputType*         device_output,
                                  const InputType     mean,
                                  const InputType     std_dev,
                                  RocrandPRNGType*    states,
                                  const LogNormalFunc lnf)
{
    const size_t offset = (GlobalSizes::items_per_block * blockIdx.x)
                          + (GlobalSizes::items_per_thread * threadIdx.x);
    const size_t state_offset = (GlobalSizes::block_size * blockIdx.x) + threadIdx.x;

    auto state = states + state_offset;
    for(size_t i = 0; i < GlobalSizes::items_per_thread; i++)
    {
        device_output[offset + i] = lnf(state, mean, std_dev);
    }
    states[state_offset] = *state;
}

template<typename OutputType,
         typename T,
         size_t OutSize,
         class RocrandPRNGType,
         class LogNormalFunc,
         class ReadMeanFunc,
         class ReadStdFunc>
void run_device_test(const LogNormalFunc lnf, const ReadMeanFunc rmf, ReadStdFunc rsf)
{
    constexpr T input_mean    = 0.5;
    constexpr T input_std_dev = 1.0;

    const T dev2 = POWF(input_std_dev, 2);

    const T expected_mean    = std::exp(input_mean + dev2 / 2);
    const T expected_std_dev = std::sqrt((std::exp(dev2) - 1) * std::exp(2 * input_mean + dev2));

    RocrandPRNGType* generators;
    HIP_CHECK(
        hipMalloc(&generators,
                  sizeof(RocrandPRNGType) * GlobalSizes::block_size * GlobalSizes::grid_size));

    GetRocrandDeviceState(generators);

    OutputType* device_output;
    HIP_CHECK(hipMalloc(&device_output, sizeof(OutputType) * GlobalSizes::size));

    log_normal_kernel<OutputType, T>
        <<<dim3(GlobalSizes::grid_size), dim3(GlobalSizes::block_size), 0, 0>>>(device_output,
                                                                                input_mean,
                                                                                input_std_dev,
                                                                                generators,
                                                                                lnf);

    std::vector<OutputType> host_output(GlobalSizes::size);

    HIP_CHECK(hipMemcpy(host_output.data(),
                        device_output,
                        sizeof(OutputType) * GlobalSizes::size,
                        hipMemcpyDeviceToHost));

    T actual_mean    = get_actual_mean<T, OutputType>(GlobalSizes::size,
                                                   OutSize,
                                                   host_output.begin(),
                                                   host_output.end(),
                                                   rmf);
    T actual_std_dev = get_actual_std_dev<T, OutputType>(GlobalSizes::size,
                                                         OutSize,
                                                         host_output.begin(),
                                                         host_output.end(),
                                                         actual_mean,
                                                         rsf);

    T mean_eps    = GET_EPS(expected_mean);
    T std_dev_eps = GET_EPS(expected_std_dev);

    ASSERT_NEAR(expected_mean, actual_mean, mean_eps);
    ASSERT_NEAR(expected_std_dev, actual_std_dev, std_dev_eps);

    HIP_CHECK(hipFree(generators));
    HIP_CHECK(hipFree(device_output));
}

template<class rocrand_state, int size>
struct log_normal
{

    template<class R>
    __device__ __host__
    auto operator()(rocrand_state* state, const R input_mean, const R input_std_dev) const
    {
        if constexpr(size == 1)
        {
            if constexpr(std::is_same_v<R, float>)
            {
                return rocrand_log_normal(state, input_mean, input_std_dev);
            }
            else
            {
                return rocrand_log_normal_double(state, input_mean, input_std_dev);
            }
        }
        else if constexpr(size == 2)
        {
            if constexpr(std::is_same_v<R, float>)
            {
                return rocrand_log_normal2(state, input_mean, input_std_dev);
            }
            else
            {
                return rocrand_log_normal_double2(state, input_mean, input_std_dev);
            }
        }
        else if constexpr(size == 4)
        {
            if constexpr(std::is_same_v<R, float>)
            {
                return rocrand_log_normal4(state, input_mean, input_std_dev);
            }
            else
            {
                return rocrand_log_normal_double4(state, input_mean, input_std_dev);
            }
        }
    }

    struct mean
    {
        template<class T>
        __device__ __host__
        auto operator()(T x) const
        {
            if constexpr(size == 1)
            {
                return x;
            }
            else if constexpr(size == 2)
            {
                return x.x + x.y;
            }
            else if constexpr(size == 4)
            {
                return x.x + x.y + x.w + x.z;
            }
        }
    };

    struct std_dev
    {
        template<class T, class R>
        __device__ __host__
        auto operator()(T x, R actual_mean) const
        {
            if constexpr(size == 1)
            {
                return POWF(x - actual_mean, 2);
            }
            else if constexpr(size == 2)
            {
                return POWF(x.x - actual_mean, 2) + POWF(x.y - actual_mean, 2);
            }
            else if constexpr(size == 4)
            {
                return POWF(x.x - actual_mean, 2) + POWF(x.y - actual_mean, 2)
                       + POWF(x.w - actual_mean, 2) + POWF(x.z - actual_mean, 2);
            }
        }
    };
};

TYPED_TEST(LogNormalTest, log_normal_device_test)
{
    using out_type            = typename TestFixture::out_type;
    using rocrand_state       = typename TestFixture::prng_type;
    constexpr size_t out_size = TestFixture::out_size;
    using T
        = std::conditional_t<(std::is_same_v<out_type, float> || std::is_same_v<out_type, float2>
                              || std::is_same_v<out_type, float4>),
                             float,
                             double>;

    using context = log_normal<rocrand_state, out_size>;
    run_device_test<out_type, T, out_size, rocrand_state>(context(),
                                                          typename context::mean(),
                                                          typename context::std_dev());
}

/* #################################################

                TEST HOST SIDE

   ###############################################*/

template<class RocrandPRNGType>
inline void GetRocrandHostState(RocrandPRNGType* host_state)
{
    if constexpr(std::is_same_v<RocrandPRNGType, rocrand_state_sobol32>)
    {
        const unsigned int* directions;
        ROCRAND_CHECK(
            rocrand_get_direction_vectors32(&directions, ROCRAND_DIRECTION_VECTORS_32_JOEKUO6));
        rocrand_init(directions, 123456, host_state);
    }
    // scrambled sobol32 case
    else if constexpr(std::is_same_v<RocrandPRNGType, rocrand_state_scrambled_sobol32>)
    {
        const unsigned int* directions;
        ROCRAND_CHECK(
            rocrand_get_direction_vectors32(&directions, ROCRAND_DIRECTION_VECTORS_32_JOEKUO6));
        rocrand_init(directions, 123456, 654321, host_state);
    }
    // sobol64 case
    else if constexpr(std::is_same_v<RocrandPRNGType, rocrand_state_sobol64>)
    {
        const unsigned long long* directions;
        ROCRAND_CHECK(
            rocrand_get_direction_vectors64(&directions, ROCRAND_DIRECTION_VECTORS_64_JOEKUO6));
        rocrand_init(directions, 123456, host_state);
    }
    // scrambled sobol64 case
    else if constexpr(std::is_same_v<RocrandPRNGType, rocrand_state_scrambled_sobol64>)
    {
        const unsigned long long* directions;
        ROCRAND_CHECK(
            rocrand_get_direction_vectors64(&directions, ROCRAND_DIRECTION_VECTORS_64_JOEKUO6));
        rocrand_init(directions, 123456, 654321, host_state);
    }
    // lfsr113 case
    else if constexpr(std::is_same_v<RocrandPRNGType, rocrand_state_lfsr113>)
    {
        rocrand_init({0xabcd, 0xdabc, 0xcdab, 0xbcda}, 0, 0, host_state);
    }
    else
    {
        rocrand_init(123456, 654321, 0, host_state);
    }
}

template<typename OutputType,
         typename T,
         size_t OutSize,
         class RocrandPRNGType,
         class LogNormalFunc,
         class ReadMeanFunc,
         class ReadStdFunc>
void run_host_test(const LogNormalFunc& lnf, const ReadMeanFunc& rmf, ReadStdFunc& rsf)
{
    constexpr size_t test_size     = 50000;
    constexpr T      input_mean    = 0.5;
    constexpr T      input_std_dev = 1.0;

    const T dev2 = POWF(input_std_dev, 2);

    const T expected_mean    = std::exp(input_mean + dev2 / 2);
    const T expected_std_dev = std::sqrt((std::exp(dev2) - 1) * std::exp(2 * input_mean + dev2));

    RocrandPRNGType generator;
    GetRocrandHostState(&generator);

    std::vector<OutputType> output(test_size);

    for(size_t i = 0; i < test_size; i++)
    {
        output[i] = lnf(&generator, input_mean, input_std_dev);
    }

    T actual_mean = std::accumulate(output.begin(),
                                    output.end(),
                                    (T)0,
                                    [=](T acc, OutputType x) { return acc + rmf(x); })
                    / static_cast<T>(test_size * OutSize);

    T actual_std_dev
        = std::accumulate(output.begin(),
                          output.end(),
                          (T)0,
                          [=](T acc, OutputType x) { return acc + rsf(x, actual_mean); });
    actual_std_dev = std::sqrt(actual_std_dev / static_cast<T>(test_size * OutSize - 1));

    T mean_eps    = GET_EPS(expected_mean);
    T std_dev_eps = GET_EPS(expected_std_dev);

    ASSERT_NEAR(expected_mean, actual_mean, mean_eps);
    ASSERT_NEAR(expected_std_dev, actual_std_dev, std_dev_eps);
}

TYPED_TEST(LogNormalTest, log_normal_host_test)
{
    using out_type            = typename TestFixture::out_type;
    using rocrand_state       = typename TestFixture::prng_type;
    constexpr size_t out_size = TestFixture::out_size;
    using T
        = std::conditional_t<(std::is_same_v<out_type, float> || std::is_same_v<out_type, float2>
                              || std::is_same_v<out_type, float4>),
                             float,
                             double>;

    if constexpr(out_size == 1)
    {
        auto mean_func = [](out_type x) { return x; };
        auto std_dev_func
            = [](out_type x, out_type actual_mean) { return POWF(x - actual_mean, 2); };
        if constexpr(std::is_same_v<out_type, float>)
        {
            run_host_test<out_type, T, out_size, rocrand_state>(
                [=](rocrand_state* state, float input_mean, float input_std_dev)
                { return rocrand_log_normal(state, input_mean, input_std_dev); },
                mean_func,
                std_dev_func);
        }
        else
        {
            run_host_test<out_type, T, out_size, rocrand_state>(
                [=](rocrand_state* state, double input_mean, double input_std_dev)
                { return rocrand_log_normal_double(state, input_mean, input_std_dev); },
                mean_func,
                std_dev_func);
        }
    }
    else if constexpr(out_size == 2)
    {
        auto mean_func    = [](out_type x) { return x.x + x.y; };
        auto std_dev_func = [](out_type x, T actual_mean)
        { return POWF(x.x - actual_mean, 2) + POWF(x.y - actual_mean, 2); };

        if constexpr(std::is_same_v<out_type, float2>)
        {
            run_host_test<out_type, T, out_size, rocrand_state>(
                [=](rocrand_state* state, float input_mean, float input_std_dev)
                { return rocrand_log_normal2(state, input_mean, input_std_dev); },
                mean_func,
                std_dev_func);
        }
        else
        {
            run_host_test<out_type, T, out_size, rocrand_state>(
                [=](rocrand_state* state, double input_mean, double input_std_dev)
                { return rocrand_log_normal_double2(state, input_mean, input_std_dev); },
                mean_func,
                std_dev_func);
        }
    }
    else
    {
        auto mean_func    = [](out_type x) { return x.x + x.y + x.w + x.z; };
        auto std_dev_func = [](out_type x, T actual_mean)
        {
            return POWF(x.x - actual_mean, 2) + POWF(x.y - actual_mean, 2)
                   + POWF(x.w - actual_mean, 2) + POWF(x.z - actual_mean, 2);
        };

        if constexpr(std::is_same_v<out_type, float4>)
        {
            run_host_test<out_type, T, out_size, rocrand_state>(
                [=](rocrand_state* state, float input_mean, float input_std_dev)
                { return rocrand_log_normal4(state, input_mean, input_std_dev); },
                mean_func,
                std_dev_func);
        }
        else
        {
            run_host_test<out_type, T, out_size, rocrand_state>(
                [=](rocrand_state* state, double input_mean, double input_std_dev)
                { return rocrand_log_normal_double4(state, input_mean, input_std_dev); },
                mean_func,
                std_dev_func);
        }
    }
}
