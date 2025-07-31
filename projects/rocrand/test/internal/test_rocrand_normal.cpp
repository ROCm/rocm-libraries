// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../test_common.hpp"

#include <cmath>
#include <numeric>
#include <random>

#undef ROCRAND_DETAIL_BM_NOT_IN_STATE
#include <rocrand/rocrand_normal.h>

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

// If x is small then get withing 0.001 otherwise 10%
#define GET_EPS(x) x < 0.01 ? 0.01 : x * 0.1

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
struct StateParams
{
    using out_type                   = OutputType;
    using rng                        = RocrandPRNGType;
    static constexpr size_t out_size = OutSize;
};

using NormalDistributionStateParam
    = ::testing::Types<StateParams<float, rocrand_state_philox4x32_10, 1>,
                       StateParams<double, rocrand_state_philox4x32_10, 1>,
                       StateParams<float, rocrand_state_mrg31k3p, 1>,
                       StateParams<double, rocrand_state_mrg31k3p, 1>,
                       StateParams<float, rocrand_state_mrg32k3a, 1>,
                       StateParams<double, rocrand_state_mrg32k3a, 1>,
                       StateParams<float, rocrand_state_xorwow, 1>,
                       StateParams<double, rocrand_state_xorwow, 1>,
                       StateParams<float, rocrand_state_sobol32, 1>,
                       StateParams<float, rocrand_state_scrambled_sobol32, 1>,
                       StateParams<float, rocrand_state_sobol64, 1>,
                       StateParams<float, rocrand_state_scrambled_sobol64, 1>,
                       StateParams<float, rocrand_state_lfsr113, 1>,
                       StateParams<float, rocrand_state_threefry2x32_20, 1>,
                       StateParams<float, rocrand_state_threefry2x64_20, 1>,
                       StateParams<float, rocrand_state_threefry4x32_20, 1>,
                       StateParams<float, rocrand_state_threefry4x64_20, 1>,
                       StateParams<double, rocrand_state_sobol32, 1>,
                       StateParams<double, rocrand_state_scrambled_sobol32, 1>,
                       StateParams<double, rocrand_state_sobol64, 1>,
                       StateParams<double, rocrand_state_scrambled_sobol64, 1>,
                       StateParams<double, rocrand_state_lfsr113, 1>,
                       StateParams<double, rocrand_state_threefry2x32_20, 1>,
                       StateParams<double, rocrand_state_threefry2x64_20, 1>,
                       StateParams<double, rocrand_state_threefry4x32_20, 1>,
                       StateParams<double, rocrand_state_threefry4x64_20, 1>,
                       StateParams<float2, rocrand_state_philox4x32_10, 2>,
                       StateParams<float2, rocrand_state_mrg31k3p, 2>,
                       StateParams<float2, rocrand_state_mrg32k3a, 2>,
                       StateParams<float2, rocrand_state_xorwow, 2>,
                       StateParams<float2, rocrand_state_lfsr113, 2>,
                       StateParams<float2, rocrand_state_threefry2x32_20, 2>,
                       StateParams<float2, rocrand_state_threefry2x64_20, 2>,
                       StateParams<float2, rocrand_state_threefry4x32_20, 2>,
                       StateParams<float2, rocrand_state_threefry4x64_20, 2>,
                       StateParams<double2, rocrand_state_philox4x32_10, 2>,
                       StateParams<double2, rocrand_state_mrg31k3p, 2>,
                       StateParams<double2, rocrand_state_mrg32k3a, 2>,
                       StateParams<double2, rocrand_state_xorwow, 2>,
                       StateParams<double2, rocrand_state_lfsr113, 2>,
                       StateParams<double2, rocrand_state_threefry2x32_20, 2>,
                       StateParams<double2, rocrand_state_threefry2x64_20, 2>,
                       StateParams<double2, rocrand_state_threefry4x32_20, 2>,
                       StateParams<double2, rocrand_state_threefry4x64_20, 2>,
                       StateParams<float4, rocrand_state_philox4x32_10, 4>,
                       StateParams<double4, rocrand_state_philox4x32_10, 4>>;

template<class StateParams>
class NormalDistributionRocRandStateTest : public ::testing::Test
{
public:
    using out_type                   = typename StateParams::out_type;
    using prng_type                  = typename StateParams::rng;
    static constexpr size_t out_size = StateParams::out_size;
};
TYPED_TEST_SUITE(NormalDistributionRocRandStateTest, NormalDistributionStateParam);

/* #################################################

                TEST HOST SIDE

   ###############################################*/

template<class RocrandPRNGType>
inline void GetHostRocrandState(RocrandPRNGType* host_state)
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
         class NormalDistFunc,
         class ReadMeanFunc,
         class ReadStdFunc>
void run_host_prng_test(const NormalDistFunc& ndf, const ReadMeanFunc& rmf, ReadStdFunc& rsf)
{
    constexpr size_t test_size        = 100000;
    const T          expected_mean    = 0;
    const T          expected_std_dev = 1;

    RocrandPRNGType generator;
    GetHostRocrandState(&generator);

    std::vector<OutputType> output(test_size);

    for(size_t i = 0; i < test_size; i++)
    {
        output[i] = ndf(&generator);
    }

    T actual_mean
        = get_actual_mean<T, OutputType>(test_size, OutSize, output.begin(), output.end(), rmf);
    T actual_std_dev = get_actual_std_dev<T, OutputType>(test_size,
                                                         OutSize,
                                                         output.begin(),
                                                         output.end(),
                                                         actual_mean,
                                                         rsf);
    T mean_eps       = GET_EPS(expected_mean);
    T std_dev_eps    = GET_EPS(expected_std_dev);

    ASSERT_NEAR(expected_mean, actual_mean, mean_eps);
    ASSERT_NEAR(expected_std_dev, actual_std_dev, std_dev_eps);
}

template<typename OutputType,
         typename InputType,
         size_t OutSize,
         class NormalDistFunc,
         class ReadMeanFunc,
         class ReadStdFunc>
void run_host_numeric_test(const NormalDistFunc& ndf, const ReadMeanFunc& rmf, ReadStdFunc& rsf)
{
    constexpr size_t test_size        = 100000;
    const double     expected_mean    = 0;
    const double     expected_std_dev = 1;

    std::vector<OutputType> output(test_size);

    std::random_device                       rd;
    std::mt19937                             gen(rd());
    std::uniform_int_distribution<InputType> dis(std::numeric_limits<InputType>::min(),
                                                 std::numeric_limits<InputType>::max());

    for(size_t i = 0; i < test_size; i++)
    {
        output[i] = ndf(dis, gen);
    }

    double actual_mean    = get_actual_mean<double, OutputType>(test_size,
                                                             OutSize,
                                                             output.begin(),
                                                             output.end(),
                                                             rmf);
    double actual_std_dev = get_actual_std_dev<double, OutputType>(test_size,
                                                                   OutSize,
                                                                   output.begin(),
                                                                   output.end(),
                                                                   actual_mean,
                                                                   rsf);
    double mean_eps       = GET_EPS(expected_mean);
    double std_dev_eps    = GET_EPS(expected_std_dev);

    ASSERT_NEAR(expected_mean, actual_mean, mean_eps);
    ASSERT_NEAR(expected_std_dev, actual_std_dev, std_dev_eps);
}

TYPED_TEST(NormalDistributionRocRandStateTest, rocrand_host_state_tests)
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
            run_host_prng_test<out_type, T, out_size, rocrand_state>(
                [=](rocrand_state* state) { return rocrand_normal(state); },
                mean_func,
                std_dev_func);
        }
        else
        {
            run_host_prng_test<out_type, T, out_size, rocrand_state>(
                [=](rocrand_state* state) { return rocrand_normal_double(state); },
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
            run_host_prng_test<out_type, T, out_size, rocrand_state>(
                [=](rocrand_state* state) { return rocrand_normal2(state); },
                mean_func,
                std_dev_func);
        }
        else
        {
            run_host_prng_test<out_type, T, out_size, rocrand_state>(
                [=](rocrand_state* state) { return rocrand_normal_double2(state); },
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
            run_host_prng_test<out_type, T, out_size, rocrand_state>(
                [=](rocrand_state* state) { return rocrand_normal4(state); },
                mean_func,
                std_dev_func);
        }
        else
        {
            run_host_prng_test<out_type, T, out_size, rocrand_state>(
                [=](rocrand_state* state) { return rocrand_normal_double4(state); },
                mean_func,
                std_dev_func);
        }
    }
}

TEST(NormalDistributionRocRandNumericTest, rocrand_host_numeric_uint_in_float_out_test)
{
    using OutputType            = float;
    using InputType             = unsigned int;
    constexpr size_t OutputSize = 1;

    auto mean_func = [](OutputType x) { return x; };
    auto std_dev_func = [](OutputType x, double actual_mean) { return POWF(x - actual_mean, 2); };

    run_host_numeric_test<OutputType, InputType, OutputSize>(
        [=](std::uniform_int_distribution<InputType>& dis, std::mt19937& gen)
        { return rocrand_device::detail::normal_distribution(dis(gen)); },
        mean_func,
        std_dev_func);
}

TEST(NormalDistributionRocRandNumericTest, rocrand_host_numeric_ullint_in_float_out_test)
{
    using OutputType            = float;
    using InputType             = unsigned long long int;
    constexpr size_t OutputSize = 1;

    auto mean_func = [](OutputType x) { return x; };
    auto std_dev_func = [](OutputType x, double actual_mean) { return POWF(x - actual_mean, 2); };

    run_host_numeric_test<OutputType, InputType, OutputSize>(
        [=](std::uniform_int_distribution<InputType>& dis, std::mt19937& gen)
        { return rocrand_device::detail::normal_distribution(dis(gen)); },
        mean_func,
        std_dev_func);
}

TEST(NormalDistributionRocRandNumericTest, rocrand_host_numeric_2uint_in_float2_out_test)
{
    using OutputType            = float2;
    using InputType             = unsigned int;
    constexpr size_t OutputSize = 2;

    auto mean_func    = [](OutputType x) { return x.x + x.y; };
    auto std_dev_func = [](OutputType x, double actual_mean)
    { return POWF(x.x - actual_mean, 2) + POWF(x.y - actual_mean, 2); };

    run_host_numeric_test<OutputType, InputType, OutputSize>(
        [=](std::uniform_int_distribution<InputType>& dis, std::mt19937& gen)
        { return rocrand_device::detail::normal_distribution2(dis(gen), dis(gen)); },
        mean_func,
        std_dev_func);
}

TEST(NormalDistributionRocRandNumericTest, rocrand_host_numeric_uint2_in_float2_out_test)
{
    using OutputType            = float2;
    using InputType             = unsigned int;
    constexpr size_t OutputSize = 2;

    auto mean_func    = [](OutputType x) { return x.x + x.y; };
    auto std_dev_func = [](OutputType x, double actual_mean)
    { return POWF(x.x - actual_mean, 2) + POWF(x.y - actual_mean, 2); };

    run_host_numeric_test<OutputType, InputType, OutputSize>(
        [=](std::uniform_int_distribution<InputType>& dis, std::mt19937& gen) {
            return rocrand_device::detail::normal_distribution2(uint2{dis(gen), dis(gen)});
        },
        mean_func,
        std_dev_func);
}

TEST(NormalDistributionRocRandNumericTest, rocrand_host_numeric_ull_in_float2_out_test)
{
    using OutputType            = float2;
    using InputType             = unsigned long long;
    constexpr size_t OutputSize = 2;

    auto mean_func    = [](OutputType x) { return x.x + x.y; };
    auto std_dev_func = [](OutputType x, double actual_mean)
    { return POWF(x.x - actual_mean, 2) + POWF(x.y - actual_mean, 2); };

    run_host_numeric_test<OutputType, InputType, OutputSize>(
        [=](std::uniform_int_distribution<InputType>& dis, std::mt19937& gen)
        { return rocrand_device::detail::normal_distribution2(dis(gen)); },
        mean_func,
        std_dev_func);
}

TEST(NormalDistributionRocRandNumericTest, rocrand_host_numeric_uint4_in_float4_out_test)
{
    using OutputType            = float4;
    using InputType             = unsigned int;
    constexpr size_t OutputSize = 4;

    auto mean_func    = [](OutputType x) { return x.w + x.x + x.y + x.z; };
    auto std_dev_func = [](OutputType x, double actual_mean)
    {
        return POWF(x.w - actual_mean, 2) + POWF(x.x - actual_mean, 2) + POWF(x.y - actual_mean, 2)
               + POWF(x.z - actual_mean, 2);
    };

    run_host_numeric_test<OutputType, InputType, OutputSize>(
        [=](std::uniform_int_distribution<InputType>& dis, std::mt19937& gen)
        {
            return rocrand_device::detail::normal_distribution4(
                uint4{dis(gen), dis(gen), dis(gen), dis(gen)});
        },
        mean_func,
        std_dev_func);
}

TEST(NormalDistributionRocRandNumericTest, rocrand_host_numeric_longlong2_in_float4_out_test)
{
    using OutputType            = float4;
    using InputType             = long long;
    constexpr size_t OutputSize = 4;

    auto mean_func    = [](OutputType x) { return x.w + x.x + x.y + x.z; };
    auto std_dev_func = [](OutputType x, double actual_mean)
    {
        return POWF(x.w - actual_mean, 2) + POWF(x.x - actual_mean, 2) + POWF(x.y - actual_mean, 2)
               + POWF(x.z - actual_mean, 2);
    };

    run_host_numeric_test<OutputType, InputType, OutputSize>(
        [=](std::uniform_int_distribution<InputType>& dis, std::mt19937& gen) {
            return rocrand_device::detail::normal_distribution4(longlong2{dis(gen), dis(gen)});
        },
        mean_func,
        std_dev_func);
}

TEST(NormalDistributionRocRandNumericTest, rocrand_host_numeric_2ull_in_float4_out_test)
{
    using OutputType            = float4;
    using InputType             = unsigned long long;
    constexpr size_t OutputSize = 4;

    auto mean_func    = [](OutputType x) { return x.w + x.x + x.y + x.z; };
    auto std_dev_func = [](OutputType x, double actual_mean)
    {
        return POWF(x.w - actual_mean, 2) + POWF(x.x - actual_mean, 2) + POWF(x.y - actual_mean, 2)
               + POWF(x.z - actual_mean, 2);
    };

    run_host_numeric_test<OutputType, InputType, OutputSize>(
        [=](std::uniform_int_distribution<InputType>& dis, std::mt19937& gen)
        { return rocrand_device::detail::normal_distribution4(dis(gen), dis(gen)); },
        mean_func,
        std_dev_func);
}

TEST(NormalDistributionRocRandNumericTest, rocrand_host_numeric_uint_in_half2_out_test)
{
    using OutputType            = __half2;
    using InputType             = unsigned int;
    constexpr size_t OutputSize = 2;

    auto mean_func = [](OutputType x) { return static_cast<float>(x.x) + static_cast<float>(x.y); };
    auto std_dev_func = [](OutputType x, double actual_mean)
    {
        float f = static_cast<float>(x.x) - actual_mean;
        float s = static_cast<float>(x.y) - actual_mean;
        return (f * f) + (s * s);
    };

    run_host_numeric_test<OutputType, InputType, OutputSize>(
        [=](std::uniform_int_distribution<InputType>& dis, std::mt19937& gen)
        { return rocrand_device::detail::normal_distribution_half2(dis(gen)); },
        mean_func,
        std_dev_func);
}

TEST(NormalDistributionRocRandNumericTest, rocrand_host_numeric_ull_in_half2_out_test)
{
    using OutputType            = __half2;
    using InputType             = unsigned long long;
    constexpr size_t OutputSize = 2;

    auto mean_func = [](OutputType x) { return static_cast<float>(x.x) + static_cast<float>(x.y); };
    auto std_dev_func = [](OutputType x, double actual_mean)
    {
        float f = static_cast<float>(x.x) - actual_mean;
        float s = static_cast<float>(x.y) - actual_mean;
        return (f * f) + (s * s);
    };

    run_host_numeric_test<OutputType, InputType, OutputSize>(
        [=](std::uniform_int_distribution<InputType>& dis, std::mt19937& gen)
        { return rocrand_device::detail::normal_distribution_half2(dis(gen)); },
        mean_func,
        std_dev_func);
}

/* #################################################

                TEST DEVICE SIDE

   ###############################################*/

struct GlobalSizes
{
    static constexpr size_t items_per_thread = 50000;
    static constexpr size_t block_size       = 16; // Number of threads
    static constexpr size_t items_per_block  = items_per_thread * block_size;
    static constexpr size_t grid_size        = 16; // Number of blocks
    static constexpr size_t size             = grid_size * items_per_block;
};

template<class RocrandPRNGType>
inline void GetDeviceRocrandState(RocrandPRNGType* device_prngs)
{
    // Initialize for device code rocrand state. Each thread will get 1 "state"
    // Assumed that device_prngs is already initialized

    std::vector<RocrandPRNGType> host_states(GlobalSizes::block_size * GlobalSizes::grid_size);

    for(size_t bi = 0; bi < GlobalSizes::grid_size; bi++)
    {
        for(size_t ti = 0; ti < GlobalSizes::block_size; ti++)
        {
            const size_t offset = bi * GlobalSizes::block_size;
            const size_t prng_offset
                = (GlobalSizes::items_per_block * bi) + (GlobalSizes::items_per_thread * ti);

            if constexpr(std::is_same_v<RocrandPRNGType, rocrand_state_sobol32>)
            {
                const unsigned int* directions;
                ROCRAND_CHECK(
                    rocrand_get_direction_vectors32(&directions,
                                                    ROCRAND_DIRECTION_VECTORS_32_JOEKUO6));

                rocrand_init(directions, prng_offset, &host_states[offset + ti]);
            }
            // scrambled sobol32 case
            else if constexpr(std::is_same_v<RocrandPRNGType, rocrand_state_scrambled_sobol32>)
            {
                const unsigned int* directions;
                ROCRAND_CHECK(
                    rocrand_get_direction_vectors32(&directions,
                                                    ROCRAND_DIRECTION_VECTORS_32_JOEKUO6));
                rocrand_init(directions, 123456, prng_offset, &host_states[offset + ti]);
            }
            // sobol64 case
            else if constexpr(std::is_same_v<RocrandPRNGType, rocrand_state_sobol64>)
            {
                const unsigned long long* directions;
                ROCRAND_CHECK(
                    rocrand_get_direction_vectors64(&directions,
                                                    ROCRAND_DIRECTION_VECTORS_64_JOEKUO6));
                rocrand_init(directions, prng_offset, &host_states[offset + ti]);
            }
            // scrambled sobol64 case
            else if constexpr(std::is_same_v<RocrandPRNGType, rocrand_state_scrambled_sobol64>)
            {
                const unsigned long long* directions;
                ROCRAND_CHECK(
                    rocrand_get_direction_vectors64(&directions,
                                                    ROCRAND_DIRECTION_VECTORS_64_JOEKUO6));
                rocrand_init(directions, 123456, prng_offset, &host_states[offset + ti]);
            }
            // lfsr113 case
            else if constexpr(std::is_same_v<RocrandPRNGType, rocrand_state_lfsr113>)
            {
                rocrand_init({0xabcd, 0xdabc, 0xcdab, 0xbcda},
                             0,
                             prng_offset,
                             &host_states[offset + ti]);
            }
            else
            {
                rocrand_init(123456, 654321, prng_offset, &host_states[offset + ti]);
            }
        }
    }

    HIP_CHECK(hipMemcpy(device_prngs,
                        host_states.data(),
                        sizeof(RocrandPRNGType) * GlobalSizes::grid_size * GlobalSizes::block_size,
                        hipMemcpyHostToDevice));
}

template<typename OutType, class RocRandPrngType, class GenFunc>
__global__
void normal_distribution_kernel(OutType*         device_output,
                                RocRandPrngType* device_prngs,
                                const GenFunc    gf)
{
    const size_t offset = (GlobalSizes::items_per_block * blockIdx.x)
                          + (GlobalSizes::items_per_thread * threadIdx.x);
    const size_t prng_offset = (GlobalSizes::block_size * blockIdx.x) + threadIdx.x;

    auto prng = device_prngs + prng_offset;
    for(size_t i = 0; i < GlobalSizes::items_per_thread; i++)
    {
        device_output[offset + i] = gf(prng);
    }

    device_prngs[prng_offset] = *prng;
}

template<typename T,
         typename OutputType,
         class RocRandPrngType,
         class GenFunc,
         class ReadMeanFunc,
         class ReadStdFunc>
void run_device_prng_test(const GenFunc      gf,
                          const ReadMeanFunc rmf,
                          const ReadStdFunc  rsf,
                          const size_t       out_size)
{
    RocRandPrngType* prngs;
    HIP_CHECK(
        hipMalloc(&prngs,
                  sizeof(RocRandPrngType) * GlobalSizes::block_size * GlobalSizes::grid_size));

    GetDeviceRocrandState(prngs);

    std::vector<OutputType> host_output(GlobalSizes::size);

    OutputType* device_output;
    HIP_CHECK(hipMalloc(&device_output, sizeof(OutputType) * GlobalSizes::size));

    hipLaunchKernelGGL(HIP_KERNEL_NAME(normal_distribution_kernel<OutputType>),
                       dim3(GlobalSizes::grid_size),
                       dim3(GlobalSizes::block_size),
                       0,
                       0,
                       device_output,
                       prngs,
                       gf);
    HIP_CHECK(hipMemcpy(host_output.data(),
                        device_output,
                        sizeof(OutputType) * GlobalSizes::size,
                        hipMemcpyDeviceToHost));

    const T expected_mean    = 0;
    const T expected_std_dev = 1;

    T actual_mean    = get_actual_mean<T, OutputType>(GlobalSizes::size,
                                                   out_size,
                                                   host_output.begin(),
                                                   host_output.end(),
                                                   rmf);
    T actual_std_dev = get_actual_std_dev<T, OutputType>(GlobalSizes::size,
                                                         out_size,
                                                         host_output.begin(),
                                                         host_output.end(),
                                                         actual_mean,
                                                         rsf);
    T mean_eps       = GET_EPS(expected_mean);
    T std_dev_eps    = GET_EPS(expected_std_dev);

    ASSERT_NEAR(expected_mean, actual_mean, mean_eps);
    ASSERT_NEAR(expected_std_dev, actual_std_dev, std_dev_eps);

    HIP_CHECK(hipFree(prngs));
    HIP_CHECK(hipFree(device_output));
}

template<class rocrand_state, int size, class out_type>
struct normal
{

    __device__ __host__
    auto operator()(rocrand_state* state) const
    {
        if constexpr(size == 1)
        {
            if constexpr(std::is_same_v<out_type, float>)
            {
                return rocrand_normal(state);
            }
            else
            {
                return rocrand_normal_double(state);
            }
        }
        else if constexpr(size == 2)
        {
            if constexpr(std::is_same_v<out_type, float>)
            {
                return rocrand_normal2(state);
            }
            else
            {
                return rocrand_normal_double2(state);
            }
        }
        else if constexpr(size == 4)
        {
            if constexpr(std::is_same_v<out_type, float>)
            {
                return rocrand_normal4(state);
            }
            else
            {
                return rocrand_normal_double4(state);
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
TYPED_TEST(NormalDistributionRocRandStateTest, rocrand_device_state_tests)
{
    using out_type            = typename TestFixture::out_type;
    using rocrand_state       = typename TestFixture::prng_type;
    constexpr size_t out_size = TestFixture::out_size;
    using T
        = std::conditional_t<(std::is_same_v<out_type, float> || std::is_same_v<out_type, float2>
                              || std::is_same_v<out_type, float4>),
                             float,
                             double>;
    using context = normal<rocrand_state, out_size, T>;

    run_device_prng_test<T, out_type, rocrand_state>(context{},
                                                     typename context::mean{},
                                                     typename context::std_dev{},
                                                     out_size);
}
